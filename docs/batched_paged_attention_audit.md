# Batched Paged Decode Attention Audit

Audit date: 2026-07-15
Based on: `paged_attention_gpu.py`, `paged_attention_ref.py`, `backend.py`, `benchmark_paged_attention.py`, `test_paged_attention_gpu.py`, `model_runner/base.py`, `model_runner/qwen_runner.py`, `cache/block_table.py`, `cache/pool.py`, `cache/cache_read.py`

## 1. Is the Triton decode kernel a true batched kernel or a Python loop?

**Verdict: True batched kernel (single kernel launch).**

The kernel `_paged_decode_kernel` (paged_attention_gpu.py:162) uses a 2D Triton program grid:

```python
grid = (total_decode, num_q_heads)
_paged_decode_kernel[grid](...)
```

Each program instance handles one `(seq_idx, head_idx)` pair:

```python
pid_seq = tl.program_id(0)  # sequence index (batch dimension)
pid_head = tl.program_id(1)  # head index
kv_head = pid_head // REPEATS  # GQA mapping
```

There is **no Python for-loop over the batch dimension**. The batching is entirely within the Triton kernel launch — `N × H` programs are dispatched in a single grid, and the GPU scheduler distributes them across SMs.

The input tensors are also correctly batched:
- `query: [total_decode, num_q_heads, head_dim]` — all sequences
- `block_table: [total_decode, max_blocks_per_seq]` — all block tables
- `kv_len_after: [total_decode]` — all context lengths
- `output: [total_decode, num_q_heads, head_dim]` — all outputs

**The reference path (`AttentionBackendRef.decode_attention`)** uses a Python for-loop (`for i in range(num_decode):`) — this is the correctness oracle, not the performance path. The benchmark's `ref_decode()` also loops, as expected for a reference.

## 2. Kernel characteristics

| Property | Value |
|----------|-------|
| Program grid | 2D: `(total_decode, num_q_heads)` |
| Per-program work | 1 seq × 1 head × iterate KV blocks |
| Online softmax | FP32 accumulation |
| GQA support | Via `REPEATS = NUM_Q_HEADS // NUM_KV_HEADS` |
| Block iteration | Triton `for` loop over blocks (known at compile-time) |
| Position iteration | Triton `for` loop over positions within each block |

## 3. How the kernel is called (call chain)

```
EngineCore.step()
  → ModelInput (contains decode_block_tables, kv_len_after)
  → QwenModelRunner.execute_model()
    → For each layer:
      → write_kv_cache()  (方案B: write-first)
      → AttentionBackendGPU.decode_attention()
        → triton_decode_attention()
          → _paged_decode_kernel[grid](...)   ← single launch
```

The model runner constructs `decode_meta` with 0-based `seq_indices` via `_build_decode_meta()` (qwen_runner.py:213), ensuring correct indexing into `decode_block_tables`.

## 4. Existing test coverage (test_paged_attention_gpu.py)

`TestDecodeAttention` has 9 tests:

| Test | Batch | Covers |
|------|-------|--------|
| `test_single_sequence` | 1 | Single seq, short context |
| `test_multi_sequence` | 2 | Two sequences |
| `test_different_lengths` | 2 | Ragged lengths (3 vs 23) |
| `test_partial_last_block` | 1 | Partial final block |
| `test_multiple_full_blocks` | 1 | 40 tokens (2.5 blocks) |
| `test_noncontiguous_block_table` | 1 | Non-contiguous physical blocks |
| `test_kv_len_zero_raises` | 1 | Error handling |
| `test_padding_blocks_ignored` | 1 | -1 padding in block_table |
| `test_long_sequence_large_cache` | 1 | 50 tokens (3+ blocks) |

### Coverage gaps

1. **No batch > 2 testing** — only batch 1 and 2 are tested
2. **No Qwen2.5-0.5B exact config** — uses 4 Q heads / 2 KV heads; Qwen has 8/2 (GQA=4)
3. **No large ragged batch** — all sequences with different context lengths (3+, 5+, 8 at once)
4. **Non-contiguous blocks only tested for batch 1** — `test_noncontiguous_block_table` is single-sequence
5. **Cross-block-boundary KV data** — no explicit test where KV data spans a block boundary with more than 2 blocks
6. **No combined ragged + non-contiguous** — worst-case scenario

## 5. Current benchmark coverage (benchmark_paged_attention.py)

| Property | Current | Missing |
|----------|---------|---------|
| Batch sizes | 1, 2, 4, 8, 16 | — |
| Context lengths | 16-1024 (powers of 2) | — |
| Same-length batch | All seqs have same ctx_length | **Ragged batch** (seqs with different lengths) |
| Contiguous blocks | Blocks are contiguous per seq | **Non-contiguous blocks** |
| Output dir | `benchmark_results/` | `benchmark_results/batched_paged_attention/` |

## 6. Recommendation

Since the kernel is already a true batched kernel:

1. **No rewrite needed** — Part 2 is skipped
2. **Enhance test coverage** — Add ragged batch, larger batches, combined ragged+noncontiguous
3. **Enhance benchmark** — Add ragged batch mode, non-contiguous block mode, output to dedicated directory
4. **Run & verify** — pytest + smoke benchmark

---

## Appendix A: Real Execution Chain Audit (Continuous Batching)

### A1. Scheduler: does it batch multiple decode requests in one step?

**Yes.** `Scheduler.schedule()` (scheduler.py:79) collects ALL running decode sequences into `result.scheduled_decode_groups` in Phase 2, then deducts their token budget in Phase 3. All decode groups are returned in a single step — no per-request loop.

### A2. EngineCore: does it execute one batched Model forward?

**Yes.** `EngineCore.step()` (engine_core.py:66) calls:

```python
model_input = self._input_builder.build(
    prefill_seqs=only_prefill_seqs,
    decode_seqs=decode_seqs,
)
model_output = self._executor.execute(model_input)
```

This is a **single** `execute()` call with ALL sequences packed into one `ModelInput`. No per-sequence loop.

### A3. ModelInput: does it contain batched tensors?

**Yes.** `ModelInputBuilder.build()` (input_builder.py:48) produces:

- `input_ids: [total_tokens]` — concatenated prefill + decode token IDs
- `positions: [total_tokens]` — absolute positions for each token
- `slot_mapping: [total_tokens]` — physical KV cache slots
- `decode_block_tables: [num_decode, max_blocks]` — padded block tables per sequence
- `attention_metadata` — with `AttentionGroup` per decode/prefill group, including `kv_len_after` per sequence

All tensors are 1D flat (not per-sequence lists), ready for batched GPU execution.

### A4. ModelRunner: does it process ALL sequences in one forward pass?

**Yes.** `QwenModelRunner.execute_model()` (qwen_runner.py:116) runs ONE layer loop for ALL tokens:

```python
for layer_idx in range(self.model.num_layers):
    # QKV projection: one call for all tokens
    q, k, v = layer.attention.qkv_proj(normed)
    # RoPE: one call for all tokens
    q = self.rope(q, positions)
    k = self.rope(k, positions)
    # Cache write: one call for all tokens
    self._attention_backend.write_kv_cache(layer_idx, k, v, slot_mapping)
    # Decode attention: one call for all decode tokens
    dec_result = self._attention_backend.decode_attention(layer_idx, dec_q, decode_meta, self._pool)
```

No per-sequence Python loop. All ops operate on flat or batched tensors.

### A5. Two execution paths: legacy vs paged

| Layer | Legacy (`executor_type="qwen"`) | Paged (`executor_type="paged"`) |
|-------|-------------------------------|-------------------------------|
| Executor | `QwenExecutor` — per‑seq HF `past_key_values` | `PagedExecutor` — unified `execute()` |
| Attention | HF Transformers SDPA | `AttentionBackend` (ref or triton) |
| KV Cache | HF `past_key_values` (per‑seq tuple) | `KVCachePool` (paged, all seqs) |
| Warm‑up | HF model forward | Triton kernel JIT |
| Batch | Python `for seq in sequences:` loop | Single batched forward pass |

The `continuous_batching.py` benchmark uses `executor_type="qwen"` by default, which is the **legacy per‑sequence path**. To benchmark paged attention, it must use `executor_type="paged"`.

### A6. Attention backend switching

`Config.attention_backend` (config.py:76) supports `"reference"` and `"triton"`. `PagedExecutor.__init__()` reads this config and creates the correct backend:

```python
self._attention_backend = AttentionBackend.create(
    model_config, backend=config.attention_backend,
)
```

No code changes needed to enable A/B switching — only the config value changes.

### A7. MetricsCollector coverage

`MetricsCollector` (metrics.py:28) already tracks:

- **TTFT** (Time To First Token) — avg/P50/P95 in ms
- **TPOT** (Time Per Output Token / inter-token latency) — avg/P50/P95 in ms
- **E2E latency** — per‑request wall time
- **Throughput** — req/s and tok/s (wall-clock and active)
- **KV utilisation** — peak/avg block usage
- **Effective batch size** — mean/max per step
- **Scheduler latency** — per‑step overhead

Missing for A/B: GPU memory tracking (`torch.cuda.max_memory_allocated/reserved`), decode step count separate from total steps.

### A8. Summary

| Question | Answer |
|----------|--------|
| Scheduler batches decode? | Yes — all decode in one `schedule()` call |
| Model forward is batched? | Yes — `QwenModelRunner.execute_model()` is single forward pass |
| Input tensors are batched? | Yes — flat `[total_tokens]` tensors |
| Supports backend switching? | Yes — `Config.attention_backend` |
| Legacy `QwenExecutor` is also batched? | No — per‑seq loop via HF `past_key_values` |
| Benchmark uses paged path? | **No** — defaults to legacy `executor_type="qwen"` |
