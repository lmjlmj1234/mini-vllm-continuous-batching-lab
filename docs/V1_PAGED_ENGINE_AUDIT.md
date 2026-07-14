# V1 Paged Engine — Current Implementation Audit

> **Phase A output.** Audit of the current mini-vLLM codebase before V1 Paged Engine changes.
> Generated: 2026-07-13

---

## 1. Working Tree Status

| Item | Value |
|------|-------|
| Branch | `feature/v1-style-real-paged-attention` (new, based on `main`) |
| Staged changes | 48 files — Docker, benchmark_results (42 files), benchmarks/, tests/test_benchmark.py |
| Unstaged changes | 11 files — config, scheduler, engine_core, qwen_executor, metrics, worker, examples, pyproject |
| Tests | **214 passed** (45s), 0 failures, 1 warning (pynvml deprecation) |
| Git log | 3 commits: `bf70c39` (core), `5338f1e` (serving), `4c35c37` (docs+profiling) |

### Uncommitted Changes Summary

- `scheduler/scheduler.py`: Added scheduler trace (enable_trace, get_and_clear_trace, _record_trace), static_batch_mode support
- `engine/engine_core.py`: Wires trace from config, passes effective_batch_size/running_count/waiting_count to metrics
- `engine/metrics.py`: Records effective_batch_size, running_count, waiting_count
- `executor/qwen_executor.py`: Added `model_path` parameter for local model (local_files_only=True), removed Chinese comments
- `config.py`: Added `model_path`, `trace_enabled`, `static_batch_mode` fields
- `examples/benchmark.py`: Full benchmark CLI with 3 modes (serial/static/continuous), 36-experiment runner
- `tests/test_benchmark.py`: 241-line test suite for benchmark CLI

---

## 2. Local Model Configuration

**Model**: `Qwen2.5-0.5B-Instruct` (Qwen2ForCausalLM)
**Local path**: `/mnt/e/vllm_awq_qwen_exp/models/Qwen2.5-0.5B-Instruct`

| Parameter | Value |
|-----------|-------|
| model_type | `qwen2` |
| num_hidden_layers | 24 |
| hidden_size | 896 |
| num_attention_heads (Q heads) | 14 |
| num_key_value_heads (KV heads) | 2 |
| head_dim | 64 (896 / 14) |
| intermediate_size | 4864 |
| vocab_size | 151936 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1,000,000.0 |
| max_position_embeddings | 32768 |
| sliding_window | 32768 (disabled via use_sliding_window=false) |
| tie_word_embeddings | True |
| torch_dtype | bfloat16 |
| hidden_act | silu |
| attention_dropout | 0.0 |
| transformers_version | 4.43.1 |
| architecture | Qwen2ForCausalLM |

**Environment**:
- GPU: NVIDIA GeForce RTX 3060 (12 GB VRAM)
- CUDA: 12.8
- PyTorch: 2.10.0+cu128
- Transformers: 4.57.6

**Key observation**: GQA with 14 Q heads and 2 KV heads (7:1 ratio). KV head_dim = 64. Block size currently 4 (config default). This model supports 24/24 layers for sliding window (max_window_layers=21 unused).

### Model File Inventory

```
config.json, generation_config.json, merges.txt, model.safetensors,
tokenizer.json, tokenizer_config.json, vocab.json
```

All required files present. `local_files_only=True` is safe.

---

## 3. Current Call Chain — Prefill & Decode

```
LLMEngine.add_request(request)
  └→ RequestQueue.add(sg)                # puts in _waiting dict

LLMEngine.step()
  └→ EngineCore.step()
       ├→ scheduler.schedule()
       │    ├→ Phase 1: finish check (running groups)
       │    ├→ Phase 2: categorize into decode/prefill-continue
       │    ├→ Phase 3: decode-first budget (deduct decode tokens)
       │    ├→ Phase 4: chunked-prefill continue (advance prefill cursor)
       │    ├→ Phase 5: admit new waiting groups (with prefix cache probe)
       │    └→ Phase 6: token counts + debug_reason
       │
       ├→ executor.prefill(only_prefill_seqs)
       │    └→ for each seq in sequences:          # ⚠ PER-SEQUENCE LOOP
       │         ├→ input_ids = tensor([[token_ids]])
       │         ├→ model(input_ids, past_key_values, use_cache=True)
       │         ├→ self._seq_kv[seq_id] = outputs.past_key_values
       │         ├→ self._block_manager.ensure_block(seq, pos)  # for each token
       │         ├→ seq.prefill_cursor = end
       │         └→ if finished: argmax → output_token_ids[0]
       │
       ├→ executor.decode(decode_seqs)
       │    └→ for each seq in sequences:          # ⚠ PER-SEQUENCE LOOP
       │         ├→ input_ids = tensor([[prev_token]])
       │         ├→ model(input_ids, past_key_values, use_cache=True)
       │         ├→ self._seq_kv[seq_id] = outputs.past_key_values
       │         ├→ argmax → next_token
       │         ├→ block_manager.ensure_block(seq, new_pos)
       │         └→ append to output_token_ids
       │
       ├→ for sg in finished: executor.cleanup_sequence(seq_id)
       │    └→ self._seq_kv.pop(seq_id, None)
       │
       └→ metrics_collector.record_step(...)
```

---

## 4. Current Data Structures & KV Flow

### `_seq_kv: Dict[str, Any]`
- **Type**: `Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]`
- HF-format `past_key_values`: tuple of 24 layers, each layer = `(key_states, value_states)`
- Each key/value shape: `[1, num_kv_heads, seq_len, head_dim]` (batch=1 for per-sequence)
- **TRUE KV source** — all KV data lives here.
- Cleaned up in `cleanup_sequence()`.
- **Not accessible** to BlockManager/BlockAllocator.

### `_kv_cache: Dict[int, List[int]]`
- **Type**: `Dict[int, List[int]]` — block_id → list of token positions
- **Metadata only** — no tensor data.
- Used to track which tokens belong to which block for stats reporting.
- Populated by `ensure_block()` → positional tracking.
- **Not a true KV cache** — does not store K/V tensors.

### `BlockAllocator`
- Manages: free-list (bool array), ref_counts (int array)
- Methods: `allocate(N)` → List[int] (physical block IDs), `free(ids)`, `increment_ref(pid)`
- **No tensor operations** — pure metadata.
- Provides `on_allocate`/`on_free` callbacks (wired to executor's prepare_block/release_block).

### `BlockManager`
- Manages: per-seq BlockTable dict, PrefixCache, shared block tracking
- Methods: `allocate_for_seq(seq)`, `ensure_block(seq, position)` → physical_block_id
- **No tensor operations**. Pure ID management.
- `ensure_block()`: checks prefix cache → hit: share (increment_ref), miss: allocator.allocate(1)
- `probe_prefix_cache()`: read-only hash probe for scheduler budget computation.

### `BlockTable`
- `List[BlockTableEntry]` with `(physical_block_id: int, is_shared: bool)`
- Methods: `add_block`, `add_shared_block`, `get_block_ids`, `get_physical_block(position)`
- **Two sources of truth**: `BlockTable` in BlockManager._tables AND `Sequence.block_table` (a mirror copy). This is a risk.

### `PrefixCache`
- `Dict[int, int]` — block_hash → physical_block_id
- Hash: Python `hash(tuple(tokens))` over block_size tokens.
- Registered at `ensure_block()` time for prompt positions (before KV data is written).
- Stale detection: probe checks `ref_count > 0` for cached entries.
- No eviction policy in current implementation.

---

## 5. Fracture Points: Management Layer vs Tensor Layer

| Aspect | Management Layer (BlockManager/Allocator) | Tensor Layer (QwenExecutor) |
|--------|------------------------------------------|----------------------------|
| KV storage | `_kv_cache: Dict[int, List[int]]` (metadata) | `_seq_kv: Dict[str, past_key_values]` (real tensors) |
| Block IDs | Allocated/freed, tracked via ref_counts | Not used in actual KV access |
| Block tables | Logical→physical mapping | Not used in model execution |
| Prefill | Not involved | HF model with past_key_values |
| Decode | Not involved | HF model with past_key_values |
| **Gap** | BlockAllocator decisions have ZERO effect on actual KV tensor layout | _seq_kv is purely HF contiguous cache, ignoring blocks |

**The fundamental fracture**: BlockAllocator allocates block IDs, but QwenExecutor ignores them for actual computation. The `_seq_kv` dictionary stores HF-format contiguous past_key_values — completely independent of the block-level abstraction.

---

## 6. Benchmark Conclusion Impact on Design

| Finding | Value | Design Impact |
|---------|-------|---------------|
| Throughput | ~1.9 req/s flat across all modes | Per-sequence model calls are the bottleneck |
| Effective batch | 1.0–4.69 (continuous) | Real batch is logical only, not GPU-level |
| TPOT scale | c=1: 19ms, c=4: 79ms | Linear scaling confirms no GPU batching |
| Scheduler overhead | <0.06ms | Negligible — model execution dominates |
| GPU memory | 12 GB, ~1 GB for weights | Sufficient for batch=4+ paged KV |

**Design conclusion**: The scheduler is efficient. The execution layer is the bottleneck. Any throughput improvement requires replacing the per-sequence HF model call with a batched execution path using real paged KV.

---

## 7. Scheduler `ScheduleResult` Fields

| Field | Type | Description |
|-------|------|-------------|
| `scheduled_prefill_groups` | `List[SequenceGroup]` | Groups selected for prefill this step |
| `scheduled_decode_groups` | `List[SequenceGroup]` | Groups selected for decode this step |
| `ignored_groups` | `List[SequenceGroup]` | Groups skipped this step |
| `finished_groups` | `List[SequenceGroup]` | Groups completed this step |
| `rejected_groups` | `List[SequenceGroup]` | Groups rejected this step |
| `preempted_groups` | `List[SequenceGroup]` | Groups preempted (currently always empty) |
| `num_batched_tokens` | `int` | Total tokens this step |
| `num_prefill_tokens` | `int` | Prefill tokens this step |
| `num_decode_tokens` | `int` | Decode tokens this step (== running request count) |
| `token_budget_remaining` | `int` | Unused token budget |
| `cached_token_count` | `int` | Tokens from prefix cache (not computed) |
| `num_uncached_prefill_tokens` | `int` | Prefill tokens needing computation |
| `matched_block_count` | `int` | Blocks shared via prefix cache |
| `debug_reason` | `str` | Human-readable scheduling summary |
| `ignored_reasons` | `Dict[str, str]` | Per-group ignore reason |

---

## 8. EngineCore Step Flow (detailed)

```python
def step(self):
    1. _check_timeouts()                          # cancel timed-out requests
    2. scheduler.schedule()                       # → ScheduleResult
    3. for sg in result.scheduled_prefill_groups:  # record first_scheduled_time
         seq.first_scheduled_time = time.time()
    4. Flatten prefill groups → List[Sequence]
    5. Flatten decode groups  → List[Sequence]
    6. if has_prefill: executor.prefill(prefill_seqs)
         for seq: set seq.first_token_time if prefill finished
    7. if has_decode:  executor.decode(decode_seqs)
    8. for sg in result.finished_groups:
         for seq: executor.cleanup_sequence(seq_id)
                  metrics_collector.register_sequence(seq)
    9. metrics.record_step(result, sched_latency, step_wall, ...)
```

---

## 9. Current Prefix Cache Risks

1. **Block registered before KV write**: PrefixCache.insert() happens in `ensure_block()`, which is called during prefill execution — but the block's KV tensor data hasn't been written yet. If another sequence shares this block, it references physical memory that is still uninitialized.

2. **Stale cache entries**: `PrefixCache` dictionary entries persist after all references are freed. `ref_count=0` entries are detected during probe (probe checks ref_count), but `insert_span` and direct `lookup` don't validate liveness.

3. **No eviction**: The cache grows monotonically with unique block hashes. No LRU, no capacity limit.

4. **Hash collision**: Python's built-in `hash()` is used — deterministic within a process but not cryptographically robust. Collisions could cause incorrect sharing.

5. **Partial block sharing**: When the last shared block has different prompt lengths between sequences, the overlap token count may exceed actual common tokens. The scheduler caps at `block_size * matched_block_count`, but this assumes the last matched block is fully shared, which may be incorrect for partial blocks.

6. **COW readiness gap**: `BlockTableEntry.is_shared` flag exists, and `BlockManager.is_block_shared()` returns it. But the executor never consumes this flag — it always writes to all positions via `_write_to_kv`, and for shared prefix blocks it **skips** via `is_block_shared()` check, leaving data uninitialized. This is correct for prompt tokens (already written by the original), but creates a correctness dependency on the original sequence staying alive.

---

## 10. Chunked Prefill Current Implementation

Scheduler-side:
- `chunked_prefill_enabled=True` → admits long prompts in chunks of `max_prefill_chunk_size`
- Sets `prefill_cursor` to the cached prefix length (or 0 if no cache)
- Phase 4 iterates prefill-continue groups, advancing cursor by chunk_size each step
- Edge case: when all prompt tokens are prefix-cached, sets cursor = prompt_len - chunk_size

Executor-side (QwenExecutor):
- Uses `seq.prefill_cursor` as start, processes `min(chunk_size, remaining)` tokens
- Calls `model(input_ids, past_key_values)` where `past_kv` is the saved HF cache
- HF model appends new token KV to past_key_values automatically
- Continues until `is_prefill_finished`, then samples first token

**Key issue**: Current chunked prefill still uses HF `past_key_values`. Each chunk just extends the HF contiguous cache. There is no integration with block-level KV storage.

---

## 11. Test Coverage Assessment

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_kv_cache_manager.py` | 12 | BlockTable, BlockAllocator, BlockManager (allocate, free, OOM, callbacks, stats) |
| `test_scheduler.py` | 9 | Admit, prefill→decode, finish, budget limits, on-demand, decode-first, chunked prefill, ignore reasons |
| `test_engine.py` | 6 | Full engine loop, continuous batching, mid-arrival, OOM, step result, KV writes |
| `test_prefix_cache.py` | 17 | PrefixCache unit, ref-count sharing, partial prefix, stale entries, probe, scheduler integration |
| `test_metrics.py` | 24 | TTFT, TPOT, throughput, no-double-count, KV utilization, scheduler latency, prefix cache metrics, profiler, serving counters |
| `test_benchmark.py` | 14 | CLI args, config propagation, JSON output, formulas, serial/continuous/static modes, trace, resource cleanup |
| `test_stage_profiler.py` | | Stage profiler recording/reporting |
| `test_request.py` | | Request/tokenizer tests |
| `test_serving_layer.py` | | HTTP serving layer tests |
| `test_fault_injection.py` | | Fault injection tests |

**Gaps**:
- No tests for QwenExecutor (requires GPU)
- No tests for `_seq_kv` lifecycle correctness
- No tests for KV tensor data integrity after manager.free() / cleanup_sequence()
- No tests that verify BlockTable is_shared flag prevents double-write
- No tests for chunked prefill edge cases (chunk size > prompt, cursor edge cases)

---

## 12. Dependency on FakeExecutor or Simulated KV

Tests that use `FakeModelExecutor` or simulated KV (i.e., tests using `Config(executor_type="fake")`):

- All `test_scheduler.py` tests
- All `test_engine.py` tests (except real-engine ones)
- All `test_metrics.py` tests
- Most `test_prefix_cache.py` tests
- All `test_benchmark.py` tests
- `test_stage_profiler.py`

These tests validate scheduling logic, metadata management, and metrics — not actual model execution. The `FakeModelExecutor` produces deterministic, fast results. For V1, these tests should continue working without modification.

---

## 13. ScheduleResult → EngineCore → Executor Interface Summary

```
ScheduleResult
 ├── scheduled_prefill_groups → flatten → List[Sequence] → executor.prefill(seqs)
 ├── scheduled_decode_groups  → flatten → List[Sequence] → executor.decode(seqs)
 └── finished_groups          → iterate → executor.cleanup_sequence(seq_id)

Executor Protocol:
  prefill(sequences: List[Sequence]) → None
  decode(sequences: List[Sequence]) → None
  cleanup_sequence(seq_id: str) → None
  tokenize(prompt: str) → List[int]
  detokenize(token_ids: List[int]) → str
  prepare_block(block_id: int) → None
  release_block(block_id: int) → None
  get_kv_stats() → Dict[str, int]
  total_tokens_processed → int
```

The interface is clean. The PagedExecutor must implement the same Protocol.

---

## 14. Files That Must NOT Be Modified

| File | Reason |
|------|--------|
| `mini_vllm/sequence/sequence.py` | Core data class — stable API |
| `mini_vllm/sequence/sequence_group.py` | Request lifecycle — stable |
| `mini_vllm/sequence/status.py` | Status enum — stable |
| `mini_vllm/sequence/sampling_params.py` | Sampling config — stable |
| `mini_vllm/scheduler/scheduler.py` | Scheduling policy — not changing |
| `mini_vllm/scheduler/schedule_result.py` | Result type — not changing |
| `mini_vllm/engine/engine.py` | Public API — preserve |
| `mini_vllm/engine/engine_core.py` | Step loop — minor adapter only |
| `mini_vllm/engine/metrics.py` | Metrics — preserve |
| `mini_vllm/engine/stage_profiler.py` | Profiler — preserve |
| `mini_vllm/serving/` | HTTP serving — not changing |
| `mini_vllm/config.py` | Add fields only, don't break existing |
| `mini_vllm/worker/fake_worker.py` | Preserve for testing |
| `mini_vllm/model/` | FakeModel — preserve for testing |

---

## 15. Summary of Key Design Decisions for V1

1. **BlockAllocator stays**: Its ref-count management, free-list, and callbacks are correct and usable for real GPU KV pool.

2. **BlockManager needs upgrade**: Must generate slot_mapping, batched block_table Tensor, context_lengths. Must bridge from metadata to tensor world.

3. **BlockTable mirroring must be resolved**: `Sequence.block_table` and `BlockManager._tables[seq_id]` are dual sources of truth. Choose one.

4. **`_seq_kv` must be eliminated**: Paged KV pool is the sole source of K/V tensors.

5. **Prefix Cache must be disabled**: Risk of stale/uninitialized blocks makes it incompatible with real paged KV until COW is implemented.

6. **QwenExecutor must be preserved as reference**: Rename to `QwenExecutorSequential` or keep as `QwenExecutor` for correctness comparison. New `PagedExecutor` is a separate implementation.

7. **GQA ratio 7:1**: Qwen2.5-0.5B has 14 Q heads and 2 KV heads. Block size must be a multiple of KV heads for aligned memory access.

8. **Memory budget**: 12 GB total, ~1 GB weights, ~2 GB runtime → ~9 GB available for KV cache. With block_size=16, block_bytes=2×2×64×2 bytes = 512 bytes per-block-token (key+value, 2 KV heads, head_dim 64, float16). Per block: 512×16 = 8 KB. ~9 GB → ~1,152,000 blocks ≈ 1.1M. More than enough even for large batches.
