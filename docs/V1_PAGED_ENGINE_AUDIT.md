# V1 Paged Engine — Current Implementation Audit

> **Phase A output — REVISION 1.** Updated per 14-point review (2026-07-13).
> Audit of the current mini-vLLM codebase before V1 Paged Engine changes.

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

**Key observation**: GQA with 14 Q heads and 2 KV heads (7:1 ratio). KV head_dim = 64.

---

## 3. Current Call Chain — Prefill & Decode (for reference)

```
LLMEngine.add_request(request)
  └→ RequestQueue.add(sg)

LLMEngine.step()
  └→ EngineCore.step()
       ├→ scheduler.schedule()  (unchanged in V1)
       ├→ executor.prefill(only_prefill_seqs)     # TWO separate calls (to be unified)
       ├→ executor.decode(decode_seqs)            # TWO separate calls (to be unified)
       ├→ cleanup finished sequences
       └→ metrics
```

**V1 target**: Replace two separate executor calls with a single `executor.execute(model_input)` call. See DESIGN.md Section 3 for the revised call chain.

---

## 4. Current Data Structures & KV Flow

### `_seq_kv: Dict[str, Any]` — TO BE ELIMINATED
- **Type**: `Dict[str, Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]`
- HF-format `past_key_values`: tuple of 24 layers, each = `(key_states, value_states)`.
- **TRUE KV source** in current code — all KV data lives here.
- **V1**: Replaced by KVCachePool.

### `_kv_cache: Dict[int, List[int]]` — TO BE ELIMINATED
- Metadata-only dict in QwenExecutor. Not a true KV cache.
- Replaced by KVCachePool with real GPU tensors.

### BlockAllocator — TO BE ENHANCED
- Pure metadata: free-list (bool array) + ref_counts (int array).
- V1: Add invariant checks, max_blocks exposure. Keep metadata-only.

### BlockManager — TO BE SIMPLIFIED
- Manages per-seq BlockTable dict (AUTHORITATIVE source).
- V1: Convert to read-only query interface. No tensor generation.

### BlockTable — TO BE ENHANCED
- `List[BlockTableEntry(physical_block_id, is_shared)]`
- V1: Add tensor export (padded for GPU kernel). Remove mirror sync to Sequence.

### PrefixCache — TO BE DISABLED
- `Dict[int, int]` — block_hash → physical_block_id.
- V1: Disabled by default. Known risks: stale entries, block registered before KV write.

### `Sequence.block_table: List[int]` — TO BE REMOVED
- **Current risk**: Dual truth source with BlockManager._tables.
- **V1**: Remove entirely. All consumers read from BlockManager.get_block_table().

---

## 5. Fracture Points: Management Layer vs Tensor Layer

| Aspect | Management Layer | Tensor Layer | V1 Resolution |
|--------|-----------------|--------------|---------------|
| KV storage | `_kv_cache: Dict[int, List[int]]` (metadata) | `_seq_kv: Dict[str, past_key_values]` (real tensors) | Unify into KVCachePool |
| Block IDs | Allocated/freed, tracked via ref_counts | Not used in actual KV access | Block IDs = pool index |
| Block tables | Logical→physical mapping | Not used in model execution | Block table = kernel input |
| Prefill | Not involved | HF model with past_key_values | Paged prefill via pool |
| Decode | Not involved | HF model with past_key_values | Paged decode via attention backend |
| **Sequence.block_table** | Mirror copy sync after each mutation | Not used | REMOVED entirely |

---

## 6. Benchmark Conclusion Impact on Design

| Finding | Value | Design Impact |
|---------|-------|---------------|
| Throughput | ~1.9 req/s flat across all modes | Per-sequence model calls are the bottleneck |
| Effective batch | 1.0–4.69 (continuous) | Real batch is logical only, not GPU-level |
| TPOT scale | c=1: 19ms, c=4: 79ms | Linear scaling confirms no GPU batching |
| Scheduler overhead | <0.06ms | Negligible |
| GPU memory | 12 GB, ~1 GB for weights | Sufficient for batch=4+ paged KV |

**Design conclusion**: The scheduler is efficient. The execution layer is the bottleneck.

---

## 7. Key Design Decisions (Revised)

1. **BlockAllocator stays metadata-only**: Its ref-count management and free-list are correct. No tensor changes.

2. **BlockManager simplified**: Removed from tensor construction role. Two read-only query methods: `get_block_table(seq_id)`, `needs_allocation(seq, position)`.

3. **Sequence.block_table REMOVED**: Single truth source in BlockManager._tables. All consumers call `get_block_table()`.

4. **`_seq_kv` eliminated**: KVCachePool is the sole K/V data source.

5. **Prefix Cache DISABLED**: Guarded by config flag. Not used in production path.

6. **QwenExecutor preserved as reference**: Unchanged. Used for correctness comparison in tests.

7. **GQA ratio 7:1**: Qwen2.5-0.5B has 14 Q heads and 2 KV heads.

8. **Memory budget**: Profile-run based initialization. No hardcoded numbers.

9. **Unified execution**: Single `execute()` per step. Single Transformer layer loop.

10. **GPU PagedAttention is mandatory**: Reference path for tests only. No silent fallback.

---

## 8. Interface Summary

### Current Executor Protocol (to be enhanced)

```python
class Executor(Protocol):
    def tokenize(self, prompt: str) -> List[int]: ...
    def detokenize(self, token_ids: List[int]) -> str: ...
    def prefill(self, sequences: List[Sequence]) -> None: ...     # TO BE REPLACED
    def decode(self, sequences: List[Sequence]) -> None: ...      # TO BE REPLACED
    def prepare_block(self, block_id: int) -> None: ...           # TO BE NO-OP
    def release_block(self, block_id: int) -> None: ...           # TO BE NO-OP
    def cleanup_sequence(self, seq_id: str) -> None: ...          # TO BE NO-OP
    def get_kv_stats(self) -> Dict[str, int]: ...
    total_tokens_processed: int
```

### Target: Single execute()

```python
class PagedExecutor:
    def execute(self, model_input: ModelInput) -> torch.Tensor: ...
    # prepare_block/release_block/cleanup_sequence = no-ops
```

---

## 9. Files That Must NOT Be Modified (Revised)

| File | Reason |
|------|--------|
| `mini_vllm/sequence/sequence.py` | Core data class. UPDATE: remove block_table field |
| `mini_vllm/sequence/sequence_group.py` | Request lifecycle |
| `mini_vllm/sequence/status.py` | Status enum |
| `mini_vllm/sequence/sampling_params.py` | Sampling config |
| `mini_vllm/scheduler/scheduler.py` | Scheduling policy — unchanged |
| `mini_vllm/scheduler/schedule_result.py` | Result type — unchanged |
| `mini_vllm/engine/engine.py` | Public API — preserve |
| `mini_vllm/engine/metrics.py` | Metrics — preserve |
| `mini_vllm/engine/stage_profiler.py` | Profiler — preserve |
| `mini_vllm/serving/` | HTTP serving — not changing |
| `mini_vllm/worker/fake_worker.py` | Preserve for testing |
| `mini_vllm/model/` | FakeModel — preserve for testing |
| `mini_vllm/executor/executor.py` | FakeModelExecutor — preserve for testing |
| `mini_vllm/executor/qwen_executor.py` | Reference implementation — preserve unchanged |

**Note**: `engine/engine_core.py` requires adaptation (single execute call). This is classified as ADAPTED, not UNCHANGED.

---

## 10. Test Coverage Assessment (Revised)

### Existing Tests (must continue passing)

| Test File | Tests | Verdict |
|-----------|-------|---------|
| `test_kv_cache_manager.py` | 12 | BlockManager API changes are backward-compatible |
| `test_scheduler.py` | 9 | No scheduler changes |
| `test_engine.py` | 6 | EngineCore changes are additive (new path) |
| `test_prefix_cache.py` | 17 | Prefix cache disabled, tests preserved |
| `test_metrics.py` | 24 | No metrics changes |
| `test_benchmark.py` | 14 | No benchmark changes |

### New Tests Required

| Area | Tests | Phase |
|------|-------|-------|
| ModelInputBuilder | Slot mapping, tensor export, sample positions | P1 |
| KVCachePool | Shape, dtype, write/read, overwrite | P2 |
| PagedAttention ref | SDPA comparison, GQA, block boundaries | P3-4 |
| Triton kernel | Reference comparison, random seeds, stress | P5 |
| ModelRunner | Single layer vs HF, full model vs HF | P6-7 |
| PagedExecutor | Full engine loop, metrics, cleanup | P8-9 |
| Chunked prefill | Full vs chunked equivalence | P9 |
| Lifecycle | All release paths (cancel, timeout, disconnect, exception) | P10 |
