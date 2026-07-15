# PagedEngine Implementation Roadmap

## Phase A — Audit & Design

**Target:** Audit existing single-sequence executor architecture, document gap between current and PagedAttention requirements, produce design documents for incremental implementation.

**Prerequisites:** None (initial phase).

**Allowed scope:**
- Review existing `FakeModelExecutor` and `QwenExecutor` code paths
- Document the `BlockTable` / `BlockAllocator` / `BlockManager` abstraction
- Identify where `Sequence` objects cross executor boundaries
- Produce `V1_PAGED_ENGINE_AUDIT.md` and `V1_PAGED_ENGINE_DESIGN.md`

**Forbidden:**
- Any implementation or refactoring of production code

**Acceptance criteria:**
- Audit document lists all files, interfaces, and data flows
- Design document proposes phased migration with concrete interfaces
- Code review of design by team

**Test requirements:** None (documentation phase).

**Status:** COMPLETED

**Completion evidence:**
- `docs/V1_PAGED_ENGINE_AUDIT.md` — full audit of v1 executor code
- `docs/V1_PAGED_ENGINE_DESIGN.md` — phased migration design

**Remaining TODOs:** None.

---

## Phase 1 — Metadata / BlockTable / Interfaces

**Target:** Establish `ModelRunnerInput` / `ModelRunnerOutput` as the sole interface between EngineCore and executor. Add `BlockTableEntry`, ref-count-based `BlockAllocator`, `PrefixCache`, `BlockManager.ensure_block_by_ids()`. All tests pass without behavior change.

**Prerequisites:** Phase A (design documents).

**Allowed scope:**
- `BlockTableEntry` dataclass with `block_id`, `ref_count`, `is_shared`
- `BlockAllocator` with ref-count-aware `allocate()` / `free()` / `free_table()`
- `BlockManager` with `ensure_block()` (by `Sequence`), `ensure_block_by_ids()` (by `seq_id`), `free()`
- `PrefixCache` with hash-based lookup, `probe()`, block sharing
- `ModelInput` / `AttentionMetadata` / `ModelRunnerOutput` / `SequenceExecutionInfo` schema dataclasses
- `ModelInputBuilder` — constructs `ModelInput` from scheduler output
- `ConfigAdapter` — bridges `Config` to `ModelConfig`
- `AttentionBackend` abstract base class
- Scheduler integration: `PrefixCache.probe()` in admission, correct `cached_len_before` / `query_len` / `kv_len_after`
- Existing `FakeModelExecutor` continues to work through old `prefill()`/`decode()` API (backward compat)
- Engine integration tests extended

**Forbidden:**
- No unified `execute()` wiring (deferred to Phase 1.5)
- No GPU KV cache pool allocation
- No PagedAttention math or Triton kernel
- No Qwen ModelRunner
- No Chunked Prefill changes beyond scheduler

**Acceptance criteria:**
- `BlockTable` truth source: `block_manager.get_block_table(seq)` replaces ad-hoc lists
- `ensure_block_by_ids()` accepts `(seq_id, position, prompt_len)` — no `Sequence` object
- All tests pass (engine integration, scheduler, cache manager, prefix cache)
- `ModelInputBuilder` correctly produces `ModelInput` for prefill and decode
- `slot_mapping` formula: `block_id * block_size + offset`

**Test requirements:**
- Full block allocation lifecycle tests (allocate, ensure, free, OOM)
- Prefix cache hit/miss/partial/stale tests
- `ensure_block_by_ids()` correctness tests
- `ModelInputBuilder` prefill / decode / mixed batch tests
- `slot_mapping` multi-block boundary tests
- `cached_len_before` semantics tests (prefill + decode)

**Status:** COMPLETED

**Completion evidence:**
- `mini_vllm/cache/block_table.py` — `BlockTable`, `BlockTableEntry`
- `mini_vllm/cache/allocator.py` — `BlockAllocator` with ref-count API
- `mini_vllm/cache/manager.py` — `BlockManager`, `ensure_block_by_ids()`
- `mini_vllm/cache/prefix_cache.py` — `PrefixCache`, `PrefixCacheProbeResult`
- `mini_vllm/model_runner/base.py` — all schema dataclasses
- `mini_vllm/engine/input_builder.py` — `ModelInputBuilder`
- `mini_vllm/attention/backend.py` — `AttentionBackend`
- Tests: `test_kv_cache_manager.py`, `test_prefix_cache.py`, `test_input_builder.py`, `test_scheduler.py`

**Remaining TODOs:** None.

---

## Phase 1.5 — Unified `execute()` Wiring

**Target:** Replace the dual `executor.prefill()` / `executor.decode()` path with a single `executor.execute(ModelInput) -> ModelRunnerOutput` call. All executors (`FakeModelExecutor`, `QwenExecutor`) share the same signature. EngineCore writes back `sampled_token_ids` to `Sequence` objects by matching `seq_id`.

**Prerequisites:** Phase 1 (interfaces defined, `ModelInputBuilder` wired).

**Allowed scope:**
- `Executor.execute()` returns `ModelRunnerOutput` (was `None`)
- `EngineCore.step()` builds `ModelInput` once, calls `execute()` once
- `EngineCore._apply_model_output()` — maps `sampled_sequence_ids` back to `Sequence` objects
- EngineCore ensures blocks before calling `ModelInputBuilder.build()`
- EngineCore advances `prefill_cursor` after execute
- Legacy `prefill()` / `decode()` methods remain on executors for backward compat but are no longer called
- Backend naming: `Config.attention_backend = "reference"` (was `"paged"`), `AttentionBackend.create(backend=...)` (was `backend_type=...`)

**Forbidden:**
- No GPU KV cache pool allocation (Phase 2)
- No PagedAttention math or Triton kernel
- No Qwen ModelRunner
- No Chunked Prefill

**Acceptance criteria:**
- `EngineCore.step()` calls `execute()` exactly once per step
- Mixed prefill+decode batches produce single `ModelInput`
- `ModelRunnerOutput.sampled_token_ids` / `sampled_sequence_ids` tuple length match
- Old `prefill()` / `decode()` are NOT called (verified by spy executor)
- All existing engine, scheduler, cache, prefix cache tests still pass

**Test requirements:**
- `SpyExecutor` variant that records call counts
- `test_execute_called_once` — single execute, no prefill/decode
- `test_mixed_batch_single_input` — prefill + decode in one ModelInput
- `test_incomplete_prefill_no_sample` — chunk boundary, no output tokens
- `test_completed_prefill_first_token` — prefill completion produces first decode token
- `test_decode_appends_token` — decode step correctly appends
- `test_sample_sequence_mapping` — seq_id-based write-back
- `test_empty_batch_no_execute` — no-op step
- `test_old_prefill_decode_not_called` — backward compat methods unused
- `test_length_assertions` — cached_len_before / query_len / kv_len_after invariants
- `test_backend_naming` — "reference" not "paged"
- `test_sequence_info_structure` / `test_model_runner_output_structure`

**Status:** COMPLETED

**Completion evidence:**
- `mini_vllm/executor/base.py` — `execute() -> ModelRunnerOutput`
- `mini_vllm/executor/executor.py` — `execute()` with `_simulate_kv_write()` / `_simulate_read_kv_bias()`
- `mini_vllm/executor/qwen_executor.py` — `execute()` with Qwen-specific block tracking
- `mini_vllm/engine/engine_core.py` — unified `step()` path, `_apply_model_output()`, block ensure
- `mini_vllm/config.py` — `attention_backend` field
- `tests/test_engine_core_unified.py` — 13 tests

**Remaining TODOs:** None.

---

### Decode Off-by-One Fix (cross-phase correction)

**Target:** Correct `cached_len_before` formula for decode steps. The first token generated by a prefill completion is the **input** to the first decode step, but it is NOT yet in the KV cache. Therefore:
```
cached_len_before = prompt_len + num_generated - 1   # (not + num_generated)
```

**Affected files:**
- `mini_vllm/engine/input_builder.py` — `_collect_decode_metadata`
- `mini_vllm/engine/engine_core.py` — decode block ensure position
- `mini_vllm/executor/executor.py` — `_simulate_read_kv_bias()` read position, `_simulate_kv_write()` write position
- `mini_vllm/executor/qwen_executor.py` — decode block tracking

**Evidence:** `TestDecodeCacheLength` (7 tests), `TestDecodeInvariants` (5 tests), `test_fake_executor_cache_length_consistency`.

**Status:** COMPLETED (integrated into Phase 1.5 delivery).

---

## Phase 2 — GPU KV Cache Pool

**Target:** Allocate the physical GPU KV cache pool (`KVCachePool`) and implement the `compute_num_gpu_blocks()` budget formula. No PagedAttention math, no Triton kernel, no Qwen ModelRunner.

**Prerequisites:** Phase 1.5 (unified `execute()` signature, `BlockManager` with `ensure_block_by_ids()`).

**Allowed scope:**
- `KVCachePool` dataclass with per-layer `[num_blocks, num_kv_heads, block_size, head_dim]` tensors
- `KVCachePool.allocate()` — `torch.empty()`, NOT `torch.zeros()`
- `KVCachePool.reset()` — zero-fill (test/debug only; NOT called on normal block free)
- `KVCachePool` accessors: `get_key_cache()`, `get_value_cache()`
- `KVCachePool` properties: `total_slots`, `bytes_per_block_per_layer`, `bytes_per_block_total`, `total_bytes`
- `compute_num_gpu_blocks()` — 10-step GPU memory budget formula:
  1. `free_bytes = cuda.mem_get_info()[0]`
  2. `post_deduction = free_bytes - peak_runtime_estimate - workspace_reserve`
  3. `safety = post_deduction * (1 - gpu_memory_utilization)`
  4. `budget = post_deduction * gpu_memory_utilization`
  5. `bytes_per_block_total = num_layers * 2 * num_kv_heads * block_size * head_dim * dtype.itemsize`
  6. `num_blocks = int(budget // bytes_per_block_total)`
  7. `num_blocks = max(num_blocks, MIN_BLOCKS)`
  8. If `< MIN_BLOCKS`, raise `RuntimeError`
- `Config.gpu_memory_utilization = 0.90` with validation
- Public exports via `cache/__init__.py` and `mini_vllm/__init__.py`

**Forbidden:**
- No PagedAttention math (block-sparse attention, block table → physical block lookup, softmax with block mask)
- No Triton kernel
- No Qwen ModelRunner
- No Chunked Prefill
- No wiring `KVCachePool` into executor or attention backend (deferred to Phase 3)
- No cache write/flush logic (deferred to Phase 3)
- No block reuse / eviction policy in pool (deferred to Phase 3)

**Acceptance criteria:**
- `KVCachePool.allocate()` produces correct shapes for all layers
- `torch.empty()` sentinel test confirms NOT `torch.zeros()`
- `reset()` zeroes all tensors (test-only usage)
- All derived properties return correct byte counts
- `compute_num_gpu_blocks()` with override returns value directly
- `compute_num_gpu_blocks()` on CPU returns `MIN_BLOCKS`
- `compute_num_gpu_blocks()` on real GPU returns ≥ `MIN_BLOCKS`
- Higher `gpu_memory_utilization` yields ≥ blocks than lower
- `Config.gpu_memory_utilization` validated in `(0, 1]`
- All 262 existing tests pass

**Test requirements:**
- 12 CPU tests + 1 GPU test (13 total):
  - Allocation: shapes, invalid params, empty sentinel (3)
  - Accessors: get_key_cache, get_value_cache (1)
  - Reset: zero-fill after non-zero write (1)
  - Properties: byte counts, dtype variation (2)
  - Override: returns value, below-min raises (2)
  - CPU fallback: returns MIN_BLOCKS (1)
  - Utilization validation: 0.0, 1.5, -0.1 raise (1)
  - GPU integration: real budget query, utilization comparison, override (1)

**Status:** COMPLETED

**Completion evidence:**
- `mini_vllm/cache/pool.py` — `KVCachePool` + `compute_num_gpu_blocks()` (9942 bytes)
- `mini_vllm/config.py` — `gpu_memory_utilization: float = 0.90` + validation
- `mini_vllm/cache/__init__.py` — exports `KVCachePool`, `compute_num_gpu_blocks`
- `mini_vllm/__init__.py` — exports same
- `tests/test_kv_cache_pool.py` — 12 collected items, 12 passed, 0 skipped

**Remaining TODOs:**
- [x] **`peak_runtime_estimate` is still `0`**: Profiled by B7 — 128 prefill + 16 decode steps on Qwen2.5-0.5B with small pool override. Measured `runtime_peak_increment=41,042,944 bytes` (0.04 GiB). Added `Config.peak_runtime_estimate: int` (default 0), wired through PagedExecutor → QwenModelRunner → `_resolve_num_blocks()` → `compute_num_gpu_blocks()`. Final budget with 3% safety margin: **159,229 blocks (7.29 GiB)** on 12 GiB GPU. Inference verified OOM-free.
- [ ] **Pool not wired into executor**: `KVCachePool` is allocated but not passed to any executor or attention backend. Phase 3 must wire it in.
- [ ] **No cache write implementation**: `slot_mapping` → KV cache write is not yet implemented. The executor cannot write computed K/V states into the pool.
- [ ] **No PagedAttention**: The block-table-driven attention computation (using physical block IDs) does not exist yet.
- [ ] **No Triton kernel**: The optimized GPU attention kernel is not implemented.
- [ ] **Block free does not reset pool slots**: On `BlockAllocator.free()`, the corresponding pool slots are NOT zeroed. The next writer will overwrite all relevant slots before any reader accesses them. This is correct by design — see docstring in `pool.py`.

---

## Phase 3 — Cache Write Reference

**Target:** Implement real K/V tensor writes into cache pool tensors using `slot_mapping`. Each token's K/V is written to the correct physical slot: `block_id = slot // block_size`, `block_offset = slot % block_size`. Prefill writes multiple tokens per sequence; decode writes one. Standalone module — not wired into executor.

**Prerequisites:** Phase 2 (`KVCachePool` allocated).

**Allowed scope:**
- New `mini_vllm/cache/cache_write.py` — `write_to_paged_cache()` pure-PyTorch function
- Single-layer API: caller passes `key_cache` and `value_cache` tensors, not the pool object
- Supports `slot == -1` skip (for prefix-cache hits)
- Supports prefill (many tokens, many slots) and decode (single token, single slot)
- `slot_mapping` → `block_id = slot // block_size`, `block_offset = slot % block_size`
- Explicit validation: shape, dtype, device, block_size, slot range, slot < -1
- Not wired into executor (standalone module + tests)

**Forbidden:**
- No attention computation (Phase 4)
- No Triton kernel (Phase 5)
- No Qwen ModelRunner (Phase 6)
- No executor wiring
- No engine changes
- No GPU-only — must work on CPU for testing
- No `KVCachePool` dependency in function signature

**Acceptance criteria:**
- Single-token decode write lands in correct block + offset
- Multi-token prefill write lands in correct blocks (possibly spanning block boundary)
- Batched sequences each write to their own slots
- All layers are written independently
- Non-contiguous physical blocks work correctly
- Repeated slot overwrite: last writer wins
- Block reuse: overwriting existing slots does not interfere with other blocks
- Unwritten slots unchanged (snapshot-based sentinel test)
- K and V are independent (writing K does not affect V)
- `slot == -1` skips write, does not modify pool
- Invalid slot (`>= total_slots`) raises `IndexError`
- `slot < -1` raises `IndexError`
- Shape/dtype/device mismatch raises `ValueError`
- `data_ptr` unchanged (in-place write)

**Test requirements:** 23 test functions across 10 test classes:
- Single-token decode, multi-token prefill, batched multi-seq, multi-layer
- Block boundary, non-contiguous blocks, repeated slot, block reuse overwrite
- Unwritten slot sentinel, K/V independence, slot=-1 skip
- Slot OOB, slot < -1
- 10 validation error sub-tests (shape, kv_heads, head_dim, block_size, dtype, device, slot length, cache shape, block_size=0)
- In-place `data_ptr` check

**Status:** COMPLETED

**Completion evidence:**
- `mini_vllm/cache/cache_write.py` — `write_to_paged_cache()` with validation
- `mini_vllm/cache/__init__.py` — export
- `mini_vllm/__init__.py` — export
- `tests/test_cache_write.py` — 23 test functions, all pass

**Remaining TODOs:** None (Phase 3 is standalone; wiring into executor deferred to Phase 6+).

---

## Milestone A — PagedAttention Correctness (COMPLETED)

**Target:** Implement the full PagedAttention reference path — cache gather/read, attention math, decode attention, GQA, causal mask, attention scale. Wire cache write into the reference AttentionBackend. Align with contiguous PyTorch SDPA output element-wise.

**Prerequisites:** Phase 2 (KVCachePool), Phase 3 (write_to_paged_cache).

**Allowed scope:**
- `cache_read.py` — `gather_paged_kv()`: gather per-sequence K/V from cache pool by `block_table` + `num_tokens`
- `paged_attention_ref.py` — `AttentionBackendRef` implementation:
  - decode attention with paged KV gather (方案 B: write-first, gather kv_len_after)
  - prefill attention with paged gather + offset-aware causal mask
  - explicit causal mask: `key_pos[j] <= query_pos[i]`, NOT `is_causal=True`
  - attention scale (`1/sqrt(head_dim)`)
  - GQA: `num_heads` → `num_kv_heads` expansion via repeat_interleave
  - non-contiguous physical blocks
  - partial last block
- Wire `write_to_paged_cache` into `AttentionBackendRef.write_kv_cache()`
- Element-wise alignment tests vs contiguous PyTorch SDPA (correct `[B,H,L,D]` layout)
- SDPA layout: input `[L,H,D]` → permute(1,0,2).unsqueeze(0) → `[1,H,L,D]` → squeeze(0).permute(1,0,2) → `[L,H,D]`
- All tests pass

**Forbidden:**
- No Triton kernel
- No Qwen model loading
- No EngineCore execution chain changes
- No FakeModelExecutor / QwenExecutor modification

**Acceptance criteria:**
- Single-sequence decode: paged attention output == contiguous SDPA output element-wise (FP16 1e-3)
- Batched decode: non-overlapping physical blocks, independent output
- Different context lengths per sequence
- Cross-block boundary (KV spans multiple blocks)
- GQA: 2 kv_heads → 8 q_heads via repeat_interleave(4)
- Non-contiguous physical blocks (blocks [7, 3, 5])
- Partial last block (incomplete final block)
- Sentinel: uninitialized pool slots are not read (kv_len after limits reads)
- 方案 B: current decode token K/V pre-written before gather
- Offset-aware causal mask: q_i sees k_0..k_{P+i} with prefix offset P
- Dimension semantics: seq_len != num_heads catches `[B,L,H,D]` vs `[B,H,L,D]` swap
- Future-token leakage: hand-verifiable oracle (position-identified V, extreme V in future)
- Full suite: 312 tests pass

**Test requirements:**
- `TestGatherPagedKV` — 7 tests: full block, partial, multi-block, non-contiguous, zero, round-trip, insufficient raises
- `TestDecodeAttention` — 8 tests: single-seq, multi-seq (non-overlapping blocks), block boundary, non-contiguous, GQA, partial last block, sentinel not read
- `TestPrefillAttention` — 6 tests: full (P=0), P=8+Q=2/3/1, multi-seq, all via offset-aware mask
- `TestDimensionSemantics` — 2 tests: decode/prefill with seq_len(1/3) ≠ num_heads(4), head_dim=4
- `TestFutureTokenLeakage` — 1 test: hand-verifiable (single head, position-identified V, extreme V in future positions)
- `TestAttentionScale` — 1 test: 1/sqrt(head_dim) vs 1.0 produces different output
- `TestCausalMask` — 1 test: paged mask vs hand-constructed tensor mask
- `TestBackendIntegration` — 3 tests: write_kv_cache, factory, allocate_pool

**Status:** COMPLETED

**Completion evidence:**
- `mini_vllm/cache/cache_read.py` — `gather_paged_kv()` with per-block loop, `block_table[logical_idx]` mapping
- `mini_vllm/attention/paged_attention_ref.py` — `AttentionBackendRef` with decode/prefill, GQA, offset-aware mask
- `mini_vllm/cache/__init__.py` — exports `gather_paged_kv`
- `mini_vllm/__init__.py` — exports `gather_paged_kv`, `AttentionBackendRef`
- `mini_vllm/attention/__init__.py` — exports `AttentionBackendRef`
- `tests/test_paged_attention_ref.py` — 27 test items, 27 passed, 0 skipped, 0 failed
- Full suite: 312 tests passed (40.23s)

**Mandatory Follow-ups (reference-only limitations):**
- `gather_paged_kv()` uses per-sequence/per-block Python `while` loop — no vectorized batch gather
- GQA uses `repeat_interleave`, which physically replicates K/V memory (not tensor-core broadcast)
- Attention uses PyTorch `scaled_dot_product_attention` — not a custom kernel
- Not wired into real Qwen ModelRunner (Milestone B)
- No Triton GPU production path (Milestone C)

---

## Milestone B — Qwen End-to-End ModelRunner (COMPLETED)

**Target:** Build the complete Qwen2.5 ModelRunner with modular transformer
components (RMSNorm, RoPE, QKV+GQA, SwiGLU MLP, LM head), weight loading from
HF checkpoint, PagedAttention via AttentionBackendRef, and EngineCore
integration via PagedExecutor — all under a single unified layer-per-sequence
loop with zero fallback to HF model forward.

**Prerequisites:** Milestone A, Phase 3.

**Status:** COMPLETED

**Completion evidence:**
- **New files (8):** `rms_norm.py`, `rotary.py`, `qkv_proj.py`, `mlp.py`,
  `weight_loader.py`, `transformer_layer.py`, `qwen_model.py`,
  `qwen_runner.py`, `paged_executor.py`, `paged_worker.py`
- **Modified files (5):** `base.py` (rope_scaling), `config_adapter.py`
  (rope_scaling), `engine.py` (paged executor_type, device parameter),
  `engine_core.py` (device → ModelInputBuilder), `worker/__init__.py`,
  `mini_vllm/__init__.py`
- **Tests (11 new files):** `test_rms_norm.py` (7), `test_rotary.py` (6),
  `test_qkv_proj.py` (6), `test_mlp.py` (6), `test_qwen_layer.py` (4),
  `test_qwen_model.py` (6), `test_weight_loader.py` (4),
  `test_qwen_runner.py` (3), `test_paged_executor.py` (2),
  `test_hf_alignment.py` (6), `test_engine_e2e.py` (4)
- **Full suite:** 366 passed, 0 failed, 188.01s (Qwen2.5-0.5B-Instruct on RTX 4090)
- **HF alignment:** All 4 levels pass with Qwen2.5-0.5B-Instruct
  - Level 1 (RMSNorm, RoPE, MLP): component-level match (atol ≤ 1e-3)
  - Level 2 (prefill logits): max_abs_diff = 0.0312 (cosine_sim ≈ 0.9997)
  - Level 3 (decode logits): max_abs_diff = 0.0342
  - Level 4 (multi-step greedy): exact token match [12095, 13, 1084, 374, 279, 7772, 3283, 304]
- **Block count verification (B7 memory profile):**
  - Qwen2.5-0.5B model on 12 GiB GPU (RTX 3060): model weight alloc ≈ 1.18 GiB
  - Free after model load (no pool): **7.81 GiB** (8,382,054,400 bytes)
  - Profiled runtime peak increment (128 prefill + 16 decode): **41,042,944 bytes (0.04 GiB)**
  - Added `Config.peak_runtime_estimate: int = 0` (default 0) — wired through PagedExecutor → QwenModelRunner → `_resolve_num_blocks()` → `compute_num_gpu_blocks()`
  - Budget formula with `gpu_memory_utilization=0.90`:
    ```
    post_deduction = free - peak_runtime_estimate - workspace(256MiB)
                 = 7.81 - 0.04 - 0.25 = 7.52 GiB
    budget        = post_deduction × 0.90 = 6.77 GiB
    num_blocks    = budget // 49152 = 147,813
    ```
  - Peak deduction impact (same snapshot): `148,564 → 147,813` (‑751 blocks = 41MB×0.90/49KB) ✅
  - **Final pool: 147,813 blocks (6.77 GiB)** — `KVCachePool.allocate()` succeeded
  - Free after pool allocation: **1.25 GiB** — OOM-free inference verified
  - `bytes_per_block_total = 24 × 2 × 2 × 4 × 64 × 2 = 49,152`
- **EngineCore E2E (4 tests):** Single prefill+8decode, continuous batching,
  mixed prefill+decode, block boundary crossing — all pass
- **Config-driven device design:** Supports `cpu`/`cuda`/`cuda:N`; validated in
  `Config.__post_init__()`; RuntimeError on unavailable CUDA; all components
  (ModelInputBuilder, PagedExecutor, QwenModelRunner, KVCachePool) share
  the same canonical `torch.device()` instance

**Known observations:**

- **F-001 — `test_qkv_deterministic` transient failure**
  Status: NOT REPRODUCIBLE
  Evidence: isolated, prefix groups, and full suite (366/366) all pass
  Action: only reopen if it occurs again with traceback and seed captured
  No `empty_cache()`, retry, or assertion relaxation added.

---

## Milestone C — GPU Production Path (COMPLETED)

**Target:** Build the production GPU path: Triton cache write, Triton PagedAttention decode kernel, GPU chunked prefill. Mixed prefill+decode scheduling. No silent fallback to reference. Benchmark and document.

**Prerequisites:** Milestone B (Qwen ModelRunner as correctness oracle), Milestone A (reference alignment).

**Allowed scope:**
- Triton cache write kernel
- Triton PagedAttention decode kernel
- GPU chunked prefill attention
- Mixed prefill + decode batching
- Correctness alignment vs reference
- Benchmark scripts
- Documentation
- Resume-ready results

**Forbidden:**
- No silent fallback to reference path
- No backward pass
- No multi-GPU

**Status:** COMPLETED (2026-07-14)

**Completion evidence:**
- `mini_vllm/attention/paged_attention_gpu.py` — C1/C2/C3 Triton kernels + `AttentionBackendGPU`
- `tests/test_paged_attention_gpu.py` — 21 GPU attention tests
- `tests/test_no_silent_fallback.py` — 6 fallback policy tests
- `tests/test_engine_e2e.py` — 4 E2E tests with triton backend: ALL PASS
- Real-model alignment: prefill + 8-step greedy decode matches HF and reference exactly
- Full regression suite: 375 pass, 18 skip — no regressions
- `docs/MILESTONE_C_PROGRESS.md` — detailed progress report

**Key bug fix:** Added `.contiguous()` to `triton_cache_write()` for key/value — non-contiguous V from fused QKV projection caused corrupted cache for tokens beyond the first.

**Comparison baseline:** Python per-sequence/per-block gather + `repeat_interleave` + PyTorch SDPA reference implementation within the project (not vLLM, FlashAttention, or other external GPU kernels).

**Deferred:** full Triton prefill kernel (future phase not yet defined); reusable benchmark script; multi-sequence prefix-gather stress benchmark.
