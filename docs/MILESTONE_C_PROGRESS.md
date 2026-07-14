# Milestone C — GPU Production Path

**Status:** COMPLETED (2026-07-14)
**Branch:** feature/v1-style-real-paged-attention
**RTX 3060** — CC 8.6, 28 SMs, 12 GiB, Triton 3.6.0

---

## Plan and acceptance criteria

Fully approved by user. See conversation (2026-07-13) for detailed plan and 8-item revision. Core rules:

- 方案 B lifecycle invariant: QKV → RoPE → **cache write** → gather from cache → attention. No circumvention.
- No silent fallback to reference backend.
- C3 (GPU Chunked Prefill) must be complete before Milestone C is marked COMPLETED.
- C3 may use PyTorch SDPA temporarily but must gather prefix from real paged cache on GPU.
- Benchmark must not preset expected speedup.

---

## Files created / modified

| File | Status | Description |
|------|--------|-------------|
| `mini_vllm/attention/paged_attention_gpu.py` | WRITTEN (~515 lines) | `AttentionBackendGPU` + C1/C2/C3 kernels |
| `mini_vllm/attention/__init__.py` | MODIFIED | Added `AttentionBackendGPU` export |
| `mini_vllm/executor/paged_executor.py` | MODIFIED | Backend selection via `config.attention_backend` (was hardcoded `"reference"`) |
| `tests/test_no_silent_fallback.py` | WRITTEN | 6 tests — CPU fallback, unsupported head_dim/block_size |
| `tests/test_paged_attention_gpu.py` | WRITTEN | 21 tests — CacheWrite, DecodeAttention, GatherPrefix, GQA, Integration |
| `benchmark_results/` | ADDED | Comprehensive benchmark report, plots, raw data |

---

## Implementation status

### C1 — Triton Cache Write (`triton_cache_write`)

**Kernel:** `_cache_write_kernel` — one program per token. Slot == -1 skip. Duplicate slot assertion in Python wrapper.

**Python wrapper:** `triton_cache_write()` in `paged_attention_gpu.py`. Validates shape, dtype, device, block_size, duplicate slots. Calls `.contiguous()` on key and value before kernel launch to support non-contiguous input tensors (e.g., V as a view of a fused QKV tensor).

**Validation:**
- 5 element-wise tests vs reference `write_to_paged_cache()`: ALL PASS
- Verified against real Qwen2.5-0.5B-Instruct model: 0 diff for all cache slots

### C2 — Triton Paged Decode (`triton_decode_attention`)

**Kernel:** `_paged_decode_kernel` — grid `(total_decode, num_q_heads)`. Online softmax in FP32. GQA via `REPEATS` constexpr. Partial last block. `-1` block skip (via `tl.where` — Triton 3.6.0 does not support `break`).

**Python wrapper:** `triton_decode_attention()`. Validates:
- head_dim ∈ {64, 128}
- block_size > 0 and power of 2
- num_q_heads % num_kv_heads == 0
- kv_len_after >= 1 (raise ValueError if any sequence has 0)
- dtype = FP16, device = CUDA

**Validation:**
- 9 element-wise tests vs reference `gather_paged_kv + SDPA`: ALL PASS
- Isolated kernel test on real model data: 0.000031 max diff vs SDPA reference
- Real-model alignment: prefill + 8-step greedy decode matches HF and reference backend exactly
- EngineCore E2E tests (single, continuous batching, block boundary, mixed): ALL PASS

### C3 — GPU Gather Prefix + PyTorch SDPA Prefill

**Gather kernel:** `_gather_prefix_kv_kernel` — grid `(total_prefix_tokens, num_kv_heads)`, reads from paged cache by `(seq_idx, local_pos)` mapping.

**Python wrapper:** `gather_prefix_kv()` — builds flat `seq_idx_map`/`local_pos_map` arrays, launches Triton kernel.

**Prefill attention:** `AttentionBackendGPU.prefill_attention()` — per-sequence PyTorch SDPA on GPU. Gathers prefix from cache via `gather_prefix_kv`, concatenates with current chunk, GQA expand, offset-aware causal mask.

**Temporary status:** PyTorch SDPA for prefill math is marked temporary; full Triton prefill kernel is a deferred optional optimization (future phase not yet defined). Gather path runs on GPU.

### C4 — AttentionBackendGPU class

Complete skeleton with `allocate_pool`, `write_kv_cache`, `decode_attention`, `prefill_attention`. Validates:
- CUDA available (init)
- head_dim ∈ {64, 128} (init)
- block_size power of 2 (allocate_pool)

**Wiring:**
- `backend.py` factory already has `triton → AttentionBackendGPU` path
- `paged_executor.py` now reads `config.attention_backend` instead of hardcoded `"reference"`

---

## Test results (2026-07-14)

### Full regression suite

| File | Tests | Status |
|------|-------|--------|
| `tests/test_paged_attention_gpu.py` | 21 (5 CacheWrite, 9 Decode, 4 Gather, 1 GQA, 2 Integration) | ALL PASS |
| `tests/test_no_silent_fallback.py` | 6 (CPU, head_dim, block_size, factory) | ALL PASS |
| `tests/test_engine_e2e.py` | 4 (single, continuous, block boundary, mixed) | ALL PASS |
| `tests/test_engine_core_unified.py` | 13 | ALL PASS |
| `tests/test_benchmark.py` | 38 | ALL PASS |
| All tests (`tests/`) | 375 pass, 18 skip | NO REGRESSIONS |

### Real-model alignment (Step 2)

- Prefill logits: 0.0 diff between triton and reference backends
- Prefill + 8-step greedy decode: ALL tokens identical to HF and reference backend
- Generated sequence: `[12095, 13, 1084, 374, 279, 7772, 3283, 304]` (prompt: "The capital of France is")

### No-silent-fallback policy (Step 4)

All error cases verified:
- Non-CUDA device: `RuntimeError("requires CUDA")`
- Unsupported head_dim (e.g., 96): `RuntimeError("head_dim=96 not supported")`
- Unsupported block_size (non-power-of-2): `RuntimeError("must be a positive power of 2")`
- Factory triton backend succeeds on CUDA: OK

---

## Critical bug fix: non-contiguous V tensor in triton cache write

**Root cause:** The `v` tensor from `QKVProjection` is a non-contiguous view of the fused QKV output (stride [1152, 64, 1] instead of [128, 64, 1] for a Qwen2.5-0.5B with hidden_size=896, num_kv_heads=2, head_dim=64). The `k` tensor becomes contiguous after RoPE, but `v` skips RoPE and remains non-contiguous. The triton `_cache_write_kernel` assumes contiguous layout, reading V from wrong memory locations for tokens beyond the first.

**Symptom:** Decode token differed between backends (13 vs 264) despite identical prefill and layer 0 slot 5 cache write. Slots 1-4 (prefill tokens 1-4) had corrupted V in the triton backend's cache.

**Fix:** Added `key = key.contiguous()` and `value = value.contiguous()` inside `triton_cache_write()` before kernel launch. The reference backend's `write_to_paged_cache()` already calls `.contiguous()`.

**Verification:**
- Before fix: V cache at slot 1 had 33.94 diff (max) and slot 4 had 47.56 diff
- After fix: all 6 written slots have 0.0 diff

---

## Known open items

- Benchmark measurement completed; reusable benchmark script remains deferred.
- Batch prefix-gather (C3 multiple sequences in one call) not stress-tested

---

## Final status

**Milestone C — COMPLETED**

Implemented:
- Triton KV cache write
- Triton paged decode
- GPU hybrid chunked prefill using paged cache + PyTorch SDPA
- no silent fallback
- real-model and EngineCore validation
- benchmark measurements

Deferred:
- full Triton prefill kernel
- reusable benchmark script
- multi-sequence prefix-gather stress benchmark

---

## Git state

Branch `feature/v1-style-real-paged-attention` with staged changes including all Milestone C files.
