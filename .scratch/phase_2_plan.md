# Phase 2: GPU KV Cache Pool (REVISION 1)

## 1. GPU Memory Budget Formula

**After model weights are loaded on GPU**, query current free memory:

```python
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
```

The **KV cache pool budget** is computed as:

```
available_for_pool =
    current_free_bytes
    - measured_peak_runtime_memory
    - workspace_reserve (fixed 256 MiB)
    - safety_margin
```

Where:

| Term | Definition |
|------|------------|
| `current_free_bytes` | `torch.cuda.mem_get_info()[0]` — memory NOT used by model weights, CUDA context, or any other tensor |
| `measured_peak_runtime_memory` | **Profile-run measurement**: one forward pass with `max_num_batched_tokens` input length. Record `torch.cuda.max_memory_allocated()` before and after; the delta is the activation + temp scratch peak. (For Phase 2, when no real model forward is wired yet, use a default estimate: `max_num_batched_tokens * hidden_size * 4 * num_layers * dtype_bytes` or 0 if no profile run available) |
| `workspace_reserve` | Fixed 256 MiB for Triton compiler cache, CUDA driver workspace, NCCL buffers — not directly measurable but needed |
| `safety_margin` | `available_for_pool * (1 - gpu_memory_utilization)` — the **configurable headroom**. `gpu_memory_utilization=0.90` means 10% of the post-profile budget is left free |

**`gpu_memory_utilization` definition**:
- Range: `(0, 1]`
- Default: `0.90`
- Meaning: what fraction of the **post-deduction** available memory to use for KV cache. `1 - gpu_memory_utilization` is the **fraction of available** left as safety margin.
- Not applied to `current_free_bytes` (which would double-count). It applies only after profile and workspace deductions.

**Summary formula with explicit steps:**

```
step 1: free_bytes = cuda.mem_get_info()[0]
step 2: peak = profile_run_peak() or estimate  (in bytes)
step 3: workspace = 256 * 1024 * 1024           (256 MiB)
step 4: post_deduction = free_bytes - peak - workspace
step 5: safety = post_deduction * (1 - gpu_memory_utilization)
step 6: budget = post_deduction - safety
         = post_deduction * gpu_memory_utilization
step 7: bytes_per_block_total = num_layers * 2 * num_kv_heads * block_size * head_dim * dtype_bytes
step 8: num_blocks = budget // bytes_per_block_total
step 9: num_blocks = max(num_blocks, MIN_BLOCKS)   # MIN_BLOCKS = 16
step 10: if num_blocks < MIN_BLOCKS: raise RuntimeError(...)
```

When `num_gpu_blocks_override` is provided, ALL of the above is skipped (including GPU query). The override value is used directly, with only the MIN_BLOCKS floor check.

## 2. Profile Run

For Phase 2, there is no real model forward wired yet (that's Phase 3+). So:

- When no real model runner exists (FakeModelExecutor path): `measured_peak_runtime_memory = 0`. The budget relies entirely on `workspace_reserve` + `safety_margin`. The user controls the override path for realistic tests.
- When a PagedWorker/QwenWorker provides a real model: a single forward call with dummy input of `max_num_batched_tokens` length is made, and peak memory is recorded. This is **not implemented in Phase 2** — it's a stub that returns 0.
- The `compute_num_gpu_blocks()` function signature allows passing `peak_runtime_estimate: int = 0`.

Implementation:

```python
def compute_num_gpu_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype,
    device: torch.device,
    gpu_memory_utilization: float = 0.90,
    peak_runtime_estimate: int = 0,      # bytes, from profile run
    workspace_reserve: int = 256 * 1024 * 1024,  # 256 MiB
    num_gpu_blocks_override: Optional[int] = None,
) -> int:
```

## 3. KV Pool Storage Layout

**Each layer gets its own pair of 4D tensors, stored as lists:**

```python
# key_caches[i] and value_caches[i] correspond to layer i

key_caches: List[torch.Tensor]   # len = num_layers
value_caches: List[torch.Tensor]  # len = num_layers

# Per-layer tensor:
key_caches[l].shape = [num_blocks, num_kv_heads, block_size, head_dim]
value_caches[l].shape = [num_blocks, num_kv_heads, block_size, head_dim]

dtype = torch.float16
device = "cuda" (production) or "cpu" (tests)
```

Not a single 5D tensor `[num_layers, num_blocks, ...]` — per-layer lists are simpler to pass to per-layer attention kernels, and avoid a giant contiguous allocation that might fragment.

**All layers' memory included in budget:**

```
bytes_per_block_total = num_layers * (2 * num_kv_heads * block_size * head_dim * dtype_bytes)

For Qwen2.5-0.5B:
  bytes_per_block_total = 24 * (2 * 2 * 4 * 64 * 2)
                       = 24 * 2048
                       = 49152 bytes per block (across all layers)
```

This is the single value used in step 7 of the budget formula.

## 4. Pool Allocation — torch.empty(), not torch.zeros()

```python
@dataclass
class KVCachePool:
    key_caches: List[torch.Tensor]
    value_caches: List[torch.Tensor]
    num_blocks: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    device: torch.device

    @staticmethod
    def allocate(...) -> KVCachePool:
        key_caches = [
            torch.empty(num_blocks, num_kv_heads, block_size, head_dim,
                        dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        value_caches = [...]
        ...
```

Correctness relies on:
- Only slot_mapping-specified positions are ever written
- Attention kernel reads only up to `kv_len_after` tokens per sequence
- Unwritten slots are NEVER accessed by the kernel (the block table + length metadata constrain reads)
- After a block is freed and reused, the new writer overwrites the relevant slots before any reader accesses them

**Sentinel test** (not in production, only in test):
- Allocate pool, fill all slots with `float('nan')` (or a sentinel value)
- Write a valid token via slot_mapping
- Read back: written slot ≠ NaN; unwritten slot == NaN
- Block reuse: free block, re-allocate, new write at different offset, old offset still has NaN

## 5. KVCachePool and BlockAllocator: Same num_blocks, Assertion

Construction order:
```python
num_blocks = compute_num_gpu_blocks(...)  # or override

allocator = BlockAllocator(num_blocks=num_blocks)
pool = KVCachePool.allocate(num_blocks=num_blocks, ...)

# Post-construction assertion:
assert pool.num_blocks == allocator.num_total_blocks, (
    f"Pool={pool.num_blocks} != Allocator={allocator.num_total_blocks}"
)
```

`physical_block_id` ∈ [0, num_blocks) indexes both:
- `allocator._free[block_id]` (free/used state)
- `pool.key_caches[layer][block_id, kv_head, offset, head_dim]` (storage)

## 6. Error Handling

| Condition | Error |
|-----------|-------|
| `compute_num_gpu_blocks` returns < 16 | `RuntimeError("GPU memory insufficient: need at least 16 blocks for KV cache, can only fit {n}")` |
| BlockAllocator.allocate() fails | `RuntimeError("OOM: no free block for seq={seq_id} position={pos}")` (pre-existing) |
| `gpu_memory_utilization` > 1.0 or ≤ 0.0 | `ValueError(f"gpu_memory_utilization must be in (0, 1], got {v}")` |
| `num_blocks == 0` in allocate_pool | `ValueError("num_blocks must be > 0")` |
| Override < 16 | Same RuntimeError as insufficient memory |
| `torch.cuda.mem_get_info()` failure | Propagates CUDA error as RuntimeError |

## 7. CPU vs GPU Tests

| Test | Device | GPU? |
|------|--------|------|
| shape, dtype, zero-init | cpu | No |
| pool_allocate_empty | cpu | No |
| scatter_write_single_token | cpu | No |
| scatter_write_multi_token | cpu | No |
| pool_blocks_match_allocator | cpu | No |
| pool_multi_layer | cpu | No |
| pool_bytes_per_block | cpu | No |
| pool_total_slots | cpu | No |
| pool_sentinel_unwritten | cpu | No |
| pool_reuse_after_free | cpu | No |
| estimate_blocks_override | cpu | No |
| estimate_blocks_low_memory | cpu | No |
| estimate_blocks_real_gpu | cuda | Yes, `@pytest.mark.gpu` |

## 8. File Inventory

### Created

| File | Content |
|------|---------|
| `mini_vllm/cache/pool.py` | `KVCachePool` + `compute_num_gpu_blocks()` |
| `tests/test_kv_cache_pool.py` | 13 tests (12 CPU, 1 GPU) |

### Modified

| File | Change |
|------|--------|
| `mini_vllm/config.py` | Add `gpu_memory_utilization: float = 0.9` + validation |
| `mini_vllm/cache/__init__.py` | Export `KVCachePool`, `compute_num_gpu_blocks` |
| `mini_vllm/__init__.py` | Export same |

## 9. Out of Scope

- PagedAttention math / Triton kernel
- `write_kv_cache()` / `decode_attention()` / `prefill_attention()` in AttentionBackend
- Qwen ModelRunner
- Profile-run execution (stub returning 0 in Phase 2)
- Pool wiring into Executor/EngineCore
- Per-block zero-on-free
- Phase 3+ work

## 10. pytest target

250 existing + 13 new = **263 passed**.
