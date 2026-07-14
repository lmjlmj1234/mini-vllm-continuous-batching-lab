"""Tests for Phase 2: GPU KV Cache Pool.

Covers ``KVCachePool`` dataclass (allocation, reset, properties) and
``compute_num_gpu_blocks()`` budget formula.  12 CPU tests + 1 GPU test.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.cache.pool import KVCachePool, compute_num_gpu_blocks, MIN_BLOCKS


# ---------------------------------------------------------------------------
# KVCachePool — allocation (3 tests)
# ---------------------------------------------------------------------------


class TestKVCachePoolAllocate:
    """Shape validation, default device, invalid parameter rejection."""

    def test_allocate_shapes(self):
        """Basic allocation shapes, dtype, device, default fallback."""
        pool = KVCachePool.allocate(
            num_layers=2, num_blocks=8, block_size=4,
            num_kv_heads=4, head_dim=128,
            dtype=torch.float16, device=torch.device("cpu"),
        )
        assert pool.num_layers == 2
        assert pool.num_blocks == 8
        assert pool.block_size == 4
        assert pool.num_kv_heads == 4
        assert pool.head_dim == 128
        assert pool.dtype == torch.float16
        assert len(pool.key_caches) == 2
        assert len(pool.value_caches) == 2
        for l in range(2):
            assert pool.key_caches[l].shape == (8, 4, 4, 128)
            assert pool.value_caches[l].shape == (8, 4, 4, 128)

        # Default device: CUDA when available, else CPU
        pool2 = KVCachePool.allocate(1, 4, 2, 2, 64)
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert pool2.device.type == expected

    def test_allocate_invalid_params(self):
        """All zero/negative params raise ValueError."""
        cases = [
            (1, 0, 4, 2, 64, "num_blocks"),
            (0, 8, 4, 2, 64, "num_layers"),
            (1, 8, 0, 2, 64, "block_size"),
            (1, 8, 4, 0, 64, "num_kv_heads"),
            (1, 8, 4, 2, 0, "head_dim"),
        ]
        for args in cases:
            with pytest.raises(ValueError, match=args[-1]):
                KVCachePool.allocate(*args[:5])

    def test_empty_sentinel(self):
        """torch.empty() means NOT all zeros on fresh allocation."""
        pool = KVCachePool.allocate(
            1, 8, 4, 2, 64, device=torch.device("cpu"))
        assert not (pool.get_key_cache(0) == 0).all(), (
            "KVCachePool used torch.zeros() — all values are 0"
        )


# ---------------------------------------------------------------------------
# KVCachePool — accessors (1 test)
# ---------------------------------------------------------------------------


class TestKVCachePoolAccessors:
    """get_key_cache, get_value_cache return correct tensor references."""

    def test_accessors(self):
        pool = KVCachePool.allocate(3, 8, 4, 2, 64, device=torch.device("cpu"))
        assert pool.get_key_cache(0) is pool.key_caches[0]
        assert pool.get_key_cache(1) is pool.key_caches[1]
        assert pool.get_value_cache(2) is pool.value_caches[2]
        assert pool.get_value_cache(2).shape == (8, 2, 4, 64)


# ---------------------------------------------------------------------------
# KVCachePool — reset (1 test)
# ---------------------------------------------------------------------------


class TestKVCachePoolReset:
    """KVCachePool.reset() zeros all tensors."""

    def test_reset_zeros(self):
        pool = KVCachePool.allocate(
            2, 4, 2, 2, 8, device=torch.device("cpu"))
        for k, v in zip(pool.key_caches, pool.value_caches):
            k.fill_(1.5)
            v.fill_(2.5)
        pool.reset()
        for k, v in zip(pool.key_caches, pool.value_caches):
            assert (k == 0).all()
            assert (v == 0).all()


# ---------------------------------------------------------------------------
# KVCachePool — derived properties (1 test)
# ---------------------------------------------------------------------------


class TestKVCachePoolProperties:
    """total_slots, bytes_per_block_*, total_bytes."""

    def test_properties(self):
        """All four derived properties return correct byte counts."""
        pool = KVCachePool.allocate(
            3, 16, 4, 4, 128, dtype=torch.float16,
            device=torch.device("cpu"),
        )
        assert pool.total_slots == 16 * 4
        assert pool.bytes_per_block_per_layer == 2 * 4 * 4 * 128 * 2
        assert pool.bytes_per_block_total == 3 * pool.bytes_per_block_per_layer
        assert pool.total_bytes == 16 * pool.bytes_per_block_total

    def test_properties_different_dtype(self):
        """bfloat16 and float32 give different byte counts."""
        pool_fp32 = KVCachePool.allocate(
            1, 8, 4, 2, 64, dtype=torch.float32,
            device=torch.device("cpu"),
        )
        pool_bf16 = KVCachePool.allocate(
            1, 8, 4, 2, 64, dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        # fp32: 4 bytes per element; bf16: 2 bytes per element
        assert pool_fp32.bytes_per_block_per_layer == 2 * pool_bf16.bytes_per_block_per_layer


# ---------------------------------------------------------------------------
# compute_num_gpu_blocks — override (1 test)
# ---------------------------------------------------------------------------


class TestComputeNumGpuBlocksOverride:
    """num_gpu_blocks_override returns value directly (bypassing GPU)."""

    def test_override_returns_value(self):
        n = compute_num_gpu_blocks(
            32, 8, 128, 16, num_gpu_blocks_override=256)
        assert n == 256

    def test_override_below_minimum_raises(self):
        with pytest.raises(RuntimeError, match="num_gpu_blocks_override"):
            compute_num_gpu_blocks(
                32, 8, 128, 16, num_gpu_blocks_override=MIN_BLOCKS - 1)


# ---------------------------------------------------------------------------
# compute_num_gpu_blocks — CPU fallback (1 test)
# ---------------------------------------------------------------------------


class TestComputeNumGpuBlocksCPU:
    """CPU fallback returns MIN_BLOCKS without querying GPU."""

    def test_cpu_fallback(self):
        n = compute_num_gpu_blocks(
            32, 8, 128, 16, device=torch.device("cpu"))
        assert n == MIN_BLOCKS


# ---------------------------------------------------------------------------
# compute_num_gpu_blocks — utilization validation (1 test)
# ---------------------------------------------------------------------------


class TestComputeNumGpuBlocksUtilization:
    """gpu_memory_utilization range validation."""

    def test_utilization_validation(self):
        for util in [0.0, 1.5, -0.1]:
            with pytest.raises(ValueError, match="gpu_memory_utilization"):
                compute_num_gpu_blocks(1, 1, 64, 4, gpu_memory_utilization=util)


# ---------------------------------------------------------------------------
# compute_num_gpu_blocks — GPU integration (1 test, skip-if-no-cuda)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestComputeNumGpuBlocksGPU:
    """Real GPU memory budget query (1 skipped on non-CUDA)."""

    def test_gpu_basic(self):
        """At least MIN_BLOCKS; higher utilization yields >= blocks."""
        n50 = compute_num_gpu_blocks(32, 8, 128, 16, gpu_memory_utilization=0.5)
        n90 = compute_num_gpu_blocks(32, 8, 128, 16, gpu_memory_utilization=0.9)
        assert n90 >= MIN_BLOCKS
        assert n90 >= n50, f"0.9->{n90} < 0.5->{n50}"

        # Override still works on GPU
        assert compute_num_gpu_blocks(
            32, 8, 128, 16, num_gpu_blocks_override=128) == 128
