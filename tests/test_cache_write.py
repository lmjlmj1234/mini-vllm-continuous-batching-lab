"""Tests for Phase 3: Cache Write Reference.

Verifies ``write_to_paged_cache()`` scatter-writes K/V tensors into
cache pool tensors by ``slot_mapping``.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.cache.cache_write import write_to_paged_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cpu_pool():
    """Return 4-block cache tensors on CPU, fp16."""
    key_cache = torch.empty(4, 2, 4, 8, dtype=torch.float16)
    value_cache = torch.empty(4, 2, 4, 8, dtype=torch.float16)
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# 1. Single-token decode
# ---------------------------------------------------------------------------

class TestSingleTokenDecode:
    def test_single_token_decode(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        num_kv_heads, head_dim = 2, 8

        key = torch.randn(1, num_kv_heads, head_dim, dtype=torch.float16)
        value = torch.randn(1, num_kv_heads, head_dim, dtype=torch.float16)
        slot_mapping = torch.tensor([5], dtype=torch.long)  # block=1, offset=1

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        block_id = 5 // 4  # 1
        offset = 5 % 4     # 1
        assert torch.equal(key_cache[1, :, 1, :], key[0])
        assert torch.equal(value_cache[1, :, 1, :], value[0])


# ---------------------------------------------------------------------------
# 2. Multi-token prefill
# ---------------------------------------------------------------------------

class TestMultiTokenPrefill:
    def test_multi_token_prefill(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        num_kv_heads, head_dim = 2, 8

        # 8 tokens: slots 0..7 → blocks 0 and 1
        key = torch.randn(8, num_kv_heads, head_dim, dtype=torch.float16)
        value = torch.randn(8, num_kv_heads, head_dim, dtype=torch.float16)
        slot_mapping = torch.arange(8, dtype=torch.long)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        for t in range(8):
            block_id = t // 4
            offset = t % 4
            assert torch.equal(key_cache[block_id, :, offset, :], key[t])
            assert torch.equal(value_cache[block_id, :, offset, :], value[t])


# ---------------------------------------------------------------------------
# 3. Batched multi-sequence
# ---------------------------------------------------------------------------

class TestBatchedMultiSequence:
    def test_batched_multi_sequence(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # 2 seqs, each 1 token → distinct slots
        key = torch.randn(2, nheads, hdim, dtype=torch.float16)
        value = torch.randn(2, nheads, hdim, dtype=torch.float16)
        slot_mapping = torch.tensor([0, 12], dtype=torch.long)
        # seq0: slot 0  (block=0, off=0)
        # seq1: slot 12 (block=3, off=0)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        assert torch.equal(key_cache[0, :, 0, :], key[0])
        assert torch.equal(value_cache[0, :, 0, :], value[0])
        assert torch.equal(key_cache[3, :, 0, :], key[1])
        assert torch.equal(value_cache[3, :, 0, :], value[1])


# ---------------------------------------------------------------------------
# 4. Multi-layer
# ---------------------------------------------------------------------------

class TestMultiLayer:
    def test_multi_layer(self):
        """3 layers, same slots, each layer has distinct values."""
        nheads, hdim = 2, 8
        key_caches = [torch.empty(4, nheads, 4, hdim, dtype=torch.float16)
                      for _ in range(3)]
        value_caches = [torch.empty(4, nheads, 4, hdim, dtype=torch.float16)
                        for _ in range(3)]

        for layer in range(3):
            key = torch.full((1, nheads, hdim), float(layer + 1),
                             dtype=torch.float16)
            value = torch.full((1, nheads, hdim), float(-(layer + 1)),
                               dtype=torch.float16)
            write_to_paged_cache(
                key, value,
                key_caches[layer], value_caches[layer],
                torch.tensor([5], dtype=torch.long),
                block_size=4,
            )

        for layer in range(3):
            expected_k = torch.full((nheads, hdim), float(layer + 1),
                                    dtype=torch.float16)
            expected_v = torch.full((nheads, hdim), float(-(layer + 1)),
                                    dtype=torch.float16)
            assert torch.equal(key_caches[layer][1, :, 1, :], expected_k)
            assert torch.equal(value_caches[layer][1, :, 1, :], expected_v)


# ---------------------------------------------------------------------------
# 5. Block boundary
# ---------------------------------------------------------------------------

class TestBlockBoundary:
    def test_block_boundary(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # 4 tokens: slot 3 (end of block 0), slot 4 (start of block 1),
        #            slot 7 (end of block 1), slot 8 (start of block 2)
        key = torch.randn(4, nheads, hdim, dtype=torch.float16)
        value = torch.randn(4, nheads, hdim, dtype=torch.float16)
        slot_mapping = torch.tensor([3, 4, 7, 8], dtype=torch.long)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        # slot 3 → block=0, off=3
        assert torch.equal(key_cache[0, :, 3, :], key[0])
        assert torch.equal(value_cache[0, :, 3, :], value[0])
        # slot 4 → block=1, off=0
        assert torch.equal(key_cache[1, :, 0, :], key[1])
        assert torch.equal(value_cache[1, :, 0, :], value[1])
        # slot 7 → block=1, off=3
        assert torch.equal(key_cache[1, :, 3, :], key[2])
        assert torch.equal(value_cache[1, :, 3, :], value[2])
        # slot 8 → block=2, off=0
        assert torch.equal(key_cache[2, :, 0, :], key[3])
        assert torch.equal(value_cache[2, :, 0, :], value[3])


# ---------------------------------------------------------------------------
# 6. Non-contiguous physical blocks
# ---------------------------------------------------------------------------

class TestNonContiguousBlocks:
    def test_non_contiguous_blocks(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # Allocate enough blocks: we need block IDs 5, 17, 3
        kc = torch.empty(32, nheads, 4, hdim, dtype=torch.float16)
        vc = torch.empty(32, nheads, 4, hdim, dtype=torch.float16)

        # slots: 5*4+0=20, 17*4+1=69, 3*4+2=14
        keys = torch.randn(3, nheads, hdim, dtype=torch.float16)
        values = torch.randn(3, nheads, hdim, dtype=torch.float16)
        slot_mapping = torch.tensor([20, 69, 14], dtype=torch.long)

        write_to_paged_cache(keys, values, kc, vc,
                             slot_mapping, block_size=4)

        assert torch.equal(kc[5, :, 0, :], keys[0])
        assert torch.equal(kc[17, :, 1, :], keys[1])
        assert torch.equal(kc[3, :, 2, :], keys[2])
        assert torch.equal(vc[5, :, 0, :], values[0])
        assert torch.equal(vc[17, :, 1, :], values[1])
        assert torch.equal(vc[3, :, 2, :], values[2])


# ---------------------------------------------------------------------------
# 7. Repeated slot overwrite
# ---------------------------------------------------------------------------

class TestRepeatedSlotOverwrite:
    def test_repeated_slot_overwrite(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # Two tokens, same slot 5
        key = torch.randn(2, nheads, hdim, dtype=torch.float16)
        value = torch.randn(2, nheads, hdim, dtype=torch.float16)
        slot_mapping = torch.tensor([5, 5], dtype=torch.long)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        # Final value is key[1] / value[1] (last writer wins)
        assert torch.equal(key_cache[1, :, 1, :], key[1])
        assert torch.equal(value_cache[1, :, 1, :], value[1])


# ---------------------------------------------------------------------------
# 8. Block reuse overwrite
# ---------------------------------------------------------------------------

class TestBlockReuseOverwrite:
    def test_block_reuse_overwrite(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # First writer fills slots 0,1,2,3
        k1 = torch.randn(4, nheads, hdim, dtype=torch.float16)
        v1 = torch.randn(4, nheads, hdim, dtype=torch.float16)
        write_to_paged_cache(k1, v1, key_cache, value_cache,
                             torch.arange(4, dtype=torch.long), block_size=4)

        # Second writer (simulating new sequence reusing block 0) overwrites
        # slots 0,1,2,3 with different values
        k2 = torch.randn(4, nheads, hdim, dtype=torch.float16)
        v2 = torch.randn(4, nheads, hdim, dtype=torch.float16)
        write_to_paged_cache(k2, v2, key_cache, value_cache,
                             torch.arange(4, dtype=torch.long), block_size=4)

        # Should be k2/v2 values now
        for off in range(4):
            assert torch.equal(key_cache[0, :, off, :], k2[off])
            assert torch.equal(value_cache[0, :, off, :], v2[off])


# ---------------------------------------------------------------------------
# 9. Unwritten slots unchanged
# ---------------------------------------------------------------------------

class TestUnwrittenSlotsUnchanged:
    def test_unwritten_slots_unchanged(self, cpu_pool):
        key_cache, value_cache = cpu_pool

        # Fill initial values everywhere
        k_init = torch.randn_like(key_cache)
        v_init = torch.randn_like(value_cache)
        key_cache.copy_(k_init)
        value_cache.copy_(v_init)

        # Write only slots 2 and 10
        key = torch.randn(2, key_cache.shape[1], key_cache.shape[3],
                          dtype=torch.float16)
        value = torch.randn(2, key_cache.shape[1], key_cache.shape[3],
                            dtype=torch.float16)
        write_to_paged_cache(key, value, key_cache, value_cache,
                             torch.tensor([2, 10], dtype=torch.long),
                             block_size=4)

        # Slots 2 and 10 are overwritten
        assert torch.equal(key_cache[0, :, 2, :], key[0])
        assert torch.equal(value_cache[0, :, 2, :], value[0])
        # slot 10 → block=2, off=2
        assert torch.equal(key_cache[2, :, 2, :], key[1])
        assert torch.equal(value_cache[2, :, 2, :], value[1])

        # All other slots unchanged
        for block in range(4):
            for off in range(4):
                slot = block * 4 + off
                if slot in (2, 10):
                    continue
                assert torch.equal(key_cache[block, :, off, :],
                                   k_init[block, :, off, :])
                assert torch.equal(value_cache[block, :, off, :],
                                   v_init[block, :, off, :])


# ---------------------------------------------------------------------------
# 10. K/V independence
# ---------------------------------------------------------------------------

class TestKVIndependence:
    def test_kv_independence(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # Same slot, K = ones, V = zeros
        key = torch.ones(1, nheads, hdim, dtype=torch.float16)
        value = torch.zeros(1, nheads, hdim, dtype=torch.float16)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             torch.tensor([5], dtype=torch.long), block_size=4)

        # K is ones, V is zeros at the same slot
        assert (key_cache[1, :, 1, :] == 1).all()
        assert (value_cache[1, :, 1, :] == 0).all()


# ---------------------------------------------------------------------------
# 11. Slot = -1 skipped
# ---------------------------------------------------------------------------

class TestSlotNegativeOneSkipped:
    def test_slot_negative_one_skipped(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        # Init with zeros
        key_cache.zero_()
        value_cache.zero_()

        # Write first token with -1 (should be skipped), then slot 5, then -1,
        # then slot 8
        key = torch.full((4, nheads, hdim), 42.0, dtype=torch.float16)
        value = torch.full((4, nheads, hdim), 42.0, dtype=torch.float16)
        slot_mapping = torch.tensor([-1, 5, -1, 8], dtype=torch.long)

        write_to_paged_cache(key, value, key_cache, value_cache,
                             slot_mapping, block_size=4)

        # Slot 5 (= block=1, off=1) should be 42
        assert (key_cache[1, :, 1, :] == 42.0).all()
        assert (value_cache[1, :, 1, :] == 42.0).all()
        # Slot 8 (= block=2, off=0) should be 42
        assert (key_cache[2, :, 0, :] == 42.0).all()
        assert (value_cache[2, :, 0, :] == 42.0).all()
        # Slot 0 (= block=0, off=0) should remain 0 (key[0] was skipped)
        assert (key_cache[0, :, 0, :] == 0).all()
        assert (value_cache[0, :, 0, :] == 0).all()


# ---------------------------------------------------------------------------
# 12. Slot out of range
# ---------------------------------------------------------------------------

class TestSlotOutOfRange:
    def test_slot_out_of_range(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)
        total_slots = 4 * 4  # 16

        with pytest.raises(IndexError, match="out of range"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([total_slots], dtype=torch.long),
                                 block_size=4)


# ---------------------------------------------------------------------------
# 13. Slot < -1
# ---------------------------------------------------------------------------

class TestSlotLessThanNegativeOne:
    def test_slot_less_than_negative_one(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)

        with pytest.raises(IndexError, match="out of range"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([-2], dtype=torch.long),
                                 block_size=4)


# ---------------------------------------------------------------------------
# 14. Shape / dtype / device errors
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_key_value_shape_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim + 1, dtype=torch.float16)  # wrong
        with pytest.raises(ValueError, match="key shape.*!=.*value shape"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_slot_mapping_length_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(2, nheads, hdim, dtype=torch.float16)
        value = torch.randn(2, nheads, hdim, dtype=torch.float16)
        with pytest.raises(ValueError, match="slot_mapping length"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_num_kv_heads_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        hdim = 8
        key = torch.randn(1, 4, hdim, dtype=torch.float16)  # 4 heads vs 2
        value = torch.randn(1, 4, hdim, dtype=torch.float16)
        with pytest.raises(ValueError, match="num_kv_heads"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_head_dim_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads = 2
        key = torch.randn(1, nheads, 16, dtype=torch.float16)  # 16 vs 8
        value = torch.randn(1, nheads, 16, dtype=torch.float16)
        with pytest.raises(ValueError, match="head_dim"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_block_size_mismatch_with_cache(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)
        # cache has block_size=4 but argument says 8
        with pytest.raises(ValueError, match="block_size"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=8)

    def test_dtype_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float32)  # fp32 vs fp16
        value = torch.randn(1, nheads, hdim, dtype=torch.float32)
        with pytest.raises(ValueError, match="dtype"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_device_mismatch(self, cpu_pool):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float16, device="cuda")
        value = torch.randn(1, nheads, hdim, dtype=torch.float16, device="cuda")
        with pytest.raises(ValueError, match="device"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=4)

    def test_block_size_zero(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)
        with pytest.raises(ValueError, match="block_size must be > 0"):
            write_to_paged_cache(key, value, key_cache, value_cache,
                                 torch.tensor([0], dtype=torch.long), block_size=0)

    def test_cache_shape_mismatch(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8
        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)
        # key_cache and value_cache shapes differ
        bad_value = torch.empty(2, nheads, 4, hdim, dtype=torch.float16)
        with pytest.raises(ValueError, match="cache shape"):
            write_to_paged_cache(key, value, key_cache, bad_value,
                                 torch.tensor([0], dtype=torch.long), block_size=4)


# ---------------------------------------------------------------------------
# 15. In-place: data_ptr unchanged
# ---------------------------------------------------------------------------

class TestInPlace:
    def test_inplace_does_not_create_new_tensor(self, cpu_pool):
        key_cache, value_cache = cpu_pool
        nheads, hdim = 2, 8

        ptr_before = key_cache.data_ptr()

        key = torch.randn(1, nheads, hdim, dtype=torch.float16)
        value = torch.randn(1, nheads, hdim, dtype=torch.float16)
        write_to_paged_cache(key, value, key_cache, value_cache,
                             torch.tensor([5], dtype=torch.long), block_size=4)

        assert key_cache.data_ptr() == ptr_before, "key_cache was re-allocated!"
