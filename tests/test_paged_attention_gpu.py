"""GPU paged attention tests — Triton kernel correctness vs reference.

Tests C1/C2/C3 kernels from ``paged_attention_gpu.py`` against the
reference implementations (``write_to_paged_cache``, ``gather_paged_kv``,
``AttentionBackendRef``).

All Triton/CUDA tests are skipped when CUDA is unavailable.  CPU fallback
tests live in ``test_no_silent_fallback.py``.
"""

from __future__ import annotations

import pytest
import torch

from mini_vllm.attention.backend import AttentionBackend
from mini_vllm.attention.paged_attention_gpu import (
    AttentionBackendGPU,
    gather_prefix_kv,
    triton_cache_write,
    triton_decode_attention,
)
from mini_vllm.attention.paged_attention_ref import AttentionBackendRef
from mini_vllm.cache.cache_read import gather_paged_kv as ref_gather_paged_kv
from mini_vllm.cache.cache_write import write_to_paged_cache
from mini_vllm.cache.pool import KVCachePool
from mini_vllm.model_runner.base import (
    AttentionGroup,
    AttentionMetadata,
    ModelConfig,
)

# ---------------------------------------------------------------------------
# Module-level skip: all tests in this file need CUDA.
# ---------------------------------------------------------------------------

cuda_available = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not cuda_available, reason="CUDA not available")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda")


def make_pool(
    block_size: int = 16,
    num_blocks: int = 64,
    num_kv_heads: int = 2,
    head_dim: int = 64,
    num_layers: int = 1,
) -> KVCachePool:
    """Create a GPU KVCachePool for testing."""
    return KVCachePool.allocate(
        num_layers=num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.float16,
        device=DEVICE,
    )


def _write_ref(key, value, pool, slot_mapping, layer=0):
    """Write to cache using the reference implementation."""
    write_to_paged_cache(
        key, value,
        pool.key_caches[layer], pool.value_caches[layer],
        slot_mapping, pool.block_size,
    )


def _write_triton(key, value, pool, slot_mapping, layer=0):
    """Write to cache using the Triton implementation."""
    triton_cache_write(
        key, value,
        pool.key_caches[layer], pool.value_caches[layer],
        slot_mapping, pool.block_size,
    )
    torch.cuda.synchronize()


def _write_and_compare(key, value, pool, slot_mapping):
    """Write to pool using both ref and triton and assert per-slot equality.

    Only compares positions specified in ``slot_mapping`` (unwritten cache
    positions contain uninitialized garbage from ``torch.empty()`` and may
    differ between pools).
    """
    pool_ref = make_pool(
        block_size=pool.block_size, num_blocks=pool.num_blocks,
        num_kv_heads=pool.num_kv_heads, head_dim=pool.head_dim,
    )
    _write_ref(key, value, pool_ref, slot_mapping)
    _write_triton(key, value, pool, slot_mapping)
    torch.cuda.synchronize()

    block_size = pool.block_size
    for i, slot in enumerate(slot_mapping.tolist()):
        if slot == -1:
            continue
        block_id = slot // block_size
        offset = slot % block_size
        triton_k = pool.key_caches[0][block_id, :, offset, :]
        ref_k = pool_ref.key_caches[0][block_id, :, offset, :]
        assert torch.equal(triton_k, ref_k), (
            f"key_cache mismatch at slot={slot} (block={block_id}, offset={offset})"
        )
        triton_v = pool.value_caches[0][block_id, :, offset, :]
        ref_v = pool_ref.value_caches[0][block_id, :, offset, :]
        assert torch.equal(triton_v, ref_v), (
            f"value_cache mismatch at slot={slot} (block={block_id}, offset={offset})"
        )


def _kv(shape, base=0.0):
    """Create deterministic K/V tensors (reproducible, not random)."""
    t = torch.empty(*shape, dtype=torch.float16, device=DEVICE)
    flat = t.flatten()
    for i in range(flat.numel()):
        flat[i] = base + float(i)
    return t


def _decode_ref(query, pool, block_table, kv_len_after, block_size, num_kv_heads, layer=0):
    """Reference decode via gather_paged_kv + SDPA (matching AttentionBackendRef)."""
    num_decode = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    n_repeats = num_heads // num_kv_heads
    scale = head_dim ** -0.5
    outputs = []
    for i in range(num_decode):
        seq_len = int(kv_len_after[i].item())
        block_ids = [int(b.item()) for b in block_table[i] if b.item() != -1]
        k, v = ref_gather_paged_kv(
            pool.key_caches[layer], pool.value_caches[layer],
            block_ids, seq_len, block_size,
        )
        k = k.repeat_interleave(n_repeats, dim=1)
        v = v.repeat_interleave(n_repeats, dim=1)
        q_sdpa = query[i].unsqueeze(0).unsqueeze(2)   # [1, H, 1, D]
        k_sdpa = k.permute(1, 0, 2).unsqueeze(0)      # [1, H, T, D]
        v_sdpa = v.permute(1, 0, 2).unsqueeze(0)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, scale=scale, is_causal=False,
        )
        outputs.append(out.squeeze(2))
    return torch.cat(outputs, dim=0)


# =============================================================================
# C1 — CacheWrite tests (5 tests)
# =============================================================================


class TestCacheWrite:
    """triton_cache_write vs write_to_paged_cache — element-wise equality."""

    @pytest.fixture(autouse=True)
    def _sync_cuda(self):
        torch.cuda.synchronize()
        yield

    def test_single_token(self):
        """Single token write matches reference."""
        pool = make_pool()
        key = _kv((1, 2, 64), base=10.0)
        value = _kv((1, 2, 64), base=100.0)
        slot = torch.tensor([5], dtype=torch.long, device=DEVICE)
        _write_and_compare(key, value, pool, slot)

    def test_multi_token_noncontiguous(self):
        """Multiple tokens written to noncontiguous slots match reference."""
        pool = make_pool(block_size=16, num_blocks=32)
        key = _kv((5, 2, 64), base=10.0)
        value = _kv((5, 2, 64), base=200.0)
        slots = torch.tensor([0, 17, 33, 50, 99], dtype=torch.long, device=DEVICE)
        _write_and_compare(key, value, pool, slots)

    def test_block_boundary(self):
        """Token at the last slot of a block writes correctly."""
        pool = make_pool(block_size=4, num_blocks=8)
        key = _kv((2, 2, 64), base=10.0)
        value = _kv((2, 2, 64), base=300.0)
        slots = torch.tensor([7, 8], dtype=torch.long, device=DEVICE)
        _write_and_compare(key, value, pool, slots)

    def test_slot_negative_one_skips(self):
        """slot=-1 skips write; other slots still written correctly."""
        pool = make_pool()
        key = _kv((3, 2, 64), base=10.0)
        value = _kv((3, 2, 64), base=400.0)
        slots = torch.tensor([0, -1, 15], dtype=torch.long, device=DEVICE)
        _write_and_compare(key, value, pool, slots)

    def test_duplicate_slot_raises(self):
        """Duplicate non--1 slots raise AssertionError."""
        pool = make_pool()
        key = _kv((2, 2, 64), base=10.0)
        value = _kv((2, 2, 64), base=500.0)
        slots = torch.tensor([5, 5], dtype=torch.long, device=DEVICE)

        with pytest.raises(AssertionError, match="Duplicate"):
            _write_triton(key, value, pool, slots)


# =============================================================================
# C2 — DecodeAttention tests (9 tests)
# =============================================================================


class TestDecodeAttention:
    """triton_decode_attention vs gather+sdpa reference."""

    @pytest.fixture(autouse=True)
    def _sync_cuda(self):
        torch.cuda.synchronize()
        yield

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.block_size = 16
        self.num_blocks = 64
        self.num_kv_heads = 2
        self.num_q_heads = 4
        self.head_dim = 64
        self.pool = make_pool(
            block_size=self.block_size,
            num_blocks=self.num_blocks,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def _write_kv_to_cache(self, num_tokens, base=10.0, slot_offset=0):
        """Write KV tokens into cache and return block table entry.

        Tokens fill slots ``[slot_offset, slot_offset + num_tokens)``.
        The returned block table lists the physical blocks covering those
        slots, suitable for decode attention.
        """
        key = _kv((num_tokens, self.num_kv_heads, self.head_dim), base=base)
        value = _kv((num_tokens, self.num_kv_heads, self.head_dim), base=base + 1000.0)
        slots = torch.arange(slot_offset, slot_offset + num_tokens, dtype=torch.long, device=DEVICE)
        _write_triton(key, value, self.pool, slots)
        # Determine physical blocks from slot range
        start_block = slot_offset // self.block_size
        end_block = (slot_offset + num_tokens - 1) // self.block_size
        block_ids = list(range(start_block, end_block + 1))
        return torch.tensor(block_ids, dtype=torch.long, device=DEVICE)

    def _make_block_table(self, block_ids_list):
        """Build a 1-row block table from a list of physical block IDs.

        Pads with -1 to ``max_blocks`` (set per-test).
        """
        return block_ids_list  # caller wraps with unsqueeze(0) and pads

    # --- individual tests ---

    def test_single_sequence(self):
        """Single decode sequence matches reference."""
        seq_len = 7
        bt = self._write_kv_to_cache(seq_len, base=10.0, slot_offset=0)
        block_table = bt.unsqueeze(0)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        query = _kv((1, self.num_q_heads, self.head_dim), base=50.0)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_multi_sequence(self):
        """Multiple decode sequences produce correct outputs."""
        seq0_len = 10
        bt0 = self._write_kv_to_cache(seq0_len, base=10.0, slot_offset=0)
        seq1_len = 15
        bt1 = self._write_kv_to_cache(seq1_len, base=200.0, slot_offset=20)

        max_blocks = max(bt0.shape[0], bt1.shape[0])
        block_table = torch.zeros(2, max_blocks, dtype=torch.long, device=DEVICE) - 1
        block_table[0, :bt0.shape[0]] = bt0
        block_table[1, :bt1.shape[0]] = bt1

        query = _kv((2, self.num_q_heads, self.head_dim), base=50.0)
        kv_len_after = torch.tensor([seq0_len, seq1_len], dtype=torch.int32, device=DEVICE)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_different_lengths(self):
        """Sequences with different KV lengths produce correct outputs."""
        seq0_len = 3
        bt0 = self._write_kv_to_cache(seq0_len, base=10.0, slot_offset=0)
        seq1_len = 23
        bt1 = self._write_kv_to_cache(seq1_len, base=300.0, slot_offset=30)

        max_blocks = max(bt0.shape[0], bt1.shape[0])
        block_table = torch.zeros(2, max_blocks, dtype=torch.long, device=DEVICE) - 1
        block_table[0, :bt0.shape[0]] = bt0
        block_table[1, :bt1.shape[0]] = bt1

        query = _kv((2, self.num_q_heads, self.head_dim), base=50.0)
        kv_len_after = torch.tensor([seq0_len, seq1_len], dtype=torch.int32, device=DEVICE)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_partial_last_block(self):
        """KV length not aligned to block_size works correctly."""
        seq_len = self.block_size + 3
        bt = self._write_kv_to_cache(seq_len, base=10.0, slot_offset=0)
        block_table = bt.unsqueeze(0)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        query = _kv((1, self.num_q_heads, self.head_dim), base=50.0)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_multiple_full_blocks(self):
        """Multiple full blocks (40 tokens = 2.5 blocks) work correctly."""
        seq_len = 40
        bt = self._write_kv_to_cache(seq_len, base=10.0, slot_offset=0)
        block_table = bt.unsqueeze(0)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        query = _kv((1, self.num_q_heads, self.head_dim), base=50.0)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_noncontiguous_block_table(self):
        """Noncontiguous physical blocks (block_table has gaps) work."""
        pool = make_pool(block_size=16, num_blocks=64, num_kv_heads=2, head_dim=64)
        # Write to block 10 (slots 160-175) and block 50 (slots 800-803)
        key0 = _kv((16, 2, 64), base=10.0)
        val0 = _kv((16, 2, 64), base=1000.0)
        slots0 = torch.arange(160, 160 + 16, dtype=torch.long, device=DEVICE)
        _write_triton(key0, val0, pool, slots0)

        key1 = _kv((3, 2, 64), base=50.0)
        val1 = _kv((3, 2, 64), base=5000.0)
        slots1 = torch.arange(800, 800 + 3, dtype=torch.long, device=DEVICE)
        _write_triton(key1, val1, pool, slots1)

        kv_len = 19
        query = _kv((1, 4, 64), base=99.0)
        block_table = torch.tensor([[10, 50]], dtype=torch.long, device=DEVICE)
        kv_len_after = torch.tensor([kv_len], dtype=torch.int32, device=DEVICE)

        ref_out = _decode_ref(query, pool, block_table, kv_len_after, 16, 2)
        triton_out = triton_decode_attention(
            query, pool.key_caches[0], pool.value_caches[0],
            block_table, kv_len_after, 16,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_kv_len_zero_raises(self):
        """kv_len_after < 1 raises ValueError."""
        pool = make_pool()
        query = _kv((1, 4, 64), base=50.0)
        block_table = torch.zeros(1, 1, dtype=torch.long, device=DEVICE) - 1
        kv_len_after = torch.tensor([0], dtype=torch.int32, device=DEVICE)

        with pytest.raises(ValueError, match="kv_len"):
            triton_decode_attention(
                query, pool.key_caches[0], pool.value_caches[0],
                block_table, kv_len_after, 16,
            )

    def test_padding_blocks_ignored(self):
        """-1 entries in block_table are ignored (padding after last block)."""
        seq_len = 5
        bt = self._write_kv_to_cache(seq_len, base=10.0, slot_offset=0)
        padded = torch.cat([bt, torch.tensor([-1, -1], dtype=torch.long, device=DEVICE)])
        block_table = padded.unsqueeze(0)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        query = _kv((1, self.num_q_heads, self.head_dim), base=50.0)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)

    def test_long_sequence_large_cache(self):
        """Longer sequence (3+ blocks) doesn't overflow or drift."""
        seq_len = 50
        bt = self._write_kv_to_cache(seq_len, base=10.0, slot_offset=0)
        block_table = bt.unsqueeze(0)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
        query = _kv((1, self.num_q_heads, self.head_dim), base=50.0)

        ref_out = _decode_ref(query, self.pool, block_table, kv_len_after,
                              self.block_size, self.num_kv_heads)
        triton_out = triton_decode_attention(
            query, self.pool.key_caches[0], self.pool.value_caches[0],
            block_table, kv_len_after, self.block_size,
        )
        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)


# =============================================================================
# C3 — GatherPrefix tests (4 tests)
# =============================================================================


class TestGatherPrefix:
    """gather_prefix_kv vs ref_gather_paged_kv — element-wise equality."""

    def test_full_prefix(self):
        """Gather all cached tokens (cached_len == kv_len)."""
        block_size = 16
        pool = make_pool(block_size=block_size, num_blocks=32, num_kv_heads=2, head_dim=64)
        seq_len = 10
        key = _kv((seq_len, 2, 64), base=10.0)
        value = _kv((seq_len, 2, 64), base=1000.0)
        slots = torch.arange(seq_len, dtype=torch.long, device=DEVICE)
        _write_triton(key, value, pool, slots)

        # Gather using gather_prefix_kv
        block_table = torch.tensor([[0]], dtype=torch.long, device=DEVICE)
        gathered_k, gathered_v = gather_prefix_kv(
            pool.key_caches[0], pool.value_caches[0],
            block_table, [seq_len], [0], block_size,
        )

        # Reference gather
        ref_k, ref_v = ref_gather_paged_kv(
            pool.key_caches[0], pool.value_caches[0],
            [0], seq_len, block_size,
        )

        assert torch.equal(gathered_k, ref_k)
        assert torch.equal(gathered_v, ref_v)

    def test_partial_large_prefix(self):
        """Gather a subset of cached tokens spanning multiple blocks."""
        block_size = 16
        pool = make_pool(block_size=block_size, num_blocks=32, num_kv_heads=2, head_dim=64)
        seq_len = 25
        key = _kv((seq_len, 2, 64), base=10.0)
        value = _kv((seq_len, 2, 64), base=1000.0)
        slots = torch.arange(seq_len, dtype=torch.long, device=DEVICE)
        _write_triton(key, value, pool, slots)

        # gather_prefix_kv with cached_len=25
        block_table = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
        gathered_k, gathered_v = gather_prefix_kv(
            pool.key_caches[0], pool.value_caches[0],
            block_table, [seq_len], [0], block_size,
        )

        ref_k, ref_v = ref_gather_paged_kv(
            pool.key_caches[0], pool.value_caches[0],
            [0, 1], seq_len, block_size,
        )

        assert torch.equal(gathered_k, ref_k)
        assert torch.equal(gathered_v, ref_v)

    def test_noncontiguous_blocks(self):
        """Gather from noncontiguous physical blocks."""
        block_size = 16
        pool = make_pool(block_size=block_size, num_blocks=64, num_kv_heads=2, head_dim=64)
        seq_len = 20
        key = _kv((seq_len, 2, 64), base=10.0)
        value = _kv((seq_len, 2, 64), base=1000.0)
        # Scatter to block 10 (slot 160-175) and block 50 (slot 800-803 for 4 tokens)
        slots0 = torch.arange(160, 160 + 16, dtype=torch.long, device=DEVICE)
        _write_triton(key[:16], value[:16], pool, slots0)
        slots1 = torch.arange(800, 800 + 4, dtype=torch.long, device=DEVICE)
        _write_triton(key[16:20], value[16:20], pool, slots1)

        # Block table: logical 0->10, logical 1->50
        block_table = torch.tensor([[10, 50]], dtype=torch.long, device=DEVICE)
        gathered_k, gathered_v = gather_prefix_kv(
            pool.key_caches[0], pool.value_caches[0],
            block_table, [seq_len], [0], block_size,
        )

        ref_k, ref_v = ref_gather_paged_kv(
            pool.key_caches[0], pool.value_caches[0],
            [10, 50], seq_len, block_size,
        )

        assert torch.equal(gathered_k, ref_k)
        assert torch.equal(gathered_v, ref_v)

    def test_zero_prefix_returns_empty(self):
        """cached_len=0 returns empty tensors with correct shape."""
        block_size = 16
        pool = make_pool(block_size=block_size, num_blocks=32, num_kv_heads=2, head_dim=64)
        block_table = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

        gathered_k, gathered_v = gather_prefix_kv(
            pool.key_caches[0], pool.value_caches[0],
            block_table, [0], [0], block_size,
        )

        assert gathered_k.shape == (0, 2, 64)
        assert gathered_v.shape == (0, 2, 64)


# =============================================================================
# GQA test (1 test)
# =============================================================================


class TestGQA:
    """GQA expansion: triton decode with GQA matches reference."""

    def test_gqa_expansion(self):
        """num_q_heads=4, num_kv_heads=2 (repeats=2) produces correct output."""
        block_size = 16
        head_dim = 64
        num_kv_heads = 2
        num_q_heads = 4
        pool = make_pool(block_size=block_size, num_blocks=32, num_kv_heads=num_kv_heads, head_dim=head_dim)

        seq_len = 8
        key = _kv((seq_len, num_kv_heads, head_dim), base=10.0)
        value = _kv((seq_len, num_kv_heads, head_dim), base=1000.0)
        slots = torch.arange(seq_len, dtype=torch.long, device=DEVICE)
        _write_triton(key, value, pool, slots)

        query = _kv((1, num_q_heads, head_dim), base=50.0)
        block_table = torch.tensor([[0]], dtype=torch.long, device=DEVICE)
        kv_len_after = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)

        ref_out = _decode_ref(query, pool, block_table, kv_len_after,
                              block_size, num_kv_heads)
        triton_out = triton_decode_attention(
            query, pool.key_caches[0], pool.value_caches[0],
            block_table, kv_len_after, block_size,
        )

        assert torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2)


# =============================================================================
# Integration tests (2 tests)
# =============================================================================


class TestIntegration:
    """AttentionBackendGPU class-level integration."""

    def test_factory_triton(self):
        """AttentionBackend.create(backend='triton') returns AttentionBackendGPU on CUDA."""
        config = ModelConfig(
            num_layers=1, hidden_size=256, num_heads=4, num_kv_heads=2,
            head_dim=64, dtype=torch.float16,
        )
        backend = AttentionBackend.create(config, backend="triton")
        assert isinstance(backend, AttentionBackendGPU)

    def test_factory_reference(self):
        """AttentionBackend.create(backend='reference') returns AttentionBackendRef."""
        config = ModelConfig(
            num_layers=1, hidden_size=256, num_heads=4, num_kv_heads=2,
            head_dim=64, dtype=torch.float16,
        )
        backend = AttentionBackend.create(config, backend="reference")
        assert isinstance(backend, AttentionBackendRef)
