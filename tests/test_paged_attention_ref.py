"""Tests for Milestone A: PagedAttention Correctness.

Covers ``gather_paged_kv``, ``AttentionBackendRef`` decode/prefill
attention, GQA, offset-aware causal mask, scale, and SDPA tensor layout.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mini_vllm.cache.cache_read import gather_paged_kv
from mini_vllm.cache.cache_write import write_to_paged_cache
from mini_vllm.attention.paged_attention_ref import AttentionBackendRef
from mini_vllm.model_runner.base import (
    ModelConfig, AttentionMetadata, AttentionGroup,
)
from mini_vllm.attention.backend import AttentionBackend as AttentionBackendFactory


# ---------------------------------------------------------------------------
# Constants  (defaults for most tests)
# ---------------------------------------------------------------------------

NUM_BLOCKS = 32
BLOCK_SIZE = 4
NUM_KV_HEADS = 2
NUM_HEADS = 8          # GQA: 8 // 2 = 4 repeats
HEAD_DIM = 32
N_REPEATS = NUM_HEADS // NUM_KV_HEADS
SCALE = HEAD_DIM ** -0.5


@pytest.fixture
def pool_tensors():
    """Return a single layer key_cache and value_cache on CPU, fp16."""
    kc = torch.empty(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                     dtype=torch.float16)
    vc = torch.empty(NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM,
                     dtype=torch.float16)
    return kc, vc


def make_kv(num_tokens: int, seed: int = 0,
            num_kv_heads: int = NUM_KV_HEADS,
            head_dim: int = HEAD_DIM) -> tuple:
    """Create random K and V tensors with known values."""
    g = torch.Generator()
    g.manual_seed(seed)
    k = torch.randn(num_tokens, num_kv_heads, head_dim,
                    dtype=torch.float16, generator=g)
    v = torch.randn(num_tokens, num_kv_heads, head_dim,
                    dtype=torch.float16, generator=g)
    return k, v


# ---------------------------------------------------------------------------
# SDPA helpers  ([B, H, L, D] layout)
# ---------------------------------------------------------------------------


def _sdpa_decode(q, k, v, scale=None):
    """SDPA for decode: Q=1, no mask."""
    qs = q.unsqueeze(2)                   # [1, H, 1, D]
    ks = k.permute(1, 0, 2).unsqueeze(0)  # [1, H, kv_len, D]
    vs = v.permute(1, 0, 2).unsqueeze(0)
    out = F.scaled_dot_product_attention(qs, ks, vs, is_causal=False, scale=scale)
    return out.squeeze(2)  # [1, H, D]


def _sdpa_prefill(q, k, v, cached, scale=None):
    """SDPA for prefill with offset-aware causal mask."""
    qs = q.permute(1, 0, 2).unsqueeze(0)       # [1, H, q_len, D]
    ks = k.permute(1, 0, 2).unsqueeze(0)       # [1, H, kv_len, D]
    vs = v.permute(1, 0, 2).unsqueeze(0)
    q_len = q.shape[0]
    kv_len = k.shape[0]
    q_pos = torch.arange(cached, cached + q_len)
    k_pos = torch.arange(kv_len)
    mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
    out = F.scaled_dot_product_attention(
        qs, ks, vs, attn_mask=mask, is_causal=False, scale=scale,
    )
    return out.squeeze(0).permute(1, 0, 2)  # [q_len, H, D]


def _make_decode_meta(seq_indices, cached_len_before, decode_block_tables):
    """Build AttentionMetadata for decode-only step."""
    dec_cached = torch.tensor(cached_len_before, dtype=torch.long)
    dec_query = torch.ones(len(seq_indices), dtype=torch.long)
    group = AttentionGroup(
        seq_indices=seq_indices, attention_type="decode_gpu",
        cached_len_before=dec_cached, query_len=dec_query,
        kv_len_after=dec_cached + dec_query,
    )
    return AttentionMetadata(
        groups=[group],
        decode_block_tables=decode_block_tables,
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
    )


def _make_prefill_meta(seq_indices, cached_len_before, query_len, prefill_block_tables):
    """Build AttentionMetadata for prefill step."""
    pref_cached = torch.tensor(cached_len_before, dtype=torch.long)
    pref_query = torch.tensor(query_len, dtype=torch.long)
    group = AttentionGroup(
        seq_indices=seq_indices, attention_type="prefill_gpu",
        cached_len_before=pref_cached, query_len=pref_query,
        kv_len_after=pref_cached + pref_query,
    )
    return AttentionMetadata(
        groups=[group],
        prefill_block_tables=prefill_block_tables,
        block_size=BLOCK_SIZE, num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
    )


def _fake_pool(key_cache, value_cache):
    """Create a minimal fake pool object."""
    from dataclasses import dataclass
    @dataclass
    class _FakePool:
        key_caches: list
        value_caches: list
        num_blocks: int
        block_size: int
        num_layers: int
        num_kv_heads: int
        head_dim: int
        device: torch.device
        dtype: torch.dtype
        def get_key_cache(self, i): return self.key_caches[i]
        def get_value_cache(self, i): return self.value_caches[i]
    return _FakePool(
        key_caches=[key_cache], value_caches=[value_cache],
        num_blocks=key_cache.shape[0], block_size=key_cache.shape[2],
        num_layers=1, num_kv_heads=key_cache.shape[1],
        head_dim=key_cache.shape[3], device=key_cache.device,
        dtype=key_cache.dtype,
    )


def _paged_backend(kc, vc):
    """Create AttentionBackendRef wired to given cache."""
    backend = AttentionBackendRef(ModelConfig(num_kv_heads=NUM_KV_HEADS))
    backend._pool = _fake_pool(kc, vc)
    backend._block_size = BLOCK_SIZE
    return backend


# ===================================================================
# gather_paged_kv tests
# ===================================================================


class TestGatherPagedKV:

    def _write_and_gather(self, kc, vc, num_tokens, seed, slots=None):
        """Write KV and return (k_in, v_in, block_ids)."""
        k_in, v_in = make_kv(num_tokens, seed=seed)
        if slots is None:
            slots = torch.arange(num_tokens, dtype=torch.long)
        write_to_paged_cache(k_in, v_in, kc, vc, slots, BLOCK_SIZE)
        block_start = slots[0].item() // BLOCK_SIZE
        num_blocks = (slots[-1].item() - slots[0].item() + BLOCK_SIZE) // BLOCK_SIZE
        block_ids = list(range(block_start, block_start + num_blocks))
        return k_in, v_in, block_ids

    def test_gather_full_block(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids = self._write_and_gather(kc, vc, BLOCK_SIZE, 10,
                                                  torch.arange(12, 16, dtype=torch.long))
        k_out, v_out = gather_paged_kv(kc, vc, bids, BLOCK_SIZE, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)
        assert torch.equal(v_out, v_in)

    def test_gather_partial_block(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids = self._write_and_gather(kc, vc, 3, 11,
                                                  torch.tensor([12, 13, 14], dtype=torch.long))
        k_out, v_out = gather_paged_kv(kc, vc, bids, 3, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)

    def test_gather_multi_block(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids = self._write_and_gather(kc, vc, 10, 12,
                                                  torch.arange(4, 14, dtype=torch.long))
        k_out, v_out = gather_paged_kv(kc, vc, bids, 10, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)

    def test_gather_non_contiguous(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in = make_kv(10, 13)
        slots = torch.tensor([28, 29, 30, 31, 12, 13, 14, 15, 60, 61], dtype=torch.long)
        write_to_paged_cache(k_in, v_in, kc, vc, slots, BLOCK_SIZE)
        k_out, v_out = gather_paged_kv(kc, vc, [7, 3, 15], 10, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)

    def test_gather_zero_tokens(self, pool_tensors):
        kc, vc = pool_tensors
        k_out, v_out = gather_paged_kv(kc, vc, [], 0, BLOCK_SIZE)
        assert k_out.shape == (0, NUM_KV_HEADS, HEAD_DIM)
        assert v_out.shape == (0, NUM_KV_HEADS, HEAD_DIM)

    def test_gather_round_trip(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids = self._write_and_gather(kc, vc, 7, 14,
                                                  torch.arange(4, 11, dtype=torch.long))
        k_out, v_out = gather_paged_kv(kc, vc, bids, 7, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)

    def test_gather_insufficient(self, pool_tensors):
        kc, vc = pool_tensors
        with pytest.raises(IndexError, match="insufficient"):
            gather_paged_kv(kc, vc, [0], 8, BLOCK_SIZE)


# ===================================================================
# Decode attention tests  (方案 B: write decode token, gather all)
# ===================================================================


class TestDecodeAttention:

    def _setup_decode(self, kc, vc, cached, seed, slot_offset=0):
        """Write cached+1 tokens and return (k_in, v_in, block_ids, kv_len_after)."""
        kv_len_after = cached + 1
        k_in, v_in = make_kv(kv_len_after, seed=seed)
        slots = torch.arange(slot_offset, slot_offset + kv_len_after, dtype=torch.long)
        write_to_paged_cache(k_in, v_in, kc, vc, slots, BLOCK_SIZE)
        block_start = slot_offset // BLOCK_SIZE
        num_blocks = (kv_len_after + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = list(range(block_start, block_start + num_blocks))
        return k_in, v_in, block_ids, kv_len_after

    def test_decode_single_seq(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids, kv_len = self._setup_decode(kc, vc, 4, 20)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_decode_meta([0], [4], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_decode_multi_seq(self, pool_tensors):
        """3 seqs with non-overlapping physical blocks."""
        kc, vc = pool_tensors
        configs = [(4, 20, 0), (8, 21, 16), (12, 22, 32)]
        q_all = torch.randn(3, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        refs = []; block_rows = []
        for idx, (cached, seed, soff) in enumerate(configs):
            k_in, v_in, bids, _ = self._setup_decode(kc, vc, cached, seed, soff)
            block_rows.append(bids + [-1] * (8 - len(bids)))
            refs.append(_sdpa_decode(q_all[idx:idx+1],
                k_in.repeat_interleave(N_REPEATS, dim=1),
                v_in.repeat_interleave(N_REPEATS, dim=1), SCALE))
        bt = torch.tensor(block_rows, dtype=torch.long)
        meta = _make_decode_meta([0, 1, 2], [4, 8, 12], bt)
        out = _paged_backend(kc, vc).decode_attention(0, q_all, meta, _paged_backend(kc, vc)._pool)
        for i in range(3):
            assert torch.allclose(out[i:i+1], refs[i], atol=1e-3), f"seq {i}"

    def test_decode_block_boundary(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in, bids, _ = self._setup_decode(kc, vc, 7, 40)
        assert bids == [0, 1]
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_decode_meta([0], [7], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_decode_non_contiguous(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in = make_kv(9, 50)  # kv_len_after=9 = cached(8)+1
        slots = torch.cat([torch.arange(28, 32), torch.arange(12, 16), torch.arange(20, 21)]).to(torch.long)
        write_to_paged_cache(k_in, v_in, kc, vc, slots, BLOCK_SIZE)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[7, 3, 5]], dtype=torch.long)  # 3 blocks = 12 slots > 9
        meta = _make_decode_meta([0], [8], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_decode_gqa(self, pool_tensors):
        """GQA: 2 kv_heads -> 8 q_heads via repeat_interleave(4)."""
        kc, vc = pool_tensors
        k_in, v_in = make_kv(7, 60)  # kv_len_after=7 = cached(6)+1
        write_to_paged_cache(k_in, v_in, kc, vc, torch.arange(7, dtype=torch.long), BLOCK_SIZE)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)
        meta = _make_decode_meta([0], [6], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_decode_partial_last_block(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in = make_kv(6, 70)  # kv_len_after=6 = cached(5)+1
        write_to_paged_cache(k_in, v_in, kc, vc, torch.arange(6, dtype=torch.long), BLOCK_SIZE)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)
        meta = _make_decode_meta([0], [5], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_decode_sentinel_not_read(self, pool_tensors):
        """Only kv_len tokens read; block padding not accessed."""
        kc, vc = pool_tensors
        k_in, v_in = make_kv(5, 80)  # kv_len_after=5 = cached(4)+1
        write_to_paged_cache(k_in, v_in, kc, vc, torch.arange(5, dtype=torch.long), BLOCK_SIZE)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)
        meta = _make_decode_meta([0], [4], bt)
        backend = _paged_backend(kc, vc)
        out = backend.decode_attention(0, q, meta, backend._pool)
        ref = _sdpa_decode(q, k_in.repeat_interleave(N_REPEATS, dim=1),
                            v_in.repeat_interleave(N_REPEATS, dim=1), SCALE)
        assert torch.allclose(out, ref, atol=1e-3)


# ===================================================================
# Prefill attention tests  (offset-aware causal mask, 方案 B)
# ===================================================================


class TestPrefillAttention:

    def _setup_prefill(self, kc, vc, cached, q_len, seed, slot_offset=0):
        total = cached + q_len
        k, v = make_kv(total, seed=seed)
        slots = torch.arange(slot_offset, slot_offset + total, dtype=torch.long)
        write_to_paged_cache(k, v, kc, vc, slots, BLOCK_SIZE)
        block_start = slot_offset // BLOCK_SIZE
        num_blocks = (total + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_ids = list(range(block_start, block_start + num_blocks))
        return k, v, block_ids

    def test_prefill_full(self, pool_tensors):
        kc, vc = pool_tensors
        k_all, v_all, bids = self._setup_prefill(kc, vc, 0, 6, 90)
        q = torch.randn(6, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_prefill_meta([0], [0], [6], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref = _sdpa_prefill(q, k_all.repeat_interleave(N_REPEATS, dim=1),
                            v_all.repeat_interleave(N_REPEATS, dim=1), cached=0, scale=SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_prefill_p8_q2(self, pool_tensors):
        kc, vc = pool_tensors
        k_all, v_all, bids = self._setup_prefill(kc, vc, 8, 2, 100)
        q = torch.randn(2, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_prefill_meta([0], [8], [2], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref = _sdpa_prefill(q, k_all[:10].repeat_interleave(N_REPEATS, dim=1),
                            v_all[:10].repeat_interleave(N_REPEATS, dim=1), cached=8, scale=SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_prefill_p8_q3(self, pool_tensors):
        kc, vc = pool_tensors
        k_all, v_all, bids = self._setup_prefill(kc, vc, 8, 3, 110)
        q = torch.randn(3, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_prefill_meta([0], [8], [3], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref = _sdpa_prefill(q, k_all[:11].repeat_interleave(N_REPEATS, dim=1),
                            v_all[:11].repeat_interleave(N_REPEATS, dim=1), cached=8, scale=SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_prefill_p8_q1(self, pool_tensors):
        kc, vc = pool_tensors
        k_all, v_all, bids = self._setup_prefill(kc, vc, 8, 1, 120)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([bids], dtype=torch.long)
        meta = _make_prefill_meta([0], [8], [1], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref = _sdpa_prefill(q, k_all[:9].repeat_interleave(N_REPEATS, dim=1),
                            v_all[:9].repeat_interleave(N_REPEATS, dim=1), cached=8, scale=SCALE)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_prefill_multi_seq(self, pool_tensors):
        kc, vc = pool_tensors
        k0, v0 = make_kv(4, 130)
        write_to_paged_cache(k0, v0, kc, vc, torch.arange(4, dtype=torch.long), BLOCK_SIZE)
        k1, v1 = make_kv(10, 131)
        write_to_paged_cache(k1, v1, kc, vc, torch.arange(16, 26, dtype=torch.long), BLOCK_SIZE)
        q0 = torch.randn(4, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        q1 = torch.randn(2, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        q = torch.cat([q0, q1], dim=0)
        bt = torch.tensor([[0, -1, -1], [4, 5, 6]], dtype=torch.long)
        meta = _make_prefill_meta([0, 1], [0, 8], [4, 2], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref0 = _sdpa_prefill(q0, k0[:4].repeat_interleave(N_REPEATS, dim=1),
                             v0[:4].repeat_interleave(N_REPEATS, dim=1), cached=0, scale=SCALE)
        ref1 = _sdpa_prefill(q1, k1[:10].repeat_interleave(N_REPEATS, dim=1),
                             v1[:10].repeat_interleave(N_REPEATS, dim=1), cached=8, scale=SCALE)
        ref = torch.cat([ref0, ref1], dim=0)
        assert torch.allclose(out, ref, atol=1e-3)


# ===================================================================
# Dimension semantic test  (seq_len != num_heads)
# ===================================================================


class TestDimensionSemantics:
    """Verify SDPA layout is [B, H, L, D] and not accidentally swapped.

    Use small HEAD_DIM=4, single kv_head, and seq_len != num_heads to
    catch layout bugs that dimension-agnostic tests might miss.
    """

    def test_decode_dim_semantics(self):
        """decode: head_dim=4, seq_len=1, num_heads=4 (diff from seq_len)."""
        NH = 4; HD = 4; NBLK = 4; BS = 2
        kc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        vc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        k_in, v_in = make_kv(3, 200, num_kv_heads=1, head_dim=HD)  # kv_len_after=3 = cached(2)+1
        write_to_paged_cache(k_in, v_in, kc, vc, torch.arange(3, dtype=torch.long), BS)
        q = torch.randn(1, NH, HD, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)  # 2 blocks = 4 slots > 3
        dec_c = torch.tensor([2], dtype=torch.long)
        dec_q = torch.ones(1, dtype=torch.long)
        grp = AttentionGroup([0], "decode_gpu", dec_c, dec_q, dec_c + dec_q)
        meta = AttentionMetadata([grp], decode_block_tables=bt,
                                 block_size=BS, num_kv_heads=1, head_dim=HD)
        b = AttentionBackendRef(ModelConfig(num_kv_heads=1))
        b._pool = _fake_pool(kc, vc)
        b._block_size = BS
        out = b.decode_attention(0, q, meta, b._pool)
        assert out.shape == (1, NH, HD), f"decode out shape {out.shape}"
        scale = HD ** -0.5
        ref = _sdpa_decode(q, k_in.repeat_interleave(NH, dim=1),
                            v_in.repeat_interleave(NH, dim=1), scale)
        assert torch.allclose(out, ref, atol=1e-3)

    def test_prefill_dim_semantics(self):
        """prefill: head_dim=4, seq_len=3, num_heads=4 (diff from seq_len)."""
        NH = 4; HD = 4; NBLK = 4; BS = 4
        kc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        vc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        k_all, v_all = make_kv(3, 210, num_kv_heads=1, head_dim=HD)
        write_to_paged_cache(k_all, v_all, kc, vc, torch.arange(3, dtype=torch.long), BS)
        q = torch.randn(3, NH, HD, dtype=torch.float16)
        bt = torch.tensor([[0]], dtype=torch.long)
        pref_c = torch.tensor([0], dtype=torch.long)
        pref_q = torch.tensor([3], dtype=torch.long)
        grp = AttentionGroup([0], "prefill_gpu", pref_c, pref_q, pref_c + pref_q)
        meta = AttentionMetadata([grp], prefill_block_tables=bt,
                                 block_size=BS, num_kv_heads=1, head_dim=HD)
        out = _paged_backend(kc, vc).prefill_attention(0, q, None, None, meta, _paged_backend(kc, vc)._pool)
        assert out.shape == (3, NH, HD), f"prefill out shape {out.shape}"
        ref = _sdpa_prefill(q, k_all.repeat_interleave(NH, dim=1),
                            v_all.repeat_interleave(NH, dim=1), cached=0, scale=HD ** -0.5)
        assert torch.allclose(out, ref, atol=1e-3)


# ===================================================================
# Future-token leakage test  (hand-verifiable, single head, small dim)
# ===================================================================


class TestFutureTokenLeakage:
    """Hand-verifiable test that q_i cannot attend to k_{>P+i}.

    Single kv_head, head_dim=4. V is set to distinct values per position.
    Future positions have extreme V. If the causal mask leaks, q0 output
    will change; if correct, only q1/q2 see the new positions.
    """

    def test_future_tokens_blocked(self):
        NH = 2; HD = 4; BS = 4; NBLK = 4
        kc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        vc = torch.empty(NBLK, 1, BS, HD, dtype=torch.float16)
        # P=4, Q=3. V at each position is distinct
        k_all = torch.zeros(7, 1, HD, dtype=torch.float16)
        v_all = torch.zeros(7, 1, HD, dtype=torch.float16)
        for pos in range(7):
            v_all[pos, 0, :] = pos + 1  # position-identifying values
            k_all[pos, 0, :] = pos + 1
        # Corrupt future positions (pos >= 5) with extreme V
        v_all[5:, 0, :] = 1e4
        write_to_paged_cache(k_all, v_all, kc, vc, torch.arange(7, dtype=torch.long), BS)
        # q_i at position P+i
        q = torch.randn(3, NH, HD, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)
        pref_c = torch.tensor([4], dtype=torch.long)
        pref_q = torch.tensor([3], dtype=torch.long)
        grp = AttentionGroup([0], "prefill_gpu", pref_c, pref_q, pref_c + pref_q)
        meta = AttentionMetadata([grp], prefill_block_tables=bt,
                                 block_size=BS, num_kv_heads=1, head_dim=HD)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        # Reference using correct mask (also blocks positions >= 5 for q0)
        k_gqa = k_all.repeat_interleave(NH, dim=1)
        v_gqa = v_all.repeat_interleave(NH, dim=1)
        ref = _sdpa_prefill(q, k_gqa, v_gqa, cached=4, scale=HD ** -0.5)
        assert torch.allclose(out, ref, atol=1e-3), "Future tokens leaked"
        # Additional: q0 should NOT have extreme V in its output
        v_without_corruption = v_all.clone()
        v_without_corruption[5:, 0, :] = torch.arange(5, 7).unsqueeze(1).expand(2, HD).to(torch.float16)
        ref_clean = _sdpa_prefill(q, k_gqa, v_without_corruption.repeat_interleave(NH, dim=1), cached=4, scale=HD ** -0.5)
        # q0 output should be same whether future V is corrupted or clean
        assert torch.allclose(out[0:1], ref_clean[0:1], atol=1e-2), "q0 affected by future V"
        assert not torch.allclose(out[1:2], ref_clean[1:2], atol=1e-1), "q1 should differ (sees pos 5)"


# ===================================================================
# Scale tests
# ===================================================================


class TestAttentionScale:
    def test_scale_is_one_over_sqrt_head_dim(self, pool_tensors):
        kc, vc = pool_tensors
        k_in, v_in = make_kv(5, 150)  # kv_len_after=5 = cached(4)+1
        write_to_paged_cache(k_in, v_in, kc, vc, torch.arange(5, dtype=torch.long), BLOCK_SIZE)
        q = torch.randn(1, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[0, 1]], dtype=torch.long)
        meta = _make_decode_meta([0], [4], bt)
        out = _paged_backend(kc, vc).decode_attention(0, q, meta, _paged_backend(kc, vc)._pool)
        k_gqa = k_in.repeat_interleave(N_REPEATS, dim=1)
        v_gqa = v_in.repeat_interleave(N_REPEATS, dim=1)
        assert torch.allclose(out, _sdpa_decode(q, k_gqa, v_gqa, SCALE), atol=1e-3)
        ref_default = _sdpa_decode(q, k_gqa, v_gqa)
        ref_no_scale = _sdpa_decode(q, k_gqa, v_gqa, scale=1.0)
        diff = torch.max(torch.abs(ref_default - ref_no_scale)).item()
        assert diff > 1e-6, f"Scale 1/sqrt(hd) vs 1.0 diff={diff}"


# ===================================================================
# Causal mask alignment tests
# ===================================================================


class TestCausalMask:
    def test_causal_mask_hand_crafted(self, pool_tensors):
        kc, vc = pool_tensors
        k_all, v_all = make_kv(10, 160)
        write_to_paged_cache(k_all, v_all, kc, vc, torch.arange(10, dtype=torch.long), BLOCK_SIZE)
        q = torch.randn(3, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
        bt = torch.tensor([[0, 1, 2]], dtype=torch.long)
        meta = _make_prefill_meta([0], [7], [3], bt)
        b = _paged_backend(kc, vc)
        out = b.prefill_attention(0, q, None, None, meta, b._pool)
        ref = _sdpa_prefill(q, k_all[:10].repeat_interleave(N_REPEATS, dim=1),
                            v_all[:10].repeat_interleave(N_REPEATS, dim=1), cached=7, scale=SCALE)
        assert torch.allclose(out, ref, atol=1e-3)


# ===================================================================
# Backend integration tests
# ===================================================================


class TestBackendIntegration:
    def test_write_kv_cache(self, pool_tensors):
        kc, vc = pool_tensors
        b = _paged_backend(kc, vc)
        k_in, v_in = make_kv(3, 170)
        b.write_kv_cache(0, k_in, v_in, torch.tensor([4, 5, 6], dtype=torch.long))
        k_out, v_out = gather_paged_kv(kc, vc, [1], 3, BLOCK_SIZE)
        assert torch.equal(k_out, k_in)
        assert torch.equal(v_out, v_in)

    def test_backend_factory(self):
        config = ModelConfig(num_kv_heads=NUM_KV_HEADS)
        backend = AttentionBackendFactory.create(config, backend="reference")
        assert isinstance(backend, AttentionBackendRef)

    def test_allocate_pool(self):
        b = AttentionBackendRef(ModelConfig(num_kv_heads=NUM_KV_HEADS))
        pool = b.allocate_pool(2, 8, 4, 2, 32, torch.float16, torch.device("cpu"))
        assert pool.num_blocks == 8
        assert len(pool.key_caches) == 2
        assert pool.key_caches[0].shape == (8, 2, 4, 32)


# ===================================================================
# Entry point
# ===================================================================


