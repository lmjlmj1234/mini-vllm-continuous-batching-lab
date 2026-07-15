"""GPU-accelerated PagedAttention backend using Triton kernels.

Implements ``AttentionBackend`` with:
- C1: Triton KV cache write (per-token scatter via slot_mapping)
- C2: Triton PagedAttention decode (online softmax, GQA)
- C3: GPU prefix gather + PyTorch SDPA prefill (temporary)

方案 B (write-first): all K/V is written to the cache before attention.
Both decode and prefill read from the cache pool.

No silent fallback to the reference backend — any failure raises.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

from ..cache.pool import KVCachePool
from ..model_runner.base import AttentionMetadata, ModelConfig
from .backend import AttentionBackend


# ==============================================================================
# C1 — Triton KV Cache Write
# ==============================================================================


@triton.jit
def _cache_write_kernel(
    # Pointers
    key_ptr,            # [num_tokens, num_kv_heads, head_dim]
    value_ptr,          # [num_tokens, num_kv_heads, head_dim]
    key_cache_ptr,      # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache_ptr,    # [num_blocks, num_kv_heads, block_size, head_dim]
    slot_mapping_ptr,   # [num_tokens] int32
    # Meta-parameters (compile-time constants)
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Triton kernel: scatter-write per-token K/V into paged cache.

    One program instance per token.  ``slot == -1`` skips the write.
    """
    pid = tl.program_id(0)  # token index

    slot = tl.load(slot_mapping_ptr + pid)
    if slot == -1:
        return

    block_id = slot // BLOCK_SIZE
    offset = slot % BLOCK_SIZE

    # Cache strides: [num_blocks, num_kv_heads, block_size, head_dim]
    # Cache layout at [block_id, kv_head, offset, :]
    # key/value layout at [pid, kv_head, :]

    # We loop over kv_heads since HEAD_DIM and BLOCK_SIZE are constexpr
    # but NUM_KV_HEADS may not fit in registers for large models.
    # For each kv_head, we load a head_dim vector and store it.
    for kv_head in range(NUM_KV_HEADS):
        # key[pid, kv_head, :]
        k_off = pid * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        k_val = tl.load(key_ptr + k_off + tl.arange(0, HEAD_DIM))

        # key_cache[block_id, kv_head, offset, :]
        kc_off = (
            block_id * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM
            + kv_head * BLOCK_SIZE * HEAD_DIM
            + offset * HEAD_DIM
        )
        tl.store(key_cache_ptr + kc_off + tl.arange(0, HEAD_DIM), k_val)

        # Same for value
        v_off = pid * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM
        v_val = tl.load(value_ptr + v_off + tl.arange(0, HEAD_DIM))

        vc_off = (
            block_id * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM
            + kv_head * BLOCK_SIZE * HEAD_DIM
            + offset * HEAD_DIM
        )
        tl.store(value_cache_ptr + vc_off + tl.arange(0, HEAD_DIM), v_val)


def triton_cache_write(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter-write per-token K/V using Triton (replaces ``write_to_paged_cache``).

    See ``tests/test_paged_attention_gpu.py::test_cache_write_alignment`` for
    element-wise equivalence with the reference implementation.

    Raises:
        AssertionError: Duplicate slots detected in ``slot_mapping``
            (Triton parallel writes to the same location are non-deterministic,
            so duplicate slots are explicitly forbidden).
        RuntimeError: Shape or dtype mismatch.
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    num_blocks = key_cache.shape[0]

    # Validate shapes
    assert key.shape == value.shape, f"key {key.shape} != value {value.shape}"
    assert key_cache.shape == value_cache.shape
    assert slot_mapping.shape == (num_tokens,)

    # Validate dtype/device consistency
    assert key.dtype == key_cache.dtype, f"key dtype {key.dtype} != cache dtype {key_cache.dtype}"
    assert value.dtype == value_cache.dtype
    assert key.device == key_cache.device, f"key device {key.device} != cache device {key_cache.device}"
    assert key.device.type == "cuda", f"key.device must be cuda, got {key.device}"

    # Validate block_size
    assert block_size == key_cache.shape[2], f"block_size {block_size} != cache dim2 {key_cache.shape[2]}"

    # Assert no duplicate non--1 slots (Triton parallel writes to the same
    # slot would race — the "last writer wins" semantic is not guaranteed).
    valid = slot_mapping[slot_mapping != -1]
    if valid.numel() > 0:
        uniq = valid.unique()
        assert uniq.numel() == valid.numel(), (
            f"Duplicate slots in slot_mapping: "
            f"{valid.numel() - uniq.numel()} duplicates. "
            f"Triton cache write forbids duplicate slots."
        )

    if num_tokens == 0:
        return

    # Ensure contiguous layout — the kernel assumes [num_tokens, num_kv_heads,
    # head_dim] with stride [num_kv_heads * head_dim, head_dim, 1].
    # Non-contiguous inputs (e.g. V as a view of a fused QKV tensor) would
    # cause the kernel to read from the wrong memory locations for tokens
    # beyond the first.
    key = key.contiguous()
    value = value.contiguous()

    grid = (num_tokens,)
    _cache_write_kernel[grid](
        key, value, key_cache, value_cache, slot_mapping,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
    )


# ==============================================================================
# C2 — Triton Paged Decode Attention
# ==============================================================================


@triton.jit
def _paged_decode_kernel(
    # Input pointers
    query_ptr,              # [total_decode, num_q_heads, head_dim]
    key_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache_ptr,        # [num_blocks, num_kv_heads, block_size, head_dim]
    block_table_ptr,        # [total_decode, max_blocks_per_seq]
    kv_len_after_ptr,       # [total_decode] int32
    output_ptr,             # [total_decode, num_q_heads, head_dim]
    max_blocks_per_seq,     # regular parameter (not constexpr)
    scale,                  # float: 1/sqrt(head_dim)
    # Compile-time constants
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    REPEATS: tl.constexpr,      # NUM_Q_HEADS // NUM_KV_HEADS
):
    """Triton kernel: PagedAttention decode.

    Each program handles one (sequence, query_head) pair.
    Online softmax in FP32.  GQA supported via ``REPEATS``.
    """
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)
    kv_head = pid_head // REPEATS

    seq_len = tl.load(kv_len_after_ptr + pid_seq)
    # seq_len >= 1 is guaranteed by the Python wrapper (ValueError otherwise)

    # Load query vector (FP16, kept in FP32 for accumulation)
    q_off = pid_seq * NUM_Q_HEADS * HEAD_DIM + pid_head * HEAD_DIM
    q = tl.load(query_ptr + q_off + tl.arange(0, HEAD_DIM)).to(tl.float32)

    # Compute cache strides (all compile-time from constexprs)
    stride_block = NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM  # bytes offset per block
    stride_kv_head = BLOCK_SIZE * HEAD_DIM                # bytes offset per kv_head
    stride_pos = HEAD_DIM                                  # bytes offset per position

    num_full_blocks = seq_len // BLOCK_SIZE
    partial_len = seq_len % BLOCK_SIZE

    # Online softmax in FP32
    m = tl.full([], -float("inf"), dtype=tl.float32)
    s = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # Iterate all blocks (full + partial)
    num_blocks_to_process = num_full_blocks + (1 if partial_len > 0 else 0)

    for block_idx in range(num_blocks_to_process):
        bid = tl.load(block_table_ptr + pid_seq * max_blocks_per_seq + block_idx)

        tokens_in_block = tl.where(
            bid >= 0,
            BLOCK_SIZE if block_idx < num_full_blocks else partial_len,
            0,
        )

        for pos in range(tokens_in_block):
            # Load k vector for this position
            k_off = (
                bid * stride_block
                + kv_head * stride_kv_head
                + pos * stride_pos
            )
            k = tl.load(key_cache_ptr + k_off + tl.arange(0, HEAD_DIM)).to(tl.float32)

            logit = tl.sum(q * k) * scale

            # Online softmax update
            m_new = tl.maximum(m, logit)
            alpha = tl.exp(m - m_new)
            beta = tl.exp(logit - m_new)
            s = s * alpha + beta
            acc = acc * alpha + tl.load(
                value_cache_ptr + k_off + tl.arange(0, HEAD_DIM)
            ).to(tl.float32) * beta
            m = m_new

    # Normalize
    output = acc / s

    # Write output (cast back to FP16)
    tl.store(
        output_ptr + q_off + tl.arange(0, HEAD_DIM),
        output.to(tl.float16),
    )


def triton_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_len_after: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """PagedAttention decode using Triton.

    Args:
        query: [total_decode, num_q_heads, head_dim] FP16
        key_cache: [num_blocks, num_kv_heads, block_size, head_dim] FP16
        value_cache: same as key_cache
        block_table: [total_decode, max_blocks_per_seq] int32, -1 padding
        kv_len_after: [total_decode] int32
        block_size: int

    Returns:
        output: [total_decode, num_q_heads, head_dim] FP16

    Raises:
        RuntimeError: Unsupported head_dim (only 64, 128).
        RuntimeError: Unsupported block_size (must be > 0 and power of 2).
        AssertionError: num_q_heads % num_kv_heads != 0.
    """
    total_decode, num_q_heads, head_dim = query.shape
    num_kv_heads = key_cache.shape[1]

    # Validate preconditions
    assert head_dim in (64, 128), f"head_dim {head_dim} not supported (only 64, 128)"
    assert block_size > 0 and (block_size & (block_size - 1)) == 0, \
        f"block_size {block_size} must be a power of 2"
    assert num_q_heads % num_kv_heads == 0, \
        f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}"
    assert query.dtype == torch.float16, f"query dtype {query.dtype} must be float16"
    assert key_cache.dtype == torch.float16
    assert value_cache.dtype == torch.float16
    assert query.device.type == "cuda"
    assert block_table.shape[0] == total_decode
    assert kv_len_after.shape == (total_decode,)

    # kv_len_after == 0 is illegal — a decode sequence must have at least
    # one token in cache (the current decode step token was just written).
    if (kv_len_after < 1).any():
        raise ValueError(
            f"kv_len_after must be >= 1 for all decode sequences, "
            f"got min={kv_len_after.min().item()}"
        )

    max_blocks_per_seq = block_table.shape[1]
    repeats = num_q_heads // num_kv_heads
    scale = head_dim ** -0.5

    output = torch.empty_like(query)

    if total_decode == 0:
        return output

    grid = (total_decode, num_q_heads)
    _paged_decode_kernel[grid](
        query, key_cache, value_cache, block_table, kv_len_after, output,
        max_blocks_per_seq, scale,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        REPEATS=repeats,
    )
    return output


# ==============================================================================
# C3 — GPU Gather Prefix + PyTorch SDPA Prefill (temporary)
# ==============================================================================


@triton.jit
def _gather_prefix_kv_kernel(
    key_cache_ptr,          # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache_ptr,        # [num_blocks, num_kv_heads, block_size, head_dim]
    block_table_ptr,        # [num_prefill_seqs, max_blocks_per_seq]
    seq_idx_map_ptr,        # [total_prefix_tokens] — seq index for each flat prefix token
    local_pos_map_ptr,      # [total_prefix_tokens] — local position within that seq
    out_k_ptr,              # [total_prefix_tokens, num_kv_heads, head_dim]
    out_v_ptr,              # [total_prefix_tokens, num_kv_heads, head_dim]
    max_blocks_per_seq,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    """Gather prefix K/V from paged cache — one program per prefix token position.

    Each program maps flat prefix token index ``pid`` to:
    - ``seq_idx = seq_idx_map[pid]``
    - ``local_pos = local_pos_map[pid]``

    Then reads K/V from the cache and writes to the contiguous output.
    """
    pid = tl.program_id(0)  # flat prefix token index
    kv_head = tl.program_id(1)  # kv head index

    seq_idx = tl.load(seq_idx_map_ptr + pid)
    local_pos = tl.load(local_pos_map_ptr + pid)

    block_idx = local_pos // BLOCK_SIZE
    offset_in_block = local_pos % BLOCK_SIZE

    bid = tl.load(block_table_ptr + seq_idx * max_blocks_per_seq + block_idx)
    if bid < 0:
        return  # invalid block

    stride_block = NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM
    stride_kv_head = BLOCK_SIZE * HEAD_DIM
    stride_pos = HEAD_DIM

    # Read K from cache
    k_off = (
        bid * stride_block
        + kv_head * stride_kv_head
        + offset_in_block * stride_pos
    )
    k_val = tl.load(key_cache_ptr + k_off + tl.arange(0, HEAD_DIM)).to(tl.float16)
    # Write to output: [total_prefix_tokens, num_kv_heads, head_dim]
    out_k_off = (
        pid * NUM_KV_HEADS * HEAD_DIM
        + kv_head * HEAD_DIM
    )
    tl.store(out_k_ptr + out_k_off + tl.arange(0, HEAD_DIM), k_val)

    # Read V from cache
    v_val = tl.load(value_cache_ptr + k_off + tl.arange(0, HEAD_DIM)).to(tl.float16)
    out_v_off = (
        pid * NUM_KV_HEADS * HEAD_DIM
        + kv_head * HEAD_DIM
    )
    tl.store(out_v_ptr + out_v_off + tl.arange(0, HEAD_DIM), v_val)


def gather_prefix_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_len_before: Sequence[int],
    seq_indices: Sequence[int],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather prefix K/V from paged cache for prefill sequences.

    Args:
        key_cache: [num_blocks, num_kv_heads, block_size, head_dim]
        value_cache: same
        block_table: [num_seqs, max_blocks_per_seq] — total seqs in model, -1 padding
        cached_len_before: per-sequence cached token count
        seq_indices: global seq indices for prefill group
        block_size: int

    Returns:
        (out_k, out_v): [total_prefix_tokens, num_kv_heads, head_dim]
    """
    num_seqs_total = block_table.shape[0]
    max_blocks_per_seq = block_table.shape[1]
    total_prefix = sum(cached_len_before)
    if total_prefix == 0:
        _, num_kv_heads, _, head_dim = key_cache.shape
        empty = torch.empty(0, num_kv_heads, head_dim, dtype=key_cache.dtype, device=key_cache.device)
        return empty, empty.clone()

    # Build mapping from flat prefix token index to (seq_idx, local_pos)
    seq_idx_map = []
    local_pos_map = []
    for i, seq_global_idx in enumerate(seq_indices):
        for pos in range(int(cached_len_before[i])):
            seq_idx_map.append(seq_global_idx)
            local_pos_map.append(pos)

    seq_idx_t = torch.tensor(seq_idx_map, dtype=torch.int32, device=key_cache.device)
    local_pos_t = torch.tensor(local_pos_map, dtype=torch.int32, device=key_cache.device)

    num_kv_heads = key_cache.shape[1]
    head_dim = key_cache.shape[3]
    out_k = torch.empty(total_prefix, num_kv_heads, head_dim, dtype=key_cache.dtype, device=key_cache.device)
    out_v = torch.empty(total_prefix, num_kv_heads, head_dim, dtype=key_cache.dtype, device=key_cache.device)

    grid = (total_prefix, num_kv_heads)
    _gather_prefix_kv_kernel[grid](
        key_cache, value_cache, block_table,
        seq_idx_t, local_pos_t,
        out_k, out_v,
        max_blocks_per_seq,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
    )
    return out_k, out_v


# ==============================================================================
# C4 — AttentionBackendGPU class
# ==============================================================================


class AttentionBackendGPU(AttentionBackend):
    """GPU-accelerated PagedAttention backend using Triton kernels.

    - C1: ``write_kv_cache`` uses Triton cache_write kernel
    - C2: ``decode_attention`` uses Triton paged decode kernel
    - C3: ``prefill_attention`` uses GPU prefix gather + PyTorch SDPA (temporary)
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._pool: Optional[KVCachePool] = None
        self._block_size: int = 0

        # Validate backend preconditions
        if not torch.cuda.is_available():
            raise RuntimeError(
                "attention_backend='triton' requires CUDA, but CUDA is not available."
            )
        try:
            import triton
        except ImportError:
            raise RuntimeError(
                "attention_backend='triton' requires the 'triton' package."
            )

        # Validate model config
        hd = config.head_dim
        if hd not in (64, 128):
            raise RuntimeError(
                f"AttentionBackendGPU: head_dim={hd} not supported "
                f"(only 64, 128). Use attention_backend='reference' for "
                f"arbitrary head_dim."
            )

    # ------------------------------------------------------------------
    # Pool allocation
    # ------------------------------------------------------------------

    def allocate_pool(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> KVCachePool:
        if block_size <= 0 or (block_size & (block_size - 1)) != 0:
            raise RuntimeError(
                f"AttentionBackendGPU: block_size={block_size} must be "
                f"a positive power of 2."
            )

        pool = KVCachePool.allocate(
            num_layers, num_blocks, block_size,
            num_kv_heads, head_dim, dtype, device,
        )
        self._pool = pool
        self._block_size = block_size
        return pool

    # ------------------------------------------------------------------
    # Cache write (C1)
    # ------------------------------------------------------------------

    def write_kv_cache(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        pool = self._pool
        assert pool is not None, "allocate_pool must be called first"
        triton_cache_write(
            key, value,
            pool.key_caches[layer_idx],
            pool.value_caches[layer_idx],
            slot_mapping,
            pool.block_size,
        )

    # ------------------------------------------------------------------
    # Decode attention (C2)
    # ------------------------------------------------------------------

    def decode_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        attn_metadata: AttentionMetadata,
        pool: KVCachePool,
    ) -> torch.Tensor:
        num_decode = query.shape[0]
        if num_decode == 0:
            return query

        decode_group = None
        for g in attn_metadata.groups:
            if g.attention_type == "decode_gpu":
                decode_group = g
                break
        if decode_group is None:
            return query

        kv_len_after = decode_group.kv_len_after.clone().to(
            dtype=torch.int32, device=query.device
        )

        return triton_decode_attention(
            query,
            pool.key_caches[layer_idx],
            pool.value_caches[layer_idx],
            attn_metadata.decode_block_tables,
            kv_len_after,
            pool.block_size,
        )

    # ------------------------------------------------------------------
    # Prefill attention (C3 — temporary: GPU gather + PyTorch SDPA)
    # ------------------------------------------------------------------

    def prefill_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
        pool: KVCachePool,
    ) -> torch.Tensor:
        total_prefill = query.shape[0]
        if total_prefill == 0:
            return query

        num_heads = query.shape[1]
        head_dim = query.shape[2]
        num_kv_heads = pool.num_kv_heads
        n_repeats = num_heads // num_kv_heads
        scale = head_dim ** -0.5

        key_cache_l = pool.key_caches[layer_idx]
        value_cache_l = pool.value_caches[layer_idx]

        output_parts = []
        token_offset = 0

        for group in attn_metadata.groups:
            if group.attention_type not in ("prefill_gpu", "prefill_ref"):
                continue

            for local_i in range(len(group.seq_indices)):
                seq_idx = int(group.seq_indices[local_i])
                cached = int(group.cached_len_before[local_i].item())
                q_len = int(group.query_len[local_i].item())
                kv_len = cached + q_len

                if cached > 0:
                    pref_k, pref_v = gather_prefix_kv(
                        key_cache_l, value_cache_l,
                        attn_metadata.prefill_block_tables,
                        [cached], [seq_idx],
                        pool.block_size,
                    )
                else:
                    _, num_kv_heads_, _, head_dim_ = key_cache_l.shape
                    pref_k = torch.empty(0, num_kv_heads_, head_dim_,
                                         dtype=key_cache_l.dtype,
                                         device=key_cache_l.device)
                    pref_v = pref_k.clone()

                chunk_k = key[token_offset:token_offset + q_len]
                chunk_v = value[token_offset:token_offset + q_len]

                full_k = torch.cat([pref_k, chunk_k], dim=0)
                full_v = torch.cat([pref_v, chunk_v], dim=0)

                full_k = full_k.repeat_interleave(n_repeats, dim=1)
                full_v = full_v.repeat_interleave(n_repeats, dim=1)

                device = query.device
                query_pos = torch.arange(cached, cached + q_len, device=device)
                key_pos = torch.arange(kv_len, device=device)
                causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)

                q_tokens = query[token_offset:token_offset + q_len]

                q_sdpa = q_tokens.permute(1, 0, 2).unsqueeze(0)
                k_sdpa = full_k.permute(1, 0, 2).unsqueeze(0)
                v_sdpa = full_v.permute(1, 0, 2).unsqueeze(0)

                attn_out = torch.nn.functional.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=causal_mask,
                    is_causal=False,
                    scale=scale,
                )
                output_parts.append(
                    attn_out.squeeze(0).permute(1, 0, 2)
                )

                token_offset += q_len

        if not output_parts:
            return query

        return torch.cat(output_parts, dim=0)
