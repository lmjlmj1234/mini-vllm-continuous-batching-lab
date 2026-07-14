"""Pure-PyTorch reference PagedAttention backend.

Implements ``AttentionBackend`` with per-sequence gather + SDPA loops.

方案 B (write-first): all K/V is written to the cache before attention is
called.  Both ``decode_attention`` and ``prefill_attention`` gather from
the cache pool.

Causal mask for prefill is explicitly computed with absolute position
offsets::

    mask[i, j] = key_position[j] <= query_position[i]

Decode (Q=1) uses no mask (full attention) since the single query token
can see all previous positions without future-leakage concerns.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..cache.cache_read import gather_paged_kv
from ..cache.cache_write import write_to_paged_cache
from ..cache.pool import KVCachePool
from ..model_runner.base import AttentionMetadata, ModelConfig
from .backend import AttentionBackend


class AttentionBackendRef(AttentionBackend):
    """Reference PagedAttention backend using pure PyTorch SDPA.

    Every sequence is processed independently in a Python loop — this is a
    correctness oracle, not a performance path.

    ``write_kv_cache`` delegates to ``write_to_paged_cache()`` (Phase 3).

    ``decode_attention``: gathers ``cached_len_before + 1`` tokens per
    sequence from the cache pool, applies GQA expansion via
    ``repeat_interleave``, and runs SDPA with no mask (Q=1 means no
    future-leakage is possible).

    ``prefill_attention``: gathers ``cached_len_before + query_len`` tokens
    per sequence, applies GQA expansion, and runs SDPA with an explicit
    offset-aware causal mask.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._pool: Optional[KVCachePool] = None
        self._block_size: int = 0

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
        pool = KVCachePool.allocate(
            num_layers, num_blocks, block_size,
            num_kv_heads, head_dim, dtype, device,
        )
        self._pool = pool
        self._block_size = block_size
        return pool

    # ------------------------------------------------------------------
    # Cache write (delegates to Phase 3 module)
    # ------------------------------------------------------------------

    def write_kv_cache(
        self,
        layer_idx: int,
        key: torch.Tensor,         # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,       # [num_tokens, num_kv_heads, head_dim]
        slot_mapping: torch.Tensor,  # [num_tokens]
    ) -> None:
        pool = self._pool
        assert pool is not None, "allocate_pool must be called first"
        write_to_paged_cache(
            key, value,
            pool.key_caches[layer_idx],
            pool.value_caches[layer_idx],
            slot_mapping,
            pool.block_size,
        )

    # ------------------------------------------------------------------
    # Decode attention  (Q=1 per sequence)
    # ------------------------------------------------------------------

    def decode_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,         # [num_decode_tokens, num_heads, head_dim]
        attn_metadata: AttentionMetadata,
        pool: KVCachePool,
    ) -> torch.Tensor:
        num_decode = query.shape[0]
        if num_decode == 0:
            return query

        num_heads = query.shape[1]
        head_dim = query.shape[2]
        num_kv_heads = pool.num_kv_heads
        n_repeats = num_heads // num_kv_heads
        scale = head_dim ** -0.5

        # Locate the decode group
        decode_group = None
        for g in attn_metadata.groups:
            if g.attention_type == "decode_gpu":
                decode_group = g
                break
        if decode_group is None:
            return query  # no decode group — no-op

        key_cache_l = pool.key_caches[layer_idx]
        value_cache_l = pool.value_caches[layer_idx]

        output_parts = []
        for i in range(num_decode):
            seq_idx = decode_group.seq_indices[i]
            cached = int(decode_group.cached_len_before[i].item())
            kv_len = cached + 1  # existing KV + this decode token

            block_row = attn_metadata.decode_block_tables[seq_idx]
            block_ids = [int(b.item()) for b in block_row if b.item() != -1]

            k, v = gather_paged_kv(
                key_cache_l, value_cache_l,
                block_ids, kv_len, pool.block_size,
            )
            # k/v: [kv_len, num_kv_heads, hd]
            # GQA expand: [kv_len, num_kv_heads, hd] → [kv_len, num_heads, hd]
            k = k.repeat_interleave(n_repeats, dim=1)
            v = v.repeat_interleave(n_repeats, dim=1)

            # SDPA expects [B, H, L, D]  —  permute from [L, H, D]
            q_sdpa = query[i].unsqueeze(0).unsqueeze(2)   # [1, num_heads, 1, hd]
            k_sdpa = k.permute(1, 0, 2).unsqueeze(0)      # [1, num_heads, kv_len, hd]
            v_sdpa = v.permute(1, 0, 2).unsqueeze(0)      # [1, num_heads, kv_len, hd]

            # Q=1: full attention, no mask needed
            attn_out = F.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, scale=scale, is_causal=False,
            )
            # attn_out: [1, num_heads, 1, head_dim]
            output_parts.append(attn_out.squeeze(2))  # [1, num_heads, hd]

        return torch.cat(output_parts, dim=0)  # [num_decode, num_heads, head_dim]

    # ------------------------------------------------------------------
    # Prefill attention (Q >= 1 per sequence)
    # ------------------------------------------------------------------

    def prefill_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,         # [num_prefill_tokens, num_heads, head_dim]
        key: torch.Tensor,           # [num_prefill_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,         # [num_prefill_tokens, num_kv_heads, head_dim]
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

                block_row = attn_metadata.prefill_block_tables[seq_idx]
                block_ids = [int(b.item()) for b in block_row if b.item() != -1]

                # Gather ALL KV from cache (方案 B)
                full_k, full_v = gather_paged_kv(
                    key_cache_l, value_cache_l,
                    block_ids, kv_len, pool.block_size,
                )
                # full_k/v: [kv_len, num_kv_heads, hd]
                full_k = full_k.repeat_interleave(n_repeats, dim=1)  # [kv_len, num_heads, hd]
                full_v = full_v.repeat_interleave(n_repeats, dim=1)

                # Offset-aware causal mask: [q_len, kv_len]
                device = query.device
                query_pos = torch.arange(cached, cached + q_len, device=device)
                key_pos = torch.arange(kv_len, device=device)
                causal_mask = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
                # shape: [q_len, kv_len] — valid as attn_mask for SDPA

                q_tokens = query[token_offset:token_offset + q_len]

                # SDPA expects [B, H, L, D]  —  permute from [L, H, D]
                q_sdpa = q_tokens.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, q_len, hd]
                k_sdpa = full_k.permute(1, 0, 2).unsqueeze(0)    # [1, num_heads, kv_len, hd]
                v_sdpa = full_v.permute(1, 0, 2).unsqueeze(0)

                attn_out = F.scaled_dot_product_attention(
                    q_sdpa, k_sdpa, v_sdpa,
                    attn_mask=causal_mask,
                    is_causal=False,
                    scale=scale,
                )
                # attn_out: [1, num_heads, q_len, head_dim]
                output_parts.append(
                    attn_out.squeeze(0).permute(1, 0, 2)  # [q_len, num_heads, hd]
                )

                token_offset += q_len

        if not output_parts:
            return query

        return torch.cat(output_parts, dim=0)  # [total_prefill, num_heads, hd]
