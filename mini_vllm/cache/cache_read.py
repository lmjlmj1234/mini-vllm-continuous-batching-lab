"""Paged KV cache gather — read contiguous K/V sequences from a paged pool.

Given a ``block_table`` (logical→physical block ID mapping) and a
``num_tokens`` count, ``gather_paged_kv()`` iterates logical blocks,
resolves each to its physical block ID, and reads the appropriate number
of tokens from ``key_cache`` / ``value_cache``.

This is the **read counterpart** of Phase 3's ``write_to_paged_cache()``.
Together they provide the full scatter/gather interface for paged KV cache.
"""

from __future__ import annotations

from typing import List, Tuple

import torch


def gather_paged_kv(
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, block_size, head_dim]
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, block_size, head_dim]
    block_table: List[int],        # physical block IDs (logical index → physical ID)
    num_tokens: int,               # number of KV tokens to gather
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather contiguous K/V sequences from a paged KV cache pool.

    Iterates logical blocks 0, 1, 2, … mapping each to a physical block
    via ``block_table[logical_idx]``.  Each block contributes up to
    ``block_size`` tokens; the last block may be partial (when
    ``num_tokens`` is not aligned to ``block_size``).

    Args:
        key_cache: Pool key cache tensor for one layer.
        value_cache: Pool value cache tensor for one layer.
        block_table: Physical block IDs in logical order.
        num_tokens: Total tokens to read from cache (must be
            ``<= len(block_table) * block_size``).
        block_size: Tokens per physical block.

    Returns:
        ``(key_out, value_out)`` each shaped
        ``[num_tokens, num_kv_heads, head_dim]``.
        When ``num_tokens == 0``, returns empty tensors with correct
        num_kv_heads and head_dim.

    Raises:
        IndexError: ``block_table`` does not cover ``num_tokens``.
    """
    if num_tokens == 0:
        return _empty_like(key_cache, value_cache)

    assert key_cache.shape == value_cache.shape
    _, num_kv_heads, _, head_dim = key_cache.shape

    remaining = num_tokens
    key_parts: List[torch.Tensor] = []
    value_parts: List[torch.Tensor] = []
    logical_idx = 0

    while remaining > 0:
        if logical_idx >= len(block_table):
            raise IndexError(
                f"block_table length {len(block_table)} insufficient "
                f"to cover {num_tokens} tokens (logical_idx={logical_idx}, "
                f"block_size={block_size})"
            )

        physical_id = block_table[logical_idx]
        tokens_this_block = min(block_size, remaining)

        # key_cache[physical_id]: [num_kv_heads, block_size, head_dim]
        # slice to [:tokens_this_block] in dim 1
        k_block = key_cache[physical_id, :, :tokens_this_block, :]
        v_block = value_cache[physical_id, :, :tokens_this_block, :]

        # Permute from [num_kv_heads, T, head_dim] → [T, num_kv_heads, head_dim]
        key_parts.append(k_block.permute(1, 0, 2))
        value_parts.append(v_block.permute(1, 0, 2))

        remaining -= tokens_this_block
        logical_idx += 1

    return torch.cat(key_parts, dim=0), torch.cat(value_parts, dim=0)


def _empty_like(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return zero-size tensors with correct num_kv_heads and head_dim."""
    _, num_kv_heads, _, head_dim = key_cache.shape
    empty = torch.empty(
        0, num_kv_heads, head_dim,
        dtype=key_cache.dtype, device=key_cache.device,
    )
    return empty, empty.clone()
