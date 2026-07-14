"""Reference (pure-PyTorch) KV cache write via slot mapping.

Each token's ``key`` and ``value`` tensor is scatter-written into the
physical cache pool at the slot specified by ``slot_mapping[i]``::

    block_id = slot // block_size
    block_offset = slot % block_size

    key_cache[block_id, :, block_offset, :] = key[i]
    value_cache[block_id, :, block_offset, :] = value[i]

``slot == -1`` skips the write (for prefix-cache hits where the token
is already present in the pool).

This module is a **standalone reference** — no executor wiring, no
attention computation, no Triton kernel.  Intended to be called by the
AttentionBackend (in Phase 6+) as part of the model forward pass.
"""

from __future__ import annotations

import torch


def write_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    """Scatter-write per-token K/V into a single layer's cache tensors.

    Args:
        key: ``[num_tokens, num_kv_heads, head_dim]`` — token(s) to write.
        value: ``[num_tokens, num_kv_heads, head_dim]`` — token(s) to write.
        key_cache: ``[num_blocks, num_kv_heads, block_size, head_dim]``.
        value_cache: ``[num_blocks, num_kv_heads, block_size, head_dim]``.
        slot_mapping: ``[num_tokens]`` — flat slot index per token.
            ``-1`` skips (prefix slot already cached).
        block_size: Number of tokens per physical block.

    Raises:
        ValueError: Shape, dtype, device, or block_size mismatch.
        IndexError: ``slot`` out of valid range (not ``-1`` and not in
            ``[0, num_blocks * block_size)``).
    """
    _validate_inputs(key, value, key_cache, value_cache, slot_mapping, block_size)

    num_tokens = key.shape[0]
    num_blocks = key_cache.shape[0]
    total_slots = num_blocks * block_size

    for t in range(num_tokens):
        slot = slot_mapping[t].item()

        if slot == -1:
            continue

        if slot < -1 or slot >= total_slots:
            raise IndexError(
                f"slot_mapping[{t}] = {slot} out of range "
                f"[0, {total_slots}) (block_size={block_size}, "
                f"num_blocks={num_blocks})"
            )

        block_id = slot // block_size
        block_offset = slot % block_size

        # key    [t] : [num_kv_heads, head_dim]
        # cache  [block_id, :, block_offset, :]
        key_cache[block_id, :, block_offset, :] = key[t]
        value_cache[block_id, :, block_offset, :] = value[t]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
) -> None:
    if key.shape != value.shape:
        raise ValueError(
            f"key shape {tuple(key.shape)} != value shape {tuple(value.shape)}"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key_cache shape {tuple(key_cache.shape)} != "
            f"value_cache shape {tuple(value_cache.shape)}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    num_tokens, num_kv_heads, head_dim = key.shape
    num_blocks_c, num_kv_heads_c, block_size_c, head_dim_c = key_cache.shape

    if len(slot_mapping) != num_tokens:
        raise ValueError(
            f"slot_mapping length {len(slot_mapping)} != "
            f"num_tokens {num_tokens}"
        )
    if num_kv_heads != num_kv_heads_c:
        raise ValueError(
            f"key num_kv_heads {num_kv_heads} != "
            f"key_cache num_kv_heads {num_kv_heads_c}"
        )
    if head_dim != head_dim_c:
        raise ValueError(
            f"key head_dim {head_dim} != key_cache head_dim {head_dim_c}"
        )
    if block_size != block_size_c:
        raise ValueError(
            f"block_size argument {block_size} != "
            f"key_cache block_size dim {block_size_c}"
        )
    if key.dtype != key_cache.dtype:
        raise ValueError(
            f"key dtype {key.dtype} != key_cache dtype {key_cache.dtype}"
        )
    if value.dtype != value_cache.dtype:
        raise ValueError(
            f"value dtype {value.dtype} != value_cache dtype {value_cache.dtype}"
        )
    if key.device != key_cache.device:
        raise ValueError(
            f"key device {key.device} != key_cache device {key_cache.device}"
        )
    if value.device != value_cache.device:
        raise ValueError(
            f"value device {value.device} != "
            f"value_cache device {value_cache.device}"
        )
