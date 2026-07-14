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

    grid = (num_tokens,)
    _cache_write_kernel[grid](
        key, value, key_cache, value_cache, slot_mapping,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        NUM_KV_HEADS=num_kv_heads,
    )
