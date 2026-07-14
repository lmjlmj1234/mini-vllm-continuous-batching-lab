"""GPU KV cache pool — one contiguous tensor pair per layer.

``KVCachePool`` is allocated once at startup, sized by
``compute_num_gpu_blocks()``.  It provides the physical storage for
PagedAttention: ``physical_block_id`` directly indexes
``key_caches[layer][block_id, kv_head, token_in_block, head_dim]``.

The pool uses ``torch.empty()`` — no zero-initialization of GB-scale
tensors.  Correctness relies on:
- Only ``slot_mapping``-specified positions are written
- Attention kernels read only up to ``kv_len_after`` per sequence
- After block reuse, the new writer overwrites all relevant slots
  before any reader accesses them
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch


# Minimum number of blocks required for any decode capacity.
MIN_BLOCKS = 16

# Reserve for Triton/CUDA workspace (256 MiB).
_WORKSPACE_RESERVE = 256 * 1024 * 1024


# ---------------------------------------------------------------------------
# KVCachePool
# ---------------------------------------------------------------------------


@dataclass
class KVCachePool:
    """GPU KV cache pool — one tensor pair per Transformer layer.

    Each layer stores K and V separately::

        key_caches[l].shape   = [num_blocks, num_kv_heads, block_size, head_dim]
        value_caches[l].shape = [num_blocks, num_kv_heads, block_size, head_dim]

    ``physical_block_id`` indexes dimension 0 of every layer's tensors.
    """

    key_caches: List[torch.Tensor]
    value_caches: List[torch.Tensor]
    num_blocks: int
    block_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def allocate(
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> KVCachePool:
        """Allocate the KV cache pool using ``torch.empty()``.

        One ``(key, value)`` tensor pair per layer::

            key_caches[l].shape   = [num_blocks, num_kv_heads, block_size, head_dim]
            value_caches[l].shape = [num_blocks, num_kv_heads, block_size, head_dim]
        """
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")

        dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        key_caches = [
            torch.empty(num_blocks, num_kv_heads, block_size, head_dim,
                        dtype=dtype, device=dev)
            for _ in range(num_layers)
        ]
        value_caches = [
            torch.empty(num_blocks, num_kv_heads, block_size, head_dim,
                        dtype=dtype, device=dev)
            for _ in range(num_layers)
        ]

        return KVCachePool(
            key_caches=key_caches,
            value_caches=value_caches,
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            device=dev,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_key_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the key cache tensor for a given layer."""
        return self.key_caches[layer_idx]

    def get_value_cache(self, layer_idx: int) -> torch.Tensor:
        """Return the value cache tensor for a given layer."""
        return self.value_caches[layer_idx]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero out all cache tensors."""
        for k, v in zip(self.key_caches, self.value_caches):
            k.zero_()
            v.zero_()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_slots(self) -> int:
        """Total number of token slots across all blocks (one layer)."""
        return self.num_blocks * self.block_size

    @property
    def bytes_per_block_per_layer(self) -> int:
        """Bytes consumed by one block in a single layer."""
        return (
            2  # K + V
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
            * self.dtype.itemsize
        )

    @property
    def bytes_per_block_total(self) -> int:
        """Bytes consumed by one block across ALL layers."""
        return self.num_layers * self.bytes_per_block_per_layer

    @property
    def total_bytes(self) -> int:
        """Total GPU memory consumed by the entire pool."""
        return self.num_blocks * self.bytes_per_block_total


# ---------------------------------------------------------------------------
# Budget computation
# ---------------------------------------------------------------------------


def compute_num_gpu_blocks(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    dtype: torch.dtype = torch.float16,
    device: Optional[torch.device] = None,
    gpu_memory_utilization: float = 0.90,
    peak_runtime_estimate: int = 0,
    workspace_reserve: int = _WORKSPACE_RESERVE,
    num_gpu_blocks_override: Optional[int] = None,
) -> int:
    """Compute how many KV cache blocks fit in available GPU memory.

    When ``num_gpu_blocks_override`` is provided, it is returned directly
    after a minimum-block check (bypassing all GPU queries).

    Otherwise the formula is::

        1. free_bytes = cuda.mem_get_info()[0]
        2. post_deduction = free_bytes - peak_runtime_estimate - workspace_reserve
        3. safety = post_deduction * (1 - gpu_memory_utilization)
        4. budget = post_deduction - safety = post_deduction * gpu_memory_utilization
        5. bytes_per_block_total = num_layers * 2 * num_kv_heads * block_size *
                                   head_dim * dtype.itemsize
        6. num_blocks = int(budget // bytes_per_block_total)
        7. num_blocks = max(num_blocks, MIN_BLOCKS)
        8. if num_blocks < MIN_BLOCKS: raise RuntimeError

    Args:
        num_layers: Number of Transformer layers.
        num_kv_heads: Number of KV heads (for GQA).
        head_dim: Dimension per attention head.
        block_size: Tokens per physical block.
        dtype: KV cache tensor dtype.
        device: Target GPU device (default: cuda:0 if available).
        gpu_memory_utilization: Fraction of post-deduction memory to use
            for KV cache.  ``(0, 1]``.  ``1.0`` = use all available
            (no safety margin).  Default 0.90.
        peak_runtime_estimate: Measured peak runtime memory in bytes
            (from profile run).  0 when no profile run is available
            (Phase 2 stub).
        workspace_reserve: Fixed reservation for Triton/CUDA workspace
            in bytes.  Default 256 MiB.
        num_gpu_blocks_override: If provided, bypass all GPU queries
            and return this value directly (after minimum check).

    Returns:
        Number of physical blocks for KV cache (guaranteed ≥ MIN_BLOCKS).

    Raises:
        RuntimeError: If computed block count < MIN_BLOCKS.
        ValueError: If ``gpu_memory_utilization`` is out of range.
    """
    if not (0 < gpu_memory_utilization <= 1.0):
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], "
            f"got {gpu_memory_utilization}"
        )

    if num_gpu_blocks_override is not None:
        if num_gpu_blocks_override < MIN_BLOCKS:
            raise RuntimeError(
                f"num_gpu_blocks_override ({num_gpu_blocks_override}) < "
                f"minimum ({MIN_BLOCKS})"
            )
        return num_gpu_blocks_override

    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if dev.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(dev)
    else:
        # CPU fallback: return minimum (used for testing on CPU)
        return MIN_BLOCKS

    # Step 2: deduct peak runtime and workspace
    post_deduction = free_bytes - peak_runtime_estimate - workspace_reserve
    if post_deduction <= 0:
        raise RuntimeError(
            f"GPU memory exhausted after deductions: free={free_bytes}, "
            f"peak={peak_runtime_estimate}, workspace={workspace_reserve}"
        )

    # Step 3-4: apply utilization
    budget = int(post_deduction * gpu_memory_utilization)

    # Step 5: bytes per block across all layers
    bytes_per_block_total = (
        num_layers * 2 * num_kv_heads * block_size * head_dim * dtype.itemsize
    )

    # Step 6-7
    num_blocks = max(MIN_BLOCKS, budget // bytes_per_block_total)

    # Step 8
    if num_blocks < MIN_BLOCKS:
        raise RuntimeError(
            f"GPU memory insufficient for KV cache: need at least "
            f"{MIN_BLOCKS} blocks, can only fit {num_blocks}. "
            f"Try reducing gpu_memory_utilization "
            f"(current={gpu_memory_utilization}) or increasing block_size."
        )

    return num_blocks
