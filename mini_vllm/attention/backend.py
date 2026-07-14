from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from ..model_runner.base import AttentionMetadata, ModelConfig


class AttentionBackend(ABC):
    """Abstract interface for PagedAttention operations.

    The backend is responsible for:
    - Allocating the GPU KV cache pool (one contiguous tensor per layer)
    - Scatter-writing per-token K/V into the pool via slot mapping
    - Running decode attention (PagedAttention, read-only from pool)
    - Running prefill attention (SDPA with paged prefix gather)

    There are two implementations:
    - ``AttentionBackendRef`` — PyTorch SDPA reference (tests only)
    - ``AttentionBackendGPU``  — Triton kernels (production, mandatory)
    """

    @abstractmethod
    def allocate_pool(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Any:
        """Allocate the GPU KV cache pool tensors.

        Returns a ``KVCachePool`` instance (or equivalent) that provides
        ``get_key_cache(layer_idx)`` and ``get_value_cache(layer_idx)``.
        """
        ...

    @abstractmethod
    def write_kv_cache(
        self,
        layer_idx: int,
        key: torch.Tensor,      # [num_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,    # [num_tokens, num_kv_heads, head_dim]
        slot_mapping: torch.Tensor,  # [num_tokens] (torch.long)
    ) -> None:
        """Scatter-write K/V tensors into the cache pool.

        Each entry in ``slot_mapping`` specifies the target physical slot
        independently — supports random-access writes for both prefill
        (many tokens per sequence) and decode (one token per sequence).
        """
        ...

    @abstractmethod
    def decode_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,            # [num_decode_tokens, num_heads, head_dim]
        attn_metadata: AttentionMetadata,
        pool: Any,
    ) -> torch.Tensor:
        """Paged decode attention.

        Reads from the KV cache pool using block tables and context
        lengths.  No contiguous KV padding — each sequence reads only
        its own blocks.
        """
        ...

    @abstractmethod
    def prefill_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,            # [num_prefill_tokens, num_heads, head_dim]
        key: torch.Tensor,              # [num_prefill_tokens, num_kv_heads, head_dim]
        value: torch.Tensor,            # [num_prefill_tokens, num_kv_heads, head_dim]
        attn_metadata: AttentionMetadata,
        pool: Any,
    ) -> torch.Tensor:
        """Prefill attention with paged prefix.

        The KV cache pool already contains the prefix K/V for each
        sequence (written in previous steps).  The current chunk's K/V
        is passed directly.  The implementation must compute attention
        over the combined prefix + chunk with causal masking.
        """
        ...

    @staticmethod
    def create(
        config: ModelConfig,
        backend: str = "triton",
    ) -> AttentionBackend:
        """Factory: create the appropriate backend.

        ``backend="triton"`` is the DEFAULT for production (GPU kernel).
        ``backend="reference"`` is for test comparison only (PyTorch SDPA).
        No silent fallback — if GPU init fails, the error propagates.
        """
        if backend in ("reference", "ref"):
            from .paged_attention_ref import AttentionBackendRef
            return AttentionBackendRef(config)
        elif backend == "triton":
            from .paged_attention_gpu import AttentionBackendGPU
            return AttentionBackendGPU(config)
        else:
            raise ValueError(f"Unknown AttentionBackend: {backend!r}")
