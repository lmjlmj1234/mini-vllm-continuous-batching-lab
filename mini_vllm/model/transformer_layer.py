from __future__ import annotations

import torch
import torch.nn as nn

from .qkv_proj import QKVProjection
from .mlp import SwiGLUMLP
from .rms_norm import RMSNorm


class QwenDecoderLayer(nn.Module):
    """Single Qwen2.5 decoder layer.

    Execution order (per layer, per step)::

        RMSNorm (input_layernorm)
        → QKV projection
        → reshape Q/K/V
        → RoPE (applied externally by ModelRunner)
        → cache write (external, via AttentionBackend)
        → paged attention (external, via AttentionBackend)
        → output projection (o_proj)
        → + residual (post-attention)
        → RMSNorm (post_attention_layernorm)
        → SwiGLU MLP
        → + residual
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = _QwenAttention(
            hidden_size, num_heads, num_kv_heads, head_dim,
        )
        self.post_attn_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Full decoder layer forward (without attention).

        Attention and RoPE are applied externally by the ModelRunner
        to handle the interaction with the paged KV cache.

        Returns:
            Updated hidden states, same shape as input.
        """
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)

        # QKV projection (RoPE + attention done externally)
        q, k, v = self.attention.qkv_proj(normed)  # [*, H, hd], [*, kv_H, hd], [*, kv_H, hd]

        return q, k, v, residual

    def post_attention(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Apply post-attention residual, RMSNorm, and MLP.

        Args:
            attn_output: ``[*, hidden_size]`` — output of ``o_proj(attention)``.
            residual: ``[*, hidden_size]`` — pre-attention residual.

        Returns:
            Updated hidden states after MLP residual.
        """
        hidden = residual + attn_output
        residual2 = hidden
        normed = self.post_attn_layernorm(hidden)
        mlp_out = self.mlp(normed)
        return residual2 + mlp_out


class _QwenAttention(nn.Module):
    """Qwen2.5 attention sub-layer (QKV + output proj only).

    RoPE and paged attention are applied by the ModelRunner.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.qkv_proj = QKVProjection(hidden_size, num_heads, num_kv_heads, head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
