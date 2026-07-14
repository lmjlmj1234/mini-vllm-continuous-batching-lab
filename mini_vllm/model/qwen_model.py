from __future__ import annotations

import torch
import torch.nn as nn

from .rms_norm import RMSNorm
from .transformer_layer import QwenDecoderLayer


class QwenModel(nn.Module):
    """Full Qwen2.5 model: embedding → N decoder layers → final RMSNorm → LM head.

    Does NOT implement HF ``past_key_values`` — K/V is managed externally via
    the paged ``AttentionBackend``.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        vocab_size: int,
        rms_norm_eps: float,
        tie_word_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tie_word_embeddings = tie_word_embeddings

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            QwenDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Full model forward returning logits.

        Args:
            input_ids: ``[batch_size, seq_len]`` token IDs.

        Returns:
            Logits ``[batch_size, seq_len, vocab_size]``.
        """
        hidden = self.embed_tokens(input_ids)  # [B, S, hidden]
        for layer in self.layers:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
            q, k, v = layer.attention.qkv_proj(hidden)
            attn_out = self._simple_attention(q, k, v)
            attn_flat = attn_out.reshape(*hidden.shape)
            attn_proj = layer.attention.o_proj(attn_flat)  # raw o_proj, not summed
            hidden = layer.post_attention(attn_proj, residual)
        hidden = self.norm(hidden)
        return self.lm_head(hidden)

    @staticmethod
    def _simple_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Simple attention for reference forward (no RoPE, no cache)."""
        import torch.nn.functional as F
        B, S, H, D = q.shape
        kv_H = k.shape[2]
        n_repeats = H // kv_H
        scale = D ** -0.5

        k_e = k.repeat_interleave(n_repeats, dim=2)
        v_e = v.repeat_interleave(n_repeats, dim=2)
        return F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k_e.permute(0, 2, 1, 3),
            v_e.permute(0, 2, 1, 3),
            is_causal=True,
            scale=scale,
        ).permute(0, 2, 1, 3)  # [B, S, H, D]
