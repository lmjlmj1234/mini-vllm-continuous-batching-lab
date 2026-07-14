from __future__ import annotations

import torch


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE).

    Follows HF Qwen2 implementation:
    - ``inv_freq = 1.0 / (theta ** (arange(0, dim, 2) / dim))``
    - cos/sin precomputed up to ``max_seq_len`` at init time.
    - RoPE formula (HF convention): ``x * cos + rotate_half(x) * sin``
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 1000000.0,
        max_seq_len: int = 32768,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        self._inv_freq = inv_freq  # [head_dim // 2]

        # Precompute cos/sin up to max_seq_len
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # [max_seq_len, head_dim//2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [max_seq_len, head_dim]
        self._cos = emb.cos().to(dtype)   # [max_seq_len, head_dim]
        self._sin = emb.sin().to(dtype)   # [max_seq_len, head_dim]

    @property
    def cos(self) -> torch.Tensor:
        return self._cos

    @property
    def sin(self) -> torch.Tensor:
        return self._sin

    def __call__(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to input tensor.

        Args:
            x: ``[total_tokens, num_heads, head_dim]`` (or ``[total_tokens, num_kv_heads, head_dim]``).
            positions: ``[total_tokens]`` — absolute positions.

        Returns:
            Rotated tensor, same shape as input.
        """
        cos = self._cos[positions]  # [total_tokens, head_dim]
        sin = self._sin[positions]  # [total_tokens, head_dim]
        return _rotate_half(x, cos, sin)


def _rotate_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE rotation following HF Qwen2 convention.

    Uses HF ``rotate_half``: ``x * cos + rotate_half(x) * sin`` where
    ``rotate_half(x) = cat([-x[half:], x[:half]], dim=-1)``.

    This keeps the half-first layout and is mathematically equivalent
    to the pair-wise rotation formula.
    """
    half = x.shape[-1] // 2
    cos = cos.unsqueeze(-2)  # [total_tokens, 1, head_dim] for broadcast
    sin = sin.unsqueeze(-2)
    x1 = x[..., :half]
    x2 = x[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
