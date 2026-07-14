from __future__ import annotations

import torch
import torch.nn as nn


class QKVProjection(nn.Module):
    """Fused QKV projection with GQA split.

    Uses a single weight tensor::

        qkv_weight [num_heads*head_dim + 2*num_kv_heads*head_dim, hidden_size]

    ``forward(x)`` returns ``(Q, K, V)`` where:
    - Q: ``[*, num_heads, head_dim]``
    - K: ``[*, num_kv_heads, head_dim]``
    - V: ``[*, num_kv_heads, head_dim]``
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        total_size = q_size + 2 * kv_size

        self.qkv_weight = nn.Parameter(torch.empty(total_size, hidden_size))
        self.qkv_bias = nn.Parameter(torch.zeros(total_size))
        self._q_slice = slice(0, q_size)
        self._k_slice = slice(q_size, q_size + kv_size)
        self._v_slice = slice(q_size + kv_size, total_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V from input.

        Args:
            x: ``[*, hidden_size]``.

        Returns:
            ``(Q, K, V)`` where:
            - Q: ``[*, num_heads, head_dim]``
            - K: ``[*, num_kv_heads, head_dim]``
            - V: ``[*, num_kv_heads, head_dim]``
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)
        qkv = x_2d @ self.qkv_weight.T + self.qkv_bias  # [*, total_size]

        q = qkv[:, self._q_slice].reshape(*orig_shape[:-1], self.num_heads, self.head_dim)
        k = qkv[:, self._k_slice].reshape(*orig_shape[:-1], self.num_kv_heads, self.head_dim)
        v = qkv[:, self._v_slice].reshape(*orig_shape[:-1], self.num_kv_heads, self.head_dim)
        return q, k, v
