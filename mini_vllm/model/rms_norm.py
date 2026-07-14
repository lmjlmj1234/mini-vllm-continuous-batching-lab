from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Computes: ``x * weight * rsqrt(mean(x^2) + eps)``
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input.

        Args:
            x: ``[*, hidden_size]``.

        Returns:
            Normalized tensor, same shape as input.
        """
        dtype = x.dtype
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x_normed).to(dtype)
