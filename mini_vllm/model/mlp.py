from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    """SwiGLU MLP with fused gate+up projection.

    Uses a single weight ``gate_up_weight`` for the fused gate+up projection::

        gate_out = x @ gate + weight  # [0:intermediate_size]
        up_out   = x @ gate_up_weight  # [intermediate_size:]
        mlp = SiLU(gate_out) * up_out

    Then ``down_proj`` projects back to ``hidden_size``.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_weight = nn.Parameter(
            torch.empty(2 * intermediate_size, hidden_size)
        )
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights for training/test stability."""
        nn.init.kaiming_uniform_(self.gate_up_weight, a=5**0.5)
        self.down_proj.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU MLP.

        Args:
            x: ``[*, hidden_size]``.

        Returns:
            ``[*, hidden_size]``.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.hidden_size)

        # Fused gate+up
        gate_up = x_2d @ self.gate_up_weight.T  # [*, 2*intermediate]
        gate = gate_up[:, :self.intermediate_size]
        up = gate_up[:, self.intermediate_size:]

        # SiLU activation + elementwise multiply
        hidden = F.silu(gate) * up  # [*, intermediate]
        out = self.down_proj(hidden)  # [*, hidden]
        return out.reshape(*orig_shape)
