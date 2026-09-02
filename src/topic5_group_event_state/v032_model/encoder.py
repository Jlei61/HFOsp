"""Event projection f_theta: standardised token -> small write vector."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EventProjection(nn.Module):
    """``Linear(D, hidden) -> GELU -> Linear(hidden, out)``; no normalisation layer."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), int(out_dim)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x.to(torch.float32))

    def weight_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.endswith("weight")]

    def bias_parameters(self) -> list[nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.endswith("bias")]
