"""Persistent causal observer ``c_{p,e}``.

The observer is inference memory, not physiology.  It accumulates past
observations and applies a penalised correction to the graph-state *estimate*
only.  The primary observer has no method that writes the resource: the flexible
control arm is a separate class so that ``correct_resource_every_event`` cannot
appear in the primary API by accident.
"""
from __future__ import annotations

import torch
from torch import nn


class PersistentObserver(nn.Module):
    """``c_e = GRU(c_{e-1}, v_e)``; forward state persists, TBPTT truncates gradient only."""

    def __init__(self, state_dim: int, observer_dim: int, mark_dim: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.observer_dim = observer_dim
        self.summary = nn.Linear(mark_dim + 3, observer_dim)
        self.cell = nn.GRUCell(observer_dim, observer_dim)
        self.node_gate = nn.Linear(mark_dim + state_dim, state_dim)
        self.global_write = nn.Linear(observer_dim, state_dim)
        self.log_gain = nn.Parameter(torch.tensor(-1.0))
        nn.init.zeros_(self.node_gate.bias)
        nn.init.zeros_(self.global_write.bias)

    def encode(self, marks: torch.Tensor, load: torch.Tensor, log_dt: torch.Tensor,
               session_open: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """``marks`` (P, N, mark_dim); padded contacts are excluded from the pool."""
        weight = node_mask.unsqueeze(-1)
        pooled = (marks * weight).sum(-2) / weight.sum(-2).clamp(min=1.0)
        extra = torch.stack([load, log_dt, session_open], dim=-1)
        return torch.tanh(self.summary(torch.cat([pooled, extra], dim=-1)))

    def update(self, observer_state: torch.Tensor, marks: torch.Tensor, load: torch.Tensor,
               log_dt: torch.Tensor, session_open: torch.Tensor,
               node_mask: torch.Tensor) -> torch.Tensor:
        v = self.encode(marks, load, log_dt, session_open, node_mask)
        return self.cell(v, observer_state)

    def correct_graph_state(self, state_minus: torch.Tensor, observer_state: torch.Tensor,
                            marks: torch.Tensor, node_mask: torch.Tensor
                            ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(H^+, correction energy)``.  Never touches the resource."""
        gain = torch.nn.functional.softplus(self.log_gain)
        node_term = torch.tanh(self.node_gate(torch.cat([marks, state_minus], dim=-1)))
        global_term = torch.tanh(self.global_write(observer_state)).unsqueeze(-2)
        delta = gain * (node_term + global_term) * node_mask.unsqueeze(-1)
        energy = (delta ** 2).sum() / node_mask.sum().clamp(min=1.0) / delta.shape[-1]
        corrected = torch.clamp(state_minus + delta, -8.0, 8.0) * node_mask.unsqueeze(-1)
        return corrected, energy


class FlexibleResourceCorrection(nn.Module):
    """Control arm only: a small, penalised observer write into the resource.

    If only this arm helps, the accepted statement is *the data need an extra
    latent coordinate*, never *resource dynamics were established*.
    """

    def __init__(self, observer_dim: int):
        super().__init__()
        self.write = nn.Linear(observer_dim, 1)
        self.log_gain = nn.Parameter(torch.tensor(-3.0))
        nn.init.zeros_(self.write.bias)

    def forward(self, resource: torch.Tensor, observer_state: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        gain = torch.sigmoid(self.log_gain) * 0.2      # hard cap on amplitude
        delta = gain * torch.tanh(self.write(observer_state)).squeeze(-1)
        penalty = (delta ** 2).mean()
        return (resource + delta).clamp(1e-3, 1.0), penalty
