"""Stable continuous-time state and causal observation correction for R1 T1.

This module deliberately contains no event-triggered jump.  Events are scored
from ``z(t-)`` but do not modify the state in T1; deterministic event history is
handled by the separately frozen baseline.
"""
from __future__ import annotations

import torch
from torch import nn


class StableGenerator(nn.Module):
    """Linear flow ``dz/dt = K (z-mu)`` with ``K = Omega - Q``.

    Time is measured in minutes.  The symmetric part of K is strictly
    negative definite, so every autonomous mode is stable without projecting
    parameters after an optimiser step.
    """

    def __init__(self, dim: int, *, min_decay_per_min: float = 1.0 / (48.0 * 60.0)):
        super().__init__()
        self.dim = int(dim)
        self.min_decay_per_min = float(min_decay_per_min)
        self.omega_raw = nn.Parameter(torch.zeros(self.dim, self.dim))
        # softplus(-4) gives an initial time constant near 55 min while the
        # lower bound still permits modes as slow as roughly 48 h.
        self.q_raw = nn.Parameter(torch.full((self.dim,), -4.0))
        self.mu = nn.Parameter(torch.zeros(self.dim))

    def matrix(self) -> torch.Tensor:
        omega = self.omega_raw - self.omega_raw.transpose(-1, -2)
        decay = torch.nn.functional.softplus(self.q_raw) + self.min_decay_per_min
        return omega - torch.diag(decay)

    def propagate(self, state: torch.Tensor,
                  delta_minutes: torch.Tensor | float) -> torch.Tensor:
        delta = torch.as_tensor(
            delta_minutes, dtype=state.dtype, device=state.device
        )
        if delta.numel() != 1:
            raise ValueError("sequential propagation requires a scalar delta")
        transition = torch.matrix_exp(self.matrix().to(state.dtype) * delta)
        return self.mu + torch.matmul(
            state - self.mu, transition.transpose(-1, -2)
        )

    def from_anchor(self, state: torch.Tensor,
                    delta_minutes: torch.Tensor) -> torch.Tensor:
        """Evaluate one corrected state at several later offsets exactly."""
        delta = torch.as_tensor(
            delta_minutes, dtype=state.dtype, device=state.device
        )
        if delta.ndim != 1:
            raise ValueError("anchor offsets must be one dimensional")
        if not len(delta):
            return state.new_empty((0, self.dim))
        if bool((delta < 0).any()):
            raise ValueError("state cannot be evaluated before its anchor")
        transition = torch.matrix_exp(
            self.matrix().to(state.dtype).unsqueeze(0) * delta[:, None, None]
        )
        return self.mu.unsqueeze(0) + torch.matmul(
            transition, (state - self.mu).unsqueeze(-1)
        ).squeeze(-1)


class ObservationCorrection(nn.Module):
    """Bounded GRU-like measurement update at an event-independent anchor."""

    def __init__(self, observation_dim: int, state_dim: int):
        super().__init__()
        joined = int(observation_dim) + int(state_dim)
        self.candidate = nn.Linear(joined, state_dim)
        self.gate = nn.Linear(joined, state_dim)
        nn.init.xavier_uniform_(self.candidate.weight, gain=0.25)
        nn.init.zeros_(self.candidate.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, state_minus: torch.Tensor, observation: torch.Tensor,
                *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return state_minus
        joined = torch.cat([observation, state_minus], dim=-1)
        candidate = torch.tanh(self.candidate(joined))
        gate = torch.sigmoid(self.gate(joined))
        return (1.0 - gate) * state_minus + gate * candidate


class ControlledPersistentState(nn.Module):
    """The sole learned state in R1 T1: stable flow plus background correction."""

    def __init__(self, observation_dim: int, state_dim: int = 8):
        super().__init__()
        self.generator = StableGenerator(state_dim)
        self.correction = ObservationCorrection(observation_dim, state_dim)

    @property
    def dim(self) -> int:
        return self.generator.dim

    def assimilate(self, state_plus: torch.Tensor, delta_minutes: float,
                   observation: torch.Tensor, *, enabled: bool = True) -> torch.Tensor:
        state_minus = self.generator.propagate(state_plus, delta_minutes)
        return self.correction(state_minus, observation, enabled=enabled)
