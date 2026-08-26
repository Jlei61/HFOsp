"""Minimal T1/T2 generator primitives and correction-off rollout contract."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class StableGenerator(nn.Module):
    """K = Omega - Q gives a stable continuous-time linear generator."""

    def __init__(self, dim: int, min_decay_per_min: float = 1.0 / (48.0 * 60.0)):
        super().__init__()
        self.dim = int(dim)
        self.min_decay = float(min_decay_per_min)
        self.omega_raw = nn.Parameter(torch.zeros(dim, dim))
        # softplus(-4) ~= 0.018 / min, an initial time constant of ~55 min.
        # Starting at zero would silently initialise every mode at 1.44 min.
        self.q_raw = nn.Parameter(torch.full((dim,), -4.0))
        self.mu = nn.Parameter(torch.zeros(dim))

    def matrix(self) -> torch.Tensor:
        omega = self.omega_raw - self.omega_raw.transpose(-1, -2)
        q = torch.nn.functional.softplus(self.q_raw) + self.min_decay
        return omega - torch.diag(q)

    def propagate(self, z: torch.Tensor, dt_minutes: torch.Tensor | float) -> torch.Tensor:
        dt = torch.as_tensor(dt_minutes, dtype=z.dtype, device=z.device)
        if dt.numel() != 1:
            raise ValueError("sequential generator propagation expects one delta-t")
        transition = torch.matrix_exp(self.matrix().to(z.dtype) * dt)
        return self.mu + torch.matmul(z - self.mu, transition.transpose(-1, -2))

    def propagate_many_from_same_state(
        self, z: torch.Tensor, dt_minutes: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one autonomous trajectory at many future times exactly.

        T1 has no event-triggered jump.  Between two measurement corrections,
        every event state therefore starts from the same corrected anchor and
        can be evaluated as one batched matrix exponential.  This is
        algebraically identical to repeated propagation because K is constant
        over the interval, while avoiding one tiny ``matrix_exp`` launch per
        high-rate IED.
        """
        dt = torch.as_tensor(dt_minutes, dtype=z.dtype, device=z.device)
        if dt.ndim != 1:
            raise ValueError("batched generator propagation expects a 1-D delta-t vector")
        if not len(dt):
            return z.new_empty((0, self.dim))
        transition = torch.matrix_exp(
            self.matrix().to(z.dtype).unsqueeze(0) * dt[:, None, None]
        )
        centred = z - self.mu
        return self.mu.unsqueeze(0) + torch.matmul(
            transition, centred.unsqueeze(-1)
        ).squeeze(-1)


class ObservationCorrection(nn.Module):
    """Observation update is separate from generator propagation by construction."""

    def __init__(self, observation_dim: int, state_dim: int):
        super().__init__()
        self.correction = nn.Linear(observation_dim + state_dim, state_dim)
        self.gate = nn.Linear(observation_dim + state_dim, state_dim)
        # Start as an identity/no-correction observer. Random initial weights
        # drove the bounded candidate to +/-1 before any scientific objective
        # was learned on dense-event records.
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -4.0)

    def forward(self, z_minus: torch.Tensor, observation: torch.Tensor,
                *, enabled: bool = True) -> torch.Tensor:
        if not enabled:
            return z_minus
        joined = torch.cat([observation, z_minus], dim=-1)
        c = torch.tanh(self.correction(joined))
        g = torch.sigmoid(self.gate(joined))
        # A residual addition accumulates roughly one update per IED and made
        # the untrained full-event smoke grow to |z|=100--314 in only 2,048
        # events.  A GRU-style convex measurement correction preserves the
        # prior when g is small while keeping repeated observations from
        # becoming an implicit event counter.
        return (1.0 - g) * z_minus + g * c


@dataclass
class ExposureState:
    value: torch.Tensor
    tau_minutes: float

    def decay(self, dt_minutes: torch.Tensor | float) -> "ExposureState":
        dt = torch.as_tensor(dt_minutes, dtype=self.value.dtype, device=self.value.device)
        return ExposureState(self.value * torch.exp(-dt / self.tau_minutes), self.tau_minutes)

    def jump(self, innovation: torch.Tensor | float) -> "ExposureState":
        eta = torch.as_tensor(innovation, dtype=self.value.dtype, device=self.value.device)
        return ExposureState(self.value + eta, self.tau_minutes)


class T1T2Core(nn.Module):
    """Minimal state core; T2 differs from T1 only by the exposure forcing edge."""

    def __init__(self, observation_dim: int, state_dim: int, *, t2: bool):
        super().__init__()
        self.generator = StableGenerator(state_dim)
        self.observer = ObservationCorrection(observation_dim, state_dim)
        self.t2 = bool(t2)
        self.exposure_to_state = nn.Parameter(torch.zeros(state_dim))

    def step(self, z_plus: torch.Tensor, dt_minutes: float,
             observation: torch.Tensor | None, exposure: ExposureState,
             *, correction_enabled: bool) -> tuple[torch.Tensor, ExposureState]:
        if self.t2:
            # Exact augmented flow for d(z-mu)/dt=K(z-mu)+B*u and du/dt=-u/tau.
            # It has the same graph depth at 1 min and 1000 min and does not use
            # a first-order approximation at long inter-event intervals.
            dim = z_plus.shape[-1]
            augmented = z_plus.new_zeros((dim + 1, dim + 1))
            augmented[:dim, :dim] = self.generator.matrix().to(z_plus.dtype)
            augmented[:dim, dim] = self.exposure_to_state
            augmented[dim, dim] = -1.0 / float(exposure.tau_minutes)
            transition = torch.matrix_exp(augmented * float(dt_minutes))
            value = torch.cat([z_plus - self.generator.mu, exposure.value.reshape(1)])
            evolved = transition @ value
            z_minus = self.generator.mu + evolved[:dim]
            decayed = ExposureState(evolved[dim], exposure.tau_minutes)
        else:
            decayed = exposure.decay(dt_minutes)
            z_minus = self.generator.propagate(z_plus, dt_minutes)
        if observation is None:
            correction_enabled = False
            observation_dim = self.observer.correction.in_features - z_minus.shape[-1]
            observation = z_minus.new_zeros((*z_minus.shape[:-1], observation_dim))
        z_new = self.observer(z_minus, observation, enabled=correction_enabled)
        return z_new, decayed


def correction_off_rollout(core: T1T2Core, z0: torch.Tensor,
                           observations: list[torch.Tensor],
                           delta_minutes: list[float], innovations: list[float],
                           tau_minutes: float, anchor_index: int) -> torch.Tensor:
    """Roll forward once; observations strictly after the anchor are ignored."""
    if not (len(observations) == len(delta_minutes) == len(innovations)):
        raise ValueError("rollout inputs have unequal length")
    z = z0
    exposure = ExposureState(torch.zeros((), dtype=z.dtype, device=z.device), tau_minutes)
    states = []
    for i, (obs, dt, eta) in enumerate(zip(observations, delta_minutes, innovations)):
        z, exposure = core.step(
            z, dt, obs, exposure, correction_enabled=(i <= anchor_index)
        )
        exposure = exposure.jump(eta)
        states.append(z)
    return torch.stack(states)
