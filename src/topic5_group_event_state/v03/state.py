"""Fixed physical-timescale cross-event state for the v0.3 pilot."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class StateConfig:
    # Primary bank is deliberately fixed and slow.  The within-event grammar
    # already owns the fast contact-sequence dynamics; these modes represent
    # cross-event memory at interpretable physical scales.  Learnable taus are
    # a later sensitivity, never evidence that a physiological constant was
    # identified.
    taus_seconds: tuple[float, ...] = (300.0, 1800.0, 7200.0, 21600.0)
    channels_per_tau: int = 4
    event_dim: int = 64
    update_hidden: int = 64
    update_fraction_numerator: float = 2.0
    update_fraction_cap: float = 0.2
    min_intensity_per_second: float = 1e-6

    @property
    def state_dim(self) -> int:
        return len(self.taus_seconds) * self.channels_per_tau


class FixedTimescaleEventState(nn.Module):
    """Low-dimensional state with fixed real-time decay and learned event update.

    Fixed taus stop a small pilot from pretending it has identified physiological
    constants.  The learned object is the event-dependent update and its useful
    functional readout, not the name or coordinate of an individual latent unit.
    """

    def __init__(self, cfg: StateConfig = StateConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        taus = torch.tensor(cfg.taus_seconds, dtype=torch.float32).repeat_interleave(
            cfg.channels_per_tau
        )
        self.register_buffer("taus", taus)
        self.register_buffer(
            "update_fraction",
            (
                float(cfg.update_fraction_numerator) / torch.sqrt(taus)
            ).clamp(max=float(cfg.update_fraction_cap)),
        )
        self.mean = nn.Parameter(torch.zeros(cfg.state_dim))
        self.initial_offset = nn.Parameter(torch.zeros(cfg.state_dim))
        self.event_norm = nn.LayerNorm(cfg.event_dim)
        self.update_net = nn.Sequential(
            nn.Linear(cfg.event_dim + cfg.state_dim, cfg.update_hidden),
            nn.GELU(),
            nn.Linear(cfg.update_hidden, 2 * cfg.state_dim),
        )
        self.intensity_norm = nn.LayerNorm(cfg.state_dim)
        self.intensity_head = nn.Linear(cfg.state_dim, 1, bias=False)
        self.register_buffer("log_base_rate", torch.tensor(math.log(1e-3)))

    def initial(self, batch: int, device: torch.device | str) -> Tensor:
        return (self.mean + self.initial_offset).unsqueeze(0).expand(batch, -1).to(device)

    def evolve(self, state: Tensor, dt_seconds: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.cfg.state_dim:
            raise ValueError("state has wrong shape")
        dt = dt_seconds.to(torch.float32).clamp_min(0.0).unsqueeze(-1)
        decay = torch.exp(-dt / self.taus)
        return self.mean + (state - self.mean) * decay

    def update(self, state_pre: Tensor, event_embedding: Tensor) -> Tensor:
        if event_embedding.shape[-1] != self.cfg.event_dim:
            raise ValueError("event embedding has wrong width")
        raw = self.update_net(torch.cat([state_pre, self.event_norm(event_embedding)], dim=-1))
        gate, candidate = raw.chunk(2, dim=-1)
        fraction = torch.sigmoid(gate) * self.update_fraction
        # GRU-style bounded correction, rather than an unbounded jump.  The
        # previous additive rule accumulated to a norm >2,000 in one real
        # training epoch, making "state" a disguised event counter.  Slower
        # physical modes also receive proportionally smaller per-event updates.
        return state_pre + fraction * (torch.tanh(candidate) - state_pre)

    def intensity(self, state: Tensor) -> Tensor:
        # The learned state is a signed modulation of the TRAIN marginal rate.
        # Subtracting the same readout at the dynamical equilibrium guarantees
        # lambda(mean) == base_rate.  Without this constraint a model trained on
        # dense local intervals assigned an arbitrary, unseen intensity to its
        # long-horizon equilibrium and made 2 h open-loop structurally invalid.
        normalized = self.intensity_norm(state)
        equilibrium = self.intensity_norm(
            self.mean.unsqueeze(0).expand(state.shape[0], -1)
        )
        residual = (
            self.intensity_head(normalized)
            - self.intensity_head(equilibrium)
        ).squeeze(-1).clamp(-5.0, 5.0)
        return torch.exp(self.log_base_rate + residual).clamp_min(
            float(self.cfg.min_intensity_per_second)
        )

    @torch.no_grad()
    def initialise_intensity_rate(self, events: int, observed_seconds: float) -> None:
        rate = max(float(events) / max(float(observed_seconds), 1.0), 1e-6)
        self.log_base_rate.copy_(torch.tensor(math.log(rate)))
        self.intensity_head.weight.mul_(0.01)


class RateNormalizedShotNoiseState(nn.Module):
    """Physical-time leaky state with rate-normalised marked-event impulses.

    Unlike the GRU-style replacement update above, an event never multiplies
    the previous state by an event-count-dependent forgetting factor.  Memory
    is lost only through ``exp(-dt/tau)``.  Dividing each bounded impulse by
    the FIT event rate times tau keeps the equilibrium scale comparable across
    patients with very different group-event rates.
    """

    def __init__(self, cfg: StateConfig = StateConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        taus = torch.tensor(cfg.taus_seconds, dtype=torch.float32).repeat_interleave(
            cfg.channels_per_tau
        )
        self.register_buffer("taus", taus)
        self.register_buffer("reference_event_rate", torch.tensor(1.0 / 60.0))
        self.mean = nn.Parameter(torch.zeros(cfg.state_dim))
        self.initial_offset = nn.Parameter(torch.zeros(cfg.state_dim))
        self.event_norm = nn.LayerNorm(cfg.event_dim)
        self.event_to_jump = nn.Sequential(
            nn.Linear(cfg.event_dim, cfg.update_hidden),
            nn.GELU(),
            nn.Linear(cfg.update_hidden, cfg.state_dim),
        )
        self.intensity_norm = nn.LayerNorm(cfg.state_dim)
        self.intensity_head = nn.Linear(cfg.state_dim, 1, bias=False)
        self.register_buffer("log_base_rate", torch.tensor(math.log(1e-3)))

    def initial(self, batch: int, device: torch.device | str) -> Tensor:
        return (self.mean + self.initial_offset).unsqueeze(0).expand(batch, -1).to(device)

    def evolve(self, state: Tensor, dt_seconds: Tensor) -> Tensor:
        if state.ndim != 2 or state.shape[-1] != self.cfg.state_dim:
            raise ValueError("state has wrong shape")
        dt = dt_seconds.to(torch.float32).clamp_min(0.0).unsqueeze(-1)
        decay = torch.exp(-dt / self.taus)
        return self.mean + (state - self.mean) * decay

    def update(self, state_pre: Tensor, event_embedding: Tensor) -> Tensor:
        if event_embedding.shape[-1] != self.cfg.event_dim:
            raise ValueError("event embedding has wrong width")
        candidate = torch.tanh(self.event_to_jump(self.event_norm(event_embedding)))
        scale = 1.0 / (self.reference_event_rate.clamp_min(1e-6) * self.taus)
        # A very sparse FIT period must not turn one event into an unbounded
        # jump.  This cap is a numerical guard, not an event-count memory rule.
        scale = scale.clamp(max=float(self.cfg.update_fraction_cap))
        return state_pre + candidate * scale

    def intensity(self, state: Tensor) -> Tensor:
        normalized = self.intensity_norm(state)
        equilibrium = self.intensity_norm(
            self.mean.unsqueeze(0).expand(state.shape[0], -1)
        )
        residual = (
            self.intensity_head(normalized) - self.intensity_head(equilibrium)
        ).squeeze(-1).clamp(-5.0, 5.0)
        return torch.exp(self.log_base_rate + residual).clamp_min(
            float(self.cfg.min_intensity_per_second)
        )

    @torch.no_grad()
    def initialise_reference_rate(self, events: int, observed_seconds: float) -> None:
        rate = max(float(events) / max(float(observed_seconds), 1.0), 1e-6)
        self.reference_event_rate.copy_(torch.tensor(rate))
        self.log_base_rate.copy_(torch.tensor(math.log(rate)))
        self.intensity_head.weight.mul_(0.01)

    @torch.no_grad()
    def initialise_intensity_rate(self, events: int, observed_seconds: float) -> None:
        self.initialise_reference_rate(events, observed_seconds)
