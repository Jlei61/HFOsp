"""M0 / M1 / M2: one skeleton, one optional feedback edge.

The whole point of this file is that the three arms differ in exactly one place.
They share the observer inputs, the free dynamics, the decoder and the training
budget; ``M1`` adds a low-capacity signed edge from an event's *occurrence and
burden* into the state transition, and ``M2`` adds a second one from its
*content*.  If the arms differed anywhere else, "M1 beat M0" would be a statement
about architecture, not about events.

Free dynamics
-------------
The state relaxes toward a background-driven target in real seconds:

    S(t + dt) = b + (S(t) - b) * exp(-dt / tau)

Two properties of that form are load-bearing, not stylistic:

*It composes exactly.*  Splitting an interval of constant ``b`` into sub-steps
leaves the result bit-for-bit identical, so ``M0``'s state cannot learn the event
count through "how many times the update ran".  A per-event additive drive -- the
shape the v0.1 model used -- does not have this property, and would have let the
common-drive arm smuggle in the very thing it is supposed to lack.

*It is linear in S.*  The impulses are input-driven, never state-gated, so a whole
chunk of the recurrence is one masked matrix multiply instead of a Python loop
over 235,000 events, and one event's signed effect on any later prediction has a
closed form -- which is what makes the impulse-response panel exact rather than
estimated.

``tau = exp(clamp(log_tau, ...))``.  ``softplus`` has already cost this project
one round of silently second-scale "slow" states.  The clamp bounds what the
model *can express*; it is not a claim about what the data identified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn


ARM_NAMES = ("M0_no_feedback", "M1_count_rate_feedback", "M2_mark_specific_feedback")

# Step kinds on the merged timeline.  Order at equal timestamps matters: the cell
# drive must be in force before anything reads it, and an anchor must be read
# *before* an event that lands on the same instant, because such an event belongs
# to the block being predicted rather than to the history predicting it.
KIND_CELL, KIND_ANCHOR, KIND_EVENT = 0, 1, 2


@dataclass
class H3Config:
    d_state: int = 48
    tau_range_s: tuple[float, float] = (60.0, 6.0 * 3600.0)
    tau_init_s: tuple[float, float] = (300.0, 3.0 * 3600.0)
    drive_hidden: int = 64
    decoder_hidden: int = 96
    adapter_rank: int = 4
    adapter_gain_init: float = 0.1
    horizons_minutes: tuple[int, ...] = (5, 30, 120)
    chunk_steps: int = 1024
    dropout: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "d_state": self.d_state,
            "tau_range_s": list(self.tau_range_s),
            "tau_init_s": list(self.tau_init_s),
            "drive_hidden": self.drive_hidden,
            "decoder_hidden": self.decoder_hidden,
            "adapter_rank": self.adapter_rank,
            "adapter_gain_init": self.adapter_gain_init,
            "horizons_minutes": list(self.horizons_minutes),
            "chunk_steps": self.chunk_steps,
            "dropout": self.dropout,
        }


# --------------------------------------------------------------------------- pieces


class BackgroundDrive(nn.Module):
    """Common drive: background SEEG + clock -> the target the state relaxes to.

    Present, and identical, in all three arms.  ``M0``'s whole case is that the
    IEDs and the future block are both readouts of this; starving it would make
    the comparison a strawman.
    """

    def __init__(self, n_features: int, cfg: H3Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + 1, cfg.drive_hidden),
            nn.GELU(),
            nn.Linear(cfg.drive_hidden, cfg.d_state),
        )

    def forward(self, features: Tensor, valid: Tensor) -> Tensor:
        # The validity flag is an input, not a mask: "no background window was
        # clean here" is itself information about the recording.
        x = torch.cat([torch.nan_to_num(features), valid.float().unsqueeze(-1)], dim=-1)
        return self.net(x)


class LowRankAdapter(nn.Module):
    """A signed, low-capacity event -> state edge.

    Low rank and a scalar gain, so the edge can be small, can be zero, and can
    point either way.  Nothing here forces an event to raise anything: a negative
    gain is as reachable as a positive one, which is required before an
    anti-seizure-like impulse response can be reported honestly.
    """

    def __init__(self, n_features: int, d_state: int, rank: int, gain_init: float):
        super().__init__()
        self.down = nn.Linear(n_features, rank, bias=True)
        self.up = nn.Linear(rank, d_state, bias=False)
        self.log_gain = nn.Parameter(torch.tensor(math.log(max(gain_init, 1e-6))))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(self.log_gain) * self.up(torch.tanh(self.down(torch.nan_to_num(x))))


class FutureBlockDecoder(nn.Module):
    """Frozen-shape readout of a state onto one future physical block.

    Two endpoints, reported separately and never summed into a single headline:
    how many events the block will contain, and -- given that it contains any --
    what they will look like.  The mark head's dimensions are grouped so that
    participation, extent, multiband and waveform/cross-band each get their own
    number without being separate models.
    """

    def __init__(self, cfg: H3Config, n_mark: int):
        super().__init__()
        self.horizons = tuple(int(h) for h in cfg.horizons_minutes)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_state, cfg.decoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.decoder_hidden, cfg.decoder_hidden),
            nn.GELU(),
        )
        self.count = nn.ModuleDict(
            {str(h): nn.Linear(cfg.decoder_hidden, 2) for h in self.horizons}
        )
        self.mark = nn.ModuleDict(
            {str(h): nn.Linear(cfg.decoder_hidden, 2 * n_mark) for h in self.horizons}
        )
        self.n_mark = int(n_mark)

    @torch.no_grad()
    def initialise(self, count_log_mu: Mapping[int, float], mark_loc: np.ndarray) -> None:
        """Start each head at its own TRAIN-split location.

        A count head initialised at 0 while its target is a block of 4,000 events
        produces a first gradient large enough to move the shared trunk into a
        regime the other head never recovers from.
        """

        loc = torch.as_tensor(mark_loc, dtype=torch.float32)
        for h in self.horizons:
            self.count[str(h)].weight.mul_(0.01)
            self.count[str(h)].bias.copy_(
                torch.tensor([float(count_log_mu[h]), 0.0], dtype=torch.float32)
            )
            self.mark[str(h)].weight.mul_(0.01)
            self.mark[str(h)].bias.copy_(
                torch.stack([loc, torch.zeros_like(loc)], dim=-1).reshape(-1)
            )

    def forward(self, state: Tensor, horizon: int) -> dict[str, Tensor]:
        h = self.trunk(state)
        count = self.count[str(int(horizon))](h)
        mark = self.mark[str(int(horizon))](h).reshape(-1, self.n_mark, 2)
        return {
            "count_log_mu": count[:, 0].clamp(-8.0, 14.0),
            "count_log_phi": count[:, 1].clamp(-4.0, 8.0),
            "mark_mu": mark[..., 0],
            "mark_log_sigma": mark[..., 1].clamp(-3.0, 2.0),
        }


# --------------------------------------------------------------------------- model


class H3Model(nn.Module):
    """One arm.  ``use_count``/``use_mark`` are the only differences between arms."""

    def __init__(
        self,
        arm: str,
        cfg: H3Config,
        n_drive_features: int,
        n_count_features: int,
        n_mark_features: int,
        generator: torch.Generator | None = None,
        mean_event_rate_hz: float = 1.0,
    ):
        super().__init__()
        if arm not in ARM_NAMES:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARM_NAMES}")
        self.arm = arm
        self.cfg = cfg
        # M2 is nested over M1 on purpose: the acceptance rule is "M2 beats M1 at
        # the same count and time", so the count path has to be present in both
        # or the increment would not be attributable to content.
        self.use_count = arm in ("M1_count_rate_feedback", "M2_mark_specific_feedback")
        self.use_mark = arm == "M2_mark_specific_feedback"

        # The event edge is expressed in units of the displacement a *typical*
        # event stream would produce, not in raw state units.  Under a constant
        # rate r the linear state settles at r * tau * u, so an unnormalised kick
        # of 0.1 lands a 0.3 Hz patient with a one-hour tau at a state of order
        # 100 -- far outside the range the decoder ever sees -- while a sparse
        # patient's edge stays negligible.  Dividing by r * tau makes the learned
        # gain dimensionless, bounded at initialisation, and comparable between a
        # patient with 2 events a second and one with one a minute.
        self.register_buffer(
            "mean_event_rate_hz", torch.tensor(float(max(mean_event_rate_hz, 1e-8)))
        )
        lo, hi = cfg.tau_init_s
        u = torch.rand(cfg.d_state, generator=generator)
        self.log_tau = nn.Parameter(math.log(lo) + u * (math.log(hi) - math.log(lo)))
        self.state_init = nn.Parameter(torch.zeros(cfg.d_state))

        self.drive = BackgroundDrive(n_drive_features, cfg)
        self.count_adapter = (
            LowRankAdapter(n_count_features, cfg.d_state, cfg.adapter_rank, cfg.adapter_gain_init)
            if self.use_count
            else None
        )
        self.mark_adapter = (
            LowRankAdapter(n_mark_features, cfg.d_state, cfg.adapter_rank, cfg.adapter_gain_init)
            if self.use_mark
            else None
        )
        self.decoder = FutureBlockDecoder(cfg, n_mark_features)

    # ---------------------------------------------------------------- dynamics

    def taus(self) -> Tensor:
        lo, hi = self.cfg.tau_range_s
        return torch.exp(self.log_tau.clamp(math.log(lo), math.log(hi)))

    def event_impulse(
        self,
        count_features: Tensor,
        mark_features: Tensor,
        *,
        enable_count: bool = True,
        enable_mark: bool = True,
    ) -> Tensor:
        """The signed state kick one event delivers, under this arm.

        ``enable_*`` exist so a *frozen* model can be replayed with the edge
        switched off inside one exposure window, which is what the
        ``no_event_feedback`` perturbation is.  They never change training.
        """

        out = torch.zeros(
            count_features.shape[0], self.cfg.d_state,
            device=count_features.device, dtype=torch.float32,
        )
        if self.count_adapter is not None and enable_count:
            out = out + self.count_adapter(count_features)
        if self.mark_adapter is not None and enable_mark:
            out = out + self.mark_adapter(mark_features)
        if self.count_adapter is None and self.mark_adapter is None:
            return out
        scale = 1.0 / (self.mean_event_rate_hz * self.taus()).clamp_min(1e-6)
        return out * scale.unsqueeze(0)

    def rollout(
        self,
        dt: Tensor,              # (S,) seconds from step k to step k+1 (last entry unused)
        drive: Tensor,           # (S, d) relaxation target in force from step k onward
        impulse: Tensor,         # (S, d) additive kick applied at step k (0 for non-events)
        want: Tensor,            # (W,) long, indices whose arriving state is returned
        *,
        state_init: Tensor | None = None,
        chunk: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Exact closed-form rollout of one coverage segment.

        Returns ``(states_at_want, final_state)``.  ``states_at_want`` is the state
        *arriving* at each requested step, before that step's own impulse -- which
        is the causal quantity: an anchor may not read the event standing on it.

        The recurrence ``y_{k+1} = a_k * y_k + v_k`` has no state-dependent term,
        so a chunk of it is a masked matrix multiply.  Weights are formed as
        ``exp(-(c_j - c_{k+1}) / tau)`` directly from cumulative time rather than
        as a ratio of two exponentials, because the ratio underflows to 0/0 across
        a recording gap and would silently zero the carry.
        """

        n_steps = int(dt.shape[0])
        d = self.cfg.d_state
        device = dt.device
        tau = self.taus().to(device)
        chunk = int(chunk or self.cfg.chunk_steps)

        y = (self.state_init if state_init is None else state_init).to(device).reshape(1, d)
        a_all = torch.exp(-dt.clamp_min(0.0).unsqueeze(-1) / tau)         # (S, d)
        v_all = a_all * impulse + (1.0 - a_all) * drive                    # (S, d)

        want = want.to(device)
        out = torch.zeros(int(want.numel()), d, device=device, dtype=torch.float32)
        cursor = 0
        for lo in range(0, n_steps, chunk):
            hi = min(lo + chunk, n_steps)
            length = hi - lo
            local = want[(want >= lo) & (want < hi)] - lo
            # cumulative arrival time of local step j, measured from the chunk's
            # own first step; c[0] = 0 and c[j] = sum of the first j intervals.
            c = torch.cat(
                [torch.zeros(1, device=device), torch.cumsum(dt[lo:hi].clamp_min(0.0), 0)]
            )  # (length + 1,)
            rows = torch.cat([local, torch.tensor([length], device=device)]).long()
            # w[r, k, :] = exp(-(c[rows[r]] - c[k + 1]) / tau), zero for k >= rows[r]
            delta = c[rows].unsqueeze(1) - c[1 : length + 1].unsqueeze(0)   # (R, length)
            mask = (torch.arange(length, device=device).unsqueeze(0) < rows.unsqueeze(1))
            w = torch.exp(-delta.clamp_min(0.0).unsqueeze(-1) / tau) * mask.unsqueeze(-1)
            contrib = torch.einsum("rkd,kd->rd", w, v_all[lo:hi])
            carry = torch.exp(-c[rows].unsqueeze(-1) / tau) * y
            states = carry + contrib
            if local.numel():
                out[cursor : cursor + local.numel()] = states[:-1]
                cursor += int(local.numel())
            y = states[-1:].contiguous()
        return out, y.reshape(d)

    # ---------------------------------------------------------------- scoring

    def score_blocks(
        self,
        states: Tensor,
        horizon: int,
        count: Tensor,
        has_events: Tensor,
        mark_mean: Tensor,
    ) -> dict[str, Tensor]:
        """Per-block log-scores.  Endpoints stay separate; nothing is summed here.

        ``count`` is negative-binomial because block counts are strongly
        overdispersed -- a Poisson would report a spurious win for whichever arm
        happened to shrink its variance.  The mark endpoint is conditional on the
        block containing events, exactly as the contract splits it.
        """

        pred = self.decoder(states, horizon)
        log_mu = pred["count_log_mu"]
        log_phi = pred["count_log_phi"]
        phi = torch.exp(log_phi)
        k = count.float()
        # NB( k ; mu, phi ) with variance mu + mu^2 / phi
        log_nb = (
            torch.lgamma(k + phi)
            - torch.lgamma(phi)
            - torch.lgamma(k + 1.0)
            + phi * (log_phi - torch.logaddexp(log_phi, log_mu))
            + k * (log_mu - torch.logaddexp(log_phi, log_mu))
        )

        mu, log_sigma = pred["mark_mu"], pred["mark_log_sigma"]
        z = (mark_mean - mu) * torch.exp(-log_sigma)
        log_mark = -0.5 * (math.log(2 * math.pi) + 2.0 * log_sigma + z**2)
        log_mark = log_mark * has_events.float().unsqueeze(-1)
        return {"count": log_nb, "mark": log_mark, "has_events": has_events}


def build_model(
    arm: str,
    cfg: H3Config,
    n_drive_features: int,
    n_count_features: int,
    n_mark_features: int,
    seed: int,
    mean_event_rate_hz: float = 1.0,
) -> H3Model:
    generator = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    return H3Model(
        arm, cfg, n_drive_features, n_count_features, n_mark_features, generator,
        mean_event_rate_hz=mean_event_rate_hz,
    )


def parameter_report(model: H3Model) -> dict[str, int]:
    groups = {"drive": 0, "decoder": 0, "count_adapter": 0, "mark_adapter": 0, "state": 0}
    for name, param in model.named_parameters():
        head = name.split(".", 1)[0]
        key = head if head in groups else "state"
        groups[key] += int(param.numel())
    groups["total"] = int(sum(p.numel() for p in model.parameters()))
    return groups
