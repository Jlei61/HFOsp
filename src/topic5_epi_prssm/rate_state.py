"""The slow state that explains *when* discharges arrive.

Kept separate from the spatial state that explains *which* contacts take part:
forcing one latent to do both would make "the state modulates the rate" and "the
state modulates the repertoire" the same claim by construction, and Topic 2 already
shows rate and participation are only partly shared.

Two contract clauses this module exists to honour, both of which an earlier version
broke:

1.  **The state handed to interval e must be measurable from history alone.**
    The first version advanced the state *through* interval e before recording it,
    so the survival term for that interval was computed from the state at its end --
    which depends on when the next discharge arrived, the very quantity being
    modelled.  That is a pseudo-likelihood that peeks at its own outcome.  Here the
    recorded row for interval e is the state at its **start**.

2.  **"The discharges inform the state" and "the discharges push the state" are
    different hypotheses and must not be bundled.**  An earlier arm let past event
    load enter the state directly while its comparison arm had no discharge
    information at all, so the contrast measured both effects at once and could
    identify neither.  Here the observer arm updates on the *innovation* -- the part
    of an event that was not predicted -- so a perfectly predicted discharge moves
    the state not at all.  The physical arm adds a push that a perfectly predicted
    discharge still delivers.  That difference is what separates the hypotheses.
"""
from __future__ import annotations

import math

import torch
from torch import nn

ARMS = ("renewal_only", "t0_exogenous_clock", "t1_observer", "t2_physical")

#: the state may forget anywhere between a minute and a day
TAU_MIN_INIT, TAU_MAX_INIT = 60.0, 86400.0
TAU_MIN, TAU_MAX = 1.0, 1.0e6
#: forgetting rate of the running load predictor the observer takes innovations against
PREDICTOR_DECAY = 0.98


class RateState(nn.Module):
    """Continuous-time leaky state, reported at the start of each interval."""

    def __init__(self, dim: int = 4, *, arm: str = "t1_observer"):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown rate-state arm {arm!r}; expected one of {ARMS}")
        self.arm = arm
        self.dim = dim
        self.log_tau = nn.Parameter(torch.log(torch.logspace(
            math.log10(TAU_MIN_INIT), math.log10(TAU_MAX_INIT), dim)))
        # Seeded small rather than zeroed: with the readout weight also at zero the
        # state would be identically zero, every gradient in the pathway would be
        # exactly zero, and the arm could never leave the origin -- an earlier version
        # reproduced its own baseline to every decimal because of this.
        self.exogenous = nn.Linear(3, dim)
        nn.init.normal_(self.exogenous.weight, std=0.1)
        nn.init.zeros_(self.exogenous.bias)
        #: observer gain on the unpredicted part of an event
        self.observer_gain = nn.Linear(1, dim, bias=False)
        nn.init.normal_(self.observer_gain.weight, std=0.02)
        #: physical push delivered by an event whether or not it was predicted
        self.physical_gain = nn.Linear(1, dim, bias=False)
        nn.init.normal_(self.physical_gain.weight, std=0.02)

    def time_constants(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_tau, math.log(TAU_MIN), math.log(TAU_MAX)))

    @staticmethod
    def _running_prediction(load: torch.Tensor, segment_start: torch.Tensor) -> torch.Tensor:
        """Causal running mean of load; the observer takes innovations against this."""
        out = torch.zeros_like(load)
        running = torch.zeros((), dtype=load.dtype, device=load.device)
        seen = 0
        for e in range(load.shape[0]):
            if bool(segment_start[e]):
                running = torch.zeros_like(running)
                seen = 0
            out[e] = running if seen else load[e]
            running = (PREDICTOR_DECAY * running + (1 - PREDICTOR_DECAY) * load[e]
                       if seen else load[e])
            seen += 1
        return out

    def forward(self, elapsed: torch.Tensor, time_of_day: torch.Tensor,
                log_since_open: torch.Tensor, load: torch.Tensor,
                segment_start: torch.Tensor) -> torch.Tensor:
        """State at the **start** of each interval, i.e. at ``t_{e-1}^+``.

        Every quantity used to build row ``e`` is measurable strictly before
        ``t_{e-1}`` ends, so the survival term for interval ``e`` never sees ``t_e``.
        """
        n = elapsed.shape[0]
        if self.arm == "renewal_only":
            return torch.zeros(n, self.dim, dtype=elapsed.dtype, device=elapsed.device)

        tau = self.time_constants()
        drive = self.exogenous(torch.stack(
            [torch.sin(time_of_day), torch.cos(time_of_day), log_since_open], dim=-1))

        if self.arm == "t0_exogenous_clock":
            jump = torch.zeros_like(drive)
        else:
            predicted = self._running_prediction(load, segment_start)
            innovation = (load - predicted).unsqueeze(-1)
            jump = self.observer_gain(innovation)
            if self.arm == "t2_physical":
                # the part a perfectly predicted discharge still delivers
                jump = jump + self.physical_gain(load.unsqueeze(-1))

        carry = torch.zeros(self.dim, dtype=elapsed.dtype, device=elapsed.device)
        rows = []
        for e in range(n):
            if bool(segment_start[e]):
                carry = torch.zeros_like(carry)
            # record BEFORE crossing interval e: this is the state at its start
            rows.append(carry)
            # now cross the interval to t_e^-, using the drive as it stood at the
            # interval's start, then apply this event's update
            decay = torch.exp(-elapsed[e] / tau)
            carry = carry * decay + drive[e] * (1.0 - decay)
            carry = carry + jump[e]
        return torch.stack(rows, dim=0)
