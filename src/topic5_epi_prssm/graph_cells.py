"""G0-G3 generator ladder on node-level graph state ``H_p(t)``.

Every cell answers the same question -- how does the slow generative state move
when no new observation arrives -- and differs only in how much structure it is
allowed.  G0 is a leaky baseline and must never be described as a graph RNN: it
provably never touches the message function (``uses_messages`` is False and the
unit tests assert the message weights stay at their initial value).
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _masked_mean(state: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Mean over real contacts only; padded lanes never dilute the pooled state."""
    weight = node_mask.unsqueeze(-1)
    return (state * weight).sum(-2) / weight.sum(-2).clamp(min=1.0)

from .contracts import FROZEN

LEVELS = ("G0", "G1", "G2", "G3")


#: Substeps used by the exponential integrator (exponential Euler).  One substep
#: is exact when the message term is frozen over the interval, which is the
#: discretisation this model is defined on.  The integrator is exact for a
#: frozen message term and unconditionally stable for any elapsed time, so the
#: substeps only refresh the message, they are not needed for stability.  A real
#: inter-event gap in this cohort reaches 5.2e5 s; an explicit Euler scheme would
#: have needed thousands of substeps there, or would have blown up.
N_EXPONENTIAL_SUBSTEPS = 1

#: log-spaced bank of initial state time constants, in seconds
TAU_MIN_INIT, TAU_MAX_INIT = 10.0, 10800.0
#: hard bounds so the exponential parametrisation cannot run away
TAU_MIN, TAU_MAX = 0.5, 1.0e6


class GraphMessage(nn.Module):
    """Shared message function; messages travel only along graph support."""

    def __init__(self, dim: int, n_relations: int = 3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_relations, dim, dim) * (0.5 / math.sqrt(dim)))
        self.gain = nn.Parameter(torch.zeros(n_relations))

    def forward(self, state: torch.Tensor, adjacency: torch.Tensor,
                edge_gate: torch.Tensor | None = None) -> torch.Tensor:
        """``state`` (P, N, D); ``adjacency`` (P, R, N, N) row-stochastic.

        A message from j to i uses ``A[r, j, i]``: the forward relation stores
        "i precedes j", so a contact already recruited sends along its own row.
        """
        adj = adjacency if edge_gate is None else adjacency * edge_gate
        gain = torch.tanh(self.gain).view(1, -1, 1, 1)
        neighbour = torch.einsum("prji,pjd->prid", adj * gain, state)
        return torch.einsum("prid,rde->pie", neighbour, self.weight)


class GeneratorCell(nn.Module):
    """One member of the G0-G3 ladder.

    ``propagate`` advances the slow generative state by a real elapsed time with
    no observation; this is the only place where physical transition happens.
    """

    def __init__(self, level: str, dim: int, n_relations: int = 3, *, use_resource: bool = False):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"unknown generator level {level!r}")
        self.level = level
        self.dim = dim
        self.use_resource = bool(use_resource)
        self.uses_messages = level in ("G1", "G2", "G3")

        # Leak / damping is present at every level.  The time constant is held in
        # log space and read back with exp, so a gradient step moves it
        # multiplicatively and the state can reach hours inside the optimisation
        # budget.  A softplus parametrisation cannot: softplus(log 300) is 5.7 s,
        # and reaching 300 s would need the raw parameter to travel to ~300,
        # which is far beyond what the budget allows.  The dimensions are
        # initialised on a log-spaced bank from TAU_MIN_INIT to TAU_MAX_INIT so
        # the generator starts able to express both fast and slow components.
        self.log_tau = nn.Parameter(
            torch.log(torch.logspace(math.log10(TAU_MIN_INIT), math.log10(TAU_MAX_INIT), dim)))
        self.rest = nn.Parameter(torch.zeros(dim))

        if self.uses_messages:
            self.message = GraphMessage(dim, n_relations)
        else:
            self.message = None

        if level in ("G2", "G3"):
            self.gate = nn.Linear(3 * dim + 1, dim)
            self.candidate = nn.Linear(2 * dim + 1, dim)
            nn.init.zeros_(self.gate.bias)
            nn.init.zeros_(self.candidate.bias)
        if level == "G1":
            self.bias_gain = nn.Parameter(torch.zeros(dim))
        if self.use_resource:
            self.resource_damping = nn.Parameter(torch.zeros(dim))
            self.resource_gain = nn.Parameter(torch.zeros(dim))

    # -- physical transition -------------------------------------------------
    def _rates(self, resource: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``resource`` already broadcast to (P, 1, 1)."""
        tau = self.time_constants()
        damping = (1.0 / tau).view(1, 1, -1) + 0.0 * resource
        gain = torch.ones_like(damping)
        if self.use_resource:
            # the resource modulates damping and recurrent gain -- never contact logits
            damping = damping * torch.exp(torch.tanh(self.resource_damping).view(1, 1, -1) * (1.0 - resource))
            gain = gain * torch.exp(-torch.nn.functional.softplus(self.resource_gain).view(1, 1, -1) * (1.0 - resource))
        return damping, gain

    def _fixed_point(self, state, adjacency, resource, edge_gate, node_mask):
        """Frozen-message relaxation target and its exponential rate.

        ``state`` (P, N, D), ``resource`` (P,), ``node_mask`` (P, N) float.
        """
        damping, gain = self._rates(resource.view(-1, 1, 1))
        if self.level == "G0":
            return self.rest.view(1, 1, -1), damping
        messages = self.message(state, adjacency, edge_gate) * gain
        if self.level == "G1":
            bias = self.bias_gain.view(1, 1, -1) * (1.0 - resource).view(-1, 1, 1)
            target = self.rest.view(1, 1, -1) + (messages + bias) / damping
            return target, damping
        pooled = _masked_mean(state, node_mask).unsqueeze(-2).expand_as(state)
        r_feat = resource.view(-1, 1, 1).expand(state.shape[0], state.shape[1], 1)
        z = torch.sigmoid(self.gate(torch.cat([state, messages, pooled, r_feat], dim=-1)))
        cand = torch.tanh(self.candidate(torch.cat([messages, pooled, r_feat], dim=-1)))
        return cand, z * damping * gain

    def time_constants(self) -> torch.Tensor:
        return torch.exp(torch.clamp(self.log_tau, math.log(TAU_MIN), math.log(TAU_MAX)))

    def propagate(self, state: torch.Tensor, delta_t: torch.Tensor, adjacency: torch.Tensor,
                  resource: torch.Tensor, node_mask: torch.Tensor,
                  edge_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Autonomous evolution over ``delta_t`` seconds -- no observation is read.

        Exponential integrator: with the message term frozen over a substep the
        remaining equation is ``dH/dt = -rate * (H - target)``, whose solution
        ``target + (H - target) exp(-rate * dt)`` is exact and bounded for any dt.
        """
        n = 1 if self.level == "G0" else N_EXPONENTIAL_SUBSTEPS
        step_shaped = (delta_t / n).view(-1, 1, 1)
        for _ in range(n):
            target, rate = self._fixed_point(state, adjacency, resource, edge_gate, node_mask)
            state = target + (state - target) * torch.exp(-torch.clamp(step_shaped * rate, max=40.0))
            state = torch.clamp(state, -8.0, 8.0) * node_mask.unsqueeze(-1)
        return state

    def stability_margin(self) -> float:
        """Smallest damping rate; positive means the linear part is contracting."""
        return float((1.0 / self.time_constants()).min().item())
