"""R0-R3 nested resource ladder.

``r`` is a bounded inhibitory/homeostatic-resource-like scalar with baseline 1.
It is never called ATP, potassium or a pump state, and it never writes directly
into contact logits: it reaches the event distribution only through the
generator's damping and recurrent gain.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .contracts import FROZEN

ARMS = ("R0", "R1", "R2", "R3")


class ResourceState(nn.Module):
    """Autonomous recovery, latent-activity consumption, impulse and exposure arms.

    R0  no resource                      r == 1 always
    R1  autonomous                       dr/dt = (1-r)/tau_r - gamma_q q(H) r
    R2  R1 + single-event depletion      r+ = r- exp(-gamma_L L_e)
    R3  R1 + integrated exposure         dr/dt -= gamma_x xbar r     (tau_r frozen first)
    """

    def __init__(self, arm: str, dim: int, *, tau_r_seconds: float | None = None,
                 freeze_tau: bool = False):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown resource arm {arm!r}")
        self.arm = arm
        tau0 = float(tau_r_seconds if tau_r_seconds is not None else FROZEN["resource_tau_grid_seconds"][2])
        self.log_tau_r = nn.Parameter(torch.tensor(math.log(tau0)), requires_grad=not freeze_tau)
        # initialised so that the autonomous equilibrium starts near r = 0.9 rather
        # than collapsing to the floor on the first multi-hour gap
        self.consumption = nn.Parameter(torch.tensor(-9.0))
        self.readout = nn.Linear(dim, 1)
        nn.init.zeros_(self.readout.bias)
        if arm in ("R2",):
            self.log_gamma_L = nn.Parameter(torch.tensor(-4.0))
        if arm in ("R3",):
            self.log_gamma_x = nn.Parameter(torch.tensor(-9.0))

    @property
    def active(self) -> bool:
        return self.arm != "R0"

    def tau_r(self) -> torch.Tensor:
        return torch.exp(self.log_tau_r).clamp(1.0, 1e5)

    def activity(self, state: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        """``q(H)`` -- bounded latent activity that consumes the resource.

        ``state`` (P, N, D), ``node_mask`` (P, N); padded lanes are excluded.
        """
        per_node = torch.sigmoid(self.readout(state)).squeeze(-1)
        return (per_node * node_mask).sum(-1) / node_mask.sum(-1).clamp(min=1.0)

    def propagate(self, resource: torch.Tensor, state: torch.Tensor, delta_t: torch.Tensor,
                  node_mask: torch.Tensor, exposure: torch.Tensor | None = None) -> torch.Tensor:
        """Exact relaxation of ``dr/dt = a - b r`` with the activity term frozen.

        ``a = 1/tau_r`` and ``b = 1/tau_r + gamma_q q(H) [+ gamma_x xbar]``.  The
        closed form ``a/b + (r - a/b) exp(-b dt)`` is exact for a frozen ``q`` and
        stays inside (0, 1] for every elapsed time, including multi-hour gaps.
        """
        if not self.active:
            return resource
        tau = self.tau_r()
        gamma_q = torch.nn.functional.softplus(self.consumption)
        q = self.activity(state, node_mask)
        a = 1.0 / tau
        b = a + gamma_q * q
        if self.arm == "R3" and exposure is not None:
            b = b + torch.nn.functional.softplus(self.log_gamma_x) * exposure
        equilibrium = a / b.clamp(min=1e-8)
        decay = torch.exp(-torch.clamp(b * delta_t, max=40.0))
        return (equilibrium + (resource - equilibrium) * decay).clamp(1e-3, 1.0)

    def absorb_event(self, resource: torch.Tensor, load: torch.Tensor) -> torch.Tensor:
        """Single-event impulse depletion.  Only R2 has this path."""
        if self.arm != "R2":
            return resource
        gamma = torch.nn.functional.softplus(self.log_gamma_L)
        return (resource * torch.exp(-gamma * load)).clamp(1e-3, 1.0)
