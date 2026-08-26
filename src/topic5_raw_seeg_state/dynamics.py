"""Stable latent dynamics for the Raw-SEEG R0.1 model (scientific spec section 5.3).

``N_ROTATION_MODES`` two-dimensional damped-rotation blocks act on the
``LATENT_DIM``-dimensional state.  For a horizon ``h`` **in minutes**::

    B_j(h) = exp(-h / tau_j) * [[cos(w_j h), -sin(w_j h)],
                                [sin(w_j h),  cos(w_j h)]]
    z_{t+h} = mu + B(h) (z_t - mu)

Every clause of the spec is enforced structurally rather than by convention:

* ``tau_j`` is learned in log space and mapped by ``exp(clamp(log_tau, ...))``.
  **Not** ``softplus``.  Epi-PRSSM v0.1 used ``softplus(log tau)``, which
  silently pinned every time constant to ~5.7 s and made that model unable to
  represent slow state at all (see the spec section 10 and memory note
  ``project_topic5_epi_prssm_v0_1_2026-08-18``).  ``exp(clamp(.))`` is a hard
  box constraint: the value can never leave [TAU_MIN_MINUTES, TAU_MAX_MINUTES]
  and the gradient is exactly zero outside it.
* ``omega_j = OMEGA_MAX_RAD_PER_MIN * tanh(omega_raw_j)`` so the rotation rate
  is bounded and ``omega_j = 0`` (pure decay) is exactly reachable.  The bound is
  the Nyquist rate of a minute-sampled state, so the shortest expressible period
  is ``2*pi/OMEGA_MAX = 2`` min; every horizon and the consistency step are whole
  minutes, and a faster mode would be an exact alias of a slower one.
* Horizon ``h`` enters the closed form directly, so ``h = 100`` is one
  evaluation of ``B(100)``; there is no python loop over ``h`` anywhere in this
  module and therefore no 100-step autograd graph.
"""

from __future__ import annotations

import math
from typing import Dict, List, Union

import torch
from torch import Tensor, nn

from . import contract

HorizonLike = Union[float, int, Tensor]

__all__ = ["DampedRotationDynamics"]


class DampedRotationDynamics(nn.Module):
    """Block-diagonal damped rotation ``z_{t+h} = mu + B(h) (z_t - mu)``.

    Parameters
    ----------
    latent_dim, n_modes:
        Default to the frozen contract values (32 and 16); ``2 * n_modes`` must
        equal ``latent_dim`` because each mode owns one 2-D block.
    identity_mode:
        Baseline #4 of the spec section 8.1 ("raw encoder + identity dynamics").
        When True ``forward`` returns ``z`` unchanged for every horizon.  The
        *same* parameters (``log_tau``, ``omega_raw``, ``mu``) are still
        registered and still initialised identically -- they are simply unused.
        This keeps the ablation a pure change of map, not of capacity, which is
        the v0.1 lesson that a capacity control must be frozen node-by-node
        rather than removed.  (The unused tensors receive no gradient, so an
        optimizer built over ``model.parameters()`` leaves them at their init.)
    """

    #: tanh(x) -> 1 only as x -> inf, so an initial |omega| exactly at the bound
    #: is not representable; the init caps the ratio here instead of saturating.
    INIT_TANH_RATIO_CAP = 0.95

    def __init__(
        self,
        latent_dim: int = contract.LATENT_DIM,
        n_modes: int = contract.N_ROTATION_MODES,
        identity_mode: bool = False,
        tau_min_minutes: float = contract.TAU_MIN_MINUTES,
        tau_max_minutes: float = contract.TAU_MAX_MINUTES,
        omega_max_rad_per_min: float = float(contract.OMEGA_MAX_RAD_PER_MIN),
    ) -> None:
        super().__init__()
        if 2 * n_modes != latent_dim:
            raise ValueError(
                f"latent_dim={latent_dim} must be 2*n_modes (n_modes={n_modes})"
            )
        if not 0.0 < tau_min_minutes < tau_max_minutes:
            raise ValueError("require 0 < tau_min_minutes < tau_max_minutes")
        self.latent_dim = int(latent_dim)
        self.n_modes = int(n_modes)
        self.identity_mode = bool(identity_mode)
        self.tau_min_minutes = float(tau_min_minutes)
        self.tau_max_minutes = float(tau_max_minutes)
        self.omega_max_rad_per_min = float(omega_max_rad_per_min)
        self._log_tau_min = math.log(self.tau_min_minutes)
        self._log_tau_max = math.log(self.tau_max_minutes)

        # -- tau: uniform in log space across the FULL [1 min, 48 h] range so
        #    every scale is represented from step 0.
        log_tau0 = torch.linspace(self._log_tau_min, self._log_tau_max, self.n_modes)
        self.log_tau = nn.Parameter(log_tau0.clone())

        # -- omega: half the modes start as pure decay (exactly 0), the other
        #    half rotate with an initial period equal to their own tau, i.e.
        #    matched to the mode's decay scale rather than faster than it.
        #
        #    Two caps apply, and both are recorded in describe_modes() provenance
        #    instead of being applied silently:
        #    (a) the shortest period the bound can express is
        #        2*pi / omega_max = 2 min under OMEGA_MAX = pi/min, which is the
        #        Nyquist period of a minute-sampled state.  A mode whose tau is
        #        shorter than that would ask for an unrepresentable (and, at
        #        integer-minute horizons, unidentifiable) period, so its
        #        requested period is raised to that floor.
        #    (b) tanh cannot reach +-1, so |omega|/omega_max is capped strictly
        #        below 1 rather than sending atanh to infinity.  Only a mode
        #        sitting on the floor from (a) is affected, and its realised
        #        initial period is slightly longer than the floor.
        tau0 = torch.exp(log_tau0)
        self.min_period_minutes = 2.0 * math.pi / self.omega_max_rad_per_min
        requested = torch.full((self.n_modes,), math.inf)          # inf == pure decay
        requested[1::2] = torch.clamp(tau0[1::2], min=self.min_period_minutes)
        omega0 = torch.zeros(self.n_modes)
        omega0[1::2] = 2.0 * math.pi / requested[1::2]
        raw_ratio = omega0 / self.omega_max_rad_per_min
        ratio = raw_ratio.clamp(-self.INIT_TANH_RATIO_CAP, self.INIT_TANH_RATIO_CAP)
        self.omega_raw = nn.Parameter(torch.atanh(ratio))
        self._init_requested_period = [float(v) for v in requested.tolist()]
        self._init_realised_period = [
            math.inf if w == 0.0 else abs(2.0 * math.pi / w)
            for w in (self.omega_max_rad_per_min * torch.tanh(self.omega_raw)).tolist()
        ]
        self._init_capped_modes = [
            i for i, (r, c) in enumerate(zip(raw_ratio.tolist(), ratio.tolist()))
            if r != c
        ]

        self.mu = nn.Parameter(torch.zeros(self.latent_dim))

    # -- effective parameters ------------------------------------------------

    @property
    def tau(self) -> Tensor:
        """Effective mode time constants in minutes, hard-clamped to contract.

        The box constraint is the *inner* clamp in log space; the outer clamp
        only removes the float32 ``exp(log(x))`` round-off (which would put tau
        ~1e-4 min above the stated bound and make any downstream
        ``tau <= TAU_MAX_MINUTES`` audit fail for no reason).
        """
        tau = torch.exp(self.log_tau.clamp(self._log_tau_min, self._log_tau_max))
        return tau.clamp(self.tau_min_minutes, self.tau_max_minutes)

    @property
    def omega(self) -> Tensor:
        """Effective rotation rates in rad/min, bounded by OMEGA_MAX_RAD_PER_MIN."""
        return self.omega_max_rad_per_min * torch.tanh(self.omega_raw)

    def is_stable(self) -> bool:
        """True iff ``exp(-h/tau) < 1`` for every mode and every ``h > 0``.

        That holds exactly when every tau is finite and strictly positive, which
        the ``exp(clamp(.))`` parametrisation guarantees by construction; the
        check is kept so a run manifest can record it as a measured fact.
        """
        tau = self.tau.detach()
        return bool(torch.isfinite(tau).all() and (tau > 0).all())

    # -- map -----------------------------------------------------------------

    def forward(self, z: Tensor, h: HorizonLike) -> Tensor:
        """Advance ``z`` by ``h`` minutes in one closed-form evaluation.

        ``h`` is a non-negative scalar or a tensor of shape ``(B,)`` (a
        per-sample horizon).  ``h`` is used directly inside the exponential and
        the rotation angle, so ``forward(z, 100)`` costs exactly as much, and
        builds exactly as deep an autograd graph, as ``forward(z, 1)``.
        """
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f"expected last dim {self.latent_dim}, got {tuple(z.shape)}")
        h_t = torch.as_tensor(h, dtype=z.dtype, device=z.device)
        if h_t.ndim == 0:
            h_t = h_t.reshape(1)
        if h_t.ndim != 1:
            raise ValueError(f"h must be a scalar or a (B,) tensor, got {tuple(h_t.shape)}")
        if not bool(torch.isfinite(h_t).all()) or bool((h_t < 0).any()):
            raise ValueError("horizon h must be finite and >= 0 (minutes)")
        if h_t.numel() > 1 and (z.ndim != 2 or h_t.shape[0] != z.shape[0]):
            # a per-sample horizon only broadcasts unambiguously against (B, L)
            raise ValueError(
                f"per-sample horizon of shape {tuple(h_t.shape)} needs z of shape "
                f"(B, {self.latent_dim}); got {tuple(z.shape)}"
            )
        if self.identity_mode:
            return z

        tau = self.tau.to(z.dtype)
        omega = self.omega.to(z.dtype)
        hh = h_t.unsqueeze(-1)                              # (B|1, 1)
        decay = torch.exp(-hh / tau)                        # (B|1, M)
        angle = omega * hh                                  # (B|1, M)
        cos, sin = torch.cos(angle), torch.sin(angle)

        centred = (z - self.mu.to(z.dtype)).reshape(*z.shape[:-1], self.n_modes, 2)
        x, y = centred[..., 0], centred[..., 1]
        rx = decay * (cos * x - sin * y)
        ry = decay * (sin * x + cos * y)
        out = torch.stack((rx, ry), dim=-1).reshape(z.shape)
        return out + self.mu.to(z.dtype)

    # -- reporting -----------------------------------------------------------

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def describe_modes(self) -> Dict[str, object]:
        """Per-mode summary for ``per_subject/<subject>/dynamics_modes.json``."""
        tau = self.tau.detach().cpu().double()
        omega = self.omega.detach().cpu().double()
        period: List[float] = []
        for w in omega.tolist():
            period.append(math.inf if w == 0.0 else abs(2.0 * math.pi / w))
        return {
            "n_modes": self.n_modes,
            "identity_mode": self.identity_mode,
            "is_stable": self.is_stable(),
            "tau_minutes": tau.tolist(),
            "omega_rad_per_min": omega.tolist(),
            "period_minutes": period,
            "mu": self.mu.detach().cpu().double().tolist(),
            "tau_bounds_minutes": [self.tau_min_minutes, self.tau_max_minutes],
            "omega_max_rad_per_min": self.omega_max_rad_per_min,
            "min_period_minutes": self.min_period_minutes,
            "init_provenance": {
                "rule": (
                    "even-index modes start at omega=0 (pure decay); odd-index "
                    "modes start with period = max(tau_j, 2*pi/omega_max), the "
                    "second term being the shortest period the rotation bound can "
                    "express (2 min at OMEGA_MAX = pi/min = the Nyquist rate of a "
                    "minute-sampled state). A mode pinned to that floor cannot be "
                    "written exactly through tanh, so its initial |omega|/omega_max "
                    f"is capped at {self.INIT_TANH_RATIO_CAP}, which lengthens its "
                    "realised initial period slightly; see capped_modes."
                ),
                "tanh_ratio_cap": self.INIT_TANH_RATIO_CAP,
                "requested_period_minutes": list(self._init_requested_period),
                "realised_period_minutes": list(self._init_realised_period),
                "capped_modes": list(self._init_capped_modes),
            },
        }

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"latent_dim={self.latent_dim}, n_modes={self.n_modes}, "
            f"identity_mode={self.identity_mode}, "
            f"tau_minutes=[{self.tau_min_minutes:g},{self.tau_max_minutes:g}]"
        )
