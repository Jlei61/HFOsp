"""Spatial Propagation Operator RNN v0.2.

v0.1 learned a free M-by-M adjacency between tissue units.  Its identifiability
check, run before any patient data was read, found edge identity and direction
of travel both unrecoverable: the fitted graph was one arbitrary member of a
large set that fit equally well.  Only the coarse ordering of how far each patch
pushes recovered, and even that turned out not to separate from where the node
sits on the axis.

So the graph is gone.  What replaces it is a handful of scalars describing how
activity spreads on the patient's plane -- how far along the axis it drifts, how
it diffuses along and across, how fast it decays, and how strongly a local
recovery field suppresses it.  Two fields evolve on a regular grid:

    a   activation, the part that can still recruit
    r   recovery, what activity leaves behind and what holds it back

The observed rank set is injected through the same fixed local electrode kernel
v0.1 used, and the prediction leaves through its transpose.  Nothing about the
contact-to-tissue map is learned, and no contact-to-contact path exists.

The point of the low dimension is not economy, it is identifiability: these are
quantities a synthetic recovery check can actually be asked about.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

# Nested family, from no spatial process to the full operator.  Each config
# freezes a strict superset of the parameters of the one before it, so a gain
# between neighbours is attributable to the component that was released.
CONFIGS = (
    "STATIC",                 # contact bias only; no field at all
    "FIELD_NULL",             # field with decay and recovery, no spatial transport
    "ISOTROPIC_DIFFUSION",    # D_parallel == D_perp, no drift
    "ANISOTROPIC_DRIFT",      # independent D_parallel, D_perp, v; no recovery
    "ANISOTROPIC_RECOVERY",   # the full operator
)

NEG_INF = -1e9


def _softplus_inv(y: float) -> float:
    """Initial raw value whose softplus is ``y``."""
    return float(np.log(np.expm1(max(y, 1e-6))))


def _logit(y: float, hi: float) -> float:
    """Initial raw value whose ``hi * sigmoid`` is ``y``."""
    z = min(max(y / hi, 1e-6), 1 - 1e-6)
    return float(np.log(z / (1 - z)))


# Explicit finite differences constrain what the coefficients may be. Two-
# dimensional diffusion by forward Euler is unstable once D_parallel + D_perp
# exceeds 1/4, and a centred advection step needs |v| below about 1/2. Fitting
# through a bounded map rather than softplus keeps every step inside those
# limits, so a large coefficient means fast transport and never numerical
# oscillation that a rectifier then dresses up as spread.
D_MAX = 0.12
V_MAX = 0.5


def rectify(x: Tensor, kind: str) -> Tensor:
    """Rectifier whose undriven field rests at exactly zero.

    Plain softplus does not: softplus(0) = log 2, so an untouched domain fills
    with a constant, recovery accumulates everywhere, and the injected rank set
    is buried under a floor that carries no information. Shifting by that
    constant restores a true rest state while keeping the soft knee.
    """
    if kind == "relu":
        return F.relu(x)
    if kind == "softplus":
        return F.relu(F.softplus(x) - float(np.log(2.0)))
    raise ValueError(f"unknown nonlinearity {kind!r}")


@dataclass
class OperatorConfig:
    variant: str
    n_contacts: int
    grid_shape: tuple[int, int]          # (n_along_axis, n_across_axis)
    microsteps: int = 3
    nonlinearity: str = "relu"
    seed: int = 0
    observation_operator: np.ndarray | None = field(default=None, repr=False)
    grid_mask: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in CONFIGS:
            raise ValueError(f"unknown variant {self.variant!r}; use one of {CONFIGS}")
        if self.variant != "STATIC" and self.observation_operator is None:
            raise ValueError(f"{self.variant} needs an observation operator")


class PropagationOperator(nn.Module):
    """Fixed finite-difference operators, scalar coefficients.

    The stencils are constants of the patient's geometry.  Only the coefficients
    in front of them are fitted, and there are six of them.
    """

    def __init__(self, config: OperatorConfig):
        super().__init__()
        self.variant = config.variant
        ny, nx = config.grid_shape
        self.grid_shape = (ny, nx)

        mask = (np.ones((ny, nx), np.float32) if config.grid_mask is None
                else np.asarray(config.grid_mask, np.float32))
        self.register_buffer("mask", torch.from_numpy(mask))

        # Free coefficients.  Positive ones are parameterised through softplus so
        # the sign constraint holds without clipping, which would kill gradients.
        self.raw_D_parallel = nn.Parameter(torch.tensor(_logit(0.04, D_MAX)))
        self.raw_D_perp = nn.Parameter(torch.tensor(_logit(0.04, D_MAX)))
        self.v = nn.Parameter(torch.tensor(0.0))              # signed drift
        self.raw_gamma_a = nn.Parameter(torch.tensor(_logit(0.20, 1.0)))
        self.raw_beta = nn.Parameter(torch.tensor(_softplus_inv(0.20)))
        self.raw_gamma_r = nn.Parameter(torch.tensor(_logit(0.30, 1.0)))
        self.raw_xi = nn.Parameter(torch.tensor(_softplus_inv(0.30)))
        self.raw_eta = nn.Parameter(torch.tensor(_softplus_inv(1.00)))

    # -- coefficients, after the variant's constraints ------------------
    @property
    def D_parallel(self) -> Tensor:
        if self.variant in ("STATIC", "FIELD_NULL"):
            return torch.zeros((), device=self.v.device)
        return D_MAX * torch.sigmoid(self.raw_D_parallel)

    @property
    def D_perp(self) -> Tensor:
        if self.variant in ("STATIC", "FIELD_NULL"):
            return torch.zeros((), device=self.v.device)
        if self.variant == "ISOTROPIC_DIFFUSION":
            # tied, not a second free number
            return D_MAX * torch.sigmoid(self.raw_D_parallel)
        return D_MAX * torch.sigmoid(self.raw_D_perp)

    @property
    def drift(self) -> Tensor:
        if self.variant in ("STATIC", "FIELD_NULL", "ISOTROPIC_DIFFUSION"):
            return torch.zeros((), device=self.v.device)
        return V_MAX * torch.tanh(self.v)

    @property
    def gamma_a(self) -> Tensor:
        return torch.sigmoid(self.raw_gamma_a)

    @property
    def beta(self) -> Tensor:
        if self.variant in ("ISOTROPIC_DIFFUSION", "ANISOTROPIC_DRIFT"):
            return torch.zeros((), device=self.v.device)
        return F.softplus(self.raw_beta)

    @property
    def gamma_r(self) -> Tensor:
        return torch.sigmoid(self.raw_gamma_r)

    @property
    def xi(self) -> Tensor:
        if self.variant in ("ISOTROPIC_DIFFUSION", "ANISOTROPIC_DRIFT"):
            return torch.zeros((), device=self.v.device)
        return F.softplus(self.raw_xi)

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.raw_eta)

    # -- fixed stencils --------------------------------------------------
    def laplacian_parallel(self, a: Tensor) -> Tensor:
        """Second difference along the propagation axis (dim -2)."""
        padded = F.pad(a.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
        return padded[:, :-2, :] - 2.0 * a + padded[:, 2:, :]

    def laplacian_perp(self, a: Tensor) -> Tensor:
        """Second difference across the axis (dim -1)."""
        padded = F.pad(a.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(1)
        return padded[:, :, :-2] - 2.0 * a + padded[:, :, 2:]

    def gradient_parallel(self, a: Tensor) -> Tensor:
        """Centred first difference along the axis; positive v moves activity +x."""
        padded = F.pad(a.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
        return 0.5 * (padded[:, 2:, :] - padded[:, :-2, :])

    def step(self, a: Tensor, r: Tensor, injection: Tensor | None,
             nonlinearity: str) -> tuple[Tensor, Tensor]:
        transport = (
            self.D_parallel * self.laplacian_parallel(a)
            + self.D_perp * self.laplacian_perp(a)
            - self.drift * self.gradient_parallel(a)
        )
        pre = (1.0 - self.gamma_a) * a + transport - self.beta * r
        if injection is not None:
            pre = pre + self.eta * injection
        a_next = rectify(pre, nonlinearity)
        a_next = a_next * self.mask
        r_next = ((1.0 - self.gamma_r) * r + self.xi * a) * self.mask
        return a_next, r_next

    def parameter_estimates(self) -> dict[str, float]:
        with torch.no_grad():
            return {
                "v": float(self.drift.item()),
                "D_parallel": float(self.D_parallel.item()),
                "D_perp": float(self.D_perp.item()),
                "gamma_a": float(self.gamma_a.item()),
                "beta": float(self.beta.item()),
                "gamma_r": float(self.gamma_r.item()),
                "xi": float(self.xi.item()),
                "eta": float(self.eta.item()),
            }


class SPOModel(nn.Module):
    """One patient, one variant.  Same next-rank / STOP interface as v0.1."""

    # Read off the STOP state: total drive, peak, how far the field has spread,
    # how much recovery has accumulated, and where in the event we are.
    N_STOP_FEATURES = 6

    def __init__(self, config: OperatorConfig):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.contact_bias = nn.Parameter(torch.zeros(config.n_contacts))
        self.stop_head = nn.Linear(self.N_STOP_FEATURES, 1)

        if config.variant == "STATIC":
            self.operator = None
            return

        self.operator = PropagationOperator(config)
        self.raw_w_a = nn.Parameter(torch.tensor(_softplus_inv(1.0)))
        self.raw_w_r = nn.Parameter(torch.tensor(_softplus_inv(0.5)))
        H = np.asarray(config.observation_operator, np.float32)
        self.register_buffer("H", torch.from_numpy(H))
        ny, nx = config.grid_shape
        if H.shape[1] != ny * nx:
            raise ValueError(
                f"observation operator has {H.shape[1]} columns but the grid holds "
                f"{ny * nx} cells"
            )
        # Distance from the axis origin, used only to report how far the field
        # spread; never a model input.
        axis = torch.arange(ny, dtype=torch.float32).view(1, ny, 1)
        self.register_buffer("axis_coordinate", axis)

    # -- state -----------------------------------------------------------
    def initial_state(self, batch: int, device: torch.device):
        if self.operator is None:
            return None
        ny, nx = self.config.grid_shape
        zeros = torch.zeros(batch, ny, nx, device=device)
        return zeros, zeros.clone()

    def _stop_features(self, a: Tensor, r: Tensor, t_norm: Tensor,
                       recruited_fraction: Tensor) -> Tensor:
        flat = a.flatten(1)
        total = flat.sum(1, keepdim=True).clamp_min(1e-6)
        mean_axis = (a * self.axis_coordinate).flatten(1).sum(1, keepdim=True) / total
        # sqrt has an infinite derivative at zero, and a rectified field is
        # exactly zero whenever nothing has been driven yet -- so the very first
        # backward pass returns NaN unless the variance is floored first.
        variance = (
            (a * (self.axis_coordinate - mean_axis.view(-1, 1, 1)) ** 2)
            .flatten(1).sum(1, keepdim=True) / total
        )
        spread = variance.clamp_min(1e-12).sqrt()
        return torch.cat([
            flat.mean(1, keepdim=True),
            flat.max(1, keepdim=True).values,
            spread,
            r.flatten(1).mean(1, keepdim=True),
            t_norm,
            recruited_fraction,
        ], dim=1)

    def step(self, state, x_t: Tensor, recruited: Tensor, t_norm: Tensor):
        frac = recruited.mean(dim=1, keepdim=True)
        if self.operator is None:
            logits = self.contact_bias.unsqueeze(0).expand(x_t.shape[0], -1)
            features = torch.cat([
                torch.zeros(x_t.shape[0], 4, device=x_t.device), t_norm, frac
            ], dim=1)
            return state, logits, self.stop_head(features).squeeze(-1)

        a, r = state
        injection = torch.einsum("bc,cm->bm", x_t, self.H).view_as(a)
        for k in range(self.config.microsteps):
            a, r = self.operator.step(
                a, r, injection if k == 0 else None, self.config.nonlinearity
            )
        emission = (F.softplus(self.raw_w_a) * a
                    - F.softplus(self.raw_w_r) * r).flatten(1)
        logits = self.contact_bias + torch.einsum("bm,cm->bc", emission, self.H)
        stop = self.stop_head(self._stop_features(a, r, t_norm, frac)).squeeze(-1)
        return (a, r), logits, stop

    def forward(self, x: Tensor, recruited: Tensor, valid: Tensor):
        batch, steps, _ = x.shape
        state = self.initial_state(batch, x.device)
        contact_logits, stop_logits = [], []
        for t in range(steps):
            t_norm = torch.full((batch, 1), t / max(steps - 1, 1), device=x.device)
            state, logits, stop = self.step(state, x[:, t], recruited[:, t], t_norm)
            contact_logits.append(logits)
            stop_logits.append(stop)
        return torch.stack(contact_logits, 1), torch.stack(stop_logits, 1)

    # -- free rollout ----------------------------------------------------
    @torch.no_grad()
    def rollout(self, seed_set: Tensor, max_steps: int = 32,
                threshold: float = 0.5) -> list[Tensor]:
        """Generate an event from a seed set with no teacher forcing.

        The model's own prediction becomes the next input, which is the only way
        to see whether the operator produces events shaped like the real ones
        rather than merely scoring the next step of a real one.
        """
        batch = seed_set.shape[0]
        state = self.initial_state(batch, seed_set.device)
        recruited = seed_set.clone()
        x_t = seed_set.clone()
        produced = [seed_set.clone()]
        # Each sequence stops on its OWN stop probability. Averaging the batch
        # ends every rollout on the same step, which makes the generated length
        # distribution a single spike and says nothing about the model.
        alive = torch.ones(batch, dtype=torch.bool, device=seed_set.device)
        lengths = torch.ones(batch, dtype=torch.long, device=seed_set.device)
        for t in range(max_steps):
            t_norm = torch.full((batch, 1), t / max(max_steps - 1, 1),
                                device=seed_set.device)
            state, logits, stop = self.step(state, x_t, recruited, t_norm)
            logits = logits.masked_fill(recruited > 0, NEG_INF)
            x_t = (torch.sigmoid(logits) > threshold).float()
            finished = (torch.sigmoid(stop) > 0.5) | (x_t.sum(-1) == 0)
            alive = alive & ~finished
            if not bool(alive.any()):
                break
            x_t = x_t * alive.unsqueeze(-1).float()
            lengths = lengths + alive.long()
            produced.append(x_t.clone())
            recruited = ((recruited + x_t) > 0).float()
        return produced, lengths

    def parameter_estimates(self) -> dict[str, float]:
        if self.operator is None:
            return {"variant": "STATIC"}
        out = self.operator.parameter_estimates()
        with torch.no_grad():
            out["w_a"] = float(F.softplus(self.raw_w_a).item())
            out["w_r"] = float(F.softplus(self.raw_w_r).item())
        # Derived readouts.  Rank is not physical time, so these are per-rank
        # effective quantities and must never be quoted as mm/s.
        K = int(self.config.microsteps)
        out["effective_axial_reach_per_rank"] = float(
            K * abs(out["v"]) + np.sqrt(max(2.0 * K * out["D_parallel"], 0.0))
        )
        out["effective_transverse_spread_per_rank"] = float(
            np.sqrt(max(2.0 * K * out["D_perp"], 0.0))
        )
        # A coefficient sitting on the stability bound is censored, not
        # estimated: the true value is somewhere at or above it and the ratio
        # built from it is a bound too. Saying so is the difference between an
        # estimate and a number that merely came out of the optimiser.
        out["D_parallel_at_bound"] = bool(out["D_parallel"] > 0.98 * D_MAX)
        out["D_perp_at_bound"] = bool(out["D_perp"] > 0.98 * D_MAX)
        out["drift_at_bound"] = bool(abs(out["v"]) > 0.98 * V_MAX)
        out["anisotropy"] = float(out["D_parallel"] / max(out["D_perp"], 1e-9))
        out["anisotropy_is_bounded_estimate"] = bool(
            out["D_parallel_at_bound"] or out["D_perp_at_bound"]
        )
        out["activation_persistence"] = float(1.0 / max(out["gamma_a"], 1e-9))
        out["recovery_strength"] = float(out["beta"] / max(out["gamma_r"], 1e-9))
        return out


def build_grid(contacts_xy: np.ndarray, sigma_mm: float,
               max_cells_per_side: int = 40,
               support_sigma: float = 3.0) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    """Regular grid over the dilated contact hull, with an in-domain mask.

    A regular grid, not the farthest-point node cloud v0.1 used: the whole point
    of this version is that the transport operators are finite differences, and
    those need a lattice with a defined along-axis and across-axis direction.
    """
    points = np.asarray(contacts_xy, float)
    reach = support_sigma * float(sigma_mm)
    lo, hi = points.min(axis=0) - reach, points.max(axis=0) + reach
    span = hi - lo
    step = max(0.5 * float(sigma_mm), float(span.max()) / max_cells_per_side)
    n_along = int(np.clip(np.ceil(span[0] / step) + 1, 4, max_cells_per_side))
    n_across = int(np.clip(np.ceil(span[1] / step) + 1, 4, max_cells_per_side))
    xs = np.linspace(lo[0], hi[0], n_along)
    ys = np.linspace(lo[1], hi[1], n_across)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    centres = np.stack([gx.ravel(), gy.ravel()], axis=-1)
    distance = np.linalg.norm(centres[:, None, :] - points[None, :, :], axis=-1)
    mask = (distance.min(axis=1) <= reach).reshape(n_along, n_across)
    return centres, (n_along, n_across), mask.astype(np.float32)
