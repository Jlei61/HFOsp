"""Dynamical motif RNN family for Topic 5.2 v0.1-r2.

Four nested main models plus three M3 alternative mechanisms.  Every model
shares the parent full-tissue leaky RNN, the contact / STOP heads, the loss and
the decoder; they differ only in how the recurrent operator is built on one
frozen local support.

Index convention (inherited from the parent and verified against
``topic5_wiring_economy_rnn._step``): the recurrent tensor is ``W[i, j]`` with
``i`` the receiving node and ``j`` the source node, and the state update uses
``h @ W.T`` so that ``(W h)_i = sum_j W[i, j] h_j``.  The column normalisation
therefore divides by a sum over receivers ``i`` and makes every source node's
outgoing weight sum to one.

Nothing in this module reads TA/TB template values, future rank sets, future
cardinality or seizure energy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

MAIN_MODELS = (
    "DM0_ISOTROPIC",
    "DM1_FREE_AXIS",
    "DM2_LOCAL_DIRECTIONAL",
    "DM3_AXIS_FEEDFORWARD_TRANSIENT",
)
M3_CONTROLS = (
    "DM3_GAIN_MEMORY",
    "DM3_SYMMETRIC_MATCHED",
    "DM3_AXIS_SHUFFLED_TRIANGULAR",
)
ALL_MODELS = MAIN_MODELS + M3_CONTROLS

# Which model each layer warm-starts from.  ``None`` means trained from scratch.
WARM_START_PARENT = {
    "DM0_ISOTROPIC": None,
    "DM1_FREE_AXIS": "DM0_ISOTROPIC",
    "DM2_LOCAL_DIRECTIONAL": "DM1_FREE_AXIS",
    "DM3_AXIS_FEEDFORWARD_TRANSIENT": "DM2_LOCAL_DIRECTIONAL",
    "DM3_GAIN_MEMORY": "DM2_LOCAL_DIRECTIONAL",
    "DM3_SYMMETRIC_MATCHED": "DM2_LOCAL_DIRECTIONAL",
    "DM3_AXIS_SHUFFLED_TRIANGULAR": "DM2_LOCAL_DIRECTIONAL",
}

# Parameters introduced by each layer.  Everything else is "shared" and carries
# the anchor penalty during joint fine-tuning.
NEW_PARAMETERS = {
    "DM0_ISOTROPIC": (),
    "DM1_FREE_AXIS": ("theta", "eta_raw"),
    "DM2_LOCAL_DIRECTIONAL": ("beta",),
    "DM3_AXIS_FEEDFORWARD_TRANSIENT": ("gamma_raw",),
    "DM3_GAIN_MEMORY": ("delta_g", "delta_kappa"),
    "DM3_SYMMETRIC_MATCHED": ("gamma_raw",),
    "DM3_AXIS_SHUFFLED_TRIANGULAR": ("gamma_raw",),
}

# Parameters that must stay non-negative.  Projected after every optimiser step
# so that the zero value is exactly representable and the nested equivalence is
# bit-exact at warm start.
NONNEGATIVE_PARAMETERS = ("eta_raw", "gamma_raw", "delta_g", "delta_kappa")

GATE_RULES = ("M2-2RANK", "M2-3RANK", "M2-ONLINE")
# GLOBAL_AXIS biases transport along one learned patient axis; EVENT_VECTOR
# uses each event's own early displacement direction and assumes no stable
# global corridor at all.
DIRECTION_MODES = ("GLOBAL_AXIS", "EVENT_VECTOR")
EPS = 1e-8


@dataclass(frozen=True)
class MotifConfig:
    model_id: str
    n_contacts: int
    n_nodes: int
    observation_operator: np.ndarray      # H, (n_contacts, n_nodes)
    node_xy_mm: np.ndarray                # (n_nodes, 2)
    local_mask: np.ndarray                # (n_nodes, n_nodes) uint8, i<-j support
    r_forward_mm: float                   # M3 forward-cone radius (frozen r_local_mm)
    sigma_s_mm: float                     # direction-evidence scale (frozen on split 1)
    seed: int = 0
    theta_init: float = 0.0
    gate_rule: str = "M2-2RANK"
    direction_mode: str = "GLOBAL_AXIS"
    stop_hidden: int = 16
    shuffle_seed: int = 0
    shuffle_distance_bins: int = 4
    shuffle_degree_bins: int = 2

    def __post_init__(self) -> None:
        if self.model_id not in ALL_MODELS:
            raise ValueError(f"unknown motif model {self.model_id!r}")
        if self.gate_rule not in GATE_RULES:
            raise ValueError(f"unknown gate rule {self.gate_rule!r}")
        if self.direction_mode not in DIRECTION_MODES:
            raise ValueError(f"unknown direction mode {self.direction_mode!r}")
        if self.observation_operator.shape != (self.n_contacts, self.n_nodes):
            raise ValueError("H must be (n_contacts, n_nodes)")
        if self.node_xy_mm.shape != (self.n_nodes, 2):
            raise ValueError("node_xy_mm must be (n_nodes, 2)")
        if self.local_mask.shape != (self.n_nodes, self.n_nodes):
            raise ValueError("local_mask must be (n_nodes, n_nodes)")
        if not float(self.r_forward_mm) > 0:
            raise ValueError("r_forward_mm must be positive")
        if not float(self.sigma_s_mm) > 0:
            raise ValueError("sigma_s_mm must be positive")


def axis_shuffled_permutation(
    node_xy_mm: np.ndarray,
    local_mask: np.ndarray,
    seed: int,
    n_distance_bins: int = 4,
    n_degree_bins: int = 2,
) -> np.ndarray:
    """Permute node order inside frozen radius/degree bins.

    The control keeps the local support, the kernel weights and triangularity,
    and destroys only the alignment between the cascade order and the patient
    axis.  The cone radius is recalibrated afterwards so the non-zero count
    matches the axis-aligned cascade (see :meth:`MotifRNN.calibrate_shuffle_radius`).
    """
    points = np.asarray(node_xy_mm, dtype=float)
    mask = np.asarray(local_mask, dtype=bool)
    n = points.shape[0]
    radius = np.linalg.norm(points - points.mean(axis=0), axis=1)
    degree = mask.sum(axis=1)
    distance_bin = np.digitize(radius, np.quantile(radius, np.linspace(0, 1, n_distance_bins + 1)[1:-1]))
    degree_bin = np.digitize(degree, np.quantile(degree, np.linspace(0, 1, n_degree_bins + 1)[1:-1]))
    rng = np.random.default_rng(seed)
    permutation = np.arange(n)
    for d in range(n_distance_bins):
        for g in range(n_degree_bins):
            members = np.flatnonzero((distance_bin == d) & (degree_bin == g))
            if members.size > 1:
                permutation[members] = rng.permutation(members)
    return permutation.astype(int)


class MotifRNN(nn.Module):
    """Full-tissue leaky RNN whose recurrent operator is a structured motif."""

    def __init__(self, config: MotifConfig):
        super().__init__()
        self.config = config
        self.model_id = config.model_id
        self.n_contacts = int(config.n_contacts)
        self.n_nodes = int(config.n_nodes)
        self.gate_rule = config.gate_rule
        torch.manual_seed(int(config.seed))

        self.register_buffer("H", torch.as_tensor(config.observation_operator, dtype=torch.float32))
        self.register_buffer("node_xy", torch.as_tensor(config.node_xy_mm, dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(np.asarray(config.local_mask) > 0, dtype=torch.float32))

        delta = self.node_xy[:, None, :] - self.node_xy[None, :, :]      # (i, j, 2) = r_i - r_j
        self.register_buffer("delta", delta)
        self.register_buffer("delta_sq", (delta ** 2).sum(-1))

        local_lengths = self.delta_sq.sqrt()[self.mask > 0]
        ell0 = float(local_lengths.median()) if local_lengths.numel() else 1.0
        self.log_ell = nn.Parameter(torch.tensor(math.log(max(ell0, 1e-3))))
        self.log_g = nn.Parameter(torch.tensor(0.0))
        self.kappa_logit = nn.Parameter(torch.zeros(1))
        self.node_bias = nn.Parameter(torch.zeros(self.n_nodes))
        self.input_gain = nn.Parameter(torch.ones(self.n_nodes))
        self.contact_bias = nn.Parameter(torch.zeros(self.n_contacts))
        self.readout_gain = nn.Parameter(torch.tensor(4.0))
        self.stop_head = nn.Sequential(
            nn.Linear(4, int(config.stop_hidden)), nn.Tanh(),
            nn.Linear(int(config.stop_hidden), 1),
        )

        # Motif parameters.  Layers that do not own a parameter register it as a
        # zero buffer so the forward code stays one expression and the nested
        # equivalence is exact rather than approximate.
        self._register_motif_parameter("theta", float(config.theta_init),
                                       owned=self.model_id != "DM0_ISOTROPIC")
        self._register_motif_parameter("eta_raw", 0.0, owned=self.model_id != "DM0_ISOTROPIC")
        self._register_motif_parameter(
            "beta", 0.0,
            owned=self.model_id not in ("DM0_ISOTROPIC", "DM1_FREE_AXIS"),
        )
        self._register_motif_parameter(
            "gamma_raw", 0.0,
            owned=self.model_id in (
                "DM3_AXIS_FEEDFORWARD_TRANSIENT",
                "DM3_SYMMETRIC_MATCHED",
                "DM3_AXIS_SHUFFLED_TRIANGULAR",
            ),
        )
        self._register_motif_parameter("delta_g", 0.0, owned=self.model_id == "DM3_GAIN_MEMORY")
        self._register_motif_parameter("delta_kappa", 0.0, owned=self.model_id == "DM3_GAIN_MEMORY")

        if self.model_id == "DM3_AXIS_SHUFFLED_TRIANGULAR":
            permutation = axis_shuffled_permutation(
                config.node_xy_mm, config.local_mask,
                int(config.shuffle_seed), int(config.shuffle_distance_bins),
                int(config.shuffle_degree_bins),
            )
            self.register_buffer("shuffle_permutation", torch.as_tensor(permutation, dtype=torch.long))
        else:
            self.register_buffer("shuffle_permutation", torch.arange(self.n_nodes, dtype=torch.long))
        self.register_buffer("r_forward_effective", torch.tensor(float(config.r_forward_mm)))

    def _register_motif_parameter(self, name: str, value: float, owned: bool) -> None:
        tensor = torch.tensor(float(value))
        if owned:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

    # ---- structured operator -------------------------------------------
    def axis_unit(self) -> tuple[Tensor, Tensor]:
        theta = self.theta
        u = torch.stack([torch.cos(theta), torch.sin(theta)])
        u_perp = torch.stack([-torch.sin(theta), torch.cos(theta)])
        return u, u_perp

    def axis_kernel(self) -> Tensor:
        """``K^axis`` (``K^iso`` when ``eta`` is zero), on the frozen support."""
        ell = torch.exp(self.log_ell)
        eta = self.eta_raw
        u, u_perp = self.axis_unit()
        parallel = self.delta @ u
        orthogonal = self.delta @ u_perp
        ell_par = ell * torch.exp(eta)
        ell_perp = ell * torch.exp(-eta)
        d2 = (parallel / ell_par) ** 2 + (orthogonal / ell_perp) ** 2
        return self.mask * torch.exp(-0.5 * d2)

    def axial_position(self) -> Tensor:
        u, _ = self.axis_unit()
        return self.node_xy @ u

    def feedforward_pair(self) -> tuple[Tensor, Tensor]:
        """``F+`` and ``F-`` on the frozen support, ordered by the patient axis."""
        kernel = self.axis_kernel()
        q = self.axial_position()
        if self.model_id == "DM3_AXIS_SHUFFLED_TRIANGULAR":
            q = q[self.shuffle_permutation]
        dq = q[:, None] - q[None, :]
        forward = ((dq > 0) & (dq < self.r_forward_effective)).to(kernel.dtype)
        f_plus = kernel * forward
        return f_plus, f_plus.transpose(0, 1)

    @torch.no_grad()
    def calibrate_shuffle_radius(self) -> dict[str, float]:
        """Match the shuffled cascade's non-zero count to the axis-aligned one.

        Permuting the axial order changes which local edges fall inside the
        forward cone.  The radius is the only free knob that restores the
        non-zero count without touching the support or the kernel weights, and
        it is chosen from geometry alone.
        """
        if self.model_id != "DM3_AXIS_SHUFFLED_TRIANGULAR":
            return {"calibrated": False}
        q = self.axial_position()
        support = self.mask > 0
        aligned_dq = q[:, None] - q[None, :]
        target = int((support & (aligned_dq > 0) & (aligned_dq < self.r_forward_effective)).sum())
        shuffled = q[self.shuffle_permutation]
        dq = shuffled[:, None] - shuffled[None, :]
        positive = dq[support & (dq > 0)]
        if positive.numel() == 0 or target == 0:
            return {"calibrated": False, "target_nonzero": target}
        ordered = torch.sort(positive).values
        index = min(max(target - 1, 0), ordered.numel() - 1)
        radius = float(ordered[index]) + 1e-9
        self.r_forward_effective.fill_(radius)
        achieved = int((support & (dq > 0) & (dq < self.r_forward_effective)).sum())
        return {
            "calibrated": True,
            "target_nonzero": target,
            "achieved_nonzero": achieved,
            "r_forward_mm": radius,
            "r_forward_axis_aligned_mm": float(self.config.r_forward_mm),
        }

    def recurrent_terms(self) -> dict[str, Tensor]:
        """Everything the step needs that does not depend on the direction gate."""
        kernel = self.axis_kernel()
        q = self.axial_position()
        terms = {
            "kernel": kernel,
            "q_centred": q - q.mean(),
            "ell": torch.exp(self.log_ell),
            "gain": torch.exp(self.log_g) * torch.exp(self.delta_g),
            "kappa": torch.sigmoid(self.kappa_logit - self.delta_kappa),
        }
        if self.model_id in ("DM3_AXIS_FEEDFORWARD_TRANSIENT",
                             "DM3_SYMMETRIC_MATCHED",
                             "DM3_AXIS_SHUFFLED_TRIANGULAR"):
            f_plus, f_minus = self.feedforward_pair()
            terms["f_plus"] = f_plus
            terms["f_minus"] = f_minus
        return terms

    def recurrent_matrix(self, s: float = 0.0) -> Tensor:
        """Dense ``W`` at one gate value.  Diagnostics and tests only."""
        terms = self.recurrent_terms()
        kernel = terms["kernel"]
        scale = torch.exp(self.beta * float(s) * terms["q_centred"] / terms["ell"])
        weighted = scale[:, None] * kernel
        normalised = weighted / (weighted.sum(dim=0, keepdim=True) + EPS)
        matrix = terms["gain"] * normalised
        if "f_plus" in terms:
            gate = float(s)
            if self.model_id == "DM3_SYMMETRIC_MATCHED":
                extra = abs(gate) * 0.5 * (terms["f_plus"] + terms["f_minus"])
            else:
                extra = max(gate, 0.0) * terms["f_plus"] + max(-gate, 0.0) * terms["f_minus"]
            matrix = matrix + self.gamma_raw * extra
        return matrix

    def direction_weight(self, displacement: Tensor, u: Tensor) -> Tensor:
        """Per-step 2-D direction weight feeding the transport bias.

        ``GLOBAL_AXIS`` projects the observed displacement onto the learned
        corridor and keeps its sign and magnitude; ``EVENT_VECTOR`` drops the
        corridor entirely and uses the event's own unit displacement.
        """
        if self.config.direction_mode == "EVENT_VECTOR":
            norm = displacement.norm(dim=-1, keepdim=True)
            return displacement / (norm + 1e-9)
        return self.direction_gate(displacement, u)[..., None] * u

    def recurrent_drive(self, h: Tensor, s: Tensor, terms: dict[str, Tensor],
                        weight: Tensor | None = None) -> Tensor:
        """``W(s) h`` without ever materialising a per-event matrix.

        Column normalisation makes the source-side factor of the directional
        kernel cancel, so ``W(s)h = g * a(s) * (K (h / c(s)))`` with ``a`` and
        ``c`` vectors.  This keeps the per-step cost at two dense mat-vecs.
        """
        kernel = terms["kernel"]
        if weight is None:
            exponent = self.beta * s[:, None] * terms["q_centred"][None, :] / terms["ell"]
        else:
            axial = weight @ self.node_xy.transpose(0, 1)
            exponent = self.beta * (axial - axial.mean(dim=1, keepdim=True)) / terms["ell"]
        scale = torch.exp(exponent - exponent.max(dim=1, keepdim=True).values)
        column = scale @ kernel + EPS                       # c_j(s), (B, n)
        drive = terms["gain"] * scale * ((h / column) @ kernel.transpose(0, 1))
        if "f_plus" in terms:
            if self.model_id == "DM3_SYMMETRIC_MATCHED":
                magnitude = s.abs()[:, None]
                extra = 0.5 * magnitude * (
                    h @ terms["f_plus"].transpose(0, 1) + h @ terms["f_minus"].transpose(0, 1)
                )
            else:
                extra = (
                    torch.clamp(s, min=0.0)[:, None] * (h @ terms["f_plus"].transpose(0, 1))
                    + torch.clamp(-s, min=0.0)[:, None] * (h @ terms["f_minus"].transpose(0, 1))
                )
            drive = drive + self.gamma_raw * extra
        return drive

    # ---- dynamics -------------------------------------------------------
    def step(self, h: Tensor, x_t: Tensor, s: Tensor, terms: dict[str, Tensor],
             weight: Tensor | None = None) -> Tensor:
        u = (x_t @ self.H) * self.input_gain
        pre = u + self.recurrent_drive(h, s, terms, weight) + self.node_bias
        kappa = terms["kappa"]
        return (1.0 - kappa) * h + kappa * torch.tanh(pre)

    def readout(self, h: Tensor) -> Tensor:
        return self.contact_bias + self.readout_gain * (h @ self.H.transpose(0, 1))

    def state_features(self, h: Tensor, t_norm: Tensor, recruited_fraction: Tensor) -> Tensor:
        return torch.stack([h.mean(-1), h.max(-1).values, t_norm, recruited_fraction], dim=-1)

    def stop_logit(self, features: Tensor) -> Tensor:
        return self.stop_head(features).squeeze(-1)

    def direction_gate(self, displacement: Tensor, u: Tensor) -> Tensor:
        """``s_k = tanh(u . a_k / sigma_s)`` for a pre-built causal displacement."""
        return torch.tanh((displacement @ u) / float(self.config.sigma_s_mm))

    def forward(
        self,
        x: Tensor,
        recruited: Tensor,
        displacement: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Teacher-forced pass.

        ``displacement`` is ``(B, steps, 2)`` and holds the causal centroid
        displacement that defines the direction gate at each step; it is built
        from the observed prefix only and is zero at the first rank.
        """
        batch, steps, _ = x.shape
        device = x.device
        h = torch.zeros(batch, self.n_nodes, device=device, dtype=x.dtype)
        terms = self.recurrent_terms()
        u, _ = self.axis_unit()
        gate = self.direction_gate(displacement, u)                 # (B, steps)
        weight = (self.direction_weight(displacement, u)
                  if self.config.direction_mode != "GLOBAL_AXIS" else None)
        denom = max(1, self.n_contacts - 1)
        logits, stops = [], []
        for t in range(steps):
            h = self.step(h, x[:, t], gate[:, t], terms,
                          None if weight is None else weight[:, t])
            logits.append(self.readout(h))
            t_norm = torch.full((batch,), t / denom, device=device, dtype=x.dtype)
            stops.append(self.stop_logit(self.state_features(h, t_norm, recruited[:, t].mean(-1))))
        return torch.stack(logits, 1), torch.stack(stops, 1), gate

    # ---- parameter bookkeeping ------------------------------------------
    def project_constraints(self) -> None:
        with torch.no_grad():
            for name in NONNEGATIVE_PARAMETERS:
                tensor = getattr(self, name)
                if isinstance(tensor, nn.Parameter):
                    tensor.clamp_(min=0.0)

    def new_parameter_names(self) -> tuple[str, ...]:
        return NEW_PARAMETERS[self.model_id]

    def shared_parameter_names(self) -> list[str]:
        new = set(self.new_parameter_names())
        return [name for name, _ in self.named_parameters() if name not in new]

    def load_warm_start(self, state: dict[str, Tensor]) -> list[str]:
        """Copy every parameter the previous layer also had; report the rest."""
        own = dict(self.named_parameters())
        buffers = dict(self.named_buffers())
        copied: list[str] = []
        with torch.no_grad():
            for name, value in state.items():
                target = own.get(name)
                if target is None:
                    target = buffers.get(name)
                    if target is None or target.shape != value.shape:
                        continue
                    target.copy_(value)
                    copied.append(name)
                    continue
                if target.shape == value.shape:
                    target.copy_(value)
                    copied.append(name)
        return copied

    def numerical_audit(self) -> dict[str, float]:
        with torch.no_grad():
            terms = self.recurrent_terms()
            kernel = terms["kernel"]
            column = kernel.sum(dim=0)
            audit = {
                "ell_mm": float(torch.exp(self.log_ell)),
                "gain": float(terms["gain"]),
                "kappa": float(terms["kappa"]),
                "theta_rad": float(self.theta) % math.pi,
                "eta": float(self.eta_raw),
                "beta": float(self.beta),
                "gamma": float(self.gamma_raw),
                "delta_g": float(self.delta_g),
                "delta_kappa": float(self.delta_kappa),
                "kernel_min_column_sum": float(column.min()),
                "kernel_nonzero": int((kernel > 0).sum()),
                "local_edges": int((self.mask > 0).sum()),
                "recurrent_row_sum_max": float(self.recurrent_matrix(0.0).sum(dim=1).max()),
            }
            if "f_plus" in terms:
                audit["feedforward_nonzero"] = int((terms["f_plus"] > 0).sum())
                audit["feedforward_row_sum_max"] = float(terms["f_plus"].sum(dim=1).max())
        return audit


class StaticReadout(nn.Module):
    """No-recurrence participation baseline (spec section 3.3)."""

    def __init__(self, n_contacts: int, covariates: np.ndarray, stop_hidden: int = 16, seed: int = 0):
        super().__init__()
        torch.manual_seed(int(seed))
        self.n_contacts = int(n_contacts)
        self.register_buffer("covariates", torch.as_tensor(covariates, dtype=torch.float32))
        n_cov = self.covariates.shape[1]
        self.contact_bias = nn.Parameter(torch.zeros(self.n_contacts))
        self.start_weight = nn.Parameter(torch.zeros(self.n_contacts, self.n_contacts))
        self.recruited_weight = nn.Parameter(torch.zeros(self.n_contacts, self.n_contacts))
        self.covariate_weight = nn.Parameter(torch.zeros(n_cov))
        self.stop_head = nn.Sequential(
            nn.Linear(4, int(stop_hidden)), nn.Tanh(), nn.Linear(int(stop_hidden), 1)
        )

    def forward(self, x: Tensor, recruited: Tensor, displacement: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, steps, _ = x.shape
        device = x.device
        start = x[:, 0]
        base = self.contact_bias + start @ self.start_weight.transpose(0, 1)
        base = base + (self.covariates @ self.covariate_weight)
        denom = max(1, self.n_contacts - 1)
        logits, stops = [], []
        for t in range(steps):
            step_logits = base + recruited[:, t] @ self.recruited_weight.transpose(0, 1)
            logits.append(step_logits)
            t_norm = torch.full((batch,), t / denom, device=device, dtype=x.dtype)
            zeros = torch.zeros_like(t_norm)
            features = torch.stack([zeros, zeros, t_norm, recruited[:, t].mean(-1)], dim=-1)
            stops.append(self.stop_head(features).squeeze(-1))
        return torch.stack(logits, 1), torch.stack(stops, 1), torch.zeros(batch, steps, device=device)

    def project_constraints(self) -> None:
        return None


class CapacityMatchedStaticReadout(nn.Module):
    """Low-rank no-recurrence baseline with no more parameters than DM0.

    ``StaticReadout`` has two unconstrained contact-by-contact matrices.  For
    many patients that baseline is larger than the tissue RNN, so a win cannot
    distinguish a useful static shortcut from raw capacity.  This variant
    factorises the start-contact and recruited-contact maps through one shared
    output basis.  The rank is selected from the DM0 parameter budget before
    any outcome is read.
    """

    def __init__(self, n_contacts: int, covariates: np.ndarray, rank: int,
                 stop_hidden: int = 16, seed: int = 0):
        super().__init__()
        if int(rank) < 1:
            raise ValueError("rank must be positive")
        torch.manual_seed(int(seed))
        self.n_contacts = int(n_contacts)
        self.rank = int(rank)
        self.register_buffer("covariates", torch.as_tensor(covariates, dtype=torch.float32))
        n_cov = int(self.covariates.shape[1])
        scale = 1.0 / math.sqrt(max(1, self.rank))
        self.contact_bias = nn.Parameter(torch.zeros(self.n_contacts))
        self.output_factor = nn.Parameter(
            scale * torch.randn(self.n_contacts, self.rank))
        self.start_factor = nn.Parameter(
            scale * torch.randn(self.n_contacts, self.rank))
        self.recruited_factor = nn.Parameter(
            scale * torch.randn(self.n_contacts, self.rank))
        self.covariate_weight = nn.Parameter(torch.zeros(n_cov))
        self.stop_head = nn.Sequential(
            nn.Linear(4, int(stop_hidden)), nn.Tanh(), nn.Linear(int(stop_hidden), 1)
        )

    def forward(self, x: Tensor, recruited: Tensor,
                displacement: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, steps, _ = x.shape
        device = x.device
        start_code = x[:, 0] @ self.start_factor
        base = self.contact_bias + start_code @ self.output_factor.transpose(0, 1)
        base = base + (self.covariates @ self.covariate_weight)
        denom = max(1, self.n_contacts - 1)
        logits, stops = [], []
        for t in range(steps):
            recruited_code = recruited[:, t] @ self.recruited_factor
            step_logits = base + recruited_code @ self.output_factor.transpose(0, 1)
            logits.append(step_logits)
            t_norm = torch.full((batch,), t / denom, device=device, dtype=x.dtype)
            zeros = torch.zeros_like(t_norm)
            features = torch.stack(
                [zeros, zeros, t_norm, recruited[:, t].mean(-1)], dim=-1)
            stops.append(self.stop_head(features).squeeze(-1))
        return (torch.stack(logits, 1), torch.stack(stops, 1),
                torch.zeros(batch, steps, device=device))

    def project_constraints(self) -> None:
        return None


def trainable_parameter_count(model: nn.Module) -> int:
    """Exact number of trainable scalar parameters."""
    return int(sum(parameter.numel() for parameter in model.parameters()
                   if parameter.requires_grad))


def capacity_matched_static_rank(
    n_contacts: int,
    covariates: np.ndarray,
    dm0_parameter_count: int,
    stop_hidden: int = 16,
) -> int:
    """Largest low rank whose exact parameter count does not exceed DM0."""
    for rank in range(int(n_contacts), 0, -1):
        candidate = CapacityMatchedStaticReadout(
            n_contacts, covariates, rank, stop_hidden=stop_hidden, seed=0)
        if trainable_parameter_count(candidate) <= int(dm0_parameter_count):
            return int(rank)
    raise ValueError(
        "DM0 parameter budget is smaller than the rank-1 static baseline; "
        "a capacity-matched comparison is not feasible for this unit")


def build_motif_event_tensors(
    ranks: np.ndarray,
    contacts_xy_mm: np.ndarray,
    gate_rule: str = "M2-2RANK",
) -> dict[str, Tensor]:
    """Padded teacher-forcing tensors plus the causal direction-gate displacement.

    The displacement at step ``t`` uses rank sets ``0..t`` only.  ``M2-2RANK``
    freezes it after rank 1, ``M2-3RANK`` after rank 2, and ``M2-ONLINE`` keeps
    updating the per-step mean displacement.  All three are zero at ``t = 0``.
    """
    if gate_rule not in GATE_RULES:
        raise ValueError(f"unknown gate rule {gate_rule!r}")
    ranks = np.asarray(ranks)
    xy = np.asarray(contacts_xy_mm, dtype=np.float32)
    n_events, n_contacts = ranks.shape
    if xy.shape != (n_contacts, 2):
        raise ValueError("contacts_xy_mm must align with the rank matrix")
    lengths = np.array([int(r[r >= 0].max()) + 1 if np.any(r >= 0) else 0 for r in ranks])
    steps = int(lengths.max())

    x = np.zeros((n_events, steps, n_contacts), np.float32)
    target = np.zeros_like(x)
    recruited = np.zeros_like(x)
    valid = np.zeros((n_events, steps), bool)
    is_last = np.zeros((n_events, steps), bool)
    centroid = np.zeros((n_events, steps, 2), np.float32)
    for e, row in enumerate(ranks):
        length = lengths[e]
        for t in range(length):
            members = row == t
            x[e, t, members] = 1.0
            centroid[e, t] = xy[members].mean(axis=0)
            recruited[e, t] = (row >= 0) & (row <= t)
            if t + 1 < length:
                target[e, t, row == t + 1] = 1.0
        valid[e, :length] = True
        if length:
            is_last[e, length - 1] = True
            centroid[e, length:] = centroid[e, length - 1]

    displacement = causal_displacement(centroid, lengths, gate_rule)
    available = (recruited == 0) & valid[:, :, None]
    return {
        "x": torch.from_numpy(x),
        "recruited": torch.from_numpy(recruited),
        "available": torch.from_numpy(available),
        "target": torch.from_numpy(target),
        "valid": torch.from_numpy(valid),
        "is_last": torch.from_numpy(is_last),
        "displacement": torch.from_numpy(displacement),
        "centroid": torch.from_numpy(centroid),
        "length": torch.from_numpy(lengths.astype(np.int64)),
    }


def causal_displacement(centroid: np.ndarray, lengths: np.ndarray, gate_rule: str) -> np.ndarray:
    """Per-step displacement vector feeding the direction gate."""
    centroid = np.asarray(centroid, dtype=np.float32)
    n_events, steps, _ = centroid.shape
    displacement = np.zeros_like(centroid)
    for t in range(1, steps):
        available = np.minimum(t, np.maximum(lengths - 1, 0))
        if gate_rule == "M2-2RANK":
            k = np.minimum(available, 1)
        elif gate_rule == "M2-3RANK":
            k = np.minimum(available, 2)
        else:
            k = available
        index = np.clip(k, 0, steps - 1)
        step_count = np.maximum(index, 1).astype(np.float32)
        picked = centroid[np.arange(n_events), index]
        displacement[:, t] = (picked - centroid[:, 0]) / step_count[:, None]
        displacement[k < 1, t] = 0.0
    return displacement


def rollout_displacement_update(
    previous: Tensor,
    centroid_start: Tensor,
    centroid_now: Tensor,
    step_index: int,
    gate_rule: str,
) -> Tensor:
    """Closed-loop counterpart of :func:`causal_displacement`."""
    if step_index < 1:
        return torch.zeros_like(previous)
    freeze_at = {"M2-2RANK": 1, "M2-3RANK": 2, "M2-ONLINE": None}[gate_rule]
    if freeze_at is not None and step_index > freeze_at:
        return previous
    return (centroid_now - centroid_start) / float(step_index)


def freeze_direction_scale(
    ranks: np.ndarray,
    contacts_xy_mm: np.ndarray,
    indices: Sequence[int],
    axis_u: np.ndarray | None = None,
) -> float:
    """Median first-step centroid displacement on the calibration split.

    The default is the displacement *magnitude*, which keeps the gate scale
    independent of the learned axis; passing an explicit axis gives the
    projected version used only for reporting.
    """
    xy = np.asarray(contacts_xy_mm, dtype=float)
    if axis_u is not None:
        u = np.asarray(axis_u, dtype=float)
        u = u / max(float(np.linalg.norm(u)), 1e-12)
    values = []
    for index in np.asarray(indices, dtype=int):
        row = np.asarray(ranks[index])
        if not np.any(row == 1):
            continue
        step = xy[row == 1].mean(axis=0) - xy[row == 0].mean(axis=0)
        values.append(abs(float(step @ u)) if axis_u is not None else float(np.linalg.norm(step)))
    if not values:
        raise RuntimeError("no calibration event exposes a second rank set")
    scale = float(np.median(values))
    return scale if scale > 1e-6 else 1e-6
