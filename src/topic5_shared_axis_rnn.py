"""Topic 5 v0.4 shared-axis structured RNN.

One learned scalar coordinate per contact defines both operators the model
may use:

* a symmetric scaffold, a Gaussian kernel of coordinate distance mixed with
  the fixed same-shaft graph;
* an antisymmetric flow, that same scaffold multiplied elementwise by an odd
  function of the signed coordinate difference.

This replaces the v0.3 rank-two endpoint bridge.  There the two operators
were built from two endpoint memberships, so the symmetric part was a
long-range bridge between the ends and the signed part was a rank-two
perturbation on top of it; the emitted fields from the two starts came out
almost identical (cohort median rank correlation +0.04).  Here the flow is a
true advection along the coordinate: every edge carries a sign set by which
side of the axis it points to, so reversing the source reverses the drift on
every edge rather than on one rank-two term.

The likelihood contract, the STOP and cardinality heads and the exact
conditional k-subset scoring are reused unchanged from v0.3, so the two
model families remain comparable on the same endpoint.

No ictal target, empirical A/B template, mean rank, SOZ or clinical label
enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.topic5_shared_scaffold_rnn import (
    EPS,
    OrdinaryDenseGRUBaseline,
    _RankSetModelInterface,
    _validate_square_adjacency,
    build_fixed_local_shaft_adjacency,
    estimate_node_hazard_bias,
    rank_sets_from_group_ids,
)


# Structural constants frozen by the v0.4 spec before any ictal value is
# read.  They are not part of any grid search.
AXIS_LENGTH_SCALE = 1.0
FLOW_DELTA = 1.0
AXIS_SMOOTHNESS_WEIGHT = 0.01
DIRECTION_GAIN = 2.0


def _offdiagonal_frobenius(matrix: Tensor, eps: float) -> Tensor:
    """Scale a matrix to unit off-diagonal Frobenius norm."""

    off = matrix - torch.diag_embed(torch.diagonal(matrix))
    return off / torch.clamp(torch.linalg.matrix_norm(off), min=eps)


def continuous_axis_operators(
    fixed_adjacency: Tensor,
    axis_coordinate: Tensor,
    *,
    gamma: Tensor | float,
    gain: Tensor | float,
    length_scale: float = AXIS_LENGTH_SCALE,
    delta: float = FLOW_DELTA,
    eps: float = EPS,
) -> dict[str, Tensor]:
    """Symmetric scaffold and its own antisymmetric flow from one coordinate.

    ``W_skew`` is ``W`` multiplied elementwise by ``tanh((s_i - s_j)/delta)``.
    That factor is odd in the index swap while ``W`` is symmetric, so the
    product is exactly antisymmetric and vanishes on the diagonal; it is not
    a second free connectivity.
    """

    fixed = _validate_square_adjacency(fixed_adjacency, eps=eps)
    coordinate = torch.as_tensor(
        axis_coordinate, device=fixed.device, dtype=fixed.dtype
    )
    if coordinate.shape != (fixed.shape[0],) or not torch.isfinite(coordinate).all():
        raise ValueError("axis_coordinate must be a finite contact vector")
    # Centre and set unit population RMS every forward pass, so the scale of
    # the raw parameter cannot trade off against the fixed length scales.
    coordinate = coordinate - coordinate.mean()
    scale = torch.sqrt(torch.mean(coordinate.square()) + eps)
    if bool(scale <= eps):
        raise ValueError("axis_coordinate must vary across contacts")
    coordinate = coordinate / scale

    mix = torch.as_tensor(gamma, device=fixed.device, dtype=fixed.dtype)
    operator_gain = torch.as_tensor(gain, device=fixed.device, dtype=fixed.dtype)
    if bool((mix < 0) | (mix > 1)):
        raise ValueError("gamma must be in [0, 1]")
    if bool(operator_gain <= 0):
        raise ValueError("gain must be positive")
    if float(length_scale) <= 0 or float(delta) <= 0:
        raise ValueError("length_scale and delta must be positive")

    difference = coordinate[:, None] - coordinate[None, :]
    axis_kernel = torch.exp(-difference.square() / (2.0 * float(length_scale) ** 2))
    axis_kernel = axis_kernel - torch.diag_embed(torch.diagonal(axis_kernel))

    # Both terms carry unit off-diagonal norm before mixing, so gamma is a
    # true mixing weight rather than a rescaling of whichever term is larger.
    local = _offdiagonal_frobenius(fixed, eps)
    axis = _offdiagonal_frobenius(axis_kernel, eps)
    adjacency = (1.0 - mix) * local + mix * axis

    degree = adjacency.sum(dim=1)
    inverse_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=eps))
    symmetric = (
        operator_gain
        * inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    skew = symmetric * torch.tanh(difference / float(delta))
    return {
        "W": symmetric,
        "W_skew": skew,
        "A": adjacency,
        "K_local": local,
        "K_axis": axis,
        "axis_coordinate": coordinate,
        "length_scale": torch.as_tensor(float(length_scale)),
        "delta": torch.as_tensor(float(delta)),
    }


def axis_smoothness_penalty(
    axis_coordinate: Tensor, local_graph: Tensor, *, weight: float, eps: float = EPS
) -> Tensor:
    """Same-shaft Dirichlet energy of the normalized coordinate.

    Only the fixed shaft graph enters, so the penalty cannot import any
    learned or target-derived structure.
    """

    coordinate = torch.as_tensor(axis_coordinate)
    coordinate = coordinate - coordinate.mean()
    coordinate = coordinate / torch.sqrt(torch.mean(coordinate.square()) + eps)
    graph = torch.as_tensor(local_graph, device=coordinate.device, dtype=coordinate.dtype)
    difference = coordinate[:, None] - coordinate[None, :]
    numerator = (graph * difference.square()).sum()
    return float(weight) * numerator / (graph.sum() + eps)


@dataclass(frozen=True)
class AxisPropagationState:
    """Rank-step state; ``direction`` is written once from the first rank set."""

    propagation: Tensor
    restraint: Tensor
    direction: Tensor
    source_initialized: Tensor


class SharedAxisPropagationRNN(_RankSetModelInterface):
    """Structured patient-specific RNN with one learned contact axis."""

    def __init__(
        self,
        *,
        fixed_adjacency: np.ndarray | Tensor,
        participation_bias: np.ndarray | Tensor,
        length_scale: float = AXIS_LENGTH_SCALE,
        delta: float = FLOW_DELTA,
        smoothness_weight: float = AXIS_SMOOTHNESS_WEIGHT,
        direction_gain: float = DIRECTION_GAIN,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(fixed_adjacency, dtype=torch.float32)
        adjacency = _validate_square_adjacency(adjacency, eps=eps)
        bias = torch.as_tensor(participation_bias, dtype=torch.float32)
        if bias.shape != (adjacency.shape[0],) or not torch.isfinite(bias).all():
            raise ValueError("participation_bias must be finite and contact-aligned")
        self.n_contacts = int(adjacency.shape[0])
        self.eps = float(eps)
        self.length_scale = float(length_scale)
        self.delta = float(delta)
        self.smoothness_weight = float(smoothness_weight)
        # Fixed, not learned: the coordinate already carries unit RMS, so a
        # learned gain here would only re-introduce the scale it removes.
        self.direction_gain = float(direction_gain)
        self.register_buffer("fixed_adjacency", adjacency)
        self.register_buffer("participation_bias", bias)

        self.axis_coordinate_raw = nn.Parameter(
            torch.empty(self.n_contacts).normal_(mean=0.0, std=1.0)
        )
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        self.gain_raw = nn.Parameter(torch.tensor(0.0))
        self.rho_p_raw = nn.Parameter(torch.tensor(-0.5))
        self.rho_gap_raw = nn.Parameter(torch.tensor(0.0))
        self.propagation_weight_raw = nn.Parameter(torch.tensor(0.5413249))
        self.restraint_weight_raw = nn.Parameter(torch.tensor(-1.2586915))
        self.flow_weight_raw = nn.Parameter(torch.tensor(-0.4327521))

        self.stop_head = nn.Linear(3, 1)
        self.cardinality_head = nn.Linear(3, self.n_contacts)

    # ------------------------------------------------------------ parameters
    @property
    def gamma(self) -> Tensor:
        return torch.sigmoid(self.gamma_raw)

    @property
    def gain(self) -> Tensor:
        return F.softplus(self.gain_raw) + self.eps

    @property
    def rho_p(self) -> Tensor:
        return (1.0 - 2.0 * self.eps) * torch.sigmoid(self.rho_p_raw)

    @property
    def rho_r(self) -> Tensor:
        available_gap = 1.0 - self.eps - self.rho_p
        return self.rho_p + available_gap * torch.sigmoid(self.rho_gap_raw)

    @property
    def propagation_weight(self) -> Tensor:
        return F.softplus(self.propagation_weight_raw) + self.eps

    @property
    def restraint_weight(self) -> Tensor:
        return F.softplus(self.restraint_weight_raw) + self.eps

    @property
    def flow_weight(self) -> Tensor:
        return F.softplus(self.flow_weight_raw) + self.eps

    def operator_components(self) -> dict[str, Tensor]:
        return continuous_axis_operators(
            self.fixed_adjacency,
            self.axis_coordinate_raw,
            gamma=self.gamma,
            gain=self.gain,
            length_scale=self.length_scale,
            delta=self.delta,
            eps=self.eps,
        )

    def smoothness_penalty(self) -> Tensor:
        return axis_smoothness_penalty(
            self.axis_coordinate_raw,
            self.fixed_adjacency,
            weight=self.smoothness_weight,
            eps=self.eps,
        )

    # ----------------------------------------------------------------- state
    def reset_state(self, batch_size: int | None = None) -> AxisPropagationState:
        shape = (
            (self.n_contacts,)
            if batch_size is None
            else (int(batch_size), self.n_contacts)
        )
        event_shape = () if batch_size is None else (int(batch_size),)
        zero = self.participation_bias.new_zeros(shape)
        return AxisPropagationState(
            zero,
            zero.clone(),
            self.participation_bias.new_zeros(event_shape),
            torch.zeros(
                event_shape, dtype=torch.bool, device=self.participation_bias.device
            ),
        )

    def observe(
        self,
        state: AxisPropagationState,
        current_set: Tensor,
        active: Tensor | None = None,
    ) -> AxisPropagationState:
        current = torch.as_tensor(
            current_set,
            device=self.participation_bias.device,
            dtype=self.participation_bias.dtype,
        )
        if current.ndim not in (1, 2) or current.shape[-1] != self.n_contacts:
            raise ValueError("current_set must be [..., contact]")
        if state.propagation.shape != current.shape or state.restraint.shape != current.shape:
            raise ValueError("state and current_set must align")
        if (
            state.direction.shape != current.shape[:-1]
            or state.source_initialized.shape != current.shape[:-1]
        ):
            raise ValueError("state direction must be one value per event")

        components = self.operator_components()
        coordinate = components["axis_coordinate"]
        symmetric, skew = components["W"], components["W_skew"]

        size = current.sum(dim=-1).clamp_min(1.0)
        # A rank set with many simultaneous contacts should not inject more
        # drive than a single-contact one; the tie-set size is modelled by the
        # cardinality head, not by the operator input.
        normalized = current / size[..., None]

        # Causal direction: only the first observed rank set may write it.
        first_rank_direction = -torch.tanh(
            self.direction_gain * (current * coordinate).sum(dim=-1) / size
        )
        direction = torch.where(
            state.source_initialized, state.direction, first_rank_direction
        )

        if current.ndim == 1:
            symmetric_drive = symmetric @ normalized
            skew_drive = skew @ normalized
        else:
            symmetric_drive = normalized @ symmetric.T
            skew_drive = normalized @ skew.T
        propagation_drive = (
            symmetric_drive + self.flow_weight * direction[..., None] * skew_drive
        )
        proposal = AxisPropagationState(
            self.rho_p * state.propagation + propagation_drive,
            self.rho_r * state.restraint + symmetric_drive,
            direction,
            torch.ones_like(state.source_initialized),
        )
        if active is None:
            return proposal
        active_bool = active.to(device=current.device, dtype=torch.bool)
        if current.ndim != 2 or active_bool.shape != current.shape[:1]:
            raise ValueError("active must align with a batched current_set")
        return AxisPropagationState(
            torch.where(active_bool[:, None], proposal.propagation, state.propagation),
            torch.where(active_bool[:, None], proposal.restraint, state.restraint),
            torch.where(active_bool, proposal.direction, state.direction),
            state.source_initialized | active_bool,
        )

    # -------------------------------------------------------------- decision
    def _summary_features(self, state: AxisPropagationState, seen: Tensor) -> Tensor:
        eligible = ~seen.bool()
        weight = eligible.to(state.propagation.dtype)
        denominator = weight.sum(dim=-1).clamp_min(1.0)
        mean_p = (state.propagation * weight).sum(dim=-1) / denominator
        mean_r = (state.restraint * weight).sum(dim=-1) / denominator
        seen_fraction = seen.to(state.propagation.dtype).mean(dim=-1)
        return torch.stack([mean_p, mean_r, seen_fraction], dim=-1)

    def decision(self, state: AxisPropagationState, seen: Tensor) -> dict[str, Tensor]:
        seen_bool = seen.to(device=self.participation_bias.device, dtype=torch.bool)
        if seen_bool.ndim not in (1, 2) or seen_bool.shape[-1] != self.n_contacts:
            raise ValueError("seen must be [..., contact]")
        if state.propagation.shape != seen_bool.shape:
            raise ValueError("state and seen must align")
        eligible = ~seen_bool
        raw_node_logits = (
            self.participation_bias
            + self.propagation_weight * state.propagation
            - self.restraint_weight * state.restraint
        )
        node_logits = raw_node_logits.masked_fill(~eligible, -torch.inf)
        summary = self._summary_features(state, seen_bool)
        stop_logit = self.stop_head(summary).squeeze(-1)
        cardinality_logits = self.cardinality_head(summary)
        n_eligible = eligible.sum(dim=-1)
        cardinality_axis = torch.arange(1, self.n_contacts + 1, device=eligible.device)
        cardinality_support = cardinality_axis <= n_eligible[..., None]
        cardinality_logits = cardinality_logits.masked_fill(
            ~cardinality_support, -torch.inf
        )
        return {
            "node_logits": node_logits,
            "raw_node_logits": raw_node_logits,
            "eligible": eligible,
            "stop_logit": stop_logit,
            "cardinality_logits": cardinality_logits,
            "cardinality_support": cardinality_support,
        }


__all__ = [
    "AXIS_LENGTH_SCALE",
    "AXIS_SMOOTHNESS_WEIGHT",
    "AxisPropagationState",
    "DIRECTION_GAIN",
    "FLOW_DELTA",
    "OrdinaryDenseGRUBaseline",
    "SharedAxisPropagationRNN",
    "axis_smoothness_penalty",
    "build_fixed_local_shaft_adjacency",
    "continuous_axis_operators",
    "estimate_node_hazard_bias",
    "rank_sets_from_group_ids",
]
