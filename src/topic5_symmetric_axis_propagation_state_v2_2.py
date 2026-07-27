"""Frozen v2.2 symmetric-axis propagation-state recurrent model.

The pseudo-time index is an observed within-event rank step.  This module has
one contact-wise propagation state, one symmetric effective scaffold, and one
scalar absorbing STOP process.  It intentionally exposes no dense recurrent
mixing, free decoder, future head, event-history input, or ictal target input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1.0e-8
VALID_PERSISTENCE_LABEL = "rank_step_persistence"


def _validate_geometry(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [contact, 3]")
    if coords.shape[0] < 2 or not np.all(np.isfinite(coords)):
        raise ValueError("at least two fully mapped contacts are required")
    return coords - np.mean(coords, axis=0, keepdims=True)


def fixed_local_scale(coords: np.ndarray, eps: float = EPS) -> float:
    """Median non-zero nearest-neighbour distance from implant geometry."""
    centered = _validate_geometry(coords)
    distances = np.linalg.norm(
        centered[:, None, :] - centered[None, :, :], axis=-1
    )
    distances[distances <= eps] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        raise ValueError("implant geometry has no non-zero contact distance")
    scale = float(np.median(nearest))
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError("invalid local geometry scale")
    return scale


def canonicalize_axis(axis: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Return a unit-vector representative of the sign-invariant physical axis."""
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("axis must be a finite vector of shape [3]")
    norm = float(np.linalg.norm(axis))
    if norm <= eps:
        raise ValueError("axis norm must be positive")
    out = axis / norm
    anchor = int(np.argmax(np.abs(out)))
    if out[anchor] < 0:
        out = -out
    return out


def validate_normalization_contract(name: str) -> None:
    """Reject normalizations that can turn a symmetric graph into a directed one."""
    if name != "symmetric_degree":
        raise ValueError("v2.2 permits only D^-1/2 A D^-1/2 normalization")


def validate_persistence_label(label: str) -> None:
    """Prevent rank-step persistence from being assigned a physical time unit."""
    if label != VALID_PERSISTENCE_LABEL:
        raise ValueError("rho_p is rank-step persistence, not a time constant")


def frozen_rollout_horizons(
    *, n_contacts: int, n_seen: int, h_train: int
) -> dict[str, int]:
    if not (0 <= n_seen <= n_contacts):
        raise ValueError("n_seen must lie in [0, n_contacts]")
    if h_train not in (0, 3, 5):
        raise ValueError("h_train must be one of 0, 3, or 5")
    return {
        "H_train": int(h_train),
        "H_eval": int(n_contacts - n_seen),
        "H_transfer": int(n_contacts - n_seen),
    }


def train_only_source_side_thresholds(
    train_source_projection: np.ndarray,
) -> dict[str, float]:
    """Frozen Q25/Q75 source-side thresholds from train events only."""
    values = np.asarray(train_source_projection, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("train source projections are empty")
    q25, q75 = np.quantile(values, [0.25, 0.75])
    return {"left_max": float(q25), "right_min": float(q75)}


def _frobenius_normalize(matrix: Tensor, eps: float) -> Tensor:
    return matrix / torch.clamp(torch.linalg.vector_norm(matrix), min=eps)


def symmetric_axis_operator(
    coords: Tensor,
    axis: Tensor,
    *,
    anisotropy_ratio: Tensor | float,
    gamma: Tensor | float,
    gain: Tensor | float,
    local_scale: Tensor | float | None = None,
    eps: float = EPS,
) -> dict[str, Tensor]:
    """Build the only allowed v2.2 symmetric effective propagation operator."""
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape [contact, 3]")
    if coords.shape[0] < 2 or not torch.isfinite(coords).all():
        raise ValueError("coords must contain at least two finite contacts")
    if axis.shape != (3,) or not torch.isfinite(axis).all():
        raise ValueError("axis must have shape [3]")

    centered = coords - coords.mean(dim=0, keepdim=True)
    axis_unit = axis / torch.clamp(torch.linalg.vector_norm(axis), min=eps)
    delta = centered[:, None, :] - centered[None, :, :]
    distance_sq = torch.sum(delta.square(), dim=-1)
    if local_scale is None:
        with torch.no_grad():
            distance = torch.sqrt(torch.clamp(distance_sq, min=0.0))
            eye = torch.eye(
                coords.shape[0], device=coords.device, dtype=torch.bool
            )
            masked = distance.masked_fill(eye | (distance <= eps), torch.inf)
            nearest = torch.min(masked, dim=1).values
            finite = nearest[torch.isfinite(nearest)]
            if finite.numel() == 0:
                raise ValueError("implant geometry has no non-zero contact distance")
            scale = torch.median(finite)
    else:
        scale = torch.as_tensor(
            local_scale, device=coords.device, dtype=coords.dtype
        )
    if not torch.isfinite(scale) or bool(scale <= eps):
        raise ValueError("local_scale must be finite and positive")

    ratio = torch.as_tensor(
        anisotropy_ratio, device=coords.device, dtype=coords.dtype
    )
    mix = torch.as_tensor(gamma, device=coords.device, dtype=coords.dtype)
    propagation_gain = torch.as_tensor(
        gain, device=coords.device, dtype=coords.dtype
    )
    if bool(ratio < 1.0):
        raise ValueError("anisotropy_ratio must be at least one")
    if bool((mix < 0.0) | (mix > 1.0)):
        raise ValueError("gamma must be in [0, 1]")
    if bool(propagation_gain <= 0.0):
        raise ValueError("gain must be positive")

    diagonal = torch.eye(
        coords.shape[0], device=coords.device, dtype=torch.bool
    )
    local = torch.exp(-distance_sq / (2.0 * scale.square())).masked_fill(
        diagonal, 0.0
    )
    parallel = torch.abs(torch.einsum("ijd,d->ij", delta, axis_unit))
    perpendicular_sq = torch.clamp(distance_sq - parallel.square(), min=0.0)
    axial = torch.exp(
        -parallel.square() / (2.0 * (ratio * scale).square())
        - perpendicular_sq / (2.0 * scale.square())
    ).masked_fill(diagonal, 0.0)

    local_bar = _frobenius_normalize(local, eps)
    axial_bar = _frobenius_normalize(axial, eps)
    adjacency = (1.0 - mix) * local_bar + mix * axial_bar
    degree = torch.sum(adjacency, dim=1)
    inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=eps))
    operator = (
        propagation_gain
        * inv_sqrt_degree[:, None]
        * adjacency
        * inv_sqrt_degree[None, :]
    )
    return {
        "W": operator,
        "A": adjacency,
        "K_local": local_bar,
        "K_axis": axial_bar,
        "axis_unit": axis_unit,
        "local_scale": scale,
    }


def rank_sets_from_group_ids(group_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    """Convert one masked event row to its ordered non-empty rank sets."""
    groups = np.asarray(group_ids, dtype=np.int64)
    if groups.ndim != 1:
        raise ValueError("group_ids must be one-dimensional")
    present = groups >= 0
    if not np.any(present):
        raise ValueError("event contains no participating contact")
    unique = np.unique(groups[present])
    if not np.array_equal(unique, np.arange(unique.size)):
        raise ValueError("participating group ids must be contiguous from zero")
    return tuple(groups == group for group in unique)


def estimate_node_hazard_bias(
    events: Iterable[np.ndarray], *, pseudocount: float = 1.0
) -> dict[str, np.ndarray]:
    """Estimate train-only discrete decision hazard, including terminal points."""
    event_rows = [np.asarray(event, dtype=np.int64) for event in events]
    if not event_rows:
        raise ValueError("at least one training event is required")
    n_contacts = int(event_rows[0].size)
    if any(row.shape != (n_contacts,) for row in event_rows):
        raise ValueError("all events must share one contact dimension")
    n_next = np.zeros(n_contacts, dtype=np.float64)
    n_eligible = np.zeros(n_contacts, dtype=np.float64)
    event_participation = np.zeros(n_contacts, dtype=np.float64)
    for row in event_rows:
        sets = rank_sets_from_group_ids(row)
        event_participation += row >= 0
        seen = np.zeros(n_contacts, dtype=bool)
        for step, current in enumerate(sets):
            seen |= current
            eligible = ~seen
            n_eligible += eligible
            if step + 1 < len(sets):
                n_next += sets[step + 1]
            # The final iteration is the real terminal decision and therefore
            # contributes to n_eligible but not n_next.
    probability = (n_next + pseudocount) / (
        n_eligible + 2.0 * pseudocount
    )
    probability = np.clip(probability, EPS, 1.0 - EPS)
    bias = np.log(probability) - np.log1p(-probability)
    return {
        "n_next": n_next,
        "n_eligible": n_eligible,
        "hazard_probability": probability,
        "bias": bias,
        "event_participation_probability": event_participation / len(event_rows),
    }


def node_bias_fingerprint(bias: np.ndarray) -> str:
    """Stable fingerprint reused by full and every control."""
    import hashlib

    array = np.ascontiguousarray(np.asarray(bias, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _log_nonempty_normalizer(log_one_minus_hazard: Tensor) -> Tensor:
    """log(1 - product(1-h)) from per-node log(1-h), stably."""
    log_empty = torch.sum(log_one_minus_hazard)
    # For finite sigmoid logits log_empty is negative.  Clamp only protects
    # pathological underflow without changing ordinary gradients.
    log_empty = torch.clamp(log_empty, max=-torch.finfo(log_empty.dtype).eps)
    return torch.log(-torch.expm1(log_empty))


def exact_one_step_log_probability(
    *,
    node_logits: Tensor,
    eligible: Tensor,
    next_set: Tensor,
    stop_logit: Tensor,
    terminal: bool,
) -> Tensor:
    """Exact v2.2 log probability for a non-empty next set or scalar STOP."""
    eligible = eligible.to(dtype=torch.bool)
    next_set = next_set.to(dtype=torch.bool)
    if node_logits.ndim != 1 or eligible.shape != node_logits.shape:
        raise ValueError("node_logits and eligible must be aligned vectors")
    if next_set.shape != node_logits.shape:
        raise ValueError("next_set must match node_logits")
    if torch.any(next_set & ~eligible):
        raise ValueError("next_set contains an ineligible contact")
    if terminal:
        if torch.any(next_set):
            raise ValueError("terminal target must be empty")
        return F.logsigmoid(stop_logit)
    if not torch.any(next_set):
        raise ValueError("non-terminal next_set must be non-empty")
    eligible_logits = node_logits[eligible]
    eligible_target = next_set[eligible].to(node_logits.dtype)
    log_hazard = F.logsigmoid(eligible_logits)
    log_one_minus = F.logsigmoid(-eligible_logits)
    bernoulli = torch.sum(
        eligible_target * log_hazard
        + (1.0 - eligible_target) * log_one_minus
    )
    log_z = _log_nonempty_normalizer(log_one_minus)
    return F.logsigmoid(-stop_logit) + bernoulli - log_z


@dataclass
class RolloutResult:
    first_arrival_mass: Tensor
    stop_before_arrival_mass: Tensor
    stop_mass: Tensor
    event_survival: Tensor
    not_arrived_survival: Tensor
    stop_probability: Tensor
    conditional_activation: Tensor
    state: Tensor

    @property
    def participation_probability(self) -> Tensor:
        return torch.sum(self.first_arrival_mass, dim=0)


def absorbing_mean_field_rollout(
    *,
    initial_state: Tensor,
    operator: Tensor,
    node_bias: Tensor,
    eligible: Tensor,
    rho_p: Tensor | float,
    c0: Tensor | float,
    c_p: Tensor | float,
    c_n: Tensor | float,
    seen_count: Tensor | float,
    horizon: int,
    eps: float = EPS,
) -> RolloutResult:
    """Absorbing mean-field first-arrival rollout from one observed prefix."""
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    n_contacts = int(initial_state.numel())
    if initial_state.shape != (n_contacts,) or operator.shape != (
        n_contacts,
        n_contacts,
    ):
        raise ValueError("state/operator shape mismatch")
    if node_bias.shape != (n_contacts,) or eligible.shape != (n_contacts,):
        raise ValueError("bias/eligible shape mismatch")
    eligible_bool = eligible.to(dtype=torch.bool)
    device, dtype = initial_state.device, initial_state.dtype
    rho = torch.as_tensor(rho_p, device=device, dtype=dtype)
    stop_c0 = torch.as_tensor(c0, device=device, dtype=dtype)
    stop_cp = torch.as_tensor(c_p, device=device, dtype=dtype)
    stop_cn = torch.as_tensor(c_n, device=device, dtype=dtype)
    base_seen = torch.as_tensor(seen_count, device=device, dtype=dtype)

    state = initial_state
    survival = torch.ones((), device=device, dtype=dtype)
    not_arrived = eligible_bool.to(dtype)
    q_steps: list[Tensor] = []
    d_steps: list[Tensor] = []
    stop_steps: list[Tensor] = []
    survival_steps: list[Tensor] = [survival]
    not_arrived_steps: list[Tensor] = [not_arrived]
    p_stop_steps: list[Tensor] = []
    activation_steps: list[Tensor] = []
    state_steps: list[Tensor] = [state]

    for _ in range(horizon):
        remaining_weight = torch.sum(not_arrived)
        weighted_mean = torch.sum(not_arrived * state) / torch.clamp(
            remaining_weight, min=eps
        )
        expected_seen = (
            base_seen + torch.sum(eligible_bool.to(dtype) - not_arrived)
        ) / float(n_contacts)
        stop_logit = stop_c0 + stop_cp * weighted_mean + stop_cn * expected_seen
        p_stop = torch.sigmoid(stop_logit)

        logits = node_bias + state
        hazard = torch.sigmoid(logits) * eligible_bool.to(dtype)
        log_empty = torch.sum(
            torch.where(
                eligible_bool,
                torch.log1p(-torch.clamp(hazard, max=1.0 - eps)),
                torch.zeros_like(hazard),
            )
        )
        z = -torch.expm1(log_empty)
        forced_stop = bool((remaining_weight <= eps) | (z <= eps))
        if forced_stop:
            p_stop = torch.ones_like(p_stop)
            conditional_hazard = torch.zeros_like(hazard)
        else:
            conditional_hazard = hazard / z

        activation = not_arrived * conditional_hazard
        q = survival * (1.0 - p_stop) * activation
        d = survival * p_stop * not_arrived
        stop_mass = survival * p_stop
        survival = survival * (1.0 - p_stop)
        not_arrived = not_arrived * (1.0 - conditional_hazard)
        state = rho * state + operator @ activation

        q_steps.append(q)
        d_steps.append(d)
        stop_steps.append(stop_mass)
        survival_steps.append(survival)
        not_arrived_steps.append(not_arrived)
        p_stop_steps.append(p_stop)
        activation_steps.append(activation)
        state_steps.append(state)

    empty_nodes = initial_state.new_empty((0, n_contacts))
    empty_scalar = initial_state.new_empty((0,))
    return RolloutResult(
        first_arrival_mass=torch.stack(q_steps) if q_steps else empty_nodes,
        stop_before_arrival_mass=torch.stack(d_steps) if d_steps else empty_nodes,
        stop_mass=torch.stack(stop_steps) if stop_steps else empty_scalar,
        event_survival=torch.stack(survival_steps),
        not_arrived_survival=torch.stack(not_arrived_steps),
        stop_probability=(
            torch.stack(p_stop_steps) if p_stop_steps else empty_scalar
        ),
        conditional_activation=(
            torch.stack(activation_steps) if activation_steps else empty_nodes
        ),
        state=torch.stack(state_steps),
    )


class SymmetricAxisPropagationStateRNN(nn.Module):
    """Minimal trainable patient-specific v2.2 model.

    Shared parameters may be supplied directly or optimized jointly across
    patients by an outer trainer.  The class itself owns exactly three
    patient-specific free objects: axis, gamma, and gain.
    """

    def __init__(
        self,
        *,
        coords: np.ndarray,
        node_bias: np.ndarray,
        shared_raw_anisotropy: nn.Parameter | Tensor | None = None,
        shared_raw_rho: nn.Parameter | Tensor | None = None,
        shared_c0: nn.Parameter | Tensor | None = None,
        shared_raw_c_p: nn.Parameter | Tensor | None = None,
        shared_raw_c_n: nn.Parameter | Tensor | None = None,
        isotropic: bool = False,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        centered = _validate_geometry(coords)
        bias = np.asarray(node_bias, dtype=np.float64)
        if bias.shape != (centered.shape[0],) or not np.all(np.isfinite(bias)):
            raise ValueError("node_bias must be finite and match contacts")
        self.register_buffer("coords", torch.as_tensor(centered, dtype=torch.float32))
        self.register_buffer("node_bias", torch.as_tensor(bias, dtype=torch.float32))
        self.register_buffer(
            "local_scale",
            torch.tensor(fixed_local_scale(centered), dtype=torch.float32),
        )
        principal = np.linalg.svd(centered, full_matrices=False)[2][0]
        self.axis_raw = nn.Parameter(torch.as_tensor(principal, dtype=torch.float32))
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        self.gain_raw = nn.Parameter(torch.tensor(0.0))
        self.isotropic = bool(isotropic)
        if self.isotropic:
            self.axis_raw.requires_grad_(False)
            self.gamma_raw.requires_grad_(False)

        self._register_shared("raw_anisotropy", shared_raw_anisotropy, 0.0)
        self._register_shared("raw_rho", shared_raw_rho, 0.0)
        self._register_shared("c0", shared_c0, -1.0)
        self._register_shared("raw_c_p", shared_raw_c_p, 0.0)
        self._register_shared("raw_c_n", shared_raw_c_n, 0.0)
        self.eps = float(eps)

    def _register_shared(
        self,
        name: str,
        supplied: nn.Parameter | Tensor | None,
        initial: float,
    ) -> None:
        if supplied is None:
            self.register_parameter(name, nn.Parameter(torch.tensor(initial)))
        elif isinstance(supplied, nn.Parameter):
            self.register_parameter(name, supplied)
        else:
            self.register_buffer(name, supplied.detach().clone())

    @property
    def axis(self) -> Tensor:
        return self.axis_raw / torch.clamp(
            torch.linalg.vector_norm(self.axis_raw), min=self.eps
        )

    @property
    def gamma(self) -> Tensor:
        if self.isotropic:
            return torch.zeros_like(self.gamma_raw)
        return torch.sigmoid(self.gamma_raw)

    @property
    def gain(self) -> Tensor:
        return F.softplus(self.gain_raw) + self.eps

    @property
    def anisotropy_ratio(self) -> Tensor:
        return 1.0 + 3.0 * torch.sigmoid(self.raw_anisotropy)

    @property
    def rho_p(self) -> Tensor:
        return 0.999 * torch.sigmoid(self.raw_rho)

    @property
    def c_p(self) -> Tensor:
        return -F.softplus(self.raw_c_p)

    @property
    def c_n(self) -> Tensor:
        return F.softplus(self.raw_c_n)

    def operator_components(self) -> dict[str, Tensor]:
        return symmetric_axis_operator(
            self.coords,
            self.axis,
            anisotropy_ratio=self.anisotropy_ratio,
            gamma=self.gamma,
            gain=self.gain,
            local_scale=self.local_scale,
            eps=self.eps,
        )

    def reset_state(self) -> Tensor:
        return torch.zeros_like(self.node_bias)

    def observe(self, state: Tensor, current_set: Tensor) -> Tensor:
        current = current_set.to(device=state.device, dtype=state.dtype)
        if current.shape != state.shape:
            raise ValueError("current_set must match state")
        return self.rho_p * state + self.operator_components()["W"] @ current

    def decision(
        self, state_after_observation: Tensor, seen: Tensor
    ) -> dict[str, Tensor]:
        seen_bool = seen.to(device=state_after_observation.device, dtype=torch.bool)
        eligible = ~seen_bool
        logits = self.node_bias + state_after_observation
        masked_logits = logits.masked_fill(~eligible, -torch.inf)
        if torch.any(eligible):
            mean_drive = torch.mean(state_after_observation[eligible])
            seen_fraction = seen_bool.to(state_after_observation.dtype).mean()
            stop_logit = self.c0 + self.c_p * mean_drive + self.c_n * seen_fraction
        else:
            stop_logit = torch.full_like(self.c0, torch.inf)
        return {
            "node_logits": masked_logits,
            "eligible": eligible,
            "stop_logit": stop_logit,
        }

    def event_log_probabilities(
        self, rank_sets: Sequence[Tensor]
    ) -> list[Tensor]:
        if not rank_sets:
            raise ValueError("event must contain at least one observed rank set")
        state = self.reset_state()
        seen = torch.zeros_like(self.node_bias, dtype=torch.bool)
        log_probabilities: list[Tensor] = []
        for index, current_set in enumerate(rank_sets):
            current = current_set.to(device=state.device, dtype=torch.bool)
            state = self.observe(state, current)
            seen = seen | current
            decision = self.decision(state, seen)
            terminal = index + 1 == len(rank_sets)
            target = (
                torch.zeros_like(seen)
                if terminal
                else rank_sets[index + 1].to(device=state.device, dtype=torch.bool)
            )
            log_probabilities.append(
                exact_one_step_log_probability(
                    node_logits=decision["node_logits"],
                    eligible=decision["eligible"],
                    next_set=target,
                    stop_logit=decision["stop_logit"],
                    terminal=terminal,
                )
            )
        return log_probabilities

    def rollout_from_prefix(
        self,
        *,
        state_after_prefix: Tensor,
        seen: Tensor,
        horizon: int,
    ) -> RolloutResult:
        eligible = ~seen.to(device=state_after_prefix.device, dtype=torch.bool)
        return absorbing_mean_field_rollout(
            initial_state=state_after_prefix,
            operator=self.operator_components()["W"],
            node_bias=self.node_bias,
            eligible=eligible,
            rho_p=self.rho_p,
            c0=self.c0,
            c_p=self.c_p,
            c_n=self.c_n,
            seen_count=seen.to(state_after_prefix.dtype).sum(),
            horizon=horizon,
            eps=self.eps,
        )


def normalized_event_nll(log_probabilities: Sequence[Tensor], eligible: Sequence[int]) -> Tensor:
    """Within-event mean of eligible-contact-normalized decision NLL."""
    if not log_probabilities or len(log_probabilities) != len(eligible):
        raise ValueError("decision log probabilities and eligible counts must align")
    terms = [
        -log_probability / max(1, int(count))
        for log_probability, count in zip(log_probabilities, eligible)
    ]
    return torch.stack(terms).mean()


def event_first_patient_mean(event_values: Sequence[Sequence[float]]) -> float:
    """Prefix/decision mean within event, then event mean within patient."""
    event_means = [
        float(np.mean(np.asarray(values, dtype=np.float64)))
        for values in event_values
        if len(values) > 0
    ]
    if not event_means:
        return float("nan")
    return float(np.mean(event_means))


def seed_median_patient_metric(seed_event_values: Sequence[Sequence[Sequence[float]]]) -> float:
    """Event-first patient metric in each seed followed by the seed median."""
    per_seed = [
        event_first_patient_mean(event_values)
        for event_values in seed_event_values
    ]
    finite = np.asarray(per_seed, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float("nan")
