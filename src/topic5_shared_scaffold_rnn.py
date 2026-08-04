"""Patient-specific shared-scaffold RNN for within-event rank propagation.

This module is intentionally separate from the earlier Topic 5 model
families.  It implements one narrow comparison contract:

* a fixed, train-only participation bias is shared by both model families;
* the structured model can mix contacts only through one symmetric scaffold;
* one learned signed contact coordinate creates a shared rank-two symmetric
  scaffold and its source-conditioned rank-two skew flow;
* two contact-wise traces separate recent propagation drive from a slower
  refractory/restraint trace;
* STOP, next-set cardinality, and contact identity are separately normalized;
* given cardinality ``k``, the next rank set is scored with the exact
  conditional distribution over all eligible ``k``-subsets.

The recurrent index is a within-event rank step, not physical time.  No ictal
target or empirical A/B template enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.propagation_skeleton_geometry import parse_shaft
from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    estimate_node_hazard_bias,
    rank_sets_from_group_ids,
)


EPS = 1.0e-8


def _validate_square_adjacency(adjacency: Tensor, *, eps: float = EPS) -> Tensor:
    matrix = torch.as_tensor(adjacency)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    if matrix.shape[0] < 2 or not torch.isfinite(matrix).all():
        raise ValueError("adjacency must contain at least two finite contacts")
    if torch.any(matrix < 0):
        raise ValueError("adjacency must be non-negative")
    if not torch.allclose(matrix, matrix.T, atol=eps, rtol=0.0):
        raise ValueError("adjacency must be symmetric")
    if not bool(torch.any(matrix > eps)):
        raise ValueError("adjacency must contain at least one edge")
    return matrix


def build_fixed_local_shaft_adjacency(
    *,
    coords: np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    distance_scale: float | None = None,
    distance_cutoff_in_scales: float = 2.5,
    shaft_neighbor_weight: float = 1.0,
    eps: float = EPS,
) -> np.ndarray:
    """Construct a fixed symmetric local/shaft graph without neural targets.

    Geometry contributes a truncated Gaussian local graph.  Consecutive
    contacts on each named shaft receive an additional fixed edge.  At least
    one of ``coords`` or ``channel_names`` must be supplied.
    """

    if coords is None and channel_names is None:
        raise ValueError("coords or channel_names is required")
    if float(distance_cutoff_in_scales) <= 0:
        raise ValueError("distance_cutoff_in_scales must be positive")
    if float(shaft_neighbor_weight) < 0:
        raise ValueError("shaft_neighbor_weight must be non-negative")

    coordinate_array = None if coords is None else np.asarray(coords, dtype=float)
    if coordinate_array is not None:
        if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 3:
            raise ValueError("coords must have shape [contact, 3]")
        n_contacts = int(coordinate_array.shape[0])
    else:
        n_contacts = len(channel_names or ())
    if channel_names is not None and len(channel_names) != n_contacts:
        raise ValueError("channel_names must align with coords")
    if n_contacts < 2:
        raise ValueError("at least two contacts are required")

    adjacency = np.zeros((n_contacts, n_contacts), dtype=np.float64)
    if coordinate_array is not None:
        mapped = np.isfinite(coordinate_array).all(axis=1)
        mapped_index = np.flatnonzero(mapped)
        if mapped_index.size >= 2:
            mapped_coords = coordinate_array[mapped]
            distance = np.linalg.norm(
                mapped_coords[:, None, :] - mapped_coords[None, :, :], axis=-1
            )
            nonzero = distance.copy()
            nonzero[nonzero <= eps] = np.inf
            nearest = np.min(nonzero, axis=1)
            nearest = nearest[np.isfinite(nearest)]
            if distance_scale is None:
                if not nearest.size:
                    raise ValueError("mapped geometry has no non-zero distance")
                scale = float(np.median(nearest))
            else:
                scale = float(distance_scale)
            if not np.isfinite(scale) or scale <= eps:
                raise ValueError("distance_scale must be finite and positive")
            local = np.exp(-0.5 * np.square(distance / scale))
            local[distance > float(distance_cutoff_in_scales) * scale] = 0.0
            np.fill_diagonal(local, 0.0)
            adjacency[np.ix_(mapped_index, mapped_index)] += local
        elif distance_scale is not None and float(distance_scale) <= eps:
            raise ValueError("distance_scale must be positive")

    if channel_names is not None and shaft_neighbor_weight > 0:
        by_shaft: dict[str, list[tuple[int, int]]] = {}
        for index, name in enumerate(channel_names):
            shaft, ordinal = parse_shaft(str(name))
            if shaft is not None and ordinal is not None:
                by_shaft.setdefault(str(shaft), []).append((int(ordinal), index))
        for members in by_shaft.values():
            ordered = sorted(members)
            for (_, left), (_, right) in zip(ordered[:-1], ordered[1:]):
                adjacency[left, right] += float(shaft_neighbor_weight)
                adjacency[right, left] += float(shaft_neighbor_weight)

    np.fill_diagonal(adjacency, 0.0)
    _validate_square_adjacency(torch.as_tensor(adjacency), eps=eps)
    return adjacency


def _frobenius_normalize(matrix: Tensor, eps: float) -> Tensor:
    return matrix / torch.clamp(torch.linalg.vector_norm(matrix), min=eps)


def source_conditioned_shared_scaffold(
    fixed_adjacency: Tensor,
    axis_coordinate: Tensor,
    *,
    endpoint_temperature: Tensor | float,
    gamma: Tensor | float,
    gain: Tensor | float,
    eps: float = EPS,
) -> dict[str, Tensor]:
    """Construct shared symmetric and skew operators from one signed axis.

    The same endpoint memberships form both analytic rank-at-most-two terms.
    Both use the degree matrix of the symmetric adjacency, so ``W`` remains
    symmetric and ``W_skew`` remains exactly antisymmetric.
    """

    fixed = _validate_square_adjacency(fixed_adjacency, eps=eps)
    coordinate = torch.as_tensor(
        axis_coordinate, device=fixed.device, dtype=fixed.dtype
    )
    if coordinate.shape != (fixed.shape[0],) or not torch.isfinite(coordinate).all():
        raise ValueError("axis_coordinate must be a finite contact vector")
    coordinate = coordinate - coordinate.mean()
    coordinate_scale = torch.sqrt(torch.mean(coordinate.square()))
    if bool(coordinate_scale <= eps):
        raise ValueError("axis_coordinate must vary across contacts")
    coordinate = coordinate / coordinate_scale
    temperature = torch.as_tensor(
        endpoint_temperature, device=fixed.device, dtype=fixed.dtype
    )
    if not torch.isfinite(temperature) or bool(temperature <= eps):
        raise ValueError("endpoint_temperature must be finite and positive")
    mix = torch.as_tensor(gamma, device=fixed.device, dtype=fixed.dtype)
    propagation_gain = torch.as_tensor(gain, device=fixed.device, dtype=fixed.dtype)
    if bool((mix < 0) | (mix > 1)):
        raise ValueError("gamma must be in [0, 1]")
    if bool(propagation_gain <= 0):
        raise ValueError("gain must be positive")

    local = _frobenius_normalize(fixed, eps)
    endpoint_minus = torch.sigmoid(-coordinate / temperature)
    endpoint_plus = torch.sigmoid(coordinate / temperature)
    axis_symmetric = (
        endpoint_plus[:, None] * endpoint_minus[None, :]
        + endpoint_minus[:, None] * endpoint_plus[None, :]
    )
    axis_skew = (
        endpoint_plus[:, None] * endpoint_minus[None, :]
        - endpoint_minus[:, None] * endpoint_plus[None, :]
    )
    axis_symmetric = _frobenius_normalize(axis_symmetric, eps)
    axis_skew = _frobenius_normalize(axis_skew, eps)
    adjacency = (1.0 - mix) * local + mix * axis_symmetric
    degree = adjacency.sum(dim=1)
    inverse_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=eps))
    symmetric_operator = (
        propagation_gain
        * inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    skew_operator = (
        propagation_gain
        * inverse_sqrt_degree[:, None]
        * axis_skew
        * inverse_sqrt_degree[None, :]
    )
    return {
        "W": symmetric_operator,
        "W_skew": skew_operator,
        "A": adjacency,
        "K_fixed": local,
        "K_axis_symmetric": axis_symmetric,
        "K_axis_skew": axis_skew,
        "axis_coordinate": coordinate,
        "endpoint_minus": endpoint_minus,
        "endpoint_plus": endpoint_plus,
        "endpoint_temperature": temperature,
    }


def _log_elementary_symmetric(log_weights: Tensor, k: int) -> Tensor:
    """Log of the order-``k`` elementary symmetric polynomial.

    The dynamic program is exact up to floating-point arithmetic and avoids
    enumerating ``n choose k`` subsets during training.
    """

    if log_weights.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    n = int(log_weights.numel())
    k = int(k)
    if k < 0 or k > n:
        raise ValueError("k must lie in [0, n]")
    # A finite log-zero avoids the undefined (-inf, -inf) gradient of
    # logaddexp in unreachable DP cells.  It is far below any representable
    # probability and therefore does not change reachable normalizers.
    log_zero = log_weights.new_tensor(-torch.finfo(log_weights.dtype).max / 4.0)
    state = [log_weights.new_zeros(())] + [log_zero for _ in range(k)]
    for item_index, value in enumerate(log_weights):
        upper = min(k, item_index + 1)
        updated = list(state)
        for order in range(1, upper + 1):
            updated[order] = torch.logaddexp(
                state[order], state[order - 1] + value
            )
        state = updated
    return state[k]


def exact_conditional_k_subset_log_probability(
    *,
    node_logits: Tensor,
    eligible: Tensor,
    next_set: Tensor,
) -> Tensor:
    """Exact ``log P(next_set | |next_set|=k, eligible, logits)``.

    For a candidate subset ``S`` of fixed size ``k``, probability is
    proportional to ``exp(sum(node_logits[S]))``.  This produces explicit
    competition between eligible contacts and supports tied rank sets.
    """

    if node_logits.ndim != 1:
        raise ValueError("node_logits must be one-dimensional")
    eligible_bool = eligible.to(device=node_logits.device, dtype=torch.bool)
    target_bool = next_set.to(device=node_logits.device, dtype=torch.bool)
    if eligible_bool.shape != node_logits.shape or target_bool.shape != node_logits.shape:
        raise ValueError("eligible and next_set must align with node_logits")
    if torch.any(target_bool & ~eligible_bool):
        raise ValueError("next_set contains an ineligible contact")
    k = int(target_bool.sum().item())
    n_eligible = int(eligible_bool.sum().item())
    if k < 1 or k > n_eligible:
        raise ValueError("next_set must be a non-empty eligible subset")
    eligible_logits = node_logits[eligible_bool]
    if not torch.isfinite(eligible_logits).all():
        raise ValueError("eligible node logits must be finite")
    numerator = node_logits[target_bool].sum()
    denominator = _log_elementary_symmetric(eligible_logits, k)
    return numerator - denominator


def batched_exact_conditional_k_subset_log_probability(
    *,
    node_logits: Tensor,
    eligible: Tensor,
    next_set: Tensor,
    active: Tensor | None = None,
) -> Tensor:
    """Batch-vectorized exact conditional subset log probabilities.

    The only Python loop is over contacts in the elementary-symmetric
    dynamic program; there is no loop over events in the batch.  Inactive
    rows (normally terminal decisions) return exactly zero.
    """

    if node_logits.ndim != 2:
        raise ValueError("node_logits must have shape [batch, contact]")
    eligible_bool = eligible.to(device=node_logits.device, dtype=torch.bool)
    target_bool = next_set.to(device=node_logits.device, dtype=torch.bool)
    if eligible_bool.shape != node_logits.shape or target_bool.shape != node_logits.shape:
        raise ValueError("eligible and next_set must align with node_logits")
    batch, n_contacts = node_logits.shape
    if active is None:
        active_bool = torch.ones(batch, dtype=torch.bool, device=node_logits.device)
    else:
        active_bool = active.to(device=node_logits.device, dtype=torch.bool)
        if active_bool.shape != (batch,):
            raise ValueError("active must have shape [batch]")
    if torch.any(target_bool & ~eligible_bool & active_bool[:, None]):
        raise ValueError("next_set contains an ineligible contact")
    cardinality = target_bool.sum(dim=1)
    n_eligible = eligible_bool.sum(dim=1)
    invalid = active_bool & ((cardinality < 1) | (cardinality > n_eligible))
    if torch.any(invalid):
        raise ValueError("active next_set must be a non-empty eligible subset")
    finite_eligible = torch.isfinite(node_logits) | ~eligible_bool | ~active_bool[:, None]
    if not torch.all(finite_eligible):
        raise ValueError("eligible node logits must be finite")

    # dp[:, q] is log e_q after the contacts processed so far.  Ineligible
    # nodes have a finite numerical log-zero: they make no measurable
    # contribution while preserving well-defined gradients.
    log_zero_value = -torch.finfo(node_logits.dtype).max / 4.0
    log_zero = node_logits.new_full((batch, n_contacts), log_zero_value)
    dp = torch.cat([node_logits.new_zeros((batch, 1)), log_zero], dim=1)
    for contact_index in range(n_contacts):
        weight = torch.where(
            eligible_bool[:, contact_index],
            node_logits[:, contact_index],
            node_logits.new_full((batch,), log_zero_value),
        )
        positive_orders = torch.logaddexp(
            dp[:, 1:], dp[:, :-1] + weight[:, None]
        )
        dp = torch.cat([dp[:, :1], positive_orders], dim=1)
    safe_cardinality = torch.where(
        active_bool, cardinality, torch.zeros_like(cardinality)
    )
    denominator = dp.gather(1, safe_cardinality[:, None]).squeeze(1)
    numerator = torch.where(target_bool, node_logits, 0.0).sum(dim=1)
    value = numerator - denominator
    return torch.where(active_bool, value, torch.zeros_like(value))


def cardinality_log_probability(cardinality_logits: Tensor, cardinality: int) -> Tensor:
    """Score ``k`` where index zero of the logits represents cardinality one."""

    if cardinality_logits.ndim != 1:
        raise ValueError("cardinality_logits must be one-dimensional")
    k = int(cardinality)
    if k < 1 or k > cardinality_logits.numel():
        raise ValueError("cardinality is outside the head support")
    if not torch.isfinite(cardinality_logits[k - 1]):
        raise ValueError("requested cardinality is ineligible")
    return F.log_softmax(cardinality_logits, dim=0)[k - 1]


def batched_cardinality_log_probability(
    cardinality_logits: Tensor,
    cardinality: Tensor,
    *,
    active: Tensor | None = None,
) -> Tensor:
    """Batch-vectorized cardinality scoring with terminal rows set to zero."""

    if cardinality_logits.ndim != 2:
        raise ValueError("cardinality_logits must have shape [batch, max_k]")
    batch, max_k = cardinality_logits.shape
    target = cardinality.to(device=cardinality_logits.device, dtype=torch.long)
    if target.shape != (batch,):
        raise ValueError("cardinality must have shape [batch]")
    if active is None:
        active_bool = torch.ones(batch, dtype=torch.bool, device=target.device)
    else:
        active_bool = active.to(device=target.device, dtype=torch.bool)
        if active_bool.shape != (batch,):
            raise ValueError("active must have shape [batch]")
    if torch.any(active_bool & ((target < 1) | (target > max_k))):
        raise ValueError("active cardinality is outside the head support")
    safe_target = torch.where(active_bool, target, torch.ones_like(target))
    selected = cardinality_logits.gather(1, (safe_target - 1)[:, None]).squeeze(1)
    if torch.any(active_bool & ~torch.isfinite(selected)):
        raise ValueError("requested cardinality is ineligible")
    # A terminal row can have no feasible cardinalities (all -inf).  Replace
    # it before log_softmax so inactive rows cannot create NaNs.
    safe_logits = torch.where(
        active_bool[:, None], cardinality_logits, torch.zeros_like(cardinality_logits)
    )
    value = F.log_softmax(safe_logits, dim=1).gather(
        1, (safe_target - 1)[:, None]
    ).squeeze(1)
    return torch.where(active_bool, value, torch.zeros_like(value))


@dataclass(frozen=True)
class StepLogProbabilities:
    """Separately reportable components of one rank-step decision."""

    total: Tensor
    stop: Tensor
    cardinality: Tensor
    conditional_contacts: Tensor
    terminal: bool
    cardinality_target: int

    def as_dict(self) -> dict[str, Tensor | bool | int]:
        return {
            "total": self.total,
            "stop": self.stop,
            "cardinality": self.cardinality,
            "conditional_contacts": self.conditional_contacts,
            "terminal": self.terminal,
            "cardinality_target": self.cardinality_target,
        }


def decomposed_one_step_log_probability(
    decision: Mapping[str, Tensor],
    *,
    next_set: Tensor,
    terminal: bool,
) -> StepLogProbabilities:
    """Score STOP, set size, and conditional contact identity separately."""

    required = {"node_logits", "eligible", "stop_logit", "cardinality_logits"}
    missing = required.difference(decision)
    if missing:
        raise KeyError(f"decision is missing {sorted(missing)}")
    node_logits = decision["node_logits"]
    target = next_set.to(device=node_logits.device, dtype=torch.bool)
    stop_logit = decision["stop_logit"].reshape(())
    zero = node_logits[torch.isfinite(node_logits)].sum() * 0.0
    if terminal:
        if torch.any(target):
            raise ValueError("terminal target must be empty")
        stop = F.logsigmoid(stop_logit)
        return StepLogProbabilities(stop, stop, zero, zero, True, 0)

    k = int(target.sum().item())
    if k < 1:
        raise ValueError("non-terminal target must be non-empty")
    stop = F.logsigmoid(-stop_logit)
    cardinality = cardinality_log_probability(decision["cardinality_logits"], k)
    contacts = exact_conditional_k_subset_log_probability(
        node_logits=node_logits,
        eligible=decision["eligible"],
        next_set=target,
    )
    return StepLogProbabilities(
        stop + cardinality + contacts,
        stop,
        cardinality,
        contacts,
        False,
        k,
    )


@dataclass(frozen=True)
class PropagationRestraintState:
    """Rank-step state of the structured model.

    ``direction`` is the frozen source-conditioned sign ``d_e``.  It is
    written once, from the first observed rank set of the event, and is
    never recomputed afterwards; ``source_initialized`` records that the
    write has happened so later rank steps cannot silently overwrite it.
    """

    propagation: Tensor
    restraint: Tensor
    direction: Tensor
    source_initialized: Tensor


class _RankSetModelInterface(nn.Module):
    """Shared event scorer; recurrent state remains model-specific and opaque."""

    n_contacts: int

    def reset_state(self):  # pragma: no cover - abstract interface
        raise NotImplementedError

    def observe(self, state, current_set: Tensor):  # pragma: no cover
        raise NotImplementedError

    def decision(self, state, seen: Tensor) -> dict[str, Tensor]:  # pragma: no cover
        raise NotImplementedError

    def event_log_probabilities(
        self, rank_sets: Sequence[Tensor]
    ) -> list[StepLogProbabilities]:
        if not rank_sets:
            raise ValueError("event must contain at least one observed rank set")
        state = self.reset_state()
        seen = torch.zeros(
            self.n_contacts,
            dtype=torch.bool,
            device=self.participation_bias.device,
        )
        scores: list[StepLogProbabilities] = []
        for index, current_set in enumerate(rank_sets):
            current = torch.as_tensor(
                current_set,
                dtype=torch.bool,
                device=self.participation_bias.device,
            )
            if current.shape != seen.shape or not torch.any(current):
                raise ValueError("every current rank set must be non-empty and contact-aligned")
            if torch.any(current & seen):
                raise ValueError("a contact cannot be recruited twice in one event")
            state = self.observe(state, current)
            seen = seen | current
            terminal = index + 1 == len(rank_sets)
            target = (
                torch.zeros_like(seen)
                if terminal
                else torch.as_tensor(
                    rank_sets[index + 1],
                    dtype=torch.bool,
                    device=seen.device,
                )
            )
            scores.append(
                decomposed_one_step_log_probability(
                    self.decision(state, seen),
                    next_set=target,
                    terminal=terminal,
                )
            )
        return scores

    def event_log_likelihood(self, rank_sets: Sequence[Tensor]) -> dict[str, Tensor]:
        scores = self.event_log_probabilities(rank_sets)
        fields = ("total", "stop", "cardinality", "conditional_contacts")
        return {
            field: torch.stack([getattr(score, field) for score in scores]).sum()
            for field in fields
        }

    def event_nll(self, rank_sets: Sequence[Tensor]) -> dict[str, Tensor]:
        likelihood = self.event_log_likelihood(rank_sets)
        return {name: -value for name, value in likelihood.items()}

    def score_group_ids(self, group_ids: np.ndarray) -> dict[str, Tensor]:
        rank_sets = [torch.as_tensor(item) for item in rank_sets_from_group_ids(group_ids)]
        return self.event_log_likelihood(rank_sets)

    def batched_event_log_likelihood(
        self, group_ids: Tensor, group_count: Tensor
    ) -> dict[str, Tensor]:
        """Vectorized likelihood for padded ``[batch, contact]`` events.

        The method loops over the at-most-``N`` observed rank steps, never
        over batch rows.  Returned likelihood components are per event; the
        count fields make event-first or decision-first aggregation explicit.
        """

        groups = torch.as_tensor(
            group_ids, device=self.participation_bias.device, dtype=torch.long
        )
        counts = torch.as_tensor(
            group_count, device=self.participation_bias.device, dtype=torch.long
        )
        if groups.ndim != 2 or groups.shape[1] != self.n_contacts:
            raise ValueError("group_ids must have shape [batch, contact]")
        batch = int(groups.shape[0])
        if counts.shape != (batch,):
            raise ValueError("group_count must have shape [batch]")
        if batch < 1 or torch.any((counts < 1) | (counts > self.n_contacts)):
            raise ValueError("every event needs 1..N contiguous rank groups")
        if torch.any((groups < -1) | (groups >= counts[:, None])):
            raise ValueError("group_ids must be -1 or below that event's group_count")
        group_axis = torch.arange(
            self.n_contacts, device=groups.device, dtype=groups.dtype
        )
        present = (groups[:, :, None] == group_axis[None, None, :]).any(dim=1)
        expected = group_axis[None, :] < counts[:, None]
        if not torch.equal(present, expected):
            raise ValueError("participating group ids must be contiguous from zero")

        state = self.reset_state(batch_size=batch)
        seen = torch.zeros(
            (batch, self.n_contacts), dtype=torch.bool, device=groups.device
        )
        components = {
            name: self.participation_bias.new_zeros(batch)
            for name in ("total", "stop", "cardinality", "conditional_contacts")
        }
        max_steps = int(counts.max().item())
        for step in range(max_steps):
            active = counts > step
            current = groups == step
            state = self.observe(state, current, active=active)
            seen = seen | current
            terminal = counts == step + 1
            nonterminal = counts > step + 1
            target = groups == step + 1
            decision = self.decision(state, seen)

            stop_value = torch.where(
                terminal,
                F.logsigmoid(decision["stop_logit"]),
                F.logsigmoid(-decision["stop_logit"]),
            )
            stop_value = torch.where(active, stop_value, torch.zeros_like(stop_value))
            cardinality_value = batched_cardinality_log_probability(
                decision["cardinality_logits"],
                target.sum(dim=1),
                active=nonterminal,
            )
            contact_value = batched_exact_conditional_k_subset_log_probability(
                node_logits=decision["node_logits"],
                eligible=decision["eligible"],
                next_set=target,
                active=nonterminal,
            )
            components["stop"] = components["stop"] + stop_value
            components["cardinality"] = (
                components["cardinality"] + cardinality_value
            )
            components["conditional_contacts"] = (
                components["conditional_contacts"] + contact_value
            )
            components["total"] = (
                components["total"]
                + stop_value
                + cardinality_value
                + contact_value
            )
        components["decision_count"] = counts
        components["nonterminal_decision_count"] = torch.clamp(counts - 1, min=0)
        return components

    def batched_event_nll(
        self,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        reduction: str = "event_first",
    ) -> dict[str, Tensor]:
        """Training loss with explicit event-first or decision-first reduction."""

        values = self.batched_event_log_likelihood(group_ids, group_count)
        decision_count = values["decision_count"].to(self.participation_bias.dtype)
        nonterminal_count = values["nonterminal_decision_count"].to(
            self.participation_bias.dtype
        )
        if reduction == "none":
            return {
                name: -values[name]
                for name in ("total", "stop", "cardinality", "conditional_contacts")
            }
        if reduction == "event_first":
            total = (-values["total"] / decision_count).mean()
            stop = (-values["stop"] / decision_count).mean()
            has_transition = nonterminal_count > 0
            if torch.any(has_transition):
                cardinality = (
                    -values["cardinality"][has_transition]
                    / nonterminal_count[has_transition]
                ).mean()
                contacts = (
                    -values["conditional_contacts"][has_transition]
                    / nonterminal_count[has_transition]
                ).mean()
            else:
                zero = total * 0.0
                cardinality, contacts = zero, zero
        elif reduction == "decision_first":
            total = -values["total"].sum() / decision_count.sum()
            stop = -values["stop"].sum() / decision_count.sum()
            denominator = nonterminal_count.sum()
            if bool(denominator > 0):
                cardinality = -values["cardinality"].sum() / denominator
                contacts = -values["conditional_contacts"].sum() / denominator
            else:
                zero = total * 0.0
                cardinality, contacts = zero, zero
        else:
            raise ValueError("reduction must be none, event_first, or decision_first")
        return {
            "total": total,
            "stop": stop,
            "cardinality": cardinality,
            "conditional_contacts": contacts,
        }


class SharedScaffoldPropagationRNN(_RankSetModelInterface):
    """Structured patient-specific RNN with no dense contact-mixing bypass."""

    def __init__(
        self,
        *,
        fixed_adjacency: np.ndarray | Tensor,
        participation_bias: np.ndarray | Tensor,
        low_rank: int = 2,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        adjacency = torch.as_tensor(fixed_adjacency, dtype=torch.float32)
        adjacency = _validate_square_adjacency(adjacency, eps=eps)
        bias = torch.as_tensor(participation_bias, dtype=torch.float32)
        if bias.shape != (adjacency.shape[0],) or not torch.isfinite(bias).all():
            raise ValueError("participation_bias must be finite and contact-aligned")
        # ``low_rank`` survives only so the frozen config, CLI and checkpoint
        # metadata keep one stable key.  The v0.3 operators are analytically
        # rank two by construction, so no other value is representable.
        if int(low_rank) != 2:
            raise ValueError("low_rank must be 2 for the source-conditioned model")
        self.n_contacts = int(adjacency.shape[0])
        self.low_rank = int(low_rank)
        self.eps = float(eps)
        self.register_buffer("fixed_adjacency", adjacency)
        # The bias is deliberately a buffer: it is estimated once from the
        # patient's training events and must remain identical in all models.
        self.register_buffer("participation_bias", bias)
        # The coordinate is centered and RMS-normalized inside the operator
        # helper, so the endpoint temperature is a fixed unit scale rather
        # than a free parameter.  Learning it would allow the memberships to
        # flatten to one half, which erases the skew flow entirely.
        self.register_buffer("endpoint_temperature", torch.tensor(1.0))

        # One signed contact coordinate per patient.  It is the only learned
        # contact-indexed object, and both operators are derived from it.
        self.axis_coordinate_raw = nn.Parameter(
            torch.empty(self.n_contacts).normal_(mean=0.0, std=1.0)
        )
        self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        self.gain_raw = nn.Parameter(torch.tensor(0.0))
        self.rho_p_raw = nn.Parameter(torch.tensor(-0.5))
        self.rho_gap_raw = nn.Parameter(torch.tensor(0.0))
        # softplus of these raw values is 1.0 and 0.25.  Equal weights made
        # the v0.2 model cancel propagation against restraint at the first
        # rank step, where both traces still carry exactly the same drive.
        self.propagation_weight_raw = nn.Parameter(torch.tensor(0.5413249))
        self.restraint_weight_raw = nn.Parameter(torch.tensor(-1.2586915))
        # softplus of these raw values is 0.5 (skew) and 2.0 (direction).
        self.skew_gain_raw = nn.Parameter(torch.tensor(-0.4327521))
        self.direction_gain_raw = nn.Parameter(torch.tensor(1.8545865))

        # These heads see only permutation-invariant state summaries.  They
        # can control event termination and tie-set size, but cannot choose a
        # contact or bypass W.
        self.stop_head = nn.Linear(3, 1)
        self.cardinality_head = nn.Linear(3, self.n_contacts)

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
        # Finite raw parameters imply a strict ordering rho_r > rho_p.
        available_gap = 1.0 - self.eps - self.rho_p
        return self.rho_p + available_gap * torch.sigmoid(self.rho_gap_raw)

    @property
    def propagation_weight(self) -> Tensor:
        return F.softplus(self.propagation_weight_raw) + self.eps

    @property
    def restraint_weight(self) -> Tensor:
        return F.softplus(self.restraint_weight_raw) + self.eps

    @property
    def skew_gain(self) -> Tensor:
        return F.softplus(self.skew_gain_raw) + self.eps

    @property
    def direction_gain(self) -> Tensor:
        return F.softplus(self.direction_gain_raw) + self.eps

    def operator_components(self) -> dict[str, Tensor]:
        return source_conditioned_shared_scaffold(
            self.fixed_adjacency,
            self.axis_coordinate_raw,
            endpoint_temperature=self.endpoint_temperature,
            gamma=self.gamma,
            gain=self.gain,
            eps=self.eps,
        )

    def reset_state(self, batch_size: int | None = None) -> PropagationRestraintState:
        shape = (
            (self.n_contacts,)
            if batch_size is None
            else (int(batch_size), self.n_contacts)
        )
        event_shape = () if batch_size is None else (int(batch_size),)
        zero = self.participation_bias.new_zeros(shape)
        return PropagationRestraintState(
            zero,
            zero.clone(),
            self.participation_bias.new_zeros(event_shape),
            torch.zeros(
                event_shape, dtype=torch.bool, device=self.participation_bias.device
            ),
        )

    def observe(
        self,
        state: PropagationRestraintState,
        current_set: Tensor,
        active: Tensor | None = None,
    ) -> PropagationRestraintState:
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
        symmetric = components["W"]
        skew = components["W_skew"]

        # Causal source direction: only the first observed rank set of the
        # event can write it, and it stays frozen for every later rank step.
        membership_gap = components["endpoint_minus"] - components["endpoint_plus"]
        observed_size = current.sum(dim=-1).clamp_min(1.0)
        first_rank_direction = torch.tanh(
            self.direction_gain * (current * membership_gap).sum(dim=-1) / observed_size
        )
        direction = torch.where(
            state.source_initialized, state.direction, first_rank_direction
        )

        if current.ndim == 1:
            symmetric_drive = symmetric @ current
            skew_drive = skew @ current
        else:
            symmetric_drive = current @ symmetric.T
            skew_drive = current @ skew.T
        # Propagation sees the shared scaffold plus the source-signed flow;
        # restraint sees the shared scaffold only and stays direction-blind.
        propagation_drive = (
            symmetric_drive + self.skew_gain * direction[..., None] * skew_drive
        )
        proposal = PropagationRestraintState(
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
        return PropagationRestraintState(
            torch.where(active_bool[:, None], proposal.propagation, state.propagation),
            torch.where(active_bool[:, None], proposal.restraint, state.restraint),
            torch.where(active_bool, proposal.direction, state.direction),
            state.source_initialized | active_bool,
        )

    def _summary_features(
        self, state: PropagationRestraintState, seen: Tensor
    ) -> Tensor:
        eligible = ~seen.bool()
        weight = eligible.to(state.propagation.dtype)
        denominator = weight.sum(dim=-1).clamp_min(1.0)
        mean_p = (state.propagation * weight).sum(dim=-1) / denominator
        mean_r = (state.restraint * weight).sum(dim=-1) / denominator
        seen_fraction = seen.to(state.propagation.dtype).mean(dim=-1)
        return torch.stack([mean_p, mean_r, seen_fraction], dim=-1)

    def decision(
        self, state: PropagationRestraintState, seen: Tensor
    ) -> dict[str, Tensor]:
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
        cardinality_axis = torch.arange(
            1, self.n_contacts + 1, device=eligible.device
        )
        cardinality_support = (
            cardinality_axis <= n_eligible[..., None]
        )
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


class OrdinaryDenseGRUBaseline(_RankSetModelInterface):
    """Unstructured dense GRU baseline with the identical scoring contract."""

    def __init__(
        self,
        *,
        participation_bias: np.ndarray | Tensor,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        bias = torch.as_tensor(participation_bias, dtype=torch.float32)
        if bias.ndim != 1 or bias.numel() < 2 or not torch.isfinite(bias).all():
            raise ValueError("participation_bias must be a finite contact vector")
        if int(hidden_size) < 1:
            raise ValueError("hidden_size must be positive")
        self.n_contacts = int(bias.numel())
        self.hidden_size = int(hidden_size)
        self.register_buffer("participation_bias", bias)
        self.gru = nn.GRUCell(self.n_contacts, self.hidden_size)
        self.contact_decoder = nn.Linear(self.hidden_size, self.n_contacts, bias=False)
        self.stop_head = nn.Linear(self.hidden_size + 1, 1)
        self.cardinality_head = nn.Linear(
            self.hidden_size + 1, self.n_contacts
        )

    def reset_state(self, batch_size: int | None = None) -> Tensor:
        shape = (
            (self.hidden_size,)
            if batch_size is None
            else (int(batch_size), self.hidden_size)
        )
        return self.participation_bias.new_zeros(shape)

    def observe(
        self, state: Tensor, current_set: Tensor, active: Tensor | None = None
    ) -> Tensor:
        current = torch.as_tensor(
            current_set,
            device=self.participation_bias.device,
            dtype=self.participation_bias.dtype,
        )
        if current.ndim not in (1, 2) or current.shape[-1] != self.n_contacts:
            raise ValueError("current_set must be [..., contact]")
        proposal = self.gru(current, state)
        if active is None:
            return proposal
        active_bool = active.to(device=current.device, dtype=torch.bool)
        if current.ndim != 2 or active_bool.shape != current.shape[:1]:
            raise ValueError("active must align with a batched current_set")
        return torch.where(active_bool[:, None], proposal, state)

    def decision(self, state: Tensor, seen: Tensor) -> dict[str, Tensor]:
        seen_bool = seen.to(device=self.participation_bias.device, dtype=torch.bool)
        if seen_bool.ndim not in (1, 2) or seen_bool.shape[-1] != self.n_contacts:
            raise ValueError("seen must be [..., contact]")
        if state.shape[:-1] != seen_bool.shape[:-1]:
            raise ValueError("state and seen batch axes must align")
        eligible = ~seen_bool
        raw_node_logits = self.participation_bias + self.contact_decoder(state)
        node_logits = raw_node_logits.masked_fill(~eligible, -torch.inf)
        head_input = torch.cat(
            [state, seen_bool.to(state.dtype).mean(dim=-1, keepdim=True)], dim=-1
        )
        stop_logit = self.stop_head(head_input).squeeze(-1)
        cardinality_logits = self.cardinality_head(head_input)
        n_eligible = eligible.sum(dim=-1)
        cardinality_axis = torch.arange(
            1, self.n_contacts + 1, device=eligible.device
        )
        cardinality_support = (
            cardinality_axis <= n_eligible[..., None]
        )
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


def brute_force_conditional_subset_probabilities(
    node_logits: Tensor, eligible: Tensor, cardinality: int
) -> dict[tuple[int, ...], Tensor]:
    """Small-graph reference used by tests and diagnostic notebooks only."""

    indices = torch.where(eligible.bool())[0].tolist()
    subsets = list(combinations(indices, int(cardinality)))
    if not subsets:
        raise ValueError("no eligible subset has the requested cardinality")
    scores = torch.stack([node_logits[list(subset)].sum() for subset in subsets])
    probabilities = torch.softmax(scores, dim=0)
    return dict(zip(subsets, probabilities))


__all__ = [
    "OrdinaryDenseGRUBaseline",
    "PropagationRestraintState",
    "SharedScaffoldPropagationRNN",
    "StepLogProbabilities",
    "batched_cardinality_log_probability",
    "batched_exact_conditional_k_subset_log_probability",
    "brute_force_conditional_subset_probabilities",
    "build_fixed_local_shaft_adjacency",
    "cardinality_log_probability",
    "decomposed_one_step_log_probability",
    "estimate_node_hazard_bias",
    "exact_conditional_k_subset_log_probability",
    "rank_sets_from_group_ids",
    "source_conditioned_shared_scaffold",
]
