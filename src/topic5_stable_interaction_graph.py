"""Stable Interaction Graph (SIG-RNN) primitives for Topic 5.

Unlike the v0.1 autonomous latent-trajectory null, every emitted rank set is
fed back through a patient-shared contact-space interaction matrix before the
next rank is generated.  ``W[target, source]`` is the frozen orientation.

SNN artifacts, geometry, clinical labels, and future contacts are not inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import json
import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_shared_propagation_field import (
    MarkovMixtureModel,
    conditional_k_subset_log_prob,
    sample_conditional_k_subset,
    training_adequacy_verdict,
)


CONTRACT_NAME = "topic5_stable_interaction_graph_rnn_v2"


def phase_basis(phi: Tensor) -> Tensor:
    """Frozen nuisance basis shared by graph and no-graph models."""
    value = phi.to(dtype=torch.get_default_dtype())
    return torch.stack(
        [value, value.square(), torch.sin(math.pi * value)], dim=-1
    )


def _safe_logit(probability: float) -> float:
    value = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    return math.log(value) - math.log1p(-value)


def uniform_provenance(
    records: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    current_source_sha256: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Return the one provenance value per key, or fail closed.

    Section 10 of the v2 contract requires the aggregator to reject mixed
    source/config.  Three failure modes are covered here because all three
    silently produce an aggregate whose recorded provenance does not describe
    the fits it summarizes:

    1. a run artifact that carries no provenance key at all,
    2. runs that disagree with each other,
    3. runs that agree with each other but were produced by a different
       revision of the code than the one now aggregating them.

    The third case is not hypothetical: an aggregate re-run after the runner
    was edited stamps the new hash onto old fits.
    """
    if not records:
        raise RuntimeError("provenance check needs at least one run artifact")
    resolved: dict[str, Any] = {}
    for key in keys:
        encoded = []
        for record in records:
            if key not in record:
                raise RuntimeError(
                    f"run artifact is missing required provenance key '{key}'"
                )
            encoded.append(json.dumps(record[key], sort_keys=True))
        unique = sorted(set(encoded))
        if len(unique) != 1:
            raise RuntimeError(
                f"aggregated runs mix '{key}': {unique}"
            )
        resolved[key] = records[0][key]
    if current_source_sha256 is not None:
        recorded = resolved.get("source_sha256")
        if recorded != dict(current_source_sha256):
            raise RuntimeError(
                "fit-time source does not match the aggregating source; "
                f"fits={recorded} aggregator={dict(current_source_sha256)}"
            )
    return resolved


class MatchedPhaseMarkovMixtureModel(MarkovMixtureModel):
    """M1/M2 control with the exact same phase nuisance basis as SIG."""

    def __init__(
        self,
        n_contacts: int,
        static_bias: Tensor | np.ndarray,
        *,
        n_components: int,
    ) -> None:
        super().__init__(
            n_contacts,
            static_bias,
            n_components=n_components,
            phase_order=3,
        )

    def component_logits(
        self,
        component: int,
        previous: Tensor,
        *,
        step: Optional[int] = None,
        group_count: Optional[Tensor] = None,
    ) -> Tensor:
        logits = (
            self.static_bias[None, :] + self.bias_offset[component][None, :]
        ).expand(previous.shape[0], -1)
        weight = previous.to(self.static_bias.dtype)
        logits = logits + weight @ self.transition[component] / weight.sum(
            1, keepdim=True
        ).clamp_min(1.0)
        if step is None or group_count is None:
            raise ValueError("matched phase control requires step/group_count")
        denominator = (group_count - 1).clamp_min(1).to(logits.dtype)
        phi = (float(step) / denominator).clamp(0.0, 1.0)
        return logits + phase_basis(phi) @ self.phase_basis[component]


class StableInteractionGraph(nn.Module):
    """Contact-space feedback graph with a matched low-rank phase nuisance."""

    def __init__(
        self,
        n_contacts: int,
        *,
        static_bias: Optional[Tensor | np.ndarray] = None,
        learn_graph: bool = True,
        phase_rank: int = 3,
        max_weight: float = 3.0,
        initial_leak: float = 0.25,
    ) -> None:
        super().__init__()
        if int(n_contacts) < 2:
            raise ValueError("n_contacts must be at least two")
        if int(phase_rank) != 3:
            raise ValueError("v2 freezes the phase basis rank at three")
        self.n_contacts = int(n_contacts)
        self.learn_graph = bool(learn_graph)
        self.max_weight = float(max_weight)
        bias = (
            np.zeros(self.n_contacts, dtype=np.float32)
            if static_bias is None
            else np.asarray(static_bias, dtype=np.float32)
        )
        if bias.shape != (self.n_contacts,):
            raise ValueError("static_bias must contain one value per contact")
        self.register_buffer("static_bias", torch.as_tensor(bias))
        diagonal_mask = ~torch.eye(self.n_contacts, dtype=torch.bool)
        self.register_buffer("off_diagonal_mask", diagonal_mask)
        if self.learn_graph:
            self.raw_weight = nn.Parameter(
                torch.empty(self.n_contacts, self.n_contacts)
            )
            nn.init.normal_(self.raw_weight, mean=0.0, std=0.04)
        else:
            self.register_parameter("raw_weight", None)
        self.phase_loading = nn.Parameter(
            torch.zeros(self.n_contacts, phase_rank)
        )
        self.leak_logit = nn.Parameter(
            torch.tensor(_safe_logit(initial_leak), dtype=torch.float32)
        )

    @property
    def leak(self) -> Tensor:
        return torch.sigmoid(self.leak_logit)

    def effective_weight(self) -> Tensor:
        if self.raw_weight is None:
            return self.static_bias.new_zeros(
                (self.n_contacts, self.n_contacts)
            )
        bounded = self.max_weight * torch.tanh(self.raw_weight)
        return torch.where(
            self.off_diagonal_mask,
            bounded,
            torch.zeros_like(bounded),
        )

    def transition(
        self,
        state: Tensor,
        previous_set: Tensor,
        *,
        weight_override: Optional[Tensor] = None,
    ) -> Tensor:
        """Feed the emitted rank set back through ``W[target, source]``."""
        if state.ndim != 2 or state.shape[1] != self.n_contacts:
            raise ValueError("state must be [batch, contact]")
        if previous_set.shape != state.shape:
            raise ValueError("previous_set must align with state")
        weight = (
            self.effective_weight()
            if weight_override is None
            else weight_override.to(device=state.device, dtype=state.dtype)
        )
        if weight.shape != (self.n_contacts, self.n_contacts):
            raise ValueError("weight override has the wrong shape")
        drive = previous_set.to(dtype=state.dtype) @ weight.T
        return self.leak * state + torch.tanh(drive)

    def emission_logits(self, state: Tensor, phi: Tensor) -> Tensor:
        basis = phase_basis(phi).to(device=state.device, dtype=state.dtype)
        return (
            self.static_bias.to(dtype=state.dtype)
            + state
            + basis @ self.phase_loading.T
        )

    def suffix_log_likelihood(
        self,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        weight_override: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Exact teacher-forced factorization of the complete suffix law."""
        if group_ids.ndim != 2 or group_ids.shape[1] != self.n_contacts:
            raise ValueError("group_ids must be [event, contact]")
        if group_count.shape != (group_ids.shape[0],):
            raise ValueError("group_count must be event aligned")
        batch = group_ids.shape[0]
        state = self.static_bias.new_zeros((batch, self.n_contacts))
        recruited = group_ids == 0
        previous = recruited.clone()
        event_log_probability = state.new_zeros(batch)
        decision_count = torch.zeros(
            batch, dtype=torch.long, device=group_ids.device
        )
        max_groups = int(group_count.max().item())
        for step in range(1, max_groups):
            active = group_count > step
            state = self.transition(
                state, previous, weight_override=weight_override
            )
            phi = step / (group_count.to(state.dtype) - 1.0).clamp_min(1.0)
            logits = self.emission_logits(state, phi)
            target = group_ids == step
            value = conditional_k_subset_log_prob(
                logits,
                target,
                ~recruited,
                target.sum(1),
            )
            event_log_probability += torch.where(
                active, value, torch.zeros_like(value)
            )
            decision_count += active.to(torch.long)
            recruited = recruited | target
            previous = target
        return {
            "event_log_probability": event_log_probability,
            "decision_count": decision_count,
        }

    def nll_per_decision(
        self,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        weight_override: Optional[Tensor] = None,
    ) -> Tensor:
        result = self.suffix_log_likelihood(
            group_ids, group_count, weight_override=weight_override
        )
        return -result["event_log_probability"].sum() / result[
            "decision_count"
        ].sum().clamp_min(1)

    @torch.no_grad()
    def rollout(
        self,
        first_set: Tensor,
        group_count: Tensor,
        cardinality_schedule: Tensor,
        *,
        generator: Optional[torch.Generator] = None,
        weight_override: Optional[Tensor] = None,
    ) -> Tensor:
        """Free rollout; every later transition uses the generated rank set."""
        if first_set.ndim != 2 or first_set.shape[1] != self.n_contacts:
            raise ValueError("first_set must be [event, contact]")
        if cardinality_schedule.shape[0] != first_set.shape[0]:
            raise ValueError("cardinality schedule must be event aligned")
        batch = first_set.shape[0]
        output = torch.full(
            first_set.shape,
            -1,
            dtype=torch.long,
            device=first_set.device,
        )
        output[first_set] = 0
        state = self.static_bias.new_zeros((batch, self.n_contacts))
        recruited = first_set.clone()
        previous = first_set.clone()
        max_groups = int(group_count.max().item())
        for step in range(1, max_groups):
            active = group_count > step
            state = self.transition(
                state, previous, weight_override=weight_override
            )
            phi = step / (group_count.to(state.dtype) - 1.0).clamp_min(1.0)
            logits = self.emission_logits(state, phi)
            cardinality = cardinality_schedule[:, step - 1].to(torch.long)
            sampled = sample_conditional_k_subset(
                logits,
                ~recruited,
                torch.where(active, cardinality, torch.zeros_like(cardinality)),
                generator=generator,
            )
            output[sampled] = step
            recruited = recruited | sampled
            previous = sampled
        return output

    def regularization(self, *, l1_weight: float = 1e-3) -> Tensor:
        if self.raw_weight is None:
            graph_penalty = self.phase_loading.new_zeros(())
        else:
            graph_penalty = self.effective_weight().abs().mean()
        return float(l1_weight) * graph_penalty + 1e-5 * (
            self.phase_loading.square().mean()
        )

    @torch.no_grad()
    def one_step_intervention_matrix(
        self,
        *,
        phi: float = 0.25,
        weight_override: Optional[Tensor] = None,
    ) -> Tensor:
        """Observable L=1 intervention with a matched candidate set.

        For source ``j``, both arms exclude ``j`` from the next-rank candidate
        set.  The only difference is whether ``e_j`` is passed through the
        shared graph, so trivial self-exclusion cannot create influence.
        """
        weight = (
            self.effective_weight()
            if weight_override is None
            else weight_override.to(self.static_bias)
        )
        influence = self.static_bias.new_zeros(
            (self.n_contacts, self.n_contacts)
        )
        base_state = self.static_bias.new_zeros((1, self.n_contacts))
        phase = self.static_bias.new_full((1,), float(phi))
        for source in range(self.n_contacts):
            previous = torch.zeros(
                (1, self.n_contacts),
                dtype=torch.bool,
                device=self.static_bias.device,
            )
            previous[0, source] = True
            do_state = self.transition(
                base_state, previous, weight_override=weight
            )
            control_state = self.transition(
                base_state,
                torch.zeros_like(previous),
                weight_override=weight,
            )
            mask = torch.ones(
                (1, self.n_contacts),
                dtype=torch.bool,
                device=self.static_bias.device,
            )
            mask[0, source] = False
            impossible = torch.finfo(self.static_bias.dtype).min / 4.0
            do_logits = torch.where(
                mask, self.emission_logits(do_state, phase), impossible
            )
            control_logits = torch.where(
                mask, self.emission_logits(control_state, phase), impossible
            )
            influence[:, source] = (
                torch.softmax(do_logits, dim=1)
                - torch.softmax(control_logits, dim=1)
            )[0]
        return influence

    @torch.no_grad()
    def empirical_one_step_intervention_matrix(
        self,
        group_ids: Tensor,
        group_count: Tensor,
    ) -> Tensor:
        """Average L=1 interventions over observed singleton-rank contexts.

        The zero-state operator above is useful as a deterministic unit test,
        but an arbitrary zero state need not be occupied by the data.  This
        operator teacher-forces the same observed prefixes through each model,
        then compares the actual previous singleton rank against a zero-input
        control at the identical pre-transition state, phase, and candidate
        set.  It therefore measures supported observable responses rather than
        extrapolation from a hand-picked latent state.
        """
        if group_ids.ndim != 2 or group_ids.shape[1] != self.n_contacts:
            raise ValueError("group_ids must be [event, contact]")
        if group_count.shape != (group_ids.shape[0],):
            raise ValueError("group_count must be event aligned")
        for event, count in zip(group_ids, group_count):
            for step in range(int(count)):
                if int(torch.sum(event == step)) != 1:
                    raise ValueError(
                        "empirical L=1 operator currently requires singleton ranks"
                    )
        batch = group_ids.shape[0]
        state = self.static_bias.new_zeros((batch, self.n_contacts))
        recruited = group_ids == 0
        previous = recruited.clone()
        numerator = self.static_bias.new_zeros(
            (self.n_contacts, self.n_contacts)
        )
        denominator = self.static_bias.new_zeros(
            (self.n_contacts, self.n_contacts)
        )
        max_groups = int(group_count.max().item())
        impossible = torch.finfo(self.static_bias.dtype).min / 4.0
        for step in range(1, max_groups):
            active = group_count > step
            phi = step / (
                group_count.to(self.static_bias.dtype) - 1.0
            ).clamp_min(1.0)
            do_state = self.transition(state, previous)
            control_state = self.transition(
                state, torch.zeros_like(previous)
            )
            candidate = ~recruited
            do_probability = torch.softmax(
                torch.where(
                    candidate,
                    self.emission_logits(do_state, phi),
                    impossible,
                ),
                dim=1,
            )
            control_probability = torch.softmax(
                torch.where(
                    candidate,
                    self.emission_logits(control_state, phi),
                    impossible,
                ),
                dim=1,
            )
            difference = do_probability - control_probability
            for source in range(self.n_contacts):
                context = active & previous[:, source]
                if not torch.any(context):
                    continue
                valid_target = candidate[context].to(self.static_bias.dtype)
                numerator[:, source] += (
                    difference[context] * valid_target
                ).sum(0)
                denominator[:, source] += valid_target.sum(0)
            target = group_ids == step
            state = do_state
            recruited = recruited | target
            previous = target
        output = self.static_bias.new_full(
            (self.n_contacts, self.n_contacts), float("nan")
        )
        valid = denominator > 0
        output[valid] = numerator[valid] / denominator[valid]
        return output

    @torch.no_grad()
    def empirical_marginal_intervention_matrix(
        self,
        group_ids: Tensor,
        group_count: Tensor,
        *,
        return_support: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Average one-sender marginal responses over occupied prefixes.

        Unlike :meth:`empirical_one_step_intervention_matrix`, this operator
        supports tied/multi-contact rank sets. For each source present in the
        previous rank, the control removes only that source while holding the
        pre-transition state, phase, candidate set, and other simultaneous
        senders fixed. Thus a tied rank is not incorrectly attributed in full
        to every member.
        """
        if group_ids.ndim != 2 or group_ids.shape[1] != self.n_contacts:
            raise ValueError("group_ids must be [event, contact]")
        if group_count.shape != (group_ids.shape[0],):
            raise ValueError("group_count must be event aligned")
        batch = group_ids.shape[0]
        state = self.static_bias.new_zeros((batch, self.n_contacts))
        recruited = group_ids == 0
        previous = recruited.clone()
        numerator = self.static_bias.new_zeros(
            (self.n_contacts, self.n_contacts)
        )
        denominator = self.static_bias.new_zeros(
            (self.n_contacts, self.n_contacts)
        )
        max_groups = int(group_count.max().item())
        impossible = torch.finfo(self.static_bias.dtype).min / 4.0
        for step in range(1, max_groups):
            active = group_count > step
            phi = step / (
                group_count.to(self.static_bias.dtype) - 1.0
            ).clamp_min(1.0)
            do_state = self.transition(state, previous)
            candidate = ~recruited
            do_probability = torch.softmax(
                torch.where(
                    candidate,
                    self.emission_logits(do_state, phi),
                    impossible,
                ),
                dim=1,
            )
            for source in range(self.n_contacts):
                context = active & previous[:, source]
                if not torch.any(context):
                    continue
                control_previous = previous[context].clone()
                control_previous[:, source] = False
                control_state = self.transition(
                    state[context], control_previous
                )
                context_candidate = candidate[context]
                control_probability = torch.softmax(
                    torch.where(
                        context_candidate,
                        self.emission_logits(control_state, phi[context]),
                        impossible,
                    ),
                    dim=1,
                )
                difference = do_probability[context] - control_probability
                valid_target = context_candidate.to(self.static_bias.dtype)
                numerator[:, source] += (
                    difference * valid_target
                ).sum(0)
                denominator[:, source] += valid_target.sum(0)
            target = group_ids == step
            state = do_state
            recruited = recruited | target
            previous = target
        output = self.static_bias.new_full(
            (self.n_contacts, self.n_contacts), float("nan")
        )
        valid = denominator > 0
        output[valid] = numerator[valid] / denominator[valid]
        if return_support:
            return output, denominator
        return output


@dataclass(frozen=True)
class SyntheticGraphDataset:
    group_ids: np.ndarray
    group_count: np.ndarray
    start_contact: np.ndarray

    def torch(self, device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
        return (
            torch.as_tensor(self.group_ids, dtype=torch.long, device=device),
            torch.as_tensor(self.group_count, dtype=torch.long, device=device),
        )


def frozen_synthetic_graph() -> dict[str, np.ndarray | float]:
    """Return the predeclared 12-contact branching graph for G0-A."""
    contacts = 12
    weight = np.zeros((contacts, contacts), dtype=np.float32)
    edges = {
        0: [(1, 2.7), (2, 2.4)],
        1: [(3, 2.8), (4, 2.2)],
        2: [(3, 2.6), (5, 2.5)],
        3: [(4, 2.3), (6, 2.8)],
        4: [(6, 2.6), (7, 2.4)],
        5: [(6, 2.5), (8, 2.7)],
        6: [(7, 2.4), (9, 2.8)],
        7: [(9, 2.5), (10, 2.6)],
        8: [(9, 2.7), (11, 2.5)],
        9: [(10, 2.7), (11, 2.5)],
        10: [(2, 1.9), (11, 2.0)],
        11: [(1, 1.9), (4, 2.0)],
    }
    for source, targets in edges.items():
        for target, value in targets:
            weight[target, source] = value
    # Weak inhibition makes non-edges distinguishable without imposing space.
    for source in range(contacts):
        for target in range(contacts):
            if source != target and weight[target, source] == 0:
                weight[target, source] = -0.20
    phase_loading = np.zeros((contacts, 3), dtype=np.float32)
    phase_loading[:, 0] = np.linspace(-0.35, 0.35, contacts)
    phase_loading[:, 1] = np.linspace(0.20, -0.20, contacts)
    phase_loading[:, 2] = 0.12 * np.sin(np.arange(contacts))
    static_bias = np.linspace(-0.15, 0.15, contacts).astype(np.float32)
    return {
        "weight": weight,
        "phase_loading": phase_loading,
        "static_bias": static_bias,
        "leak": 0.25,
    }


def independent_synthetic_graph(seed: int = 20260801) -> dict[str, np.ndarray | float]:
    """Create the pre-frozen independent ring-plus-branch confirmation graph."""
    contacts = 12
    rng = np.random.default_rng(int(seed))
    weight = np.full((contacts, contacts), -0.20, dtype=np.float32)
    np.fill_diagonal(weight, 0.0)
    for source in range(contacts):
        target_ring = (source + 1) % contacts
        weight[target_ring, source] = float(rng.uniform(2.35, 2.75))
        offsets = [2, 3, 4, 5]
        rng.shuffle(offsets)
        target_branch = (source + offsets[0]) % contacts
        if target_branch == target_ring:
            target_branch = (source + offsets[1]) % contacts
        weight[target_branch, source] = float(rng.uniform(2.0, 2.65))
    phase_loading = rng.normal(0.0, 0.14, size=(contacts, 3)).astype(np.float32)
    phase_loading[:, 0] += np.linspace(-0.25, 0.25, contacts)
    static_bias = rng.normal(0.0, 0.10, size=contacts).astype(np.float32)
    static_bias -= static_bias.mean()
    return {
        "weight": weight,
        "phase_loading": phase_loading,
        "static_bias": static_bias,
        "leak": 0.30,
        "graph_seed": int(seed),
    }


def simulate_synthetic_events(
    n_events: int,
    *,
    starts: Sequence[int],
    seed: int,
    min_groups: int = 6,
    max_groups: int = 7,
    graph: Optional[dict[str, np.ndarray | float]] = None,
) -> SyntheticGraphDataset:
    """Generate stochastic branching events from the frozen G0-A graph."""
    if not starts:
        raise ValueError("starts cannot be empty")
    truth = frozen_synthetic_graph() if graph is None else graph
    weight = np.asarray(truth["weight"])
    loading = np.asarray(truth["phase_loading"])
    bias = np.asarray(truth["static_bias"])
    leak = float(truth["leak"])
    contacts = weight.shape[0]
    rng = np.random.default_rng(int(seed))
    groups = np.full((int(n_events), contacts), -1, dtype=np.int16)
    counts = rng.integers(
        int(min_groups), int(max_groups) + 1, size=int(n_events), endpoint=False
    ).astype(np.int16)
    start_values = rng.choice(np.asarray(starts, dtype=int), size=int(n_events))
    for event_index, (start, count) in enumerate(zip(start_values, counts)):
        groups[event_index, start] = 0
        recruited = np.zeros(contacts, dtype=bool)
        recruited[start] = True
        previous = np.zeros(contacts, dtype=np.float32)
        previous[start] = 1.0
        state = np.zeros(contacts, dtype=np.float32)
        for step in range(1, int(count)):
            state = leak * state + np.tanh(weight @ previous)
            phi = step / max(int(count) - 1, 1)
            basis = np.asarray(
                [phi, phi**2, math.sin(math.pi * phi)], dtype=np.float32
            )
            logits = bias + state + loading @ basis
            logits[recruited] = -np.inf
            finite = np.isfinite(logits)
            probability = np.zeros(contacts, dtype=float)
            shifted = logits[finite] - np.max(logits[finite])
            probability[finite] = np.exp(shifted)
            probability /= probability.sum()
            selected = int(rng.choice(contacts, p=probability))
            groups[event_index, selected] = step
            recruited[selected] = True
            previous.fill(0.0)
            previous[selected] = 1.0
    return SyntheticGraphDataset(
        group_ids=groups,
        group_count=counts,
        start_contact=start_values.astype(np.int16),
    )


def cardinality_schedule(
    group_ids: np.ndarray, group_count: np.ndarray
) -> np.ndarray:
    max_suffix = max(int(np.max(group_count)) - 1, 0)
    output = np.zeros((len(group_ids), max_suffix), dtype=np.int16)
    for step in range(1, max_suffix + 1):
        output[:, step - 1] = np.sum(group_ids == step, axis=1)
    return output


def precedence_matrix(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids)
    contacts = groups.shape[1]
    numerator = np.zeros((contacts, contacts), dtype=float)
    denominator = np.zeros((contacts, contacts), dtype=float)
    for event in groups:
        present = event >= 0
        both = present[:, None] & present[None, :]
        denominator += both
        numerator += both & (event[:, None] < event[None, :])
    output = np.full((contacts, contacts), np.nan, dtype=float)
    valid = denominator > 0
    output[valid] = numerator[valid] / denominator[valid]
    np.fill_diagonal(output, np.nan)
    return output


def precedence_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def rank_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float).ravel()
    b = np.asarray(right, dtype=float).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 3:
        return float("nan")

    def ranks(value: np.ndarray) -> np.ndarray:
        order = np.argsort(value, kind="stable")
        sorted_value = value[order]
        result = np.empty(len(value), dtype=float)
        start = 0
        while start < len(value):
            stop = start + 1
            while stop < len(value) and sorted_value[stop] == sorted_value[start]:
                stop += 1
            result[order[start:stop]] = 0.5 * (start + stop - 1)
            start = stop
        return result

    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def top_positive_overlap(
    truth: np.ndarray, estimate: np.ndarray, *, fraction: float = 0.20
) -> float:
    true_value = np.asarray(truth, dtype=float)
    estimated = np.asarray(estimate, dtype=float)
    mask = ~np.eye(true_value.shape[0], dtype=bool)
    indices = np.flatnonzero(mask.ravel())
    size = max(1, int(math.ceil(float(fraction) * len(indices))))
    true_top = set(indices[np.argsort(true_value.ravel()[indices])[-size:]])
    estimate_top = set(
        indices[np.argsort(estimated.ravel()[indices])[-size:]]
    )
    return float(len(true_top & estimate_top) / min(len(true_top), len(estimate_top)))


@dataclass
class FitResult:
    model: StableInteractionGraph
    history: list[dict[str, Any]]
    best_epoch: int
    best_validation_nll: float
    learning_rate: float
    recovery_depth: int
    adequacy: dict[str, Any]
    best_optimizer_state: dict[str, Any]


def fit_synthetic_sig(
    train: SyntheticGraphDataset,
    validation: SyntheticGraphDataset,
    *,
    seed: int,
    learn_graph: bool,
    max_epochs: int = 300,
    patience: int = 35,
    learning_rate: float = 0.03,
    l1_weight: float = 5e-4,
    batch_size: int = 256,
    recovery_depth: int = 0,
    static_bias: Optional[np.ndarray] = None,
    convergence_tolerance: float = 0.002,
    minimum_relative_improvement: float = 0.001,
    minimum_training_epochs: int = 20,
    minimum_best_epoch: int = 3,
    maximum_recovery_depth: int = 2,
) -> FitResult:
    """Fit one SIG model with a fail-closed validation-selected checkpoint.

    The initial validation score is recorded before the first update.  A run
    that reaches the epoch budget while its validation curve is still
    improving is not called converged; it is re-fit from the same deterministic
    initialization with a larger budget.  An early optimum is re-fit with a
    smaller learning rate.  This keeps the human SIG comparison on the same
    training-adequacy contract as the phase/template ladder.
    """
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    truth = frozen_synthetic_graph()
    bias = (
        np.asarray(truth["static_bias"])
        if static_bias is None
        else np.asarray(static_bias)
    )
    model = StableInteractionGraph(
        train.group_ids.shape[1],
        static_bias=bias,
        learn_graph=learn_graph,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-5
    )
    train_groups, train_counts = train.torch()
    validation_groups, validation_counts = validation.torch()
    generator = torch.Generator().manual_seed(int(seed) + 101)
    best_state = copy.deepcopy(model.state_dict())
    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
    model.eval()
    with torch.no_grad():
        initial_validation = float(
            model.nll_per_decision(validation_groups, validation_counts)
        )
    best_epoch = 0
    best_validation = initial_validation
    history: list[dict[str, Any]] = [
        {
            "epoch": 0.0,
            "train_nll_per_decision": None,
            "validation_nll_per_decision": initial_validation,
        }
    ]
    stale = 0
    stopped_by_patience = False
    for epoch in range(1, int(max_epochs) + 1):
        order = torch.randperm(
            len(train_groups), generator=generator
        )
        model.train()
        losses = []
        for start in range(0, len(order), int(batch_size)):
            index = order[start : start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            nll = model.nll_per_decision(
                train_groups[index], train_counts[index]
            )
            loss = nll + model.regularization(l1_weight=l1_weight)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite SIG training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(nll.detach()))
        model.eval()
        with torch.no_grad():
            validation_nll = float(
                model.nll_per_decision(
                    validation_groups, validation_counts
                )
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_nll_per_decision": float(np.mean(losses)),
                "validation_nll_per_decision": validation_nll,
            }
        )
        if validation_nll < best_validation - 1e-5:
            best_validation = validation_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch >= 60 and stale >= int(patience):
            stopped_by_patience = True
            break
    model.load_state_dict(best_state)
    adequacy = training_adequacy_verdict(
        [row["validation_nll_per_decision"] for row in history],
        patience=int(patience),
        tolerance=float(convergence_tolerance),
        initial_validation_nll=initial_validation,
        minimum_relative_improvement=float(minimum_relative_improvement),
        minimum_epochs=int(minimum_training_epochs),
        minimum_best_epoch=int(minimum_best_epoch),
        stopped_by_patience=stopped_by_patience,
    )
    # A null model can legitimately plateau at its zero-effect initialization.
    # This is optimization adequacy, not evidence of a learned effect.  Accept
    # it only when patience actually fired; budget exhaustion remains a fail.
    if (
        adequacy["verdict"] == "NO_LEARNING_PROGRESS"
        and stopped_by_patience
        and len(history) - 1 >= int(minimum_training_epochs)
    ):
        adequacy = {
            **adequacy,
            "converged": True,
            "verdict": "PLATEAU_NO_MATERIAL_GAIN",
        }
    if not adequacy["converged"] and int(recovery_depth) < int(
        maximum_recovery_depth
    ):
        early = adequacy["verdict"] == "EARLY_OPTIMUM_UNVERIFIED"
        next_learning_rate = (
            float(learning_rate) / 3.0 if early else float(learning_rate)
        )
        next_epochs = (
            int(max_epochs) if early else int(max_epochs) * 2
        )
        return fit_synthetic_sig(
            train,
            validation,
            seed=seed,
            learn_graph=learn_graph,
            max_epochs=next_epochs,
            patience=patience,
            learning_rate=next_learning_rate,
            l1_weight=l1_weight,
            batch_size=batch_size,
            recovery_depth=int(recovery_depth) + 1,
            static_bias=bias,
            convergence_tolerance=convergence_tolerance,
            minimum_relative_improvement=minimum_relative_improvement,
            minimum_training_epochs=minimum_training_epochs,
            minimum_best_epoch=minimum_best_epoch,
            maximum_recovery_depth=maximum_recovery_depth,
        )
    if not adequacy["converged"] or not np.isfinite(best_validation):
        raise RuntimeError(
            "SIG fit did not establish a usable optimum: "
            f"{adequacy['verdict']}"
        )
    return FitResult(
        model=model,
        history=history,
        best_epoch=int(best_epoch),
        best_validation_nll=float(best_validation),
        learning_rate=float(learning_rate),
        recovery_depth=int(recovery_depth),
        adequacy=adequacy,
        best_optimizer_state=best_optimizer_state,
    )
