"""Exact target-free rollout utilities for the shared-scaffold RNN.

The conditional contact-set sampler targets the same distribution used by
the training likelihood: for fixed cardinality ``k``, a subset has mass
proportional to ``exp(sum(logit_i))``.  This is not the Plackett--Luce law
produced by Gumbel top-k sampling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def exact_conditional_k_subset_sample(
    *,
    node_logits: Tensor,
    eligible: Tensor,
    cardinality: Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Sample exact fixed-cardinality subsets with an elementary-symmetric DP."""

    if node_logits.ndim != 2:
        raise ValueError("node_logits must have shape [batch, contact]")
    batch, n_contacts = node_logits.shape
    eligible_bool = eligible.to(device=node_logits.device, dtype=torch.bool)
    target_k = cardinality.to(device=node_logits.device, dtype=torch.long)
    if eligible_bool.shape != node_logits.shape or target_k.shape != (batch,):
        raise ValueError("eligible/cardinality must align with node_logits")
    n_eligible = eligible_bool.sum(dim=1)
    if torch.any((target_k < 0) | (target_k > n_eligible)):
        raise ValueError("cardinality must lie within each eligible set")
    if torch.any(~torch.isfinite(node_logits) & eligible_bool):
        raise ValueError("eligible node logits must be finite")

    log_weight = node_logits.masked_fill(~eligible_bool, -torch.inf)
    # dp[:, i, q] = log e_q of the first i contacts.  Rollout runs under
    # no_grad, so retaining this finite-horizon table is inexpensive and
    # makes exact backward sampling straightforward.
    dp = node_logits.new_full((batch, n_contacts + 1, n_contacts + 1), -torch.inf)
    dp[:, 0, 0] = 0.0
    for contact_index in range(n_contacts):
        dp[:, contact_index + 1] = dp[:, contact_index]
        upper = contact_index + 1
        dp[:, contact_index + 1, 1 : upper + 1] = torch.logaddexp(
            dp[:, contact_index, 1 : upper + 1],
            dp[:, contact_index, :upper]
            + log_weight[:, contact_index, None],
        )

    selected = torch.zeros_like(eligible_bool)
    remaining = target_k.clone()
    rows = torch.arange(batch, device=node_logits.device)
    for contact_index in range(n_contacts - 1, -1, -1):
        active = remaining > 0
        safe_remaining = torch.clamp(remaining, min=1)
        denominator = dp[rows, contact_index + 1, safe_remaining]
        numerator = (
            log_weight[:, contact_index]
            + dp[rows, contact_index, safe_remaining - 1]
        )
        probability = torch.where(
            active,
            torch.exp(numerator - denominator).clamp(0.0, 1.0),
            torch.zeros_like(numerator),
        )
        probability = torch.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)
        draw = torch.rand(
            batch,
            device=node_logits.device,
            dtype=node_logits.dtype,
            generator=generator,
        )
        take = active & eligible_bool[:, contact_index] & (draw < probability)
        selected[:, contact_index] = take
        remaining = remaining - take.to(remaining.dtype)
    if torch.any(remaining != 0) or not torch.equal(selected.sum(dim=1), target_k):
        raise RuntimeError("exact subset backward sampler did not satisfy cardinality")
    return selected


@dataclass(frozen=True)
class RolloutResult:
    event_group_ids: np.ndarray
    event_group_count: np.ndarray
    first_arrival_mass: np.ndarray
    source_at_step_zero: np.ndarray
    cumulative_participation_include_source: np.ndarray
    cumulative_participation_post_source: np.ndarray
    stop_step_histogram: np.ndarray


@torch.no_grad()
def rollout_from_source_pool(
    model,
    *,
    source_pool: np.ndarray | Tensor,
    horizon: int,
    n_rollouts: int,
    seed: int,
    batch_size: int = 512,
) -> RolloutResult:
    """Roll out STOP, cardinality, and exact conditional contact subsets."""

    if int(horizon) < 1 or int(n_rollouts) < 1 or int(batch_size) < 1:
        raise ValueError("horizon, n_rollouts, and batch_size must be positive")
    device = model.participation_bias.device
    n_contacts = int(model.n_contacts)
    source = torch.as_tensor(source_pool, device=device, dtype=torch.bool)
    if source.shape != (n_contacts,) or not torch.any(source):
        raise ValueError("source_pool must be a non-empty contact vector")
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    all_groups: list[np.ndarray] = []
    all_counts: list[np.ndarray] = []
    # Rows are future steps 1..H.  Observed source contacts live at t=0 and
    # must remain zero here so the frozen field reducer can add them exactly.
    arrival_counts = np.zeros((int(horizon), n_contacts), dtype=np.int64)
    stop_histogram = np.zeros(int(horizon) + 2, dtype=np.int64)
    remaining_rollouts = int(n_rollouts)
    model.eval()
    while remaining_rollouts:
        current_batch = min(int(batch_size), remaining_rollouts)
        current = source.unsqueeze(0).expand(current_batch, -1).clone()
        seen = current.clone()
        state = model.reset_state(batch_size=current_batch)
        state = model.observe(
            state,
            current,
            active=torch.ones(current_batch, dtype=torch.bool, device=device),
        )
        groups = torch.full(
            (current_batch, n_contacts),
            -1,
            dtype=torch.int16,
            device=device,
        )
        groups[current] = 0
        group_count = torch.ones(current_batch, dtype=torch.int16, device=device)
        alive = torch.ones(current_batch, dtype=torch.bool, device=device)
        stopped_at = torch.full(
            (current_batch,), int(horizon) + 1, dtype=torch.long, device=device
        )
        for step in range(1, int(horizon) + 1):
            decision = model.decision(state, seen)
            has_eligible = decision["eligible"].any(dim=1)
            stop_probability = torch.sigmoid(decision["stop_logit"])
            stop_draw = torch.rand(
                current_batch,
                device=device,
                dtype=stop_probability.dtype,
                generator=generator,
            )
            stop = alive & ((stop_draw < stop_probability) | ~has_eligible)
            stopped_at = torch.where(
                stop & (stopped_at == int(horizon) + 1),
                torch.full_like(stopped_at, step),
                stopped_at,
            )
            continuing = alive & ~stop
            next_set = torch.zeros_like(seen)
            if torch.any(continuing):
                rows = torch.where(continuing)[0]
                cardinality_logits = decision["cardinality_logits"][rows]
                cardinality_probability = torch.softmax(cardinality_logits, dim=1)
                sampled_k = torch.multinomial(
                    cardinality_probability,
                    1,
                    generator=generator,
                ).squeeze(1) + 1
                sampled = exact_conditional_k_subset_sample(
                    node_logits=decision["node_logits"][rows],
                    eligible=decision["eligible"][rows],
                    cardinality=sampled_k,
                    generator=generator,
                )
                next_set[rows] = sampled
                groups[next_set] = int(step)
                group_count[rows] += 1
                arrival_counts[step - 1] += sampled.sum(dim=0).cpu().numpy().astype(np.int64)
            state = model.observe(state, next_set, active=continuing)
            seen = seen | next_set
            alive = continuing
            if not torch.any(alive):
                break
        for value in stopped_at.cpu().numpy():
            stop_histogram[int(value)] += 1
        all_groups.append(groups.cpu().numpy())
        all_counts.append(group_count.cpu().numpy())
        remaining_rollouts -= current_batch

    first_arrival_mass = arrival_counts.astype(np.float64) / float(n_rollouts)
    source_at_step_zero = source.cpu().numpy().astype(np.float64)
    return RolloutResult(
        event_group_ids=np.row_stack(all_groups),
        event_group_count=np.concatenate(all_counts),
        first_arrival_mass=first_arrival_mass,
        source_at_step_zero=source_at_step_zero,
        cumulative_participation_include_source=np.maximum(
            first_arrival_mass.sum(axis=0), source_at_step_zero
        ),
        cumulative_participation_post_source=first_arrival_mass.sum(axis=0),
        stop_step_histogram=stop_histogram,
    )


__all__ = [
    "RolloutResult",
    "exact_conditional_k_subset_sample",
    "rollout_from_source_pool",
]
