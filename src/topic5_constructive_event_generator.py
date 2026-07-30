"""Source-conditioned constructive generation for Topic 5.

This module deliberately stays inside one interictal population event.  It
combines a train-only patient contact scaffold, the history residual of a
frozen linear-state model, and a train-only event-progress termination hazard.
No state is carried between events and no ictal value is consumed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import Tensor
except ImportError:  # pragma: no cover
    torch = None
    Tensor = object


VALID_CONDITIONS = {
    "full_constructive",
    "static_only",
    "static_shuffle",
    "history_h1",
    "history_h2",
    "constant_stop",
    "no_termination",
}


def train_static_log_scaffold(
    group_ids: np.ndarray,
    train_indices: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Smoothed train-only log participation frequency by contact."""
    groups = np.asarray(group_ids, dtype=int)[np.asarray(train_indices, dtype=int)]
    if groups.ndim != 2 or groups.shape[0] == 0:
        raise ValueError("train group_ids must be a non-empty event x contact array")
    if float(alpha) <= 0:
        raise ValueError("alpha must be positive")
    count = np.sum(groups >= 0, axis=0, dtype=float)
    probability = (count + float(alpha)) / (
        groups.shape[0] + float(alpha) * groups.shape[1]
    )
    return np.log(np.clip(probability, 1e-12, None))


def train_progress_hazard(
    group_count: np.ndarray,
    train_indices: np.ndarray,
    *,
    max_groups: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Estimate P(STOP after t observed rank sets | length >= t)."""
    length = np.asarray(group_count, dtype=int)[np.asarray(train_indices, dtype=int)]
    if length.size == 0 or np.any(length < 1):
        raise ValueError("training event lengths must be positive")
    if int(max_groups) < int(np.max(length)):
        raise ValueError("max_groups is smaller than an observed training event")
    hazard = np.ones(int(max_groups) + 1, dtype=np.float64)
    hazard[0] = 0.0
    for step in range(1, int(max_groups) + 1):
        at_risk = int(np.sum(length >= step))
        stopped = int(np.sum(length == step))
        if at_risk:
            hazard[step] = (stopped + float(alpha)) / (
                at_risk + 2.0 * float(alpha)
            )
        else:
            hazard[step] = 1.0
    hazard[-1] = 1.0
    return np.clip(hazard, 0.0, 1.0)


def constant_stop_hazard(group_count: np.ndarray, train_indices: np.ndarray) -> float:
    """Geometric hazard with the train-only mean event length."""
    length = np.asarray(group_count, dtype=float)[np.asarray(train_indices, dtype=int)]
    if length.size == 0 or not np.all(np.isfinite(length)):
        raise ValueError("training event lengths must be finite and non-empty")
    return float(np.clip(1.0 / max(float(np.mean(length)), 1.0), 1e-6, 1.0))


def contact_shaft(name: str) -> str:
    """Return a conservative shaft label without using a patient identifier."""
    match = re.match(r"^([^0-9]+)", str(name).strip())
    return match.group(1).upper() if match else str(name).strip().upper()


def shaft_preserving_permutation(
    contact_names: Sequence[str],
    *,
    seed: int,
) -> np.ndarray:
    """Deterministically permute contact fields only within electrode shafts."""
    names = [str(name) for name in contact_names]
    permutation = np.arange(len(names), dtype=int)
    rng = np.random.default_rng(int(seed))
    shafts: Dict[str, list[int]] = {}
    for index, name in enumerate(names):
        shafts.setdefault(contact_shaft(name), []).append(index)
    for indices in shafts.values():
        if len(indices) > 1:
            shuffled = rng.permutation(indices)
            permutation[np.asarray(indices, dtype=int)] = shuffled
    return permutation


def categorical_from_uniform(probability: np.ndarray, uniform: np.ndarray) -> np.ndarray:
    """Inverse-CDF sampling with one shared uniform per row."""
    probability = np.asarray(probability, dtype=np.float64)
    uniform = np.asarray(uniform, dtype=np.float64)
    if probability.ndim != 2 or uniform.shape != (probability.shape[0],):
        raise ValueError("probability must be [row, action] and uniform [row]")
    if np.any(probability < 0) or not np.all(np.isfinite(probability)):
        raise ValueError("probabilities must be finite and nonnegative")
    total = probability.sum(axis=1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("every row needs positive probability mass")
    normalized = probability / total
    cdf = np.cumsum(normalized, axis=1)
    cdf[:, -1] = 1.0
    clipped = np.clip(uniform, 0.0, np.nextafter(1.0, 0.0))
    return np.sum(cdf < clipped[:, None], axis=1).astype(np.int64)


@dataclass(frozen=True)
class ConstructiveRollout:
    event_group_ids: np.ndarray
    event_group_count: np.ndarray
    event_participant_count: np.ndarray
    revealed_source_mask: np.ndarray
    uniforms_sha256: str


def _replay_hidden(
    model,
    embedding: Tensor,
    contact_mask: Tensor,
    initial_hidden: Tensor,
    history_sets: Sequence[Tensor],
    history_prefixes: Sequence[Tensor],
    *,
    window: int,
) -> Tensor:
    start = max(0, len(history_sets) - int(window))
    hidden = initial_hidden
    for current, recruited in zip(
        history_sets[start:], history_prefixes[start:]
    ):
        hidden = model._advance(
            embedding,
            current,
            recruited,
            hidden,
            contact_mask,
        )
    return hidden


@torch.no_grad()
def source_conditioned_rollout(
    model,
    contact_features: Tensor,
    contact_mask: Tensor,
    local_offset: Tensor,
    source_mask: np.ndarray,
    uniforms: np.ndarray,
    static_log_scaffold: np.ndarray,
    progress_hazard: np.ndarray,
    *,
    condition: str,
    static_permutation: Optional[np.ndarray] = None,
    constant_hazard: Optional[float] = None,
    batch_size: int = 1024,
    uniforms_sha256: str = "",
) -> ConstructiveRollout:
    """Generate suffixes after revealing the held-out first rank set.

    The same ``uniforms`` can be passed to every condition to provide paired
    common-random-number interventions.
    """
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown rollout condition: {condition}")
    source = np.asarray(source_mask, dtype=bool)
    random_uniforms = np.asarray(uniforms, dtype=np.float64)
    static = np.asarray(static_log_scaffold, dtype=np.float64)
    hazard = np.asarray(progress_hazard, dtype=np.float64)
    if source.ndim != 2 or not np.all(source.any(axis=1)):
        raise ValueError("every event needs a non-empty revealed source set")
    n_events, n_contacts = source.shape
    if random_uniforms.shape != (n_events, n_contacts):
        raise ValueError("uniforms must be event x contact")
    if static.shape != (n_contacts,):
        raise ValueError("static scaffold must align with contacts")
    if contact_features.shape[-2] != n_contacts:
        raise ValueError("contact features and source mask are misaligned")
    if len(hazard) <= n_contacts:
        raise ValueError("progress hazard must cover all possible rank counts")
    if condition == "static_shuffle":
        if static_permutation is None:
            raise ValueError("static_shuffle requires a frozen permutation")
        permutation = np.asarray(static_permutation, dtype=int)
        if sorted(permutation.tolist()) != list(range(n_contacts)):
            raise ValueError("static permutation is invalid")
        static = static[permutation]
    if condition == "constant_stop" and constant_hazard is None:
        raise ValueError("constant_stop requires a frozen constant hazard")

    model.eval()
    device = contact_features.device
    output_groups = np.full((n_events, n_contacts), -1, dtype=np.int16)
    output_count = np.ones(n_events, dtype=np.int16)
    output_participants = source.sum(axis=1).astype(np.int16)

    for batch_start in range(0, n_events, int(batch_size)):
        batch_stop = min(batch_start + int(batch_size), n_events)
        batch_source_np = source[batch_start:batch_stop]
        batch_uniforms = random_uniforms[batch_start:batch_stop]
        current_batch = batch_stop - batch_start

        features = contact_features[:1].expand(current_batch, -1, -1)
        mask = contact_mask[:1].expand(current_batch, -1)
        embedding, encoder_input = model._encode(features, local_offset)
        initial_hidden = model._initial_hidden(embedding, mask)
        all_contact_logits, _ = model._decode(
            embedding,
            encoder_input,
            initial_hidden,
            mask,
        )

        # ``torch.as_tensor`` may share CPU memory with the NumPy source.
        # Clone before in-place recruitment updates so the revealed-source
        # contract remains immutable.
        recruited = torch.as_tensor(
            batch_source_np, dtype=torch.bool, device=device
        ).clone()
        last_set = recruited.clone()
        history_sets = [last_set.clone()]
        history_prefixes = [recruited.clone()]
        full_hidden = model._advance(
            embedding,
            last_set,
            recruited,
            initial_hidden,
            mask,
        )
        alive = torch.ones(current_batch, dtype=torch.bool, device=device)
        group_count = torch.ones(current_batch, dtype=torch.long, device=device)
        participant_count = recruited.sum(1).to(torch.long)

        local_groups = np.full(
            (current_batch, n_contacts), -1, dtype=np.int16
        )
        local_groups[batch_source_np] = 0

        for generation_step in range(n_contacts - 1):
            candidate = mask & ~recruited
            no_candidate = ~candidate.any(1)
            alive = alive & ~no_candidate
            if not torch.any(alive):
                break

            if condition == "history_h1":
                hidden = _replay_hidden(
                    model,
                    embedding,
                    mask,
                    initial_hidden,
                    history_sets,
                    history_prefixes,
                    window=1,
                )
            elif condition == "history_h2":
                hidden = _replay_hidden(
                    model,
                    embedding,
                    mask,
                    initial_hidden,
                    history_sets,
                    history_prefixes,
                    window=2,
                )
            else:
                hidden = full_hidden

            contact_logits, _ = model._decode(
                embedding,
                encoder_input,
                hidden,
                candidate,
            )
            if condition == "static_only":
                residual = torch.zeros_like(contact_logits)
            else:
                residual = contact_logits - all_contact_logits
            score = residual + torch.as_tensor(
                static, dtype=contact_logits.dtype, device=device
            )[None, :]
            score = score.masked_fill(~candidate, -1e9)
            contact_probability = torch.softmax(score, dim=1).cpu().numpy()

            count_np = group_count.cpu().numpy()
            if condition == "no_termination":
                stop_probability = np.zeros(current_batch, dtype=np.float64)
            elif condition == "constant_stop":
                stop_probability = np.full(
                    current_batch, float(constant_hazard), dtype=np.float64
                )
            else:
                stop_probability = hazard[count_np]
            stop_probability = np.where(
                alive.cpu().numpy(), stop_probability, 1.0
            )
            probability = np.column_stack(
                [
                    stop_probability,
                    (1.0 - stop_probability[:, None]) * contact_probability,
                ]
            )
            action = categorical_from_uniform(
                probability,
                batch_uniforms[:, generation_step],
            )
            action = np.where(alive.cpu().numpy(), action, 0)
            chose_contact_np = action > 0
            chose_stop_np = ~chose_contact_np
            alive_np = alive.cpu().numpy() & ~chose_stop_np

            new_set = torch.zeros_like(recruited)
            if np.any(alive_np):
                row_np = np.flatnonzero(alive_np)
                contact_np = action[row_np] - 1
                local_groups[row_np, contact_np] = count_np[row_np].astype(
                    np.int16
                )
                row = torch.as_tensor(row_np, dtype=torch.long, device=device)
                contact = torch.as_tensor(
                    contact_np, dtype=torch.long, device=device
                )
                new_set[row, contact] = True
                recruited[row, contact] = True
                group_count[row] += 1
                participant_count[row] += 1

            alive = torch.as_tensor(alive_np, dtype=torch.bool, device=device)
            history_sets.append(new_set.clone())
            history_prefixes.append(recruited.clone())
            updated_hidden = model._advance(
                embedding,
                new_set,
                recruited,
                full_hidden,
                mask,
            )
            full_hidden = torch.where(
                alive[:, None], updated_hidden, full_hidden
            )

        output_groups[batch_start:batch_stop] = local_groups
        output_count[batch_start:batch_stop] = (
            group_count.cpu().numpy().astype(np.int16)
        )
        output_participants[batch_start:batch_stop] = (
            participant_count.cpu().numpy().astype(np.int16)
        )

    if not np.all(output_groups[source] == 0):
        raise RuntimeError("a revealed source was not retained at rank zero")
    if np.any(output_participants != np.sum(output_groups >= 0, axis=1)):
        raise RuntimeError("participant counts are inconsistent")
    return ConstructiveRollout(
        event_group_ids=output_groups,
        event_group_count=output_count,
        event_participant_count=output_participants,
        revealed_source_mask=source,
        uniforms_sha256=str(uniforms_sha256),
    )


def remove_revealed_source(
    group_ids: np.ndarray,
    source_mask: np.ndarray,
) -> np.ndarray:
    """Return suffix-only group ranks, reindexed from zero per event."""
    groups = np.asarray(group_ids, dtype=int).copy()
    source = np.asarray(source_mask, dtype=bool)
    if groups.shape != source.shape:
        raise ValueError("groups and source masks must align")
    groups[source] = -1
    for event in range(groups.shape[0]):
        valid = groups[event] >= 0
        if np.any(valid):
            unique = np.unique(groups[event, valid])
            mapping = {int(value): index for index, value in enumerate(unique)}
            groups[event, valid] = np.asarray(
                [mapping[int(value)] for value in groups[event, valid]],
                dtype=int,
            )
    return groups.astype(np.int16)


def event_length_wasserstein(
    predicted_count: np.ndarray,
    observed_count: np.ndarray,
) -> float:
    """Normalized one-dimensional Wasserstein distance between event lengths."""
    from scipy.stats import wasserstein_distance

    predicted = np.asarray(predicted_count, dtype=float)
    observed = np.asarray(observed_count, dtype=float)
    scale = max(float(np.max(np.r_[predicted, observed])), 1.0)
    return float(wasserstein_distance(predicted, observed) / scale)


def stop_hazard_curve(group_count: np.ndarray, *, max_groups: int) -> np.ndarray:
    """Empirical termination hazard for posterior predictive checks."""
    length = np.asarray(group_count, dtype=int)
    out = np.full(int(max_groups) + 1, np.nan, dtype=float)
    for step in range(1, int(max_groups) + 1):
        at_risk = int(np.sum(length >= step))
        if at_risk:
            out[step] = float(np.sum(length == step) / at_risk)
    return out
