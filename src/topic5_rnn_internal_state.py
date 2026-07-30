"""Utilities for target-blind Topic 5 GRU hidden-state audits.

The functions in this module never read seizure targets.  They operate on the
frozen interictal rank-sequence checkpoints and preserve event-first scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass(frozen=True)
class PCAState:
    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray


def deterministic_event_sample(indices: np.ndarray, limit: int) -> np.ndarray:
    """Return a chronological, approximately uniform deterministic sample."""
    values = np.asarray(indices, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("indices must be one dimensional")
    if int(limit) < 1:
        raise ValueError("limit must be positive")
    if len(values) <= int(limit):
        return values.copy()
    positions = np.linspace(0, len(values) - 1, int(limit))
    positions = np.unique(np.rint(positions).astype(np.int64))
    if len(positions) != int(limit):
        raise RuntimeError("deterministic sample did not retain requested size")
    return values[positions]


def split_train80(
    train_indices: np.ndarray, fraction_train: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """Split the frozen chronological train80 into train60/validation20."""
    values = np.asarray(train_indices, dtype=np.int64)
    cut = int(np.floor(len(values) * float(fraction_train)))
    cut = min(max(cut, 1), len(values) - 1)
    return values[:cut], values[cut:]


@torch.no_grad()
def teacher_forced_hidden(
    model: torch.nn.Module,
    contact_features: torch.Tensor,
    local_offset: torch.Tensor,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_indices: np.ndarray,
    *,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the hidden state immediately before every next-action decision."""
    model.eval()
    device = next(model.parameters()).device
    features_one = contact_features.to(device=device, dtype=torch.float32)
    if features_one.ndim != 2:
        raise ValueError("contact_features must be [contact, feature]")
    offset = local_offset.to(device=device, dtype=torch.float32)
    indices = np.asarray(event_indices, dtype=np.int64)
    states: list[np.ndarray] = []
    state_events: list[np.ndarray] = []
    state_steps: list[np.ndarray] = []
    for start in range(0, len(indices), int(batch_size)):
        current_indices = indices[start : start + int(batch_size)]
        groups = torch.as_tensor(
            np.asarray(group_ids[current_indices], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        counts = torch.as_tensor(
            np.asarray(group_count[current_indices], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        batch = len(current_indices)
        features = features_one.unsqueeze(0).expand(batch, -1, -1)
        contact_mask = torch.ones(
            (batch, features.shape[1]), dtype=torch.bool, device=device
        )
        embedding, _ = model._encode(features, offset)
        hidden = model._initial_hidden(embedding, contact_mask)
        recruited = torch.zeros_like(contact_mask)
        maximum = int(counts.max().item())
        for step in range(maximum + 1):
            active = counts >= step
            if torch.any(active):
                states.append(hidden[active].cpu().numpy().astype(np.float32))
                state_events.append(current_indices[active.cpu().numpy()])
                state_steps.append(
                    np.full(int(active.sum().item()), step, dtype=np.int16)
                )
            if step == maximum:
                break
            current = (groups == step) & contact_mask
            advances = counts > step
            updated_recruited = recruited | current
            updated_hidden = model._advance(
                embedding,
                current,
                updated_recruited,
                hidden,
                contact_mask,
            )
            hidden = torch.where(advances[:, None], updated_hidden, hidden)
            recruited = torch.where(
                advances[:, None], updated_recruited, recruited
            )
    return (
        np.row_stack(states),
        np.concatenate(state_events).astype(np.int64),
        np.concatenate(state_steps).astype(np.int16),
    )


@torch.no_grad()
def teacher_forced_probability_fields(
    model: torch.nn.Module,
    contact_features: torch.Tensor,
    local_offset: torch.Tensor,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_indices: np.ndarray,
    *,
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """Aggregate one-step probabilities along each observed heldout prefix.

    ``union_participation`` is ``1-prod_t(1-p_i,t)`` along the observed event
    path, including the terminal decision. It is a bounded diagnostic field,
    not a free-running generative participation probability.
    """
    model.eval()
    device = next(model.parameters()).device
    features_one = contact_features.to(device=device, dtype=torch.float32)
    if features_one.ndim != 2:
        raise ValueError("contact_features must be [contact, feature]")
    offset = local_offset.to(device=device, dtype=torch.float32)
    indices = np.asarray(event_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("event_indices must be a nonempty vector")
    union_sum = np.zeros(features_one.shape[0], dtype=np.float64)
    probability_sum = np.zeros(features_one.shape[0], dtype=np.float64)
    per_event_mass = []
    for start in range(0, len(indices), int(batch_size)):
        current = indices[start : start + int(batch_size)]
        groups = torch.as_tensor(
            np.asarray(group_ids[current], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        counts = torch.as_tensor(
            np.asarray(group_count[current], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        batch = len(current)
        features = features_one.unsqueeze(0).expand(batch, -1, -1)
        contact_mask = torch.ones(
            (batch, features.shape[1]), dtype=torch.bool, device=device
        )
        output = model(
            contact_features=features,
            contact_mask=contact_mask,
            group_ids=groups,
            group_count=counts,
            local_offset=offset,
        )
        action_logits = torch.cat(
            [output["stop_logits"].unsqueeze(-1), output["contact_logits"]],
            dim=-1,
        )
        contact_probability = torch.softmax(action_logits, dim=-1)[..., 1:]
        steps = torch.arange(
            contact_probability.shape[1], device=device
        ).unsqueeze(0)
        valid = steps <= counts.unsqueeze(1)
        masked = contact_probability * valid.unsqueeze(-1)
        union = 1.0 - torch.prod(
            torch.where(
                valid.unsqueeze(-1),
                1.0 - contact_probability,
                torch.ones_like(contact_probability),
            ),
            dim=1,
        )
        summed = masked.sum(dim=1)
        union_sum += union.sum(dim=0).cpu().numpy()
        probability_sum += summed.sum(dim=0).cpu().numpy()
        per_event_mass.append(union.sum(dim=1).cpu().numpy())
    denominator = float(len(indices))
    return {
        "union_participation": (union_sum / denominator).astype(np.float32),
        "summed_next_probability": (
            probability_sum / denominator
        ).astype(np.float32),
        "event_union_mass": np.concatenate(per_event_mass).astype(np.float32),
    }


@torch.no_grad()
def prefix_intervention_outputs(
    model: torch.nn.Module,
    contact_features: torch.Tensor,
    local_offset: torch.Tensor,
    group_ids: torch.Tensor,
    group_count: torch.Tensor,
    *,
    intervention: str,
    reset_after_rank: int | None = None,
) -> dict[str, torch.Tensor]:
    """Replay causal prefixes under an explicit history intervention.

    The eligible-contact mask always follows the complete observed prefix.
    Interventions change only the rank-set tokens reaching recurrent state.
    Therefore an omitted early token cannot become an eligible future action.

    ``reset_after_rank=k`` means that predictions through rank ``k`` use the
    ordinary prefix; before predicting rank ``k+1`` the hidden state is reset,
    and only subsequently observed tokens are replayed.
    """
    if intervention not in {
        "ordered",
        "reverse_prefix",
        "drop_earliest",
        "reset_after_rank",
    }:
        raise ValueError(f"unknown intervention: {intervention}")
    if intervention == "reset_after_rank" and (
        reset_after_rank is None or int(reset_after_rank) < 0
    ):
        raise ValueError("reset_after_rank requires a nonnegative rank")
    model.eval()
    groups = group_ids.to(dtype=torch.long)
    counts = group_count.to(dtype=torch.long)
    if groups.ndim != 2 or counts.shape != (groups.shape[0],):
        raise ValueError("group_ids/group_count shape mismatch")
    batch, n_contacts = groups.shape
    features_one = contact_features.to(
        device=groups.device, dtype=torch.float32
    )
    if features_one.ndim == 2:
        features = features_one.unsqueeze(0).expand(batch, -1, -1)
    elif features_one.ndim == 3 and features_one.shape[0] == batch:
        features = features_one
    else:
        raise ValueError("contact_features must be [contact, feature] or batched")
    mask = torch.ones((batch, n_contacts), dtype=torch.bool, device=groups.device)
    embedding, encoder_input = model._encode(features, local_offset)
    maximum = int(counts.max().item())
    contact_logits = []
    stop_logits = []
    candidate_masks = []
    for prediction_step in range(maximum + 1):
        recruited = (groups >= 0) & (groups < prediction_step) & mask
        hidden = model._initial_hidden(embedding, mask)
        history = list(range(prediction_step))
        if intervention == "reverse_prefix":
            history = history[::-1]
        elif intervention == "drop_earliest":
            history = history[1:]
        elif intervention == "reset_after_rank":
            reset = int(reset_after_rank)
            if prediction_step > reset:
                history = list(range(reset + 1, prediction_step))
        replayed = torch.zeros_like(mask)
        for rank_step in history:
            current = (groups == rank_step) & mask
            replayed = replayed | current
            if intervention == "reverse_prefix":
                # Progress must be causal in the intervened sequence. Keeping
                # the original rank-specific progress would reveal the
                # token's pre-intervention position and partly undo reversal.
                recruited_at_rank = replayed
            else:
                # Drop/reset retain the observable complete-prefix progress
                # while removing identity-bearing history from the state.
                recruited_at_rank = (
                    (groups >= 0) & (groups <= rank_step) & mask
                )
            hidden = model._advance(
                embedding,
                current,
                recruited_at_rank,
                hidden,
                mask,
            )
        candidate = mask & ~recruited
        action, stop = model._decode(
            embedding, encoder_input, hidden, candidate
        )
        contact_logits.append(action)
        stop_logits.append(stop)
        candidate_masks.append(candidate)
    return {
        "contact_logits": torch.stack(contact_logits, dim=1),
        "stop_logits": torch.stack(stop_logits, dim=1),
        "candidate_mask": torch.stack(candidate_masks, dim=1),
    }


@torch.no_grad()
def readout_relevant_local_memory(
    model: torch.nn.Module,
    contact_features: torch.Tensor,
    local_offset: torch.Tensor,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_indices: np.ndarray,
    *,
    max_events: int = 24,
) -> dict[str, float]:
    """Measure local rank-step memory only through the contact readout.

    The diagnostic uses the exact local Jacobian of the fitted nongated
    recurrence and the frozen hidden-to-contact-logit map. It is invariant to
    contact ordering and never reads an ictal target. The values describe
    event-indexed retention in model coordinates, not a biological time
    constant.
    """
    model.eval()
    device = next(model.parameters()).device
    features = contact_features.to(device=device, dtype=torch.float32)
    if features.ndim != 2:
        raise ValueError("contact_features must be [contact, feature]")
    offset = local_offset.to(device=device, dtype=torch.float32)
    contact_mask = torch.ones(
        (1, features.shape[0]), dtype=torch.bool, device=device
    )
    embedding, _ = model._encode(features.unsqueeze(0), offset)
    readout = (
        embedding[0] @ model.action_query.weight
    ) / np.sqrt(float(model.contact_embedding_dim))
    readout_np = readout.detach().cpu().numpy().astype(np.float64)
    readout_norm = float(np.linalg.norm(readout_np))
    if readout_norm <= 1.0e-12:
        raise RuntimeError("degenerate hidden-to-contact readout")

    indices = deterministic_event_sample(
        np.asarray(event_indices, dtype=np.int64),
        min(int(max_events), len(event_indices)),
    )
    groups_all = np.asarray(group_ids, dtype=np.int64)
    counts_all = np.asarray(group_count, dtype=np.int64)
    if hasattr(model, "persistence"):
        jacobian_np = np.diag(
            model.persistence.detach().cpu().numpy().astype(np.float64)
        )
        propagated = readout_np @ jacobian_np
        propagated_norm = float(np.linalg.norm(propagated))
        denominator = readout_norm * propagated_norm
        return {
            "n_sampled_events": int(len(indices)),
            "n_sampled_transitions": int(np.sum(counts_all[indices])),
            "readout_retention_median": propagated_norm / readout_norm,
            "readout_alignment_median": (
                float(np.sum(readout_np * propagated) / denominator)
                if denominator > 1.0e-12
                else np.nan
            ),
            "state_gain_median": float(
                np.linalg.norm(jacobian_np) / np.sqrt(jacobian_np.shape[0])
            ),
            "local_spectral_radius_median": float(
                np.max(np.abs(np.linalg.eigvals(jacobian_np)))
            ),
        }
    retention = []
    alignment = []
    state_gain = []
    spectral_radius = []
    for event_index in indices:
        groups = torch.as_tensor(
            groups_all[event_index : event_index + 1],
            dtype=torch.long,
            device=device,
        )
        count = int(counts_all[event_index])
        hidden = model._initial_hidden(embedding, contact_mask)
        recruited = torch.zeros_like(contact_mask)
        for step in range(count):
            current = (groups == step) & contact_mask
            recruited = recruited | current
            updated = model._advance(
                embedding,
                current,
                recruited,
                hidden,
                contact_mask,
            )
            if hasattr(model, "rnn"):
                derivative = 1.0 - updated[0].square()
                jacobian = derivative[:, None] * model.rnn.weight_hh
            elif hasattr(model, "alpha") and hasattr(model, "decay"):
                alpha = model.alpha
                proposal = (
                    updated[0] - (1.0 - alpha) * hidden[0]
                ) / alpha.clamp_min(1.0e-6)
                recurrent = -torch.diag(model.decay)
                if int(getattr(model, "recurrent_rank", 0)):
                    recurrent = recurrent + (
                        model.mode_u @ model.mode_v.T
                    ) / np.sqrt(float(model.recurrent_rank))
                jacobian = (
                    (1.0 - alpha) * torch.eye(
                        model.hidden_size, device=device
                    )
                    + alpha
                    * (1.0 - proposal.square())[:, None]
                    * recurrent
                )
            else:
                raise TypeError(
                    "readout memory diagnostic supports linear, vanilla and "
                    "low-rank nongated recurrences"
                )
            jacobian_np = jacobian.detach().cpu().numpy().astype(np.float64)
            propagated = readout_np @ jacobian_np
            propagated_norm = float(np.linalg.norm(propagated))
            retention.append(propagated_norm / readout_norm)
            denominator = readout_norm * propagated_norm
            alignment.append(
                float(np.sum(readout_np * propagated) / denominator)
                if denominator > 1.0e-12
                else np.nan
            )
            state_gain.append(
                float(np.linalg.norm(jacobian_np) / np.sqrt(jacobian_np.shape[0]))
            )
            spectral_radius.append(
                float(np.max(np.abs(np.linalg.eigvals(jacobian_np))))
            )
            hidden = updated
    if not retention:
        raise RuntimeError("no eligible rank-step transitions for memory audit")
    return {
        "n_sampled_events": int(len(indices)),
        "n_sampled_transitions": int(len(retention)),
        "readout_retention_median": float(np.nanmedian(retention)),
        "readout_alignment_median": float(np.nanmedian(alignment)),
        "state_gain_median": float(np.nanmedian(state_gain)),
        "local_spectral_radius_median": float(np.nanmedian(spectral_radius)),
    }


def prefix_observables(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_index: np.ndarray,
    step: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reconstruct causal prefix features and future targets."""
    groups = np.asarray(group_ids, dtype=np.int64)[np.asarray(event_index, int)]
    counts = np.asarray(group_count, dtype=np.int64)[np.asarray(event_index, int)]
    steps = np.asarray(step, dtype=np.int64)
    n_contacts = groups.shape[1]
    recruited = (groups >= 0) & (groups < steps[:, None])
    last_set = (
        (steps[:, None] > 0) & (groups == (steps[:, None] - 1))
    )
    candidate = ~recruited
    progress = recruited.sum(1) / float(n_contacts)
    new_fraction = last_set.sum(1) / float(n_contacts)
    terminal = steps == counts
    next_action = np.full(len(steps), n_contacts, dtype=np.int64)
    nonterminal = ~terminal
    if np.any(nonterminal):
        target = groups[nonterminal] == steps[nonterminal, None]
        if np.any(target.sum(1) < 1):
            raise RuntimeError("nonterminal prefix lacks a next contact")
        next_action[nonterminal] = np.argmax(target, axis=1)

    future = (groups >= steps[:, None]).astype(np.float32)
    remaining_score = np.zeros_like(future, dtype=np.float32)
    for row in range(len(steps)):
        remaining_groups = int(counts[row] - steps[row])
        valid = groups[row] >= steps[row]
        if not np.any(valid):
            continue
        if remaining_groups <= 1:
            remaining_score[row, valid] = 1.0
        else:
            remaining_score[row, valid] = (
                1.0
                - (groups[row, valid] - steps[row])
                / float(remaining_groups - 1)
            )
    return {
        "recruited": recruited.astype(np.float32),
        "last_set": last_set.astype(np.float32),
        "candidate": candidate,
        "progress": progress.astype(np.float32),
        "new_fraction": new_fraction.astype(np.float32),
        "terminal": terminal,
        "next_action": next_action,
        "future_participation": future,
        "remaining_score": remaining_score,
    }


def observable_design(observables: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Build matched causal probe features."""
    base = np.column_stack(
        [observables["progress"], observables["new_fraction"]]
    )
    if mode == "progress":
        return base.astype(np.float32)
    if mode == "last_set":
        return np.column_stack([base, observables["last_set"]]).astype(np.float32)
    if mode == "unordered":
        return np.column_stack(
            [base, observables["last_set"], observables["recruited"]]
        ).astype(np.float32)
    raise ValueError(f"unknown observable mode: {mode}")


def fit_pca(hidden: np.ndarray) -> PCAState:
    values = np.asarray(hidden, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("hidden must be a nontrivial [sample, state] matrix")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    components = eigenvectors[:, order].T
    return PCAState(
        mean=mean.astype(np.float32),
        components=components.astype(np.float32),
        eigenvalues=eigenvalues.astype(np.float64),
    )


def pca_summary(pca: PCAState) -> dict[str, float | int]:
    values = np.asarray(pca.eigenvalues, dtype=np.float64)
    total = float(values.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "k80": 0,
            "k90": 0,
            "k95": 0,
        }
    fraction = np.cumsum(values) / total
    effective = total * total / float(np.sum(values * values))
    return {
        "effective_rank": float(effective),
        "k80": int(np.searchsorted(fraction, 0.80) + 1),
        "k90": int(np.searchsorted(fraction, 0.90) + 1),
        "k95": int(np.searchsorted(fraction, 0.95) + 1),
    }


def project_reconstruct(
    hidden: np.ndarray, pca: PCAState, n_components: int
) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float64)
    k = min(max(int(n_components), 0), pca.components.shape[0])
    centered = values - pca.mean
    if k == 0:
        return np.broadcast_to(pca.mean, values.shape).copy().astype(np.float32)
    basis = np.asarray(pca.components[:k], dtype=np.float64)
    return (
        pca.mean + (centered @ basis.T) @ basis
    ).astype(np.float32)


def variance_fidelity(
    hidden: np.ndarray, reconstructed: np.ndarray, reference_mean: np.ndarray
) -> float:
    values = np.asarray(hidden, dtype=np.float64)
    fitted = np.asarray(reconstructed, dtype=np.float64)
    mean = np.asarray(reference_mean, dtype=np.float64)
    denominator = np.sum((values - mean) ** 2)
    if denominator <= 0:
        return float("nan")
    return float(1.0 - np.sum((values - fitted) ** 2) / denominator)


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment without forming sample Gram matrices."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must contain the same observations")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(
        y.T @ y, ord="fro"
    )
    return float(cross / denominator) if denominator > 0 else float("nan")


def subspace_overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Mean squared cosine between two row-orthonormal bases."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[0]:
        raise ValueError("subspaces must have the same dimension")
    return float(np.linalg.norm(a @ b.T, ord="fro") ** 2 / a.shape[0])


def event_first_mean(
    values: np.ndarray, event_index: np.ndarray
) -> float:
    scores = np.asarray(values, dtype=np.float64)
    events = np.asarray(event_index, dtype=np.int64)
    unique, inverse = np.unique(events, return_inverse=True)
    totals = np.bincount(inverse, weights=scores)
    counts = np.bincount(inverse)
    return float(np.mean(totals / counts)) if len(unique) else float("nan")


@torch.no_grad()
def decode_hidden_nll(
    model: torch.nn.Module,
    contact_features: torch.Tensor,
    local_offset: torch.Tensor,
    hidden: np.ndarray,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    event_index: np.ndarray,
    step: np.ndarray,
    *,
    batch_size: int = 8192,
) -> tuple[float, np.ndarray]:
    """Decode arbitrary frozen states under the model's original action head."""
    model.eval()
    device = next(model.parameters()).device
    features = contact_features.to(device=device, dtype=torch.float32).unsqueeze(0)
    offset = local_offset.to(device=device, dtype=torch.float32)
    mask = torch.ones(
        (1, features.shape[1]), dtype=torch.bool, device=device
    )
    embedding, encoder_input = model._encode(features, offset)
    states = np.asarray(hidden, dtype=np.float32)
    events = np.asarray(event_index, dtype=np.int64)
    steps = np.asarray(step, dtype=np.int64)
    nll_parts: list[np.ndarray] = []
    for start in range(0, len(states), int(batch_size)):
        stop = min(start + int(batch_size), len(states))
        current_events = events[start:stop]
        current_steps = steps[start:stop]
        current_groups = np.asarray(group_ids[current_events], dtype=np.int64)
        current_counts = np.asarray(group_count[current_events], dtype=np.int64)
        recruited = (current_groups >= 0) & (
            current_groups < current_steps[:, None]
        )
        candidate = torch.as_tensor(
            ~recruited, dtype=torch.bool, device=device
        )
        state = torch.as_tensor(
            states[start:stop], dtype=torch.float32, device=device
        )
        batch_embedding = embedding.expand(len(state), -1, -1)
        batch_input = encoder_input.expand(len(state), -1, -1)
        contact_logits, stop_logits = model._decode(
            batch_embedding, batch_input, state, candidate
        )
        denominator = torch.logsumexp(
            torch.cat([stop_logits[:, None], contact_logits], dim=1), dim=1
        )
        terminal = torch.as_tensor(
            current_steps == current_counts, dtype=torch.bool, device=device
        )
        target = torch.as_tensor(
            current_groups == current_steps[:, None],
            dtype=torch.bool,
            device=device,
        )
        contact_numerator = torch.logsumexp(
            contact_logits.masked_fill(~target, -1.0e9), dim=1
        )
        numerator = torch.where(terminal, stop_logits, contact_numerator)
        nll_parts.append((denominator - numerator).cpu().numpy())
    per_prefix = np.concatenate(nll_parts).astype(np.float64)
    return event_first_mean(per_prefix, events), per_prefix


def random_orthonormal_bases(
    dimension: int, rank: int, seeds: Iterable[int]
) -> list[np.ndarray]:
    bases = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        matrix = rng.standard_normal((int(dimension), int(rank)))
        q, _ = np.linalg.qr(matrix)
        bases.append(q.T.astype(np.float32))
    return bases
