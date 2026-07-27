"""Utilities for Topic-5 axis-positive read-back and static transfer v2.4."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


EPS = 1.0e-12


def unit_vector(vector: np.ndarray | Iterable[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("axis must be a finite 3-vector")
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        raise ValueError("axis norm must be positive")
    return value / norm


def sign_invariant_cosine(
    first: np.ndarray | Iterable[float],
    second: np.ndarray | Iterable[float],
) -> float:
    return float(abs(np.dot(unit_vector(first), unit_vector(second))))


def candidate_alignment_summary(
    selected: np.ndarray,
    reference: np.ndarray,
    candidates: np.ndarray,
) -> dict[str, float]:
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidates must have shape [directions, 3]")
    values = np.asarray(
        [sign_invariant_cosine(axis, reference) for axis in candidates],
        dtype=np.float64,
    )
    observed = sign_invariant_cosine(selected, reference)
    return {
        "selected_abs_cosine": observed,
        "candidate_median_abs_cosine": float(np.median(values)),
        "alignment_margin": float(observed - np.median(values)),
        "candidate_empirical_p_upper": float(
            (1 + np.count_nonzero(values >= observed)) / (len(values) + 1)
        ),
        "candidate_percentile": float(np.mean(values <= observed)),
    }


def normalized_rank_distribution(groups: np.ndarray) -> np.ndarray:
    """Return [contact, nonparticipation + 10 joint normalized-rank bins]."""
    groups = np.asarray(groups, dtype=np.int64)
    if groups.ndim != 2 or len(groups) == 0:
        raise ValueError("groups must have shape [events, contacts] and be non-empty")
    n_events, n_contacts = groups.shape
    counts = np.zeros((n_contacts, 11), dtype=np.float64)
    for event in groups:
        participating = event >= 0
        counts[~participating, 0] += 1.0
        if not np.any(participating):
            continue
        maximum = int(event[participating].max())
        denominator = max(maximum, 1)
        for contact in np.flatnonzero(participating):
            position = float(event[contact]) / denominator
            rank_bin = min(int(np.floor(position * 10.0)), 9)
            counts[contact, 1 + rank_bin] += 1.0
    distribution = counts / float(n_events)
    if not np.allclose(distribution.sum(axis=1), 1.0, atol=1.0e-10):
        raise RuntimeError("rank-distribution rows do not sum to one")
    return distribution


def empirical_rank_distribution(
    groups: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    groups = np.asarray(groups, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("indices must be a non-empty vector")
    return normalized_rank_distribution(groups[indices])


def paired_rollout_design(
    groups: np.ndarray,
    train_indices: np.ndarray,
    *,
    n_rollouts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample train-event source/length priors and paired categorical uniforms."""
    groups = np.asarray(groups, dtype=np.int64)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    if n_rollouts < 1 or len(train_indices) < 1:
        raise ValueError("rollouts and train indices must be non-empty")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(train_indices, size=n_rollouts, replace=True)
    uniforms = rng.random((n_rollouts, groups.shape[1]), dtype=np.float64)
    return sampled.astype(np.int64), uniforms


def rollout_model_distribution(
    model: torch.nn.Module,
    groups: np.ndarray,
    sampled_events: np.ndarray,
    uniforms: np.ndarray,
    *,
    node_only: bool = False,
) -> np.ndarray:
    """Free-roll a fitted v2.3 model using paired empirical source/length priors."""
    groups = np.asarray(groups, dtype=np.int64)
    sampled_events = np.asarray(sampled_events, dtype=np.int64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if uniforms.shape != (len(sampled_events), groups.shape[1]):
        raise ValueError("uniform stream shape mismatch")

    device = model.node_logit.device
    dtype = model.node_logit.dtype
    symmetric, directed = model.operators()
    template = torch.as_tensor(
        groups[sampled_events], dtype=torch.long, device=device
    )
    simulated = torch.full_like(template, -1)
    source = template == 0
    simulated[source] = 0
    maximum_rank = template.max(dim=1).values
    propagation = torch.zeros(
        simulated.shape, dtype=dtype, device=device
    )
    competition = torch.zeros_like(propagation)
    source_count = source.sum(dim=1).clamp_min(1).to(dtype)
    source_projection = (
        source.to(dtype) * model.projection[None, :]
    ).sum(dim=1) / source_count
    source_scale = model.projection.std(unbiased=False).clamp_min(1.0e-8)
    source_score = torch.tanh(source_projection / source_scale)
    uniform_tensor = torch.as_tensor(
        uniforms, dtype=dtype, device=device
    )
    with torch.no_grad():
        for step in range(int(maximum_rank.max().item())):
            active = maximum_rank > step
            current = simulated == step
            current_count = current.sum(dim=1).clamp_min(1).to(dtype)
            x = current.to(dtype) / current_count[:, None]
            propagation = model.rho_propagation * propagation + x @ symmetric
            competition = model.rho_competition * competition + x @ symmetric
            if node_only:
                score = model.node_logit[None, :].expand_as(propagation)
            else:
                score = (
                    model.node_logit[None, :]
                    + model.gain_propagation * propagation
                    - model.gain_competition * competition
                    + model.source_beta
                    * source_score[:, None]
                    * (x @ directed)
                )
            eligible = simulated < 0
            masked = score.masked_fill(~eligible, -torch.inf)
            probabilities = torch.softmax(masked, dim=1)
            cdf = torch.cumsum(probabilities, dim=1)
            selected = torch.sum(
                cdf < uniform_tensor[:, step, None], dim=1
            ).clamp_max(groups.shape[1] - 1)
            active_rows = torch.nonzero(active, as_tuple=False).flatten()
            simulated[active_rows, selected[active_rows]] = step + 1
    return normalized_rank_distribution(simulated.detach().cpu().numpy())


def robust_patient_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        raise ValueError("at least two finite values are required")
    median = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - median)))
    scale = 1.4826 * mad
    if scale <= EPS:
        scale = float(np.std(values[finite]))
    if scale <= EPS:
        raise ValueError("patient target is constant")
    result = np.full_like(values, np.nan, dtype=np.float64)
    result[finite] = (values[finite] - median) / scale
    return result


def weighted_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    test_x: np.ndarray,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit a weighted ridge with training-only standardization and free intercept."""
    train_x = np.asarray(train_x, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.float64)
    train_weight = np.asarray(train_weight, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    if (
        train_x.ndim != 2
        or test_x.ndim != 2
        or train_x.shape[1] != test_x.shape[1]
        or train_y.shape != (len(train_x),)
        or train_weight.shape != (len(train_x),)
        or np.any(train_weight <= 0)
    ):
        raise ValueError("invalid weighted-ridge arrays")
    total_weight = float(train_weight.sum())
    mean = np.sum(train_x * train_weight[:, None], axis=0) / total_weight
    variance = (
        np.sum(
            (train_x - mean) ** 2 * train_weight[:, None], axis=0
        )
        / total_weight
    )
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= EPS] = 1.0
    standardized = (train_x - mean) / scale
    test_standardized = (test_x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    test_design = np.column_stack(
        [np.ones(len(test_standardized)), test_standardized]
    )
    root_weight = np.sqrt(train_weight)
    weighted_design = design * root_weight[:, None]
    weighted_target = train_y * root_weight
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ weighted_target,
    )
    return test_design @ coefficients
