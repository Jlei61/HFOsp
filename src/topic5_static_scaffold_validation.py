"""Utilities for fixed-readout static-scaffold validation."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

from .propagation_skeleton_geometry import parse_shaft


def centered_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    ranked = rankdata(values)
    return ranked - ranked.mean()


def shaft_groups(names: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(np.asarray(names).astype(str)):
        parsed, _ = parse_shaft(name)
        grouped[str(parsed or name)].append(index)
    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in grouped.items()
    }


def coherent_index_null(
    names: np.ndarray,
    *,
    n_draws: int,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create coherent patient-level field permutations.

    Returned rows index a field vector. The same row is intended to be used for
    all seizures and all compared models in one patient.
    """
    names = np.asarray(names).astype(str)
    rng = np.random.default_rng(int(seed))
    n_contacts = len(names)
    identity = np.arange(n_contacts, dtype=np.int64)
    groups = shaft_groups(names)
    output = np.tile(identity, (n_draws, 1))
    if mode == "all_contact":
        for draw in range(n_draws):
            output[draw] = rng.permutation(n_contacts)
        return output, {
            "eligible": True,
            "mode": mode,
            "n_shafts": len(groups),
            "movable_fraction": 1.0,
        }

    movable = sum(len(indices) for indices in groups.values() if len(indices) >= 2)
    if mode in {"within_shaft_circular", "within_shaft_dihedral"}:
        eligible = movable >= 4 and movable / max(n_contacts, 1) >= 0.5
        for draw in range(n_draws):
            changed = False
            for indices in groups.values():
                length = len(indices)
                if length < 2:
                    continue
                shift = int(rng.integers(0, length))
                reverse = bool(rng.integers(0, 2)) if mode.endswith("dihedral") else False
                source = indices[::-1] if reverse else indices
                source = np.roll(source, shift)
                output[draw, indices] = source
                changed |= bool(shift or reverse)
            if not changed:
                candidates = [indices for indices in groups.values() if len(indices) >= 2]
                chosen = candidates[int(rng.integers(0, len(candidates)))]
                output[draw, chosen] = np.roll(chosen, 1)
        return output, {
            "eligible": bool(eligible),
            "mode": mode,
            "n_shafts": len(groups),
            "movable_fraction": float(movable / max(n_contacts, 1)),
        }

    if mode == "equal_size_shaft_profile":
        by_size: dict[int, list[np.ndarray]] = defaultdict(list)
        for indices in groups.values():
            by_size[len(indices)].append(indices)
        exchangeable = sum(
            size * len(bucket)
            for size, bucket in by_size.items()
            if len(bucket) >= 2
        )
        eligible = exchangeable >= 4 and exchangeable / max(n_contacts, 1) >= 0.5
        for draw in range(n_draws):
            for bucket in by_size.values():
                if len(bucket) < 2:
                    continue
                mapping = rng.permutation(len(bucket))
                for destination, source_index in zip(bucket, mapping):
                    output[draw, destination] = bucket[int(source_index)]
        return output, {
            "eligible": bool(eligible),
            "mode": mode,
            "n_shafts": len(groups),
            "movable_fraction": float(exchangeable / max(n_contacts, 1)),
        }
    raise ValueError(f"unknown null mode: {mode}")


def fit_rbf_lengthscale(field: np.ndarray, coords: np.ndarray) -> float:
    """Estimate an RBF scale from the interictal field only."""
    field = np.asarray(field, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    distances = pdist(coords)
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("degenerate contact geometry")
    semivariance = 0.5 * pdist(field[:, None], metric="sqeuclidean")
    base = float(np.median(positive))
    candidates = base * np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
    best_scale = candidates[0]
    best_loss = np.inf
    for scale in candidates:
        basis = 1.0 - np.exp(-(distances**2) / (2.0 * scale**2))
        denominator = float(basis @ basis)
        amplitude = (
            max(float(basis @ semivariance) / denominator, 0.0)
            if denominator > 0
            else 0.0
        )
        loss = float(np.mean((semivariance - amplitude * basis) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_scale = float(scale)
    return best_scale


def geometry_smooth_surrogates(
    field: np.ndarray,
    coords: np.ndarray,
    *,
    standard_normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Generate target-blind, rank-matched smooth field surrogates."""
    field = np.asarray(field, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    standard_normal = np.asarray(standard_normal, dtype=np.float64)
    if standard_normal.shape[1] != len(field):
        raise ValueError("standard-normal/contact shape mismatch")
    if not np.all(np.isfinite(coords)):
        raise ValueError("non-finite geometry")
    lengthscale = fit_rbf_lengthscale(field, coords)
    distance = squareform(pdist(coords))
    covariance = np.exp(-(distance**2) / (2.0 * lengthscale**2))
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    transform = eigenvector @ np.diag(np.sqrt(np.maximum(eigenvalue, 1.0e-8)))
    latent = standard_normal @ transform.T
    order = np.argsort(latent, axis=1)
    sorted_field = np.sort(field)
    result = np.empty_like(latent)
    np.put_along_axis(
        result,
        order,
        np.broadcast_to(sorted_field, result.shape),
        axis=1,
    )
    return result, lengthscale


def score_signed_field(
    field: np.ndarray,
    target: np.ndarray,
    null_fields: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Score a fixed field and coherent null fields across seizures."""
    field = np.asarray(field, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    null_fields = np.asarray(null_fields, dtype=np.float64)
    target_rank = np.row_stack([centered_rank(row) for row in target])
    target_norm = np.linalg.norm(target_rank, axis=1)
    field_rank = centered_rank(field)
    field_norm = float(np.linalg.norm(field_rank))
    if field_norm <= 0:
        raise ValueError("constant fixed field")
    observed_per_seizure = (
        target_rank @ field_rank
    ) / np.maximum(target_norm * field_norm, 1.0e-12)
    null_rank = np.row_stack([centered_rank(row) for row in null_fields])
    null_norm = np.linalg.norm(null_rank, axis=1)
    correlations = (null_rank @ target_rank.T) / np.maximum(
        null_norm[:, None] * target_norm[None, :], 1.0e-12
    )
    return {
        "observed_signed": float(np.median(observed_per_seizure)),
        "observed_absolute": float(np.median(np.abs(observed_per_seizure))),
        "null_signed": np.median(correlations, axis=1),
        "null_absolute": np.median(np.abs(correlations), axis=1),
    }


def participation_rate(group_ids: np.ndarray) -> np.ndarray:
    """Return the event-first empirical participation rate per contact."""
    groups = np.asarray(group_ids)
    if groups.ndim != 2 or not len(groups):
        raise ValueError("group_ids must be a nonempty [event, contact] matrix")
    return np.mean(groups >= 0, axis=0, dtype=np.float64)


def event_brier(
    field: np.ndarray, group_ids: np.ndarray
) -> float:
    """Event-first mean contact Brier score for a fixed participation field."""
    field = np.asarray(field, dtype=np.float64)
    observed = (np.asarray(group_ids) >= 0).astype(np.float64)
    if observed.ndim != 2 or observed.shape[1] != len(field):
        raise ValueError("field/group_ids contact shape mismatch")
    return float(np.mean(np.mean((observed - field[None, :]) ** 2, axis=1)))


def beta_binomial_participation(
    group_ids: np.ndarray, concentration: float
) -> np.ndarray:
    """Shrink contact rates toward the patient-wide event participation rate."""
    groups = np.asarray(group_ids)
    rate = participation_rate(groups)
    prior_mean = float(np.mean(rate))
    concentration = float(concentration)
    if concentration < 0:
        raise ValueError("concentration must be nonnegative")
    return (
        np.sum(groups >= 0, axis=0, dtype=np.float64)
        + concentration * prior_mean
    ) / (len(groups) + concentration)


def contact_graph(
    names: np.ndarray,
    *,
    coords: np.ndarray | None = None,
    mode: str,
) -> np.ndarray:
    """Build a target-blind shaft or geometry graph."""
    names = np.asarray(names).astype(str)
    n_contacts = len(names)
    weight = np.zeros((n_contacts, n_contacts), dtype=np.float64)
    if mode == "shaft":
        for indices in shaft_groups(names).values():
            if len(indices) < 2:
                continue
            for left, right in zip(indices[:-1], indices[1:]):
                weight[int(left), int(right)] = 1.0
                weight[int(right), int(left)] = 1.0
    elif mode == "geometry":
        if coords is None:
            raise ValueError("geometry graph requires coordinates")
        coords = np.asarray(coords, dtype=np.float64)
        if coords.shape != (n_contacts, 3) or not np.all(np.isfinite(coords)):
            raise ValueError("coordinates must be finite [contact, 3]")
        distance = squareform(pdist(coords))
        positive = distance[distance > 0]
        if not len(positive):
            raise ValueError("degenerate contact geometry")
        nearest = np.where(distance > 0, distance, np.inf).min(axis=1)
        scale = float(np.median(nearest[np.isfinite(nearest)]))
        weight = np.exp(-(distance**2) / (2.0 * max(scale, 1.0e-8) ** 2))
        np.fill_diagonal(weight, 0.0)
    else:
        raise ValueError(f"unknown graph mode: {mode}")
    return weight


def laplacian_smooth(
    field: np.ndarray, graph: np.ndarray, penalty: float
) -> np.ndarray:
    """Smooth a contact field with ``(I + penalty * L)^-1``."""
    field = np.asarray(field, dtype=np.float64)
    graph = np.asarray(graph, dtype=np.float64)
    if graph.shape != (len(field), len(field)):
        raise ValueError("graph/field shape mismatch")
    penalty = float(penalty)
    if penalty < 0:
        raise ValueError("penalty must be nonnegative")
    laplacian = np.diag(graph.sum(axis=1)) - graph
    smoothed = np.linalg.solve(
        np.eye(len(field), dtype=np.float64) + penalty * laplacian,
        field,
    )
    return np.clip(smoothed, 1.0e-6, 1.0 - 1.0e-6)


def contact_rank_categories(
    group_ids: np.ndarray, n_rank_bins: int = 10
) -> np.ndarray:
    """Encode nonparticipation as 0 and participating local rank as 1..B."""
    groups = np.asarray(group_ids, dtype=np.int64)
    if groups.ndim != 2:
        raise ValueError("group_ids must be [event, contact]")
    bins = int(n_rank_bins)
    if bins < 2:
        raise ValueError("n_rank_bins must be at least two")
    categories = np.zeros(groups.shape, dtype=np.int16)
    for event_index, event in enumerate(groups):
        participating = event >= 0
        if not np.any(participating):
            continue
        maximum = int(np.max(event[participating]))
        if maximum <= 0:
            local = np.zeros(np.count_nonzero(participating), dtype=np.int64)
        else:
            local = np.floor(
                event[participating] / float(maximum) * bins
            ).astype(np.int64)
        categories[event_index, participating] = 1 + np.clip(
            local, 0, bins - 1
        )
    return categories


def dirichlet_contact_rank_distribution(
    categories: np.ndarray,
    concentration: float,
    *,
    n_rank_bins: int = 10,
) -> np.ndarray:
    """Estimate a smoothed contact x (nonparticipation + rank-bin) table."""
    values = np.asarray(categories, dtype=np.int64)
    if values.ndim != 2 or not len(values):
        raise ValueError("categories must be nonempty [event, contact]")
    n_categories = int(n_rank_bins) + 1
    if np.any((values < 0) | (values >= n_categories)):
        raise ValueError("category lies outside the frozen rank-bin support")
    counts = np.column_stack(
        [np.sum(values == category, axis=0) for category in range(n_categories)]
    ).astype(np.float64)
    pooled = counts.sum(axis=0)
    prior = pooled / max(float(pooled.sum()), 1.0)
    concentration = float(concentration)
    if concentration < 0:
        raise ValueError("concentration must be nonnegative")
    estimate = counts + concentration * prior[None, :]
    estimate /= np.maximum(estimate.sum(axis=1, keepdims=True), 1.0e-12)
    return estimate


def categorical_event_nll(
    distribution: np.ndarray, categories: np.ndarray
) -> float:
    """Event-first categorical NLL for a fixed contact x category table."""
    probability = np.asarray(distribution, dtype=np.float64)
    values = np.asarray(categories, dtype=np.int64)
    if values.ndim != 2 or probability.shape[0] != values.shape[1]:
        raise ValueError("distribution/categories contact shape mismatch")
    selected = probability[
        np.broadcast_to(np.arange(values.shape[1]), values.shape),
        values,
    ]
    per_event = -np.mean(np.log(np.clip(selected, 1.0e-12, 1.0)), axis=1)
    return float(np.mean(per_event))


def partial_rank_score(
    field: np.ndarray,
    target: np.ndarray,
    covariates: np.ndarray,
    *,
    min_residual_df: int = 3,
    n_null_draws: int = 0,
    null_seed: int = 0,
) -> dict[str, Any]:
    """Compute seizure-first partial Spearman scores for one covariate block."""
    field = np.asarray(field, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != len(field):
        raise ValueError("target must be [seizure, contact] aligned to field")
    if covariates.ndim == 1:
        covariates = covariates[:, None]
    if covariates.shape[0] != len(field):
        raise ValueError("covariates/field contact shape mismatch")
    finite = (
        np.isfinite(field)
        & np.all(np.isfinite(target), axis=0)
        & np.all(np.isfinite(covariates), axis=1)
    )
    y = field[finite]
    t = target[:, finite]
    x = covariates[finite]
    if len(y) < 4:
        return {"eligible": False, "reason": "fewer_than_four_contacts"}
    scale = np.std(x, axis=0)
    informative = scale > 1.0e-12
    if not np.any(informative):
        return {
            "eligible": False,
            "reason": "constant_covariate",
            "n_contacts": int(len(y)),
        }
    x = x[:, informative]
    design = np.column_stack([np.ones(len(y)), x])
    design_rank = int(np.linalg.matrix_rank(design))
    residual_df = int(len(y) - design_rank)
    if residual_df < int(min_residual_df):
        return {
            "eligible": False,
            "reason": "insufficient_residual_df",
            "n_contacts": int(len(y)),
            "design_rank": design_rank,
            "residual_df": residual_df,
        }
    projector = design @ np.linalg.pinv(design)
    residualizer = np.eye(len(y)) - projector
    field_residual = residualizer @ centered_rank(y)
    field_norm = float(np.linalg.norm(field_residual))
    if field_norm <= 1.0e-12:
        return {
            "eligible": False,
            "reason": "field_fully_explained_by_covariates",
            "n_contacts": int(len(y)),
            "design_rank": design_rank,
            "residual_df": residual_df,
        }
    scores = []
    for seizure in t:
        target_residual = residualizer @ centered_rank(seizure)
        denominator = field_norm * float(np.linalg.norm(target_residual))
        scores.append(
            float(field_residual @ target_residual / denominator)
            if denominator > 1.0e-12
            else np.nan
        )
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores[np.isfinite(scores)]
    if not len(scores):
        return {
            "eligible": False,
            "reason": "no_finite_seizure_score",
            "n_contacts": int(len(y)),
            "design_rank": design_rank,
            "residual_df": residual_df,
        }
    result: dict[str, Any] = {
        "eligible": True,
        "n_contacts": int(len(y)),
        "design_rank": design_rank,
        "residual_df": residual_df,
        "signed_rho": float(np.median(scores)),
        "absolute_rho": float(np.median(np.abs(scores))),
        "per_seizure_signed_rho": scores,
    }
    if int(n_null_draws) > 0:
        rng = np.random.default_rng(int(null_seed))
        permutations = np.row_stack(
            [rng.permutation(len(y)) for _ in range(int(n_null_draws))]
        )
        permuted = field_residual[permutations] @ residualizer.T
        permuted_norm = np.linalg.norm(permuted, axis=1)
        target_residual = np.row_stack(
            [residualizer @ centered_rank(seizure) for seizure in t]
        )
        target_norm = np.linalg.norm(target_residual, axis=1)
        correlations = (
            permuted @ target_residual.T
        ) / np.maximum(
            permuted_norm[:, None] * target_norm[None, :], 1.0e-12
        )
        null_signed = np.median(correlations, axis=1)
        null_absolute = np.median(np.abs(correlations), axis=1)
        signed_null_median = float(np.median(null_signed))
        absolute_null_median = float(np.median(null_absolute))
        result.update(
            {
                "n_null_draws": int(n_null_draws),
                "signed_null_median": signed_null_median,
                "absolute_null_median": absolute_null_median,
                "signed_margin": float(
                    result["signed_rho"] - signed_null_median
                ),
                "absolute_margin": float(
                    result["absolute_rho"] - absolute_null_median
                ),
                "signed_empirical_p": float(
                    (
                        1
                        + np.count_nonzero(
                            null_signed >= result["signed_rho"]
                        )
                    )
                    / (int(n_null_draws) + 1)
                ),
                "absolute_empirical_p": float(
                    (
                        1
                        + np.count_nonzero(
                            null_absolute >= result["absolute_rho"]
                        )
                    )
                    / (int(n_null_draws) + 1)
                ),
            }
        )
    return result
