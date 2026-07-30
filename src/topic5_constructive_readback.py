"""Train-only read-back metrics for constructive within-event generation.

The helpers in this module never alter or train the event generator.  They
define unsupervised propagation modes and a physical displacement axis from
chronological train80 human events, then score held-out human or generated
events in those frozen coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.lagpat_rank_audit import (
    build_masked_kmeans_features,
    mask_phantom_ranks,
)


def group_feature_matrix(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=int)
    if groups.ndim != 2:
        raise ValueError("group_ids must be event x contact")
    return build_masked_kmeans_features(
        groups.T.astype(float),
        (groups >= 0).T,
        impute="event_median",
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.sum(valid) < 3:
        return float("nan")
    if np.nanstd(left[valid]) == 0 or np.nanstd(right[valid]) == 0:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def _cluster_templates(
    group_ids: np.ndarray,
    labels: np.ndarray,
    *,
    n_clusters: int = 2,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=int)
    labels = np.asarray(labels, dtype=int)
    masked = mask_phantom_ranks(
        groups.T.astype(float),
        (groups >= 0).T,
        normalize=True,
    ).T
    templates = np.full((int(n_clusters), groups.shape[1]), np.nan, dtype=float)
    for cluster in range(int(n_clusters)):
        selected = masked[labels == cluster]
        if selected.size:
            with np.errstate(invalid="ignore"):
                templates[cluster] = np.nanmean(selected, axis=0)
    return templates


def _match_centers(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(
        np.asarray(reference)[:, None, :] - np.asarray(candidate)[None, :, :],
        axis=2,
    )
    row, col = linear_sum_assignment(distance)
    mapping = np.empty(candidate.shape[0], dtype=int)
    mapping[col] = row
    return mapping


@dataclass(frozen=True)
class TrainModeReadback:
    kmeans: KMeans
    templates: np.ndarray
    silhouette: float
    minimum_cluster_fraction: float
    cross_half_ari: float
    train_template_correlation: float
    reliable: bool


def fit_train_mode_readback(
    train_groups: np.ndarray,
    *,
    random_state: int = 0,
) -> TrainModeReadback:
    groups = np.asarray(train_groups, dtype=int)
    features = group_feature_matrix(groups)
    if groups.shape[0] < 20:
        raise ValueError("at least 20 train events are required")
    model = KMeans(n_clusters=2, n_init=20, random_state=int(random_state))
    labels = model.fit_predict(features)
    counts = np.bincount(labels, minlength=2)
    minimum_fraction = float(np.min(counts) / labels.size)
    # Silhouette is a read-back reliability diagnostic, not a fitted target.
    # Bound its pairwise-distance memory while retaining a deterministic,
    # patient-balanced estimate; KMeans itself still uses every train event.
    silhouette_sample = min(features.shape[0], 5000)
    silhouette = (
        float(
            silhouette_score(
                features,
                labels,
                sample_size=silhouette_sample,
                random_state=int(random_state),
            )
        )
        if np.unique(labels).size == 2
        else float("nan")
    )
    templates = _cluster_templates(groups, labels)
    template_correlation = _safe_spearman(templates[0], templates[1])

    midpoint = groups.shape[0] // 2
    first = features[:midpoint]
    second = features[midpoint:]
    first_model = KMeans(
        n_clusters=2, n_init=20, random_state=int(random_state)
    ).fit(first)
    second_model = KMeans(
        n_clusters=2, n_init=20, random_state=int(random_state)
    ).fit(second)
    second_mapping = _match_centers(
        first_model.cluster_centers_,
        second_model.cluster_centers_,
    )
    second_native = second_mapping[second_model.labels_]
    second_cross = first_model.predict(second)
    first_mapping = _match_centers(
        second_model.cluster_centers_,
        first_model.cluster_centers_,
    )
    first_native = first_mapping[first_model.labels_]
    first_cross = second_model.predict(first)
    cross_half_ari = float(
        np.mean(
            [
                adjusted_rand_score(second_native, second_cross),
                adjusted_rand_score(first_native, first_cross),
            ]
        )
    )
    reliable = bool(
        np.isfinite(silhouette)
        and silhouette >= 0.10
        and minimum_fraction >= 0.10
        and cross_half_ari >= 0.10
    )
    return TrainModeReadback(
        kmeans=model,
        templates=templates,
        silhouette=silhouette,
        minimum_cluster_fraction=minimum_fraction,
        cross_half_ari=cross_half_ari,
        train_template_correlation=template_correlation,
        reliable=reliable,
    )


def evaluate_mode_readback(
    readback: TrainModeReadback,
    group_ids: np.ndarray,
) -> dict[str, float | np.ndarray]:
    groups = np.asarray(group_ids, dtype=int)
    features = group_feature_matrix(groups)
    distance = readback.kmeans.transform(features)
    labels = np.argmin(distance, axis=1)
    sorted_distance = np.sort(distance, axis=1)
    margin = (sorted_distance[:, 1] - sorted_distance[:, 0]) / np.maximum(
        sorted_distance.sum(axis=1), 1e-12
    )
    templates = _cluster_templates(groups, labels)
    template_match = np.asarray(
        [
            _safe_spearman(templates[cluster], readback.templates[cluster])
            for cluster in range(2)
        ],
        dtype=float,
    )
    return {
        "labels": labels,
        "templates": templates,
        "mode1_fraction": float(np.mean(labels == 1)),
        "assignment_margin_mean": float(np.mean(margin)),
        "template_match_to_train": float(np.nanmean(template_match)),
    }


def mode_distribution_errors(
    reference: dict[str, float | np.ndarray],
    candidate: dict[str, float | np.ndarray],
) -> dict[str, float]:
    ref_templates = np.asarray(reference["templates"], dtype=float)
    candidate_templates = np.asarray(candidate["templates"], dtype=float)
    correlations = np.asarray(
        [
            _safe_spearman(ref_templates[cluster], candidate_templates[cluster])
            for cluster in range(2)
        ]
    )
    return {
        "mode_prevalence_error": float(
            abs(
                float(candidate["mode1_fraction"])
                - float(reference["mode1_fraction"])
            )
        ),
        "mode_margin_error": float(
            abs(
                float(candidate["assignment_margin_mean"])
                - float(reference["assignment_margin_mean"])
            )
        ),
        "template_error": float(1.0 - np.nanmean(correlations)),
        "template_correlation": float(np.nanmean(correlations)),
    }


def first_order_transition(
    group_ids: np.ndarray,
    *,
    event_chunk_size: int = 1024,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=int)
    n_contacts = groups.shape[1]
    count = np.zeros((n_contacts, n_contacts), dtype=float)
    for start in range(0, groups.shape[0], int(event_chunk_size)):
        chunk = groups[start : start + int(event_chunk_size)]
        source_rank = chunk[:, :, None]
        target_rank = chunk[:, None, :]
        adjacent = (source_rank >= 0) & (target_rank == source_rank + 1)
        # Rank ties split one transition unit over the Cartesian product of
        # source and target sets, exactly matching the event-loop definition.
        same_rank = chunk[:, :, None] == chunk[:, None, :]
        rank_size = np.sum(same_rank & (source_rank >= 0), axis=2)
        denominator = (
            rank_size[:, :, None].astype(float)
            * rank_size[:, None, :].astype(float)
        )
        weighted = np.divide(
            adjacent,
            denominator,
            out=np.zeros(adjacent.shape, dtype=float),
            where=adjacent & (denominator > 0),
        )
        count += weighted.sum(axis=0)
    denominator = count.sum(axis=1, keepdims=True)
    return np.divide(count, denominator, out=np.zeros_like(count), where=denominator > 0)


def transition_errors(
    observed_groups: np.ndarray,
    candidate_groups: np.ndarray,
) -> dict[str, float]:
    observed = first_order_transition(observed_groups)
    candidate = first_order_transition(candidate_groups)
    valid = (observed.sum(axis=1) > 0) | (candidate.sum(axis=1) > 0)
    valid_matrix = np.broadcast_to(valid[:, None], observed.shape)
    if not np.any(valid_matrix):
        return {
            "transition_mae": float("nan"),
            "transition_correlation": float("nan"),
        }
    left = observed[valid_matrix]
    right = candidate[valid_matrix]
    return {
        "transition_mae": float(np.mean(np.abs(left - right))),
        "transition_correlation": _safe_spearman(left, right),
    }


def source_sink_displacements(
    group_ids: np.ndarray,
    group_count: np.ndarray,
    coords: np.ndarray,
) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=int)
    counts = np.asarray(group_count, dtype=int)
    coords = np.asarray(coords, dtype=float)
    if coords.shape != (groups.shape[1], 3) or not np.all(np.isfinite(coords)):
        return np.empty((0, 3), dtype=float)
    output = []
    for event, length in zip(groups, counts):
        source = np.flatnonzero(event == 0)
        sink = np.flatnonzero(event == int(length) - 1)
        if source.size and sink.size:
            vector = np.mean(coords[sink], axis=0) - np.mean(
                coords[source], axis=0
            )
            if np.linalg.norm(vector) > 1e-8:
                output.append(vector)
    return np.asarray(output, dtype=float).reshape(-1, 3)


@dataclass(frozen=True)
class TrainAxisReadback:
    axis: np.ndarray
    explained_variance_fraction: float
    projection_scale: float
    reliable: bool
    n_train_vectors: int


def fit_train_axis_readback(
    train_groups: np.ndarray,
    train_count: np.ndarray,
    coords: np.ndarray,
) -> TrainAxisReadback:
    vectors = source_sink_displacements(train_groups, train_count, coords)
    if vectors.shape[0] < 2:
        return TrainAxisReadback(
            axis=np.full(3, np.nan),
            explained_variance_fraction=float("nan"),
            projection_scale=float("nan"),
            reliable=False,
            n_train_vectors=int(vectors.shape[0]),
        )
    covariance = vectors.T @ vectors / vectors.shape[0]
    eigenvalue, eigenvector = np.linalg.eigh(covariance)
    axis = eigenvector[:, int(np.argmax(eigenvalue))]
    anchor = int(np.argmax(np.abs(axis)))
    if axis[anchor] < 0:
        axis = -axis
    explained = float(np.max(eigenvalue) / max(np.sum(eigenvalue), 1e-12))
    projection = vectors @ axis
    scale = float(max(np.percentile(projection, 90) - np.percentile(projection, 10), 1.0))
    return TrainAxisReadback(
        axis=axis,
        explained_variance_fraction=explained,
        projection_scale=scale,
        reliable=bool(vectors.shape[0] >= 50 and explained >= 0.50),
        n_train_vectors=int(vectors.shape[0]),
    )


def evaluate_axis_readback(
    readback: TrainAxisReadback,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    coords: np.ndarray,
) -> dict[str, float | np.ndarray]:
    vectors = source_sink_displacements(group_ids, group_count, coords)
    if vectors.size == 0 or not np.all(np.isfinite(readback.axis)):
        return {
            "projection": np.asarray([], dtype=float),
            "axis_concentration": float("nan"),
            "positive_fraction": float("nan"),
            "negative_count": 0,
            "positive_count": 0,
        }
    projection = vectors @ readback.axis / readback.projection_scale
    norm = np.linalg.norm(vectors, axis=1)
    return {
        "projection": projection,
        "axis_concentration": float(
            np.mean(np.abs(vectors @ readback.axis) / np.maximum(norm, 1e-12))
        ),
        "positive_fraction": float(np.mean(projection > 0)),
        "negative_count": int(np.sum(projection < 0)),
        "positive_count": int(np.sum(projection > 0)),
    }


def axis_distribution_errors(
    reference: dict[str, float | np.ndarray],
    candidate: dict[str, float | np.ndarray],
) -> dict[str, float]:
    observed = np.asarray(reference["projection"], dtype=float)
    predicted = np.asarray(candidate["projection"], dtype=float)
    if observed.size == 0 or predicted.size == 0:
        return {
            "signed_axis_wasserstein": float("nan"),
            "axis_concentration_error": float("nan"),
            "axis_side_prevalence_error": float("nan"),
        }
    return {
        "signed_axis_wasserstein": float(
            wasserstein_distance(observed, predicted)
        ),
        "axis_concentration_error": float(
            abs(
                float(reference["axis_concentration"])
                - float(candidate["axis_concentration"])
            )
        ),
        "axis_side_prevalence_error": float(
            abs(
                float(reference["positive_fraction"])
                - float(candidate["positive_fraction"])
            )
        ),
    }
