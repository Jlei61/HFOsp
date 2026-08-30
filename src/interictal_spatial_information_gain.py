"""Held-out test of spatial information in interictal propagation modes.

The temporal arm reproduces the masked-rank KMeans representation used for
interictal propagation templates.  The hybrid arm adds a patient-specific
single-event propagation direction, but only while fitting the training-fold
clusters.  Held-out events are assigned from rank templates alone and their
directions are opened only for scoring.  This keeps the spatial outcome out of
the held-out classifier.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from src.interictal_propagation import (
    assign_events_to_templates,
    build_cluster_templates,
)
from src.lagpat_rank_audit import build_masked_kmeans_features
from src.topic5_interictal_direction_rose import fit_event_directions_3d


METHOD_TEMPORAL = "timing_only"
METHOD_HYBRID = "timing_plus_space"
ALL_EVENT_SPATIAL_POLICY = "all_events_missing_spatial_view"


def _canonicalize_full_fit(
    ranks: np.ndarray,
    bools: np.ndarray,
    raw_labels: np.ndarray,
    coords: np.ndarray,
) -> Dict[str, Any]:
    """Order a deterministic K=2 fit by prevalence and build its templates."""
    raw_counts = np.bincount(np.asarray(raw_labels, int), minlength=2)
    order = np.asarray(
        sorted(range(2), key=lambda cluster: (-int(raw_counts[cluster]), cluster)),
        int,
    )
    inverse = np.empty(2, int)
    inverse[order] = np.arange(2)
    labels = inverse[np.asarray(raw_labels, int)]
    counts = raw_counts[order]
    templates = build_cluster_templates(ranks, bools, labels, n_clusters=2)
    axes = fit_event_directions_3d(
        templates.T, np.asarray(coords, float), min_contacts=3
    )["directions"]
    if not np.isfinite(axes).all():
        raise ValueError("a full-fit template axis is not estimable")
    supports = np.vstack([
        np.mean(np.asarray(bools, bool)[:, labels == cluster], axis=1)
        for cluster in (0, 1)
    ])
    return {
        "labels": labels,
        "templates": templates,
        "axes": axes,
        "cluster_counts": counts,
        "supports": supports,
        "cluster_order_from_raw_kmeans": order,
        "template_label_rule": "A=more events; B=fewer events",
    }


def fit_full_temporal_template_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    coords: np.ndarray,
    *,
    min_cluster_events: int = 20,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Fit the same-sample Timing-only audit comparator for a full hybrid fit."""
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 2 or bool_values.shape != rank_values.shape:
        raise ValueError("ranks and bools must align as contacts x events")
    if xyz.shape != (rank_values.shape[0], 3):
        raise ValueError("coords must align with contacts")
    temporal = build_masked_kmeans_features(
        rank_values, bool_values, impute="event_median"
    )
    raw_labels = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=int(random_state),
    ).fit_predict(temporal)
    raw_counts = np.bincount(raw_labels, minlength=2)
    if int(raw_counts.min()) < int(min_cluster_events):
        raise ValueError(
            "timing_only full fit: cluster support below "
            f"{min_cluster_events}: {raw_counts.tolist()}"
        )
    return {
        "method": METHOD_TEMPORAL,
        **_canonicalize_full_fit(
            rank_values, bool_values, raw_labels, xyz
        ),
        "fit_role": "same-sample audit comparator for timing_plus_space full fit",
    }


def fit_full_spatial_template_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    coords: np.ndarray,
    *,
    min_cluster_events: int = 20,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Fit the publication-facing Timing+Space templates on all input events.

    This is the full-data counterpart of the training-side hybrid arm used by
    :func:`compute_crossfit_spatial_information_gain`.  It is intentionally not
    a held-out estimate: cross-fit establishes that space adds information,
    whereas this function freezes the final interictal templates for downstream
    projection.  Cluster A is defined as the more prevalent event class, matching
    the manuscript label contract; ties retain deterministic KMeans order.
    """
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 2 or bool_values.shape != rank_values.shape:
        raise ValueError("ranks and bools must align as contacts x events")
    n_contacts, n_events = rank_values.shape
    if direction_values.shape != (n_events, 3):
        raise ValueError("directions must align with events")
    if xyz.shape != (n_contacts, 3):
        raise ValueError("coords must align with contacts")
    if not np.isfinite(direction_values).all():
        raise ValueError("full-fit directions must be finite")

    temporal = build_masked_kmeans_features(
        rank_values, bool_values, impute="event_median"
    )
    hybrid, spatial_scale = build_hybrid_training_features(
        temporal, direction_values
    )
    raw_labels = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=int(random_state),
    ).fit_predict(hybrid)
    raw_counts = np.bincount(raw_labels, minlength=2)
    if int(raw_counts.min()) < int(min_cluster_events):
        raise ValueError(
            "timing_plus_space full fit: cluster support below "
            f"{min_cluster_events}: {raw_counts.tolist()}"
        )

    canonical = _canonicalize_full_fit(
        rank_values, bool_values, raw_labels, xyz
    )
    return {
        "method": METHOD_HYBRID,
        **canonical,
        "spatial_scale": float(spatial_scale),
        "fit_role": "full interictal fit after independent cross-fit validation",
    }


def equal_view_spatial_scale(
    temporal_features: np.ndarray,
    directions: np.ndarray,
) -> float:
    """Scale the spatial block to equal the temporal block's train variance.

    The scale is fitted on a training fold only.  It makes the sum of feature
    variances equal between the full temporal block and the three-dimensional
    direction block, so no tunable spatial weight is selected from outcomes.
    """
    temporal = np.asarray(temporal_features, float)
    spatial = np.asarray(directions, float)
    if temporal.ndim != 2 or spatial.ndim != 2 or spatial.shape[1] != 3:
        raise ValueError("temporal features and directions must be 2-D")
    if temporal.shape[0] != spatial.shape[0] or temporal.shape[0] < 2:
        raise ValueError("temporal features and directions must align")
    if not np.isfinite(temporal).all() or not np.isfinite(spatial).all():
        raise ValueError("training features must be finite")
    temporal_variance = float(np.var(temporal, axis=0, ddof=0).sum())
    spatial_variance = float(np.var(spatial, axis=0, ddof=0).sum())
    if temporal_variance <= 1e-12 or spatial_variance <= 1e-12:
        raise ValueError("both feature views must have non-zero variance")
    return float(np.sqrt(temporal_variance / spatial_variance))


def build_hybrid_training_features(
    temporal_features: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Append an equal-total-variance spatial direction block."""
    scale = equal_view_spatial_scale(temporal_features, directions)
    hybrid = np.column_stack(
        [np.asarray(temporal_features, float), scale * np.asarray(directions, float)]
    )
    return hybrid, scale


def _fit_missing_view_hybrid_labels(
    temporal_features: np.ndarray,
    directions: np.ndarray,
    *,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Fit K=2 while allowing the spatial view to be absent per event.

    Every event contributes its temporal distance.  An event contributes an
    additional spatial distance only when its direction is finite.  Spatial
    centroids are consequently estimated from the direction-estimable members
    of each cluster, whereas temporal centroids use every cluster member.  This
    is a masked-view objective; missing directions are never encoded as zero
    vectors and no event is removed from clustering.
    """
    temporal = np.asarray(temporal_features, float)
    spatial = np.asarray(directions, float)
    if temporal.ndim != 2 or spatial.shape != (temporal.shape[0], 3):
        raise ValueError("temporal features and directions must align")
    if temporal.shape[0] < 2 or not np.isfinite(temporal).all():
        raise ValueError("all-event temporal features must be finite")
    spatial_valid = np.isfinite(spatial).all(axis=1)
    if int(spatial_valid.sum()) < 2:
        raise ValueError("fewer than two events have an estimable spatial view")

    # Fit the view weight only on events for which both views are observed.
    spatial_scale = equal_view_spatial_scale(
        temporal[spatial_valid], spatial[spatial_valid]
    )
    scaled_spatial = np.full_like(spatial, np.nan, dtype=float)
    scaled_spatial[spatial_valid] = spatial_scale * spatial[spatial_valid]

    best: Dict[str, Any] | None = None
    for init in range(int(n_init)):
        labels = KMeans(
            n_clusters=2,
            n_init=1,
            random_state=int(random_state) + init,
        ).fit_predict(temporal)
        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            counts = np.bincount(labels, minlength=2)
            spatial_counts = np.asarray([
                np.sum((labels == cluster) & spatial_valid)
                for cluster in (0, 1)
            ], int)
            if int(counts.min()) == 0 or int(spatial_counts.min()) == 0:
                break
            temporal_centers = np.vstack([
                temporal[labels == cluster].mean(axis=0)
                for cluster in (0, 1)
            ])
            spatial_centers = np.vstack([
                scaled_spatial[(labels == cluster) & spatial_valid].mean(axis=0)
                for cluster in (0, 1)
            ])
            distances = np.stack([
                np.sum((temporal - temporal_centers[cluster]) ** 2, axis=1)
                for cluster in (0, 1)
            ], axis=1)
            for cluster in (0, 1):
                distances[spatial_valid, cluster] += np.sum(
                    (
                        scaled_spatial[spatial_valid]
                        - spatial_centers[cluster]
                    ) ** 2,
                    axis=1,
                )
            updated = np.argmin(distances, axis=1)
            if np.array_equal(updated, labels):
                labels = updated
                break
            labels = updated

        counts = np.bincount(labels, minlength=2)
        spatial_counts = np.asarray([
            np.sum((labels == cluster) & spatial_valid)
            for cluster in (0, 1)
        ], int)
        if int(counts.min()) == 0 or int(spatial_counts.min()) == 0:
            continue
        temporal_centers = np.vstack([
            temporal[labels == cluster].mean(axis=0)
            for cluster in (0, 1)
        ])
        spatial_centers = np.vstack([
            scaled_spatial[(labels == cluster) & spatial_valid].mean(axis=0)
            for cluster in (0, 1)
        ])
        distances = np.stack([
            np.sum((temporal - temporal_centers[cluster]) ** 2, axis=1)
            for cluster in (0, 1)
        ], axis=1)
        for cluster in (0, 1):
            distances[spatial_valid, cluster] += np.sum(
                (scaled_spatial[spatial_valid] - spatial_centers[cluster]) ** 2,
                axis=1,
            )
        objective = float(np.sum(distances[np.arange(len(labels)), labels]))
        candidate = {
            "labels": labels.copy(),
            "temporal_centers": temporal_centers,
            "spatial_centers_scaled": spatial_centers,
            "cluster_counts": counts,
            "cluster_spatial_counts": spatial_counts,
            "spatial_valid": spatial_valid,
            "spatial_scale": float(spatial_scale),
            "objective": objective,
            "n_iter": int(iterations),
        }
        if best is None or objective < float(best["objective"]) - 1e-12:
            best = candidate
    if best is None:
        raise ValueError("masked-view KMeans could not form two spatially supported clusters")
    return best


def fit_full_all_event_spatial_template_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    coords: np.ndarray,
    *,
    min_cluster_events: int = 20,
    min_spatial_cluster_events: int = 3,
    random_state: int = 0,
) -> Dict[str, Any]:
    """Fit final Timing+Space templates without deleting spatial-missing events."""
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 2 or bool_values.shape != rank_values.shape:
        raise ValueError("ranks and bools must align as contacts x events")
    if direction_values.shape != (rank_values.shape[1], 3):
        raise ValueError("directions must align with events")
    if xyz.shape != (rank_values.shape[0], 3):
        raise ValueError("coords must align with contacts")
    temporal = build_masked_kmeans_features(
        rank_values, bool_values, impute="event_median"
    )
    fitted = _fit_missing_view_hybrid_labels(
        temporal,
        direction_values,
        random_state=random_state,
    )
    counts = np.asarray(fitted["cluster_counts"], int)
    spatial_counts = np.asarray(fitted["cluster_spatial_counts"], int)
    if int(counts.min()) < int(min_cluster_events):
        raise ValueError(
            "all-event timing_plus_space full fit: cluster support below "
            f"{min_cluster_events}: {counts.tolist()}"
        )
    if int(spatial_counts.min()) < int(min_spatial_cluster_events):
        raise ValueError(
            "all-event timing_plus_space full fit: spatial support below "
            f"{min_spatial_cluster_events}: {spatial_counts.tolist()}"
        )
    canonical = _canonicalize_full_fit(
        rank_values,
        bool_values,
        np.asarray(fitted["labels"], int),
        xyz,
    )
    order = np.asarray(canonical["cluster_order_from_raw_kmeans"], int)
    return {
        "method": METHOD_HYBRID,
        **canonical,
        "cluster_spatial_counts": spatial_counts[order],
        "n_spatial_view_events": int(np.sum(fitted["spatial_valid"])),
        "n_spatial_missing_events": int(np.sum(~fitted["spatial_valid"])),
        "spatial_scale": float(fitted["spatial_scale"]),
        "masked_view_objective": float(fitted["objective"]),
        "masked_view_n_iter": int(fitted["n_iter"]),
        "event_policy": ALL_EVENT_SPATIAL_POLICY,
        "fit_role": "all-interictal-event fit with an optional spatial view",
    }


def _fit_mode_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    coords: np.ndarray,
    train_indices: np.ndarray,
    *,
    method: str,
    min_cluster_events: int,
    random_state: int,
) -> Dict[str, Any]:
    train = np.asarray(train_indices, int)
    features = build_masked_kmeans_features(
        ranks[:, train], bools[:, train], impute="event_median"
    )
    spatial_scale = float("nan")
    if method == METHOD_HYBRID:
        features, spatial_scale = build_hybrid_training_features(
            features, directions[train]
        )
    elif method != METHOD_TEMPORAL:
        raise ValueError(f"unknown method: {method}")

    labels = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=int(random_state),
    ).fit_predict(features)
    counts = np.bincount(labels, minlength=2)
    if int(counts.min()) < int(min_cluster_events):
        raise ValueError(
            f"{method}: train cluster support below {min_cluster_events}: "
            f"{counts.tolist()}"
        )

    templates = build_cluster_templates(
        ranks[:, train], bools[:, train], labels, n_clusters=2
    )
    axes = fit_event_directions_3d(
        templates.T, np.asarray(coords, float), min_contacts=3
    )["directions"]
    if not np.isfinite(axes).all():
        raise ValueError(f"{method}: a train template gradient is not estimable")
    return {
        "method": method,
        "labels": labels,
        "templates": templates,
        "axes": axes,
        "train_cluster_counts": counts,
        "spatial_scale": spatial_scale,
    }


def _score_model(
    model: Mapping[str, Any],
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    test_indices: np.ndarray,
    *,
    min_cluster_events: int,
    min_shared_channels: int,
) -> Dict[str, Any]:
    test = np.asarray(test_indices, int)
    assignments = assign_events_to_templates(
        ranks[:, test],
        bools[:, test],
        np.asarray(model["templates"], float),
        min_shared_channels=min_shared_channels,
    )
    test_directions = np.asarray(directions, float)[test]
    axes = np.asarray(model["axes"], float)
    cluster_scores = np.full(2, np.nan, float)
    cluster_counts = np.zeros(2, int)
    for cluster in (0, 1):
        selected = assignments == cluster
        cluster_counts[cluster] = int(selected.sum())
        if cluster_counts[cluster] < int(min_cluster_events):
            raise ValueError(
                f"{model['method']}: held-out cluster support below "
                f"{min_cluster_events}: {cluster_counts.tolist()}"
            )
        cluster_scores[cluster] = float(
            np.mean(test_directions[selected] @ axes[cluster])
        )
    return {
        "direction_score": float(np.mean(cluster_scores)),
        "cluster_scores": cluster_scores,
        "test_cluster_counts": cluster_counts,
        "assignments": assignments,
        "test_directions": test_directions,
        "axes": axes,
    }


def blockwise_permutation(block_ids: Sequence[int], rng: np.random.Generator) -> np.ndarray:
    """Permute event rows only within their recording block."""
    blocks = np.asarray(block_ids)
    if blocks.ndim != 1:
        raise ValueError("block_ids must be one-dimensional")
    permutation = np.arange(len(blocks), dtype=int)
    for block in np.unique(blocks):
        indices = np.flatnonzero(blocks == block)
        permutation[indices] = rng.permutation(indices)
    return permutation


def permute_train_directions_within_blocks(
    directions: np.ndarray,
    block_ids: Sequence[int],
    train_indices: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Break timing--direction pairing without changing spatial missingness.

    Only finite direction vectors belonging to the same recording block and
    training fold exchange rows.  Held-out rows and every non-finite direction
    row are left byte-for-byte unchanged.
    """
    values = np.asarray(directions, float)
    blocks = np.asarray(block_ids)
    train = np.asarray(train_indices, int)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("directions must have shape (n_events, 3)")
    if blocks.shape != (values.shape[0],):
        raise ValueError("block_ids must align with directions")
    if train.ndim != 1 or np.any(train < 0) or np.any(train >= values.shape[0]):
        raise ValueError("train_indices must be valid one-dimensional indices")

    shuffled = values.copy()
    finite = np.isfinite(values).all(axis=1)
    train_blocks = blocks[train]
    for block in np.unique(train_blocks):
        recipients = train[(train_blocks == block) & finite[train]]
        if recipients.size > 1:
            shuffled[recipients] = values[rng.permutation(recipients)]
    return shuffled


def _score_with_directions(
    evaluation: Mapping[str, Any],
    directions: np.ndarray,
) -> float:
    assignments = np.asarray(evaluation["assignments"], int)
    axes = np.asarray(evaluation["axes"], float)
    values = np.asarray(directions, float)
    scores = [
        float(np.mean(values[assignments == cluster] @ axes[cluster]))
        for cluster in (0, 1)
    ]
    return float(np.mean(scores))


def _alternating_block_folds(block_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    blocks = np.asarray(block_ids)
    unique_blocks = np.unique(blocks)
    if unique_blocks.size < 2:
        raise ValueError("at least two recording blocks are required")
    fold_a = np.flatnonzero(np.isin(blocks, unique_blocks[0::2]))
    fold_b = np.flatnonzero(np.isin(blocks, unique_blocks[1::2]))
    if not fold_a.size or not fold_b.size:
        raise ValueError("alternating block split produced an empty fold")
    return [(fold_a, fold_b), (fold_b, fold_a)]


def _match_hybrid_labels_to_temporal(
    temporal_model: Mapping[str, Any],
    hybrid_model: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve KMeans label symmetry by maximum train-event agreement."""
    temporal_labels = np.asarray(temporal_model["labels"], int)
    hybrid_labels = np.asarray(hybrid_model["labels"], int)
    if temporal_labels.shape != hybrid_labels.shape:
        raise ValueError("temporal and hybrid train labels must align")
    identity = float(np.mean(temporal_labels == hybrid_labels))
    flipped = float(np.mean(temporal_labels == (1 - hybrid_labels)))
    swap = bool(flipped > identity)
    if not swap:
        return {**hybrid_model, "label_swap_to_temporal": False,
                "train_label_overlap": identity}
    return {
        **hybrid_model,
        "labels": 1 - hybrid_labels,
        "templates": np.asarray(hybrid_model["templates"], float)[[1, 0]],
        "axes": np.asarray(hybrid_model["axes"], float)[[1, 0]],
        "train_cluster_counts": np.asarray(
            hybrid_model["train_cluster_counts"], int
        )[[1, 0]],
        **({
            "train_cluster_spatial_counts": np.asarray(
                hybrid_model["train_cluster_spatial_counts"], int
            )[[1, 0]],
        } if "train_cluster_spatial_counts" in hybrid_model else {}),
        "label_swap_to_temporal": True,
        "train_label_overlap": flipped,
    }


def fit_evaluate_crossfit_fold(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    block_ids: Sequence[int],
    coords: np.ndarray,
    *,
    fold_index: int,
    min_cluster_events: int = 20,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    """Fit both arms on one block fold and evaluate rank-only held-out labels.

    Hybrid cluster IDs are matched to temporal cluster IDs by maximum overlap
    on the shared training events.  This relabeling changes no score; it only
    makes method-to-method rose comparisons visually meaningful.
    """
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    blocks = np.asarray(block_ids)
    xyz = np.asarray(coords, float)
    folds = _alternating_block_folds(blocks)
    if fold_index not in range(len(folds)):
        raise ValueError(f"fold_index must be 0 or 1, got {fold_index}")
    train, test = folds[fold_index]

    temporal = _fit_mode_model(
        rank_values,
        bool_values,
        direction_values,
        xyz,
        train,
        method=METHOD_TEMPORAL,
        min_cluster_events=min_cluster_events,
        random_state=0,
    )
    hybrid_raw = _fit_mode_model(
        rank_values,
        bool_values,
        direction_values,
        xyz,
        train,
        method=METHOD_HYBRID,
        min_cluster_events=min_cluster_events,
        random_state=0,
    )
    hybrid = _match_hybrid_labels_to_temporal(temporal, hybrid_raw)
    fitted = {METHOD_TEMPORAL: temporal, METHOD_HYBRID: hybrid}
    evaluated = {
        method: _score_model(
            model,
            rank_values,
            bool_values,
            direction_values,
            test,
            min_cluster_events=min_cluster_events,
            min_shared_channels=min_shared_channels,
        )
        for method, model in fitted.items()
    }
    return {
        "fold": int(fold_index),
        "train_indices": train,
        "test_indices": test,
        "train_blocks": np.unique(blocks[train]),
        "test_blocks": np.unique(blocks[test]),
        "models": fitted,
        "evaluations": evaluated,
        "train_label_ami": float(adjusted_mutual_info_score(
            temporal["labels"], hybrid["labels"]
        )),
        "train_label_overlap": float(hybrid["train_label_overlap"]),
        "hybrid_label_swap_to_temporal": bool(hybrid["label_swap_to_temporal"]),
    }


def _fit_all_event_mode_model(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    coords: np.ndarray,
    train_indices: np.ndarray,
    *,
    method: str,
    min_cluster_events: int,
    min_spatial_cluster_events: int,
    random_state: int,
) -> Dict[str, Any]:
    """Fit one training-fold model while retaining every training event."""
    train = np.asarray(train_indices, int)
    temporal = build_masked_kmeans_features(
        ranks[:, train], bools[:, train], impute="event_median"
    )
    spatial_scale = float("nan")
    spatial_counts = np.zeros(2, int)
    if method == METHOD_TEMPORAL:
        labels = KMeans(
            n_clusters=2,
            n_init=10,
            random_state=int(random_state),
        ).fit_predict(temporal)
    elif method == METHOD_HYBRID:
        missing_view = _fit_missing_view_hybrid_labels(
            temporal,
            np.asarray(directions, float)[train],
            random_state=random_state,
        )
        labels = np.asarray(missing_view["labels"], int)
        spatial_scale = float(missing_view["spatial_scale"])
        spatial_counts = np.asarray(
            missing_view["cluster_spatial_counts"], int
        )
        if int(spatial_counts.min()) < int(min_spatial_cluster_events):
            raise ValueError(
                "all-event timing_plus_space: train spatial support below "
                f"{min_spatial_cluster_events}: {spatial_counts.tolist()}"
            )
    else:
        raise ValueError(f"unknown method: {method}")

    counts = np.bincount(labels, minlength=2)
    if int(counts.min()) < int(min_cluster_events):
        raise ValueError(
            f"all-event {method}: train cluster support below "
            f"{min_cluster_events}: {counts.tolist()}"
        )
    templates = build_cluster_templates(
        ranks[:, train], bools[:, train], labels, n_clusters=2
    )
    axes = fit_event_directions_3d(
        templates.T, np.asarray(coords, float), min_contacts=3
    )["directions"]
    if not np.isfinite(axes).all():
        raise ValueError(f"all-event {method}: a train template axis is not estimable")
    return {
        "method": method,
        "labels": labels,
        "templates": templates,
        "axes": axes,
        "train_cluster_counts": counts,
        "train_cluster_spatial_counts": spatial_counts,
        "n_train_spatial_view_events": int(
            np.isfinite(np.asarray(directions, float)[train]).all(axis=1).sum()
        ),
        "spatial_scale": spatial_scale,
    }


def _evaluate_all_event_models(
    models: Mapping[str, Mapping[str, Any]],
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    test_indices: np.ndarray,
    *,
    min_shared_channels: int,
    min_score_events_per_cluster: int,
) -> Dict[str, Any]:
    """Score both arms on one shared minimally estimable event denominator."""
    test = np.asarray(test_indices, int)
    all_assignments = {
        method: assign_events_to_templates(
            ranks[:, test],
            bools[:, test],
            np.asarray(model["templates"], float),
            min_shared_channels=min_shared_channels,
        )
        for method, model in models.items()
    }
    test_directions = np.asarray(directions, float)[test]
    direction_estimable = np.isfinite(test_directions).all(axis=1)
    common = direction_estimable.copy()
    for assignments in all_assignments.values():
        common &= np.asarray(assignments, int) >= 0
    if not np.any(common):
        raise ValueError("no common direction-estimable held-out events")

    output: Dict[str, Any] = {
        "common_score_mask": common,
        "common_test_directions": test_directions[common],
        "n_test_events": int(test.size),
        "n_test_direction_estimable": int(direction_estimable.sum()),
        "n_test_common_assigned": int(common.sum()),
    }
    for method, model in models.items():
        assignments = np.asarray(all_assignments[method], int)
        score_assignments = assignments[common]
        axes = np.asarray(model["axes"], float)
        cluster_scores = np.full(2, np.nan, float)
        score_counts = np.zeros(2, int)
        contributions = np.empty(int(common.sum()), float)
        for cluster in (0, 1):
            selected = score_assignments == cluster
            score_counts[cluster] = int(selected.sum())
            if score_counts[cluster] < int(min_score_events_per_cluster):
                raise ValueError(
                    f"all-event {method}: common held-out score support below "
                    f"{min_score_events_per_cluster}: {score_counts.tolist()}"
                )
            values = output["common_test_directions"][selected] @ axes[cluster]
            contributions[selected] = values
            cluster_scores[cluster] = float(np.mean(values))
        output[method] = {
            "direction_score": float(np.mean(cluster_scores)),
            "event_weighted_direction_score": float(np.mean(contributions)),
            "cluster_scores": cluster_scores,
            "score_cluster_counts": score_counts,
            "test_cluster_counts_all": np.asarray([
                np.sum(assignments == cluster) for cluster in (0, 1)
            ], int),
            "n_test_unassigned": int(np.sum(assignments < 0)),
            "assignment_coverage": float(np.mean(assignments >= 0)),
            "score_assignments": score_assignments,
            "axes": axes,
        }
    return output


def _score_all_event_model_on_fixed_mask(
    model: Mapping[str, Any],
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    test_indices: np.ndarray,
    fixed_score_mask: np.ndarray,
    *,
    min_shared_channels: int,
    min_score_events_per_cluster: int,
) -> Dict[str, Any]:
    """Score one refitted model on an already frozen held-out denominator."""
    test = np.asarray(test_indices, int)
    fixed = np.asarray(fixed_score_mask, bool)
    if fixed.shape != (test.size,):
        raise ValueError("fixed_score_mask must align with held-out events")
    assignments = assign_events_to_templates(
        np.asarray(ranks, float)[:, test],
        np.asarray(bools, bool)[:, test],
        np.asarray(model["templates"], float),
        min_shared_channels=min_shared_channels,
    )
    if np.any(assignments[fixed] < 0):
        raise ValueError("refitted model does not cover the fixed score denominator")
    score_assignments = assignments[fixed]
    heldout_directions = np.asarray(directions, float)[test][fixed]
    axes = np.asarray(model["axes"], float)
    cluster_scores = np.full(2, np.nan, float)
    score_counts = np.zeros(2, int)
    contributions = np.empty(int(fixed.sum()), float)
    for cluster in (0, 1):
        selected = score_assignments == cluster
        score_counts[cluster] = int(selected.sum())
        if score_counts[cluster] < int(min_score_events_per_cluster):
            raise ValueError(
                "refitted model has insufficient fixed-denominator support: "
                f"{score_counts.tolist()}"
            )
        values = heldout_directions[selected] @ axes[cluster]
        contributions[selected] = values
        cluster_scores[cluster] = float(np.mean(values))
    return {
        "direction_score": float(np.mean(cluster_scores)),
        "event_weighted_direction_score": float(np.mean(contributions)),
        "cluster_scores": cluster_scores,
        "score_cluster_counts": score_counts,
    }


def fit_evaluate_all_event_crossfit_fold(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    block_ids: Sequence[int],
    coords: np.ndarray,
    *,
    fold_index: int,
    min_cluster_events: int = 20,
    min_spatial_cluster_events: int = 3,
    min_score_events_per_cluster: int = 3,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    """Fit on all train events and score a shared minimal-direction subset."""
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    blocks = np.asarray(block_ids)
    xyz = np.asarray(coords, float)
    folds = _alternating_block_folds(blocks)
    if fold_index not in range(len(folds)):
        raise ValueError(f"fold_index must be 0 or 1, got {fold_index}")
    train, test = folds[fold_index]
    temporal = _fit_all_event_mode_model(
        rank_values,
        bool_values,
        direction_values,
        xyz,
        train,
        method=METHOD_TEMPORAL,
        min_cluster_events=min_cluster_events,
        min_spatial_cluster_events=min_spatial_cluster_events,
        random_state=0,
    )
    hybrid_raw = _fit_all_event_mode_model(
        rank_values,
        bool_values,
        direction_values,
        xyz,
        train,
        method=METHOD_HYBRID,
        min_cluster_events=min_cluster_events,
        min_spatial_cluster_events=min_spatial_cluster_events,
        random_state=0,
    )
    hybrid = _match_hybrid_labels_to_temporal(temporal, hybrid_raw)
    models = {METHOD_TEMPORAL: temporal, METHOD_HYBRID: hybrid}
    evaluated = _evaluate_all_event_models(
        models,
        rank_values,
        bool_values,
        direction_values,
        test,
        min_shared_channels=min_shared_channels,
        min_score_events_per_cluster=min_score_events_per_cluster,
    )
    return {
        "fold": int(fold_index),
        "train_indices": train,
        "test_indices": test,
        "train_blocks": np.unique(blocks[train]),
        "test_blocks": np.unique(blocks[test]),
        "models": models,
        "evaluations": evaluated,
        "train_label_ami": float(adjusted_mutual_info_score(
            temporal["labels"], hybrid["labels"]
        )),
        "train_label_overlap": float(hybrid["train_label_overlap"]),
        "hybrid_label_swap_to_temporal": bool(hybrid["label_swap_to_temporal"]),
    }


def _score_available_directions(
    evaluation: Mapping[str, Any],
    directions: np.ndarray,
) -> tuple[float, float]:
    assignments = np.asarray(evaluation["score_assignments"], int)
    axes = np.asarray(evaluation["axes"], float)
    values = np.asarray(directions, float)
    contributions = np.empty(len(assignments), float)
    cluster_scores = []
    for cluster in (0, 1):
        selected = assignments == cluster
        scores = values[selected] @ axes[cluster]
        contributions[selected] = scores
        cluster_scores.append(float(np.mean(scores)))
    return float(np.mean(cluster_scores)), float(np.mean(contributions))


def compute_crossfit_all_event_spatial_information_gain(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    block_ids: Sequence[int],
    coords: np.ndarray,
    *,
    min_cluster_events: int = 20,
    min_spatial_cluster_events: int = 3,
    min_score_events_per_cluster: int = 3,
    min_shared_channels: int = 3,
    n_null: int = 1000,
    n_train_spatial_null: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Held-out Timing versus Timing+Space comparison with all-event fitting.

    Events lacking a mathematical direction remain in both training arms and
    affect the temporal centroids, templates and supports.  Their spatial view
    is simply absent.  Direction score is necessarily evaluated on the common
    minimally direction-estimable held-out subset after both models freeze.
    """
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    blocks = np.asarray(block_ids)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 2 or bool_values.shape != rank_values.shape:
        raise ValueError("ranks and bools must align as contacts x events")
    n_events = rank_values.shape[1]
    if direction_values.shape != (n_events, 3) or blocks.shape != (n_events,):
        raise ValueError("directions and blocks must align with events")
    if xyz.shape != (rank_values.shape[0], 3):
        raise ValueError("coords must align with contacts")
    if n_null < 1:
        raise ValueError("n_null must be positive")
    if n_train_spatial_null < 0:
        raise ValueError("n_train_spatial_null must be non-negative")
    n_direction_estimable = int(np.isfinite(direction_values).all(axis=1).sum())
    if n_direction_estimable < 2:
        raise ValueError("fewer than two events have a minimally estimable direction")

    folds = _alternating_block_folds(blocks)
    fold_rows = []
    null_primary_timing = []
    null_primary_hybrid = []
    null_event_timing = []
    null_event_hybrid = []
    train_null_primary_hybrid = []
    train_null_event_hybrid = []
    rng = np.random.default_rng(seed)
    train_null_rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), 0x54534E])
    )
    for fold_index, (train, test) in enumerate(folds):
        fold = fit_evaluate_all_event_crossfit_fold(
            rank_values,
            bool_values,
            direction_values,
            blocks,
            xyz,
            fold_index=fold_index,
            min_cluster_events=min_cluster_events,
            min_spatial_cluster_events=min_spatial_cluster_events,
            min_score_events_per_cluster=min_score_events_per_cluster,
            min_shared_channels=min_shared_channels,
        )
        models = fold["models"]
        evaluated = fold["evaluations"]
        temporal_eval = evaluated[METHOD_TEMPORAL]
        hybrid_eval = evaluated[METHOD_HYBRID]
        n_common = int(evaluated["n_test_common_assigned"])
        fold_rows.append({
            "fold": int(fold_index),
            "train_parity": "even_indexed_blocks" if fold_index == 0 else "odd_indexed_blocks",
            "n_train_events": int(len(train)),
            "n_test_events": int(len(test)),
            "n_train_blocks": int(np.unique(blocks[train]).size),
            "n_test_blocks": int(np.unique(blocks[test]).size),
            "n_test_direction_estimable": int(evaluated["n_test_direction_estimable"]),
            "n_test_common_assigned": n_common,
            "timing_only_score": float(temporal_eval["direction_score"]),
            "timing_plus_space_score": float(hybrid_eval["direction_score"]),
            "spatial_information_gain": float(
                hybrid_eval["direction_score"] - temporal_eval["direction_score"]
            ),
            "timing_only_event_weighted_score": float(
                temporal_eval["event_weighted_direction_score"]
            ),
            "timing_plus_space_event_weighted_score": float(
                hybrid_eval["event_weighted_direction_score"]
            ),
            "event_weighted_gain": float(
                hybrid_eval["event_weighted_direction_score"]
                - temporal_eval["event_weighted_direction_score"]
            ),
            "timing_only_assignment_coverage": float(temporal_eval["assignment_coverage"]),
            "timing_plus_space_assignment_coverage": float(hybrid_eval["assignment_coverage"]),
            "train_label_ami": float(fold["train_label_ami"]),
            "train_label_overlap": float(fold["train_label_overlap"]),
            "hybrid_label_swap_to_temporal": bool(fold["hybrid_label_swap_to_temporal"]),
            "timing_only_train_cluster_counts": models[METHOD_TEMPORAL]["train_cluster_counts"],
            "timing_plus_space_train_cluster_counts": models[METHOD_HYBRID]["train_cluster_counts"],
            "timing_plus_space_train_cluster_spatial_counts": models[METHOD_HYBRID]["train_cluster_spatial_counts"],
            "timing_only_score_cluster_counts": temporal_eval["score_cluster_counts"],
            "timing_plus_space_score_cluster_counts": hybrid_eval["score_cluster_counts"],
            "spatial_scale": float(models[METHOD_HYBRID]["spatial_scale"]),
        })

        if n_train_spatial_null:
            shuffled_primary = np.empty(n_train_spatial_null, float)
            shuffled_event = np.empty(n_train_spatial_null, float)
            accepted = 0
            attempts = 0
            max_attempts = max(10 * int(n_train_spatial_null), 100)
            last_error: Exception | None = None
            while accepted < int(n_train_spatial_null) and attempts < max_attempts:
                attempts += 1
                shuffled_directions = permute_train_directions_within_blocks(
                    direction_values,
                    blocks,
                    train,
                    train_null_rng,
                )
                try:
                    null_hybrid_raw = _fit_all_event_mode_model(
                        rank_values,
                        bool_values,
                        shuffled_directions,
                        xyz,
                        train,
                        method=METHOD_HYBRID,
                        min_cluster_events=min_cluster_events,
                        min_spatial_cluster_events=min_spatial_cluster_events,
                        random_state=0,
                    )
                    null_hybrid = _match_hybrid_labels_to_temporal(
                        models[METHOD_TEMPORAL], null_hybrid_raw
                    )
                    null_evaluation = _score_all_event_model_on_fixed_mask(
                        null_hybrid,
                        rank_values,
                        bool_values,
                        direction_values,
                        test,
                        np.asarray(evaluated["common_score_mask"], bool),
                        min_shared_channels=min_shared_channels,
                        min_score_events_per_cluster=min_score_events_per_cluster,
                    )
                except ValueError as exc:
                    last_error = exc
                    continue
                shuffled_primary[accepted] = float(
                    null_evaluation["direction_score"]
                )
                shuffled_event[accepted] = float(
                    null_evaluation["event_weighted_direction_score"]
                )
                accepted += 1
            if accepted < int(n_train_spatial_null):
                raise ValueError(
                    "training-side spatial shuffle produced only "
                    f"{accepted}/{n_train_spatial_null} valid refits after "
                    f"{attempts} attempts; last error={last_error}"
                )
            train_null_primary_hybrid.append(shuffled_primary)
            train_null_event_hybrid.append(shuffled_event)

        primary_timing = np.empty(n_null, float)
        primary_hybrid = np.empty(n_null, float)
        event_timing = np.empty(n_null, float)
        event_hybrid = np.empty(n_null, float)
        common_directions = np.asarray(evaluated["common_test_directions"], float)
        common_blocks = blocks[test][np.asarray(evaluated["common_score_mask"], bool)]
        for draw in range(n_null):
            permutation = blockwise_permutation(common_blocks, rng)
            shuffled = common_directions[permutation]
            primary_timing[draw], event_timing[draw] = _score_available_directions(
                temporal_eval, shuffled
            )
            primary_hybrid[draw], event_hybrid[draw] = _score_available_directions(
                hybrid_eval, shuffled
            )
        null_primary_timing.append(primary_timing)
        null_primary_hybrid.append(primary_hybrid)
        null_event_timing.append(event_timing)
        null_event_hybrid.append(event_hybrid)

    timing_only = float(np.mean([row["timing_only_score"] for row in fold_rows]))
    timing_plus_space = float(np.mean([
        row["timing_plus_space_score"] for row in fold_rows
    ]))
    common_counts = np.asarray([
        row["n_test_common_assigned"] for row in fold_rows
    ], float)
    event_timing_score = float(np.average(
        [row["timing_only_event_weighted_score"] for row in fold_rows],
        weights=common_counts,
    ))
    event_hybrid_score = float(np.average(
        [row["timing_plus_space_event_weighted_score"] for row in fold_rows],
        weights=common_counts,
    ))
    primary_null_timing = np.mean(np.vstack(null_primary_timing), axis=0)
    primary_null_hybrid = np.mean(np.vstack(null_primary_hybrid), axis=0)
    event_null_timing = np.average(
        np.vstack(null_event_timing), axis=0, weights=common_counts
    )
    event_null_hybrid = np.average(
        np.vstack(null_event_hybrid), axis=0, weights=common_counts
    )
    if n_train_spatial_null:
        train_spatial_null_hybrid = np.mean(
            np.vstack(train_null_primary_hybrid), axis=0
        )
        train_spatial_null_event_hybrid = np.average(
            np.vstack(train_null_event_hybrid), axis=0, weights=common_counts
        )
    else:
        train_spatial_null_hybrid = np.empty(0, float)
        train_spatial_null_event_hybrid = np.empty(0, float)
    return {
        "status": "ok",
        "n_events": int(n_events),
        "n_blocks": int(np.unique(blocks).size),
        "n_direction_estimable": n_direction_estimable,
        "direction_estimable_fraction": float(n_direction_estimable / n_events),
        "folds": fold_rows,
        "timing_only_score": timing_only,
        "timing_plus_space_score": timing_plus_space,
        "spatial_information_gain": timing_plus_space - timing_only,
        "event_weighted_timing_only_score": event_timing_score,
        "event_weighted_timing_plus_space_score": event_hybrid_score,
        "event_weighted_spatial_information_gain": event_hybrid_score - event_timing_score,
        "mean_train_label_ami": float(np.mean([
            row["train_label_ami"] for row in fold_rows
        ])),
        "direction_shuffle_null_timing_only_score": primary_null_timing,
        "direction_shuffle_null_timing_plus_space_score": primary_null_hybrid,
        "direction_shuffle_null_gain": primary_null_hybrid - primary_null_timing,
        "event_weighted_direction_shuffle_null_timing_only_score": event_null_timing,
        "event_weighted_direction_shuffle_null_timing_plus_space_score": event_null_hybrid,
        "event_weighted_direction_shuffle_null_gain": event_null_hybrid - event_null_timing,
        "train_spatial_shuffle_null_timing_plus_space_score": train_spatial_null_hybrid,
        "train_spatial_shuffle_null_gain_vs_timing": (
            train_spatial_null_hybrid - timing_only
        ),
        "event_weighted_train_spatial_shuffle_null_timing_plus_space_score": (
            train_spatial_null_event_hybrid
        ),
        "event_weighted_train_spatial_shuffle_null_gain_vs_timing": (
            train_spatial_null_event_hybrid - event_timing_score
        ),
        "contract": {
            "split": "two-way alternating recording-block cross-fit",
            "training_event_universe": "all interictal events; no geometry or LOCO event exclusion",
            "training_spatial_view": "optional unit 3D early-to-late event gradient when mathematically estimable",
            "missing_spatial_view": "masked from spatial distance and centroid; timing view retained",
            "hybrid_view_weight": "equal total train-fold variance on events with both views",
            "heldout_assignment": "rank-template distance only; no held-out coordinates or directions",
            "direction_score_denominator": "same common minimally direction-estimable and rank-assignable held-out events in both arms",
            "direction_minimal_definition": "at least three mapped participating contacts, nonconstant ranks and nonzero fitted gradient",
            "hard_geometry_loco_qc_used": False,
            "primary_direction_score": "equal-fold, equal-cluster mean signed cosine, matching Figure 2B",
            "sensitivity_direction_score": "event-weighted signed cosine with fold weights equal to common scored events",
            "null": "common held-out directions shuffled within recording block after model freeze",
            "training_spatial_shuffle_control": (
                "within each training fold, finite direction vectors are permuted "
                "within recording block while missingness stays fixed; only Hybrid "
                "is refitted, held-out assignment remains rank-only, and scoring "
                "uses real held-out directions on the observed fixed denominator"
            ),
            "min_cluster_events": int(min_cluster_events),
            "min_spatial_cluster_events": int(min_spatial_cluster_events),
            "min_score_events_per_cluster": int(min_score_events_per_cluster),
            "min_shared_channels": int(min_shared_channels),
            "n_null": int(n_null),
            "n_train_spatial_null": int(n_train_spatial_null),
            "seed": int(seed),
        },
    }


def compute_crossfit_spatial_information_gain(
    ranks: np.ndarray,
    bools: np.ndarray,
    directions: np.ndarray,
    block_ids: Sequence[int],
    coords: np.ndarray,
    *,
    min_cluster_events: int = 20,
    min_shared_channels: int = 3,
    n_null: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare timing-only and timing-plus-space on held-out blocks.

    Spatial directions enter the hybrid KMeans only on the training side.  Both
    arms create rank templates from their training labels, assign held-out
    events using rank distance only, and use the same template-gradient axis
    estimator.  The primary score is the equal-cluster mean signed cosine on
    held-out event directions.  The null permutes those outcome directions
    within held-out recording blocks after every model is frozen.
    """
    rank_values = np.asarray(ranks, float)
    bool_values = np.asarray(bools, bool)
    direction_values = np.asarray(directions, float)
    blocks = np.asarray(block_ids)
    xyz = np.asarray(coords, float)
    if rank_values.ndim != 2 or bool_values.shape != rank_values.shape:
        raise ValueError("ranks and bools must align as contacts x events")
    n_events = rank_values.shape[1]
    if direction_values.shape != (n_events, 3) or blocks.shape != (n_events,):
        raise ValueError("directions and blocks must align with events")
    if xyz.shape != (rank_values.shape[0], 3):
        raise ValueError("coords must align with contacts")
    if not np.isfinite(direction_values).all():
        raise ValueError("all analyzed event directions must be finite")
    if n_null < 1:
        raise ValueError("n_null must be positive")

    folds = _alternating_block_folds(blocks)
    fold_rows = []
    fold_null = []
    fold_null_temporal = []
    fold_null_hybrid = []
    rng = np.random.default_rng(seed)
    for fold_index, (train, test) in enumerate(folds):
        fold = fit_evaluate_crossfit_fold(
            rank_values,
            bool_values,
            direction_values,
            blocks,
            xyz,
            fold_index=fold_index,
            min_cluster_events=min_cluster_events,
            min_shared_channels=min_shared_channels,
        )
        fitted = fold["models"]
        evaluated = fold["evaluations"]

        temporal_score = float(evaluated[METHOD_TEMPORAL]["direction_score"])
        hybrid_score = float(evaluated[METHOD_HYBRID]["direction_score"])
        fold_rows.append({
            "fold": int(fold_index),
            "train_parity": "even_indexed_blocks" if fold_index == 0 else "odd_indexed_blocks",
            "n_train_events": int(len(train)),
            "n_test_events": int(len(test)),
            "n_train_blocks": int(np.unique(blocks[train]).size),
            "n_test_blocks": int(np.unique(blocks[test]).size),
            "timing_only_score": temporal_score,
            "timing_plus_space_score": hybrid_score,
            "spatial_information_gain": hybrid_score - temporal_score,
            "train_label_ami": float(fold["train_label_ami"]),
            "train_label_overlap": float(fold["train_label_overlap"]),
            "hybrid_label_swap_to_temporal": bool(
                fold["hybrid_label_swap_to_temporal"]
            ),
            "timing_only_train_cluster_counts": fitted[METHOD_TEMPORAL][
                "train_cluster_counts"
            ],
            "timing_plus_space_train_cluster_counts": fitted[METHOD_HYBRID][
                "train_cluster_counts"
            ],
            "timing_only_test_cluster_counts": evaluated[METHOD_TEMPORAL][
                "test_cluster_counts"
            ],
            "timing_plus_space_test_cluster_counts": evaluated[METHOD_HYBRID][
                "test_cluster_counts"
            ],
            "spatial_scale": float(fitted[METHOD_HYBRID]["spatial_scale"]),
        })

        null_temporal = np.empty(n_null, float)
        null_hybrid = np.empty(n_null, float)
        test_blocks = blocks[test]
        test_directions = direction_values[test]
        for draw in range(n_null):
            permutation = blockwise_permutation(test_blocks, rng)
            shuffled = test_directions[permutation]
            null_temporal[draw] = _score_with_directions(
                evaluated[METHOD_TEMPORAL], shuffled
            )
            null_hybrid[draw] = _score_with_directions(
                evaluated[METHOD_HYBRID], shuffled
            )
        fold_null_temporal.append(null_temporal)
        fold_null_hybrid.append(null_hybrid)
        fold_null.append(null_hybrid - null_temporal)

    timing_only = float(np.mean([row["timing_only_score"] for row in fold_rows]))
    timing_plus_space = float(
        np.mean([row["timing_plus_space_score"] for row in fold_rows])
    )
    return {
        "status": "ok",
        "n_events": int(n_events),
        "n_blocks": int(np.unique(blocks).size),
        "folds": fold_rows,
        "timing_only_score": timing_only,
        "timing_plus_space_score": timing_plus_space,
        "spatial_information_gain": timing_plus_space - timing_only,
        "mean_train_label_ami": float(np.mean([
            row["train_label_ami"] for row in fold_rows
        ])),
        "direction_shuffle_null_timing_only_score": np.mean(
            np.vstack(fold_null_temporal), axis=0
        ),
        "direction_shuffle_null_timing_plus_space_score": np.mean(
            np.vstack(fold_null_hybrid), axis=0
        ),
        "direction_shuffle_null_gain": np.mean(np.vstack(fold_null), axis=0),
        "contract": {
            "split": "two-way alternating recording-block cross-fit",
            "training_spatial_view": "unit 3D early-to-late event gradient",
            "hybrid_view_weight": "equal total train-fold variance per temporal/spatial block",
            "heldout_assignment": "rank-template distance only; no held-out coordinates or directions",
            "axis_estimator": "same rank-template least-squares gradient in both arms",
            "direction_score": "equal-cluster mean signed cosine on held-out QC-clean events",
            "null": "held-out event directions shuffled within recording block after model freeze",
            "min_cluster_events": int(min_cluster_events),
            "min_shared_channels": int(min_shared_channels),
            "n_null": int(n_null),
            "seed": int(seed),
        },
    }
