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
