"""Topic 1 cluster geometry visualization.

Provides two embedding views of PR-2 / PR-2.5 cluster decomposition:

1. **PCA on the KMeans feature matrix** (``compute_pca_embedding``) —
   the all-events native view in the space KMeans actually clustered.
   Uses the full ``lagPatRank`` matrix that KMeans was trained on
   (non-participating channels keep their legacy fallback ranks; the
   ``np.where(isfinite, ..., 0.0)`` in ``compute_adaptive_cluster_stereotypy``
   is a defensive no-op for cohort subjects whose lagPatRank is all-finite).
2. **Classical MDS on the template-matching distance** (``classical_mds``) —
   audit view under the masked shared-channel mean squared deviation
   identical to ``interictal_propagation.assign_events_to_templates``.
   Computed on a subsample for memory.

The two metrics are intentionally NOT identical:

* KMeans Euclidean uses **all** channels' ranks (including non-participating
  positions' fallback ranks from the legacy lagPat producer); the metric is
  ``sum_c (rank_x[c] - rank_y[c])²`` over all ``n_ch`` channels.
* Template matching uses **only** channels participating in BOTH inputs;
  the metric is ``mean_{c shared} (rank_x[c] - rank_y[c])²``.

Their disagreement on per-event cluster assignment is the audit signal.

Design doc: docs/archive/topic1/propagation/cluster_geometry_viz_plan_2026-05-06.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

from src.interictal_propagation import (
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
)

DEFAULT_MIN_SHARED_CHANNELS = 3
DEFAULT_N_MAX_FOR_MDS = 8000
DEFAULT_SUBSAMPLE_SEED = 0
DEFAULT_MIN_VALID_DISTANCE_FRACTION = 0.5
DEFAULT_STRESS_WARN_THRESHOLD = 0.3
DEFAULT_IMPUTATION_WARN_THRESHOLD = 0.20


# ---------------------------------------------------------------------------
# Single-pair masked distance
# ---------------------------------------------------------------------------


def compute_masked_distance(
    x_rank: np.ndarray,
    y_rank: np.ndarray,
    x_bool: np.ndarray,
    y_bool: np.ndarray,
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> float:
    """Masked shared-channel root-mean-squared rank deviation.

    Returns ``np.nan`` when fewer than ``min_shared`` channels are jointly
    participating + finite-rank in both inputs.

    The metric is symmetric, non-negative, and zero on x == y when both
    bool vectors agree. It does NOT generally satisfy the triangle inequality
    (a known property of masked metrics; classical MDS handles this via
    eigenvalue clipping).
    """
    x_rank = np.asarray(x_rank, dtype=float)
    y_rank = np.asarray(y_rank, dtype=float)
    x_bool = np.asarray(x_bool, dtype=bool)
    y_bool = np.asarray(y_bool, dtype=bool)

    shared = (
        x_bool
        & y_bool
        & np.isfinite(x_rank)
        & np.isfinite(y_rank)
    )
    n_shared = int(shared.sum())
    if n_shared < min_shared:
        return float("nan")
    diff = x_rank[shared] - y_rank[shared]
    return float(np.sqrt(np.mean(diff * diff)))


# ---------------------------------------------------------------------------
# Event-to-template distance matrix
# ---------------------------------------------------------------------------


def compute_event_template_distances(
    ranks: np.ndarray,
    bools: np.ndarray,
    templates_real: np.ndarray,
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> np.ndarray:
    """Event × cluster distance matrix under the unified metric.

    Parameters
    ----------
    ranks : (n_ch, n_events) — channel ranks per event
    bools : (n_ch, n_events) — channel participation per event
    templates_real : (n_clusters, n_ch) — mean-rank templates with NaN where
        no event in the cluster has the channel participating

    Returns
    -------
    D : (n_events, n_clusters) float32 — distances; NaN where fewer than
        ``min_shared`` channels are jointly available.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    templates_real = np.asarray(templates_real, dtype=float)

    n_ch, n_events = ranks.shape
    n_clusters, n_ch_t = templates_real.shape
    if n_ch != n_ch_t:
        raise ValueError(
            f"channel dim mismatch: ranks has {n_ch}, templates has {n_ch_t}"
        )

    template_bools = np.isfinite(templates_real)  # (n_clusters, n_ch)
    rank_finite = np.isfinite(ranks)  # (n_ch, n_events)

    out = np.full((n_events, n_clusters), np.nan, dtype=np.float32)

    for k in range(n_clusters):
        tk = templates_real[k]  # (n_ch,)
        tb = template_bools[k]  # (n_ch,)
        # shared: (n_ch, n_events) — per channel, per event flag
        shared = bools & tb[:, None] & rank_finite & np.isfinite(tk)[:, None]
        n_shared_per_event = shared.sum(axis=0)  # (n_events,)
        # diff matrix (n_ch, n_events) with zeros where not shared
        diff = (ranks - tk[:, None]) * shared
        sum_sq = (diff * diff).sum(axis=0)
        valid_event = n_shared_per_event >= min_shared
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sq = np.where(
                valid_event,
                sum_sq / np.maximum(n_shared_per_event, 1),
                np.nan,
            )
        out[:, k] = np.sqrt(mean_sq).astype(np.float32, copy=False)

    return out


def compute_template_template_distances(
    templates_real: np.ndarray,
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> np.ndarray:
    """Template × template distance matrix under the unified metric.

    Diagonal is forced to zero. Off-diagonal NaN persists when two templates
    share fewer than ``min_shared`` finite channels.
    """
    templates_real = np.asarray(templates_real, dtype=float)
    n_clusters, n_ch = templates_real.shape
    template_bools = np.isfinite(templates_real)
    D = np.full((n_clusters, n_clusters), np.nan, dtype=np.float32)
    np.fill_diagonal(D, 0.0)
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            d = compute_masked_distance(
                templates_real[i],
                templates_real[j],
                template_bools[i],
                template_bools[j],
                min_shared=min_shared,
            )
            D[i, j] = np.float32(d)
            D[j, i] = np.float32(d)
    return D


# ---------------------------------------------------------------------------
# Event-event distance matrix (vectorized loop)
# ---------------------------------------------------------------------------


def compute_event_event_distances(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: Sequence[int],
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> np.ndarray:
    """Event × event symmetric distance matrix on a subset of events.

    Returns float32 matrix of shape (n, n) with diagonal forced to zero.
    Off-diagonal NaN persists where events share fewer than ``min_shared``
    channels.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    idx = np.asarray(event_indices, dtype=int)
    n = idx.size

    sub_ranks = ranks[:, idx].astype(np.float32, copy=False)  # (n_ch, n)
    sub_bools = bools[:, idx]  # (n_ch, n)
    sub_finite = np.isfinite(sub_ranks)

    D = np.full((n, n), np.nan, dtype=np.float32)
    np.fill_diagonal(D, 0.0)

    for i in range(n - 1):
        xr = sub_ranks[:, i : i + 1]
        xb = sub_bools[:, i : i + 1]
        xf = sub_finite[:, i : i + 1]

        rest_r = sub_ranks[:, i + 1 :]
        rest_b = sub_bools[:, i + 1 :]
        rest_f = sub_finite[:, i + 1 :]

        shared = xb & rest_b & xf & rest_f  # (n_ch, n-i-1)
        n_shared = shared.sum(axis=0).astype(np.int32)
        diff = (xr - rest_r) * shared
        sum_sq = (diff * diff).sum(axis=0)
        valid = n_shared >= min_shared
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sq = np.where(valid, sum_sq / np.maximum(n_shared, 1), np.nan)
        d = np.sqrt(mean_sq).astype(np.float32)
        D[i, i + 1 :] = d
        D[i + 1 :, i] = d

    return D


# ---------------------------------------------------------------------------
# Augmented (E + k) × (E + k) distance matrix
# ---------------------------------------------------------------------------


def compute_augmented_distance_matrix(
    ranks: np.ndarray,
    bools: np.ndarray,
    templates_real: np.ndarray,
    event_indices: Sequence[int],
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> np.ndarray:
    """Build the (E + k) × (E + k) augmented distance matrix.

    Layout: rows/cols 0..E-1 are events (in order of ``event_indices``),
    rows/cols E..E+k-1 are clusters in order ``templates_real`` provides.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    idx = np.asarray(event_indices, dtype=int)
    E = idx.size
    k = templates_real.shape[0]

    D_ee = compute_event_event_distances(ranks, bools, idx, min_shared=min_shared)
    D_et = compute_event_template_distances(
        ranks[:, idx], bools[:, idx], templates_real, min_shared=min_shared
    )  # (E, k)
    D_tt = compute_template_template_distances(templates_real, min_shared=min_shared)

    D = np.full((E + k, E + k), np.nan, dtype=np.float32)
    D[:E, :E] = D_ee
    D[:E, E:] = D_et
    D[E:, :E] = D_et.T
    D[E:, E:] = D_tt
    np.fill_diagonal(D, 0.0)
    return D


# ---------------------------------------------------------------------------
# Classical MDS
# ---------------------------------------------------------------------------


def classical_mds(
    D: np.ndarray,
    n_components: int = 2,
    min_valid_distance_fraction: float = DEFAULT_MIN_VALID_DISTANCE_FRACTION,
) -> Dict[str, Any]:
    """Classical MDS via double-centering + top eigenpair extraction.

    NaN handling:
      1. Rows whose valid-distance fraction < ``min_valid_distance_fraction``
         are flagged as ``low_coverage`` indices in the output (still embedded
         using imputation, but caller may want to mark them).
      2. Remaining NaN off-diagonal entries are imputed with the median of
         finite off-diagonal pairwise distances.

    Returns dict with keys:
      - ``Y`` : (n, n_components) float64 embedding
      - ``stress`` : sqrt(sum((D - D_Y)²) / sum(D²)) on imputed D
      - ``eigvals`` : top-``n_components`` eigenvalues of double-centred B
      - ``imputed_fraction`` : fraction of off-diagonal entries that were
        imputed (excluding diagonal)
      - ``low_coverage_indices`` : list of row indices whose valid fraction
        was below the threshold (the embedding still includes them)
    """
    D = np.asarray(D, dtype=np.float64).copy()
    n = D.shape[0]
    if n < n_components + 1:
        raise ValueError(f"need at least n_components+1 rows, got n={n}")

    # Force diagonal to zero
    np.fill_diagonal(D, 0.0)

    # Detect low-coverage rows
    finite_mask = np.isfinite(D)
    np.fill_diagonal(finite_mask, True)
    valid_fraction = finite_mask.sum(axis=1) / n
    low_coverage_indices = list(np.where(valid_fraction < min_valid_distance_fraction)[0].tolist())

    # Compute imputation value: median of off-diagonal finite distances
    off_diag_mask = ~np.eye(n, dtype=bool)
    finite_offdiag = np.isfinite(D) & off_diag_mask
    n_offdiag = int(off_diag_mask.sum())
    n_finite_offdiag = int(finite_offdiag.sum())
    n_imputed = n_offdiag - n_finite_offdiag
    imputed_fraction = float(n_imputed / max(n_offdiag, 1))

    if n_finite_offdiag == 0:
        # Degenerate: no valid distances at all. Return zeros.
        return {
            "Y": np.zeros((n, n_components), dtype=np.float64),
            "stress": float("nan"),
            "eigvals": np.zeros(n_components, dtype=np.float64),
            "imputed_fraction": 1.0,
            "low_coverage_indices": low_coverage_indices,
        }

    median_d = float(np.nanmedian(D[finite_offdiag]))
    D_imp = np.where(np.isfinite(D), D, median_d)
    np.fill_diagonal(D_imp, 0.0)

    # Double centering using row/column means
    D2 = D_imp * D_imp
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    total_mean = D2.mean()
    B = -0.5 * (D2 - row_mean - col_mean + total_mean)

    # Symmetrize B for numerical safety
    B = 0.5 * (B + B.T)

    # Top n_components eigenpairs (subset_by_index counts from low to high)
    eigvals_top, eigvecs_top = eigh(B, subset_by_index=[n - n_components, n - 1])
    # eigh returns ascending; reverse to descending
    eigvals_top = eigvals_top[::-1]
    eigvecs_top = eigvecs_top[:, ::-1]

    # Some eigenvalues may be slightly negative for non-Euclidean masked
    # metrics; clip them to 0 for the embedding (loses information but is
    # the standard cmdscale fallback).
    eigvals_clipped = np.maximum(eigvals_top, 0.0)
    Y = eigvecs_top * np.sqrt(eigvals_clipped)[None, :]

    # Stress on imputed D
    diffs = Y[:, None, :] - Y[None, :, :]
    D_Y = np.sqrt((diffs * diffs).sum(axis=-1))
    denom = float((D_imp * D_imp).sum())
    if denom > 0:
        stress = float(np.sqrt(((D_imp - D_Y) ** 2).sum() / denom))
    else:
        stress = float("nan")

    return {
        "Y": Y,
        "stress": stress,
        "eigvals": eigvals_top,
        "imputed_fraction": imputed_fraction,
        "low_coverage_indices": low_coverage_indices,
    }


# ---------------------------------------------------------------------------
# PCA on the KMeans feature matrix (all-events native view)
# ---------------------------------------------------------------------------


def compute_pca_embedding(
    ranks: np.ndarray,
    valid_event_indices: Sequence[int],
    templates_real: np.ndarray,
    n_components: int = 2,
) -> Dict[str, Any]:
    """PCA on the same feature matrix KMeans was trained on.

    Reproduces the imputation that ``compute_adaptive_cluster_stereotypy``
    uses (``np.where(isfinite, x, 0.0)``); for cohort subjects whose
    lagPatRank is all-finite (the typical case), this is a no-op.

    Templates are projected as additional rows in the same feature space;
    template NaN values are imputed to 0 (matching KMeans's defensive
    fallback). All events are embedded — no subsampling.

    Returns
    -------
    dict with keys ``Y_events`` (n_events, n_components),
    ``Y_templates`` (n_clusters, n_components), and
    ``explained_variance_ratio`` (n_components,).
    """
    ranks = np.asarray(ranks, dtype=float)
    idx = np.asarray(valid_event_indices, dtype=int)

    feat_events = ranks[:, idx].T  # (n_events, n_ch)
    feat_events = np.where(np.isfinite(feat_events), feat_events, 0.0)

    feat_templates = np.asarray(templates_real, dtype=float).copy()  # (k, n_ch)
    feat_templates = np.where(np.isfinite(feat_templates), feat_templates, 0.0)

    n_events = feat_events.shape[0]
    if n_events < n_components + 1:
        Y_events = np.zeros((n_events, n_components), dtype=np.float64)
        Y_templates = np.zeros((feat_templates.shape[0], n_components), dtype=np.float64)
        return {
            "Y_events": Y_events,
            "Y_templates": Y_templates,
            "explained_variance_ratio": np.zeros(n_components, dtype=np.float64),
        }

    pca = PCA(n_components=n_components, random_state=0)
    Y_events = pca.fit_transform(feat_events)
    Y_templates = pca.transform(feat_templates)

    return {
        "Y_events": Y_events,
        "Y_templates": Y_templates,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


# ---------------------------------------------------------------------------
# Per-event silhouette (template-prototype version)
# ---------------------------------------------------------------------------


def compute_per_event_silhouette(
    d_within: np.ndarray,
    d_min_other: np.ndarray,
) -> np.ndarray:
    """Per-event silhouette using template-prototype distances.

    s_i = (b_i - a_i) / max(a_i, b_i)
      a_i = d_within[i] = distance to assigned template
      b_i = d_min_other[i] = distance to nearest non-assigned template

    NaN propagates if either input is NaN; if max(a,b) == 0, silhouette = 0.
    """
    a = np.asarray(d_within, dtype=float)
    b = np.asarray(d_min_other, dtype=float)
    denom = np.maximum(a, b)
    out = np.full_like(a, np.nan, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    nz = valid & (denom > 0)
    out[nz] = (b[nz] - a[nz]) / denom[nz]
    out[valid & (denom == 0)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Subject-level pipeline
# ---------------------------------------------------------------------------


def compute_subject_geometry(
    ranks: np.ndarray,
    bools: np.ndarray,
    channel_names: Sequence[str],
    adaptive_labels: np.ndarray,
    chosen_k: int,
    valid_event_indices: Sequence[int],
    event_abs_times: Optional[np.ndarray] = None,
    block_ids: Optional[np.ndarray] = None,
    min_shared: int = DEFAULT_MIN_SHARED_CHANNELS,
    max_events_for_mds: int = DEFAULT_N_MAX_FOR_MDS,
    subsample_seed: int = DEFAULT_SUBSAMPLE_SEED,
    n_min_events_total: int = 50,
) -> Dict[str, Any]:
    """End-to-end per-subject geometry pipeline.

    Steps:
      1. Build templates_real via ``build_cluster_templates`` (NaN where
         no event in the cluster has the channel participating).
      2. Reassign events to templates via ``assign_events_to_templates``
         (the audit metric); compare to ``adaptive_labels``.
      3. Compute event-template distances; derive d_within / d_min_other /
         silhouette per event.
      4. Subsample events for the MDS step if needed; build augmented
         distance matrix; run classical MDS.
      5. Aggregate per-subject summary fields.

    Parameters
    ----------
    ranks, bools : (n_ch, n_events) — full-cohort propagation arrays
    adaptive_labels : (n_valid_events,) — PR-2 cluster labels for the
        valid-event subset (NOT the full event array)
    valid_event_indices : (n_valid_events,) — indices into ranks/bools for
        the events the labels apply to

    Returns
    -------
    dict with keys: ``status``, plus subject-level summary fields. If
    ``status != "ok"``, ``excluded_reason`` is set and the rest is partial.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    valid_idx = np.asarray(valid_event_indices, dtype=int)
    labels = np.asarray(adaptive_labels, dtype=int)
    n_ch, n_events_total = ranks.shape

    if labels.size != valid_idx.size:
        raise ValueError("adaptive_labels and valid_event_indices length mismatch")

    if valid_idx.size < n_min_events_total:
        return {
            "status": "excluded",
            "excluded_reason": "too_few_events",
            "n_events_total": int(valid_idx.size),
        }

    # Step 1: rebuild real-valued templates from the valid-event subset
    sub_ranks = ranks[:, valid_idx]
    sub_bools = bools[:, valid_idx]
    templates_real = build_cluster_templates(sub_ranks, sub_bools, labels, chosen_k)

    # Detect all-NaN clusters (all events in cluster k have no participating channel)
    all_nan_clusters = []
    for k in range(chosen_k):
        if not np.any(np.isfinite(templates_real[k])):
            all_nan_clusters.append(int(k))
    if all_nan_clusters:
        return {
            "status": "excluded",
            "excluded_reason": "all_nan_template",
            "failed_clusters": all_nan_clusters,
            "n_events_total": int(valid_idx.size),
        }

    # Step 2: reassign events under the audit metric
    matching_labels = assign_events_to_templates(
        sub_ranks, sub_bools, templates_real, min_shared_channels=min_shared
    )
    # matching_labels can be -1 if no template is reachable; treat as disagreement
    agreement_mask = matching_labels == labels
    agreement_overall = float(agreement_mask.mean()) if agreement_mask.size else float("nan")

    # Step 3: event-template distances + template-template distances + per-event readout
    D_et = compute_event_template_distances(
        sub_ranks, sub_bools, templates_real, min_shared=min_shared
    )  # (E, k)
    D_tt = compute_template_template_distances(templates_real, min_shared=min_shared)  # (k, k)
    n_events_valid = D_et.shape[0]

    d_within = np.full(n_events_valid, np.nan, dtype=np.float32)
    d_min_other = np.full(n_events_valid, np.nan, dtype=np.float32)
    for i in range(n_events_valid):
        ki = labels[i]
        if 0 <= ki < chosen_k:
            d_within[i] = D_et[i, ki]
        if chosen_k > 1:
            mask = np.ones(chosen_k, dtype=bool)
            if 0 <= ki < chosen_k:
                mask[ki] = False
            others = D_et[i, mask]
            if np.any(np.isfinite(others)):
                d_min_other[i] = float(np.nanmin(others))

    silhouette = compute_per_event_silhouette(d_within, d_min_other)

    # n_participating per event (on valid subset)
    n_participating = sub_bools.sum(axis=0).astype(np.int32)

    # Boundary fraction by n_participating bin
    bins = {"3-4": (3, 4), "5-8": (5, 8), "9+": (9, n_ch + 1)}
    boundary_fraction = {}
    for name, (lo, hi) in bins.items():
        in_bin = (n_participating >= lo) & (n_participating <= hi)
        if int(in_bin.sum()) > 0:
            boundary_fraction[name] = float(
                (~agreement_mask[in_bin]).mean()
            )
        else:
            boundary_fraction[name] = None

    # Step 4a: PCA on the KMeans feature matrix (all events, no subsample)
    pca_out = compute_pca_embedding(
        ranks=ranks,
        valid_event_indices=valid_idx,
        templates_real=templates_real,
        n_components=2,
    )
    pca_Y_events = pca_out["Y_events"]
    pca_Y_templates = pca_out["Y_templates"]

    # Step 4b: build augmented distance matrix; subsample if needed
    rng = np.random.default_rng(subsample_seed)
    if n_events_valid > max_events_for_mds:
        mds_event_local_idx = np.sort(
            rng.choice(n_events_valid, max_events_for_mds, replace=False)
        )
        subsampled = True
    else:
        mds_event_local_idx = np.arange(n_events_valid)
        subsampled = False

    mds_event_idx_in_full = valid_idx[mds_event_local_idx]
    D_aug = compute_augmented_distance_matrix(
        ranks, bools, templates_real, mds_event_idx_in_full, min_shared=min_shared
    )

    mds_out = classical_mds(D_aug, n_components=2)
    Y = mds_out["Y"]
    E_used = mds_event_local_idx.size

    # Split Y into events and templates parts
    Y_events_subset = Y[:E_used]  # only the subsampled events
    Y_templates = Y[E_used:]

    # For events not in the subsample, mds_x/mds_y = NaN
    mds_x = np.full(n_events_valid, np.nan, dtype=np.float64)
    mds_y = np.full(n_events_valid, np.nan, dtype=np.float64)
    mds_x[mds_event_local_idx] = Y_events_subset[:, 0]
    mds_y[mds_event_local_idx] = Y_events_subset[:, 1]

    # Step 5: aggregate per-event records + summary
    events_records: List[Dict[str, Any]] = []
    abs_times = (
        np.asarray(event_abs_times, dtype=float)
        if event_abs_times is not None
        else np.full(n_events_total, np.nan, dtype=float)
    )
    blk_ids = (
        np.asarray(block_ids, dtype=int)
        if block_ids is not None
        else np.full(n_events_total, -1, dtype=int)
    )
    for i in range(n_events_valid):
        full_idx = int(valid_idx[i])
        events_records.append(
            {
                "event_idx": full_idx,
                "kmeans_label": int(labels[i]),
                "matching_label": int(matching_labels[i]),
                "agreement": bool(agreement_mask[i]),
                "d_within": _safe_float(d_within[i]),
                "d_min_other": _safe_float(d_min_other[i]),
                "d_to_each_template": [_safe_float(v) for v in D_et[i].tolist()],
                "silhouette": _safe_float(silhouette[i]),
                "n_participating": int(n_participating[i]),
                "pca_x": _safe_float(pca_Y_events[i, 0]),
                "pca_y": _safe_float(pca_Y_events[i, 1]),
                "mds_x": _safe_float(mds_x[i]),
                "mds_y": _safe_float(mds_y[i]),
                "block_id": int(blk_ids[full_idx]) if 0 <= full_idx < blk_ids.size else -1,
                "event_abs_time": _safe_float(abs_times[full_idx]) if 0 <= full_idx < abs_times.size else float("nan"),
            }
        )

    # Per-cluster IQR over channel ranks (for plot panel c half-band)
    template_iqr_low = np.full((chosen_k, n_ch), np.nan, dtype=float)
    template_iqr_high = np.full((chosen_k, n_ch), np.nan, dtype=float)
    for k in range(chosen_k):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_ranks = sub_ranks[:, mask]
        cluster_bools = sub_bools[:, mask]
        for c in range(n_ch):
            participating = cluster_bools[c]
            if int(participating.sum()) < 3:
                continue
            vals = cluster_ranks[c, participating]
            template_iqr_low[k, c] = float(np.percentile(vals, 25))
            template_iqr_high[k, c] = float(np.percentile(vals, 75))

    templates_records = [
        {
            "cluster_id": int(k),
            "pca_x": float(pca_Y_templates[k, 0]),
            "pca_y": float(pca_Y_templates[k, 1]),
            "mds_x": float(Y_templates[k, 0]),
            "mds_y": float(Y_templates[k, 1]),
            "template_rank_real": templates_real[k].tolist(),
            "template_iqr_low": template_iqr_low[k].tolist(),
            "template_iqr_high": template_iqr_high[k].tolist(),
            "n_finite_channels": int(np.sum(np.isfinite(templates_real[k]))),
        }
        for k in range(chosen_k)
    ]

    finite_sil = silhouette[np.isfinite(silhouette)]
    sil_median = float(np.median(finite_sil)) if finite_sil.size else float("nan")
    sil_iqr = (
        [float(np.percentile(finite_sil, 25)), float(np.percentile(finite_sil, 75))]
        if finite_sil.size
        else [float("nan"), float("nan")]
    )

    # Inter-cluster Spearman r matrix (re-derived from templates_real for record)
    inter_corr = _template_pair_spearman(templates_real)

    return {
        "status": "ok",
        "n_events_total": int(n_events_valid),
        "n_events_used_for_mds": int(E_used),
        "subsampled": bool(subsampled),
        "subsample_seed": int(subsample_seed),
        "chosen_k": int(chosen_k),
        "stress": float(mds_out["stress"]),
        "imputed_fraction": float(mds_out["imputed_fraction"]),
        "imputation_warning": bool(
            mds_out["imputed_fraction"] > DEFAULT_IMPUTATION_WARN_THRESHOLD
        ),
        "stress_warning": bool(mds_out["stress"] > DEFAULT_STRESS_WARN_THRESHOLD),
        "pca_explained_variance_ratio": [float(x) for x in pca_out["explained_variance_ratio"]],
        "silhouette_median": sil_median,
        "silhouette_iqr": sil_iqr,
        "agreement_overall": agreement_overall,
        "boundary_fraction_by_nparticipating": boundary_fraction,
        "channel_names": list(channel_names),
        "events": events_records,
        "templates": templates_records,
        "templates_real": templates_real.tolist(),
        "template_template_distance": D_tt.tolist(),
        "inter_cluster_corr_matrix": inter_corr.tolist(),
        "low_coverage_mds_indices": mds_out["low_coverage_indices"],
    }


def _template_pair_spearman(templates_real: np.ndarray) -> np.ndarray:
    """Spearman r between each pair of templates on jointly-finite channels.

    Diagonal is 1.0; pairs with fewer than 3 jointly finite channels = NaN.
    """
    n_clusters = templates_real.shape[0]
    out = np.full((n_clusters, n_clusters), np.nan, dtype=float)
    for i in range(n_clusters):
        out[i, i] = 1.0
        for j in range(i + 1, n_clusters):
            both = np.isfinite(templates_real[i]) & np.isfinite(templates_real[j])
            if int(both.sum()) >= 3:
                a = templates_real[i, both]
                b = templates_real[j, both]
                if np.std(a) > 1e-12 and np.std(b) > 1e-12:
                    r, _ = spearmanr(a, b)
                    out[i, j] = float(r)
                    out[j, i] = float(r)
    return out


def _safe_float(x: float) -> float:
    """JSON-safe float: NaN/Inf become None-coercible Python floats but stay
    NaN/Inf so downstream serializers (the runners use ``json.dumps`` with
    ``allow_nan=False``→False fallback) can decide.
    """
    return float(x)


# ---------------------------------------------------------------------------
# Cohort summary aggregator
# ---------------------------------------------------------------------------


def summarize_cohort_geometry(
    per_subject_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-subject geometry results into cohort summary.

    Skips subjects whose ``status != "ok"``; their reasons are recorded
    in ``excluded``.
    """
    included: Dict[str, Dict[str, Any]] = {}
    excluded: Dict[str, Dict[str, Any]] = {}
    for name, res in per_subject_results.items():
        if res.get("status") == "ok":
            included[name] = {
                "silhouette_median": res["silhouette_median"],
                "silhouette_iqr": res["silhouette_iqr"],
                "agreement_overall": res["agreement_overall"],
                "stable_k": res["chosen_k"],
                "stress": res["stress"],
                "imputed_fraction": res["imputed_fraction"],
                "imputation_warning": res["imputation_warning"],
                "stress_warning": res["stress_warning"],
                "n_events": res["n_events_total"],
                "subsampled": res["subsampled"],
                "boundary_fraction_by_nparticipating": res[
                    "boundary_fraction_by_nparticipating"
                ],
            }
        else:
            excluded[name] = {
                "excluded_reason": res.get("excluded_reason", "unknown"),
                "details": {
                    k: v for k, v in res.items() if k != "events"
                },
            }

    # Joint silhouette vs agreement Spearman
    sil_vals = [v["silhouette_median"] for v in included.values()]
    agr_vals = [v["agreement_overall"] for v in included.values()]
    sil_arr = np.asarray(sil_vals, dtype=float)
    agr_arr = np.asarray(agr_vals, dtype=float)
    paired = np.isfinite(sil_arr) & np.isfinite(agr_arr)
    if int(paired.sum()) >= 3:
        r, p = spearmanr(sil_arr[paired], agr_arr[paired])
        joint = {"r": float(r), "p": float(p), "n": int(paired.sum())}
    else:
        joint = {"r": float("nan"), "p": float("nan"), "n": int(paired.sum())}

    # Boundary fraction by n_part bins, cohort-level
    bin_keys = ["3-4", "5-8", "9+"]
    boundary_by_bin: Dict[str, List[float]] = {b: [] for b in bin_keys}
    for v in included.values():
        bf = v.get("boundary_fraction_by_nparticipating", {})
        for b in bin_keys:
            val = bf.get(b)
            if val is not None and np.isfinite(val):
                boundary_by_bin[b].append(float(val))

    high_stress = [name for name, v in included.items() if v.get("stress_warning")]
    high_imputation = [
        name for name, v in included.items() if v.get("imputation_warning")
    ]

    return {
        "n_subjects_included": len(included),
        "n_subjects_excluded": len(excluded),
        "per_subject": included,
        "excluded": excluded,
        "joint_silhouette_vs_agreement_spearman": joint,
        "boundary_fraction_by_nparticipating": boundary_by_bin,
        "subjects_high_stress": high_stress,
        "subjects_high_imputation": high_imputation,
    }


# ---------------------------------------------------------------------------
# Convenience: use the standard PR-2 valid-event indexing
# ---------------------------------------------------------------------------


def derive_valid_events_for_pr2(
    bools: np.ndarray,
    min_participating: int = DEFAULT_MIN_SHARED_CHANNELS,
) -> np.ndarray:
    """Mirror ``_valid_event_indices`` from interictal_propagation."""
    return _valid_event_indices(bools, min_participating=min_participating)
