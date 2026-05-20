"""Topic 4 — propagation-state attractor diagnostics (Topic 1 → Topic 4 bridge).

Step 1 deliverables:
- build_rank_feature_matrix : PR-2-consistent X_rank with NaN -> 0
- fit_pca                   : top-k PCA in feature space (default k=3)
- fit_principal_curve       : Hastie-Stuetzle iterative smoothing spline in PC space
- compute_gof               : var_explained_curve, residual stats
- compute_angle_to_kmeans_axis : angle between curve tangent and KMeans axis
- run_step1_subject         : end-to-end per-subject Step 1

Locked design (per user 2026-05-10, conversation Q1+Q2):
- Feature space = lagPatRank.T (PR-2 KMeans contract;
  src/interictal_propagation.py:1215). H3 tests PR-2 stable_k=2's dynamics, so
  the feature space MUST match PR-2's KMeans input.
- NaN -> 0.0 for inactive channels (PR-2 sentinel; "inactive treated as
  earliest"). This is part of PR-2's state definition, not biology — the H3
  claim is therefore "PR-2 rank-space propagation states (rank order +
  participation pattern) show metastable transition dynamics".

Sensitivities (deferred to separate functions, not in main path):
- Sensitivity A: active-only / participation-matched rank geometry.
- Sensitivity B: relative_lag with channel-mean imputation (NOT NaN -> 0).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)


# Topic 4 main-analysis gate. Tighter than PR-2's default min_participating=3
# to keep PCA-3 well-conditioned (an event with n_participating=3 has at most
# 3 informative entries in the rank vector; PCA-3 on such events would be
# rank-deficient).
TOPIC4_MIN_PARTICIPATING = 6


def build_rank_feature_matrix(
    ranks: np.ndarray,
    bools: np.ndarray,
    *,
    min_participating: int = TOPIC4_MIN_PARTICIPATING,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build PR-2-consistent rank feature matrix with NaN -> 0.

    Mirrors `compute_kmeans_cluster_stereotypy` lines 1215-1217 in
    `src/interictal_propagation.py`:
        rank_subset = ranks[:, valid_events].T
        rank_subset = np.where(np.isfinite(rank_subset), rank_subset, 0.0)

    Parameters
    ----------
    ranks : (n_chan, n_events) lagPatRank with NaN for inactive entries.
    bools : (n_chan, n_events) participation mask.
    min_participating : Topic 4's tighter eligibility gate (default 6).

    Returns
    -------
    X : (n_eligible, n_chan_union) feature matrix, NaN-filled-to-zero.
    eligible_idx : (n_eligible,) indices into the original event axis.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if ranks.shape != bools.shape:
        raise ValueError("ranks and bools must share shape")
    n_part = bools.sum(axis=0).astype(int)
    eligible_idx = np.where(n_part >= int(min_participating))[0]
    if eligible_idx.size == 0:
        return np.zeros((0, ranks.shape[0]), dtype=float), eligible_idx
    X = ranks[:, eligible_idx].T
    X = np.where(np.isfinite(X), X, 0.0)
    return X, eligible_idx


def fit_pca(X: np.ndarray, *, n_components: int = 3) -> Dict[str, Any]:
    """Center X and compute top-`n_components` PCA via SVD.

    Returns a dict with:
        scores                   : (n_events, n_comp) PC scores
        components               : (n_comp, n_features) row-major PC directions
        explained_variance       : (n_comp,) per-PC variance (n-1 normalisation)
        explained_variance_ratio : (n_comp,) per-PC fraction of total variance
        total_variance           : float (sum over all dims)
        mean                     : (n_features,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    n_events, n_features = X.shape
    if n_events < 2 or n_features < 1:
        raise ValueError("Need n_events >= 2 and n_features >= 1")
    mean = X.mean(axis=0)
    Xc = X - mean
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    n_comp = int(min(n_components, S.size))
    denom = max(n_events - 1, 1)
    explained_var = (S[:n_comp] ** 2) / denom
    total_var = float(np.sum(Xc ** 2) / denom)
    explained_var_ratio = explained_var / max(total_var, 1e-12)
    scores = U[:, :n_comp] * S[:n_comp]
    return {
        "scores": scores,
        "components": Vt[:n_comp].copy(),
        "explained_variance": explained_var,
        "explained_variance_ratio": explained_var_ratio,
        "total_variance": total_var,
        "mean": mean,
    }


def _project_points_to_curve(
    points: np.ndarray,
    curve_grid_pts: np.ndarray,
    s_grid: np.ndarray,
    *,
    chunk_size: int = 8192,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto a curve sampled on `s_grid`. Returns (new_s, residuals).

    Chunked to keep peak memory ~ chunk_size * len(s_grid) * d * 8 bytes.
    """
    n = points.shape[0]
    new_s = np.empty(n, dtype=float)
    resid = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        diffs = points[start:end, None, :] - curve_grid_pts[None, :, :]
        d2 = np.sum(diffs * diffs, axis=2)
        idx = np.argmin(d2, axis=1)
        new_s[start:end] = s_grid[idx]
        resid[start:end] = np.sqrt(d2[np.arange(end - start), idx])
    return new_s, resid


def fit_principal_curve(
    pc_scores: np.ndarray,
    *,
    smoothing: Optional[float] = None,
    max_iter: int = 15,
    tol: float = 1e-4,
    grid_n: int = 400,
    spline_k: int = 3,
    spline_max_n: int = 20000,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Fit Hastie-Stuetzle principal curve in PC-score space.

    Initialization: PC1 score (each point's parameter = its PC1 coordinate).
    Iteration: project points to current curve to get new s; fit smoothing
    spline per dimension on (s, score_d); resample on dense grid; re-project.

    For n > `spline_max_n`, the spline is fit on a stride-subsampled subset
    along sorted s (step = n // spline_max_n). Projection always uses the full
    n. Subsampling is necessary because UnivariateSpline cost grows
    superlinearly with knot count, and on n=160k it dominates total runtime.

    Returns dict with: s, residuals, residual_mean_sq, n_iter, converged,
    splines, s_grid, curve_grid_pts, spline_n_used.
    """
    points = np.asarray(pc_scores, dtype=float)
    if points.ndim != 2:
        raise ValueError("pc_scores must be 2D")
    n, d = points.shape
    if n < spline_k + 1:
        raise ValueError(f"Need at least {spline_k + 1} points for spline_k={spline_k}")

    s = points[:, 0].copy()

    prev_resid = float("inf")
    converged = False
    splines: Optional[List[UnivariateSpline]] = None
    s_grid: Optional[np.ndarray] = None
    curve_grid_pts: Optional[np.ndarray] = None
    new_s = s.copy()
    residuals = np.zeros(n)
    final_iter = 0
    spline_n_used = 0

    for it in range(max_iter):
        final_iter = it
        order = np.argsort(s)
        s_sorted = s[order]
        scores_sorted = points[order]

        if n > spline_max_n:
            stride = max(1, n // spline_max_n)
            sub_idx = np.arange(0, n, stride)
            s_sub = s_sorted[sub_idx]
            scores_sub = scores_sorted[sub_idx]
        else:
            s_sub = s_sorted
            scores_sub = scores_sorted

        unique_mask = np.concatenate([[True], np.diff(s_sub) > 1e-12])
        if int(unique_mask.sum()) < spline_k + 1:
            break
        s_unique = s_sub[unique_mask]
        scores_unique = scores_sub[unique_mask]
        spline_n_used = int(s_unique.size)

        try:
            splines = [
                UnivariateSpline(s_unique, scores_unique[:, di],
                                 s=smoothing, k=spline_k)
                for di in range(d)
            ]
        except Exception as exc:
            logger.warning("Spline fit failed at iter %d: %s", it, exc)
            break

        s_grid = np.linspace(float(s_unique.min()), float(s_unique.max()), grid_n)
        curve_grid_pts = np.column_stack([spl(s_grid) for spl in splines])

        new_s, residuals = _project_points_to_curve(points, curve_grid_pts, s_grid)
        cur_resid = float(np.mean(residuals ** 2))
        if abs(prev_resid - cur_resid) / max(prev_resid, 1e-12) < tol:
            converged = True
            s = new_s
            break
        prev_resid = cur_resid
        s = new_s

    return {
        "s": new_s,
        "residuals": residuals,
        "residual_mean_sq": float(np.mean(residuals ** 2)) if residuals.size else float("nan"),
        "n_iter": int(final_iter + 1),
        "converged": bool(converged),
        "splines": splines,
        "s_grid": s_grid,
        "curve_grid_pts": curve_grid_pts,
        "spline_n_used": spline_n_used,
    }


def compute_gof(
    pc_explained_variance: np.ndarray,
    curve_residuals: np.ndarray,
    n_events: int,
) -> Dict[str, Any]:
    """Goodness-of-fit for principal curve in PC-k subspace.

    var_explained_curve = 1 - (sample_var(residual) / total_pc_variance)

    sample_var(residual) is computed with the n-1 normalisation to match the
    PCA explained_variance scale.
    """
    total_pc_var = float(np.sum(pc_explained_variance))
    if curve_residuals.size == 0:
        return {
            "var_explained_curve": float("nan"),
            "residual_median": float("nan"),
            "residual_p95": float("nan"),
            "residual_max": float("nan"),
            "total_pc_variance": total_pc_var,
            "n_events": 0,
        }
    n = int(n_events)
    sum_sq = float(np.sum(curve_residuals ** 2))
    var_unexplained = sum_sq / max(n - 1, 1)
    var_explained = 1.0 - var_unexplained / max(total_pc_var, 1e-12)
    return {
        "var_explained_curve": float(np.clip(var_explained, -1.0, 1.0)),
        "residual_median": float(np.median(curve_residuals)),
        "residual_p95": float(np.quantile(curve_residuals, 0.95)),
        "residual_max": float(np.max(curve_residuals)),
        "total_pc_variance": total_pc_var,
        "n_events": n,
    }


def compute_angle_to_kmeans_axis(
    pca_components: np.ndarray,
    template_0: np.ndarray,
    template_1: np.ndarray,
    *,
    splines: Optional[List[UnivariateSpline]] = None,
    s_eval: Optional[float] = None,
) -> Dict[str, Any]:
    """Angle (deg) between principal curve tangent and KMeans (template) axis.

    KMeans axis is template_1 - template_0 in feature space; we project to PC
    space (axis_pc = pca_components @ axis). The curve tangent at `s_eval` is
    the per-dim spline derivative; if unavailable we fall back to PC1.

    Direction signs are arbitrary, so the returned angle is in [0, 90].
    """
    template_0 = np.asarray(template_0, dtype=float).ravel()
    template_1 = np.asarray(template_1, dtype=float).ravel()
    pca_components = np.asarray(pca_components, dtype=float)
    if template_0.shape != template_1.shape:
        raise ValueError("templates must share shape")
    if template_0.shape[0] != pca_components.shape[1]:
        raise ValueError("templates must align with feature dim of pca_components")

    axis = template_1 - template_0
    axis_pc = pca_components @ axis
    axis_pc_norm = float(np.linalg.norm(axis_pc))
    axis_full_norm = float(np.linalg.norm(axis))
    if axis_pc_norm < 1e-12:
        return {
            "angle_deg": float("nan"),
            "axis_pc_norm": axis_pc_norm,
            "axis_full_norm": axis_full_norm,
            "axis_explained_in_pc": float(axis_pc_norm / max(axis_full_norm, 1e-12)),
            "tangent_source": "n/a",
            "warning": "axis_zero_or_outside_pc_subspace",
        }

    n_comp = pca_components.shape[0]
    tangent: Optional[np.ndarray] = None
    tangent_source = "pc1_fallback"
    if splines is not None and s_eval is not None and len(splines) == n_comp:
        try:
            tangent = np.array(
                [float(spl.derivative()(s_eval)) for spl in splines], dtype=float
            )
            tangent_source = "spline_derivative"
        except Exception:
            tangent = None
    if tangent is None or float(np.linalg.norm(tangent)) < 1e-12:
        tangent = np.zeros(n_comp)
        tangent[0] = 1.0
        tangent_source = "pc1_fallback"

    tangent_norm = float(np.linalg.norm(tangent))
    cos_signed = float(np.dot(axis_pc, tangent) / (axis_pc_norm * tangent_norm))
    cos_signed = max(-1.0, min(1.0, cos_signed))
    angle_deg = float(np.degrees(np.arccos(abs(cos_signed))))
    return {
        "angle_deg": angle_deg,
        "cos_signed": cos_signed,
        "axis_pc_norm": axis_pc_norm,
        "axis_full_norm": axis_full_norm,
        "axis_explained_in_pc": float(axis_pc_norm / max(axis_full_norm, 1e-12)),
        "tangent_source": tangent_source,
        "s_eval": float(s_eval) if s_eval is not None else None,
    }


def _build_label_transition_count(
    labels: np.ndarray, block_ids: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Within-block consecutive label transitions; rows = source, cols = target.

    Pairs across block boundaries are dropped. Pairs with any label < 0 are
    dropped (sentinel for unlabeled events).
    """
    labels = np.asarray(labels, dtype=int)
    block_ids = np.asarray(block_ids, dtype=int)
    if labels.shape != block_ids.shape:
        raise ValueError("labels and block_ids must share shape")
    if labels.size < 2:
        return np.zeros((n_clusters, n_clusters), dtype=int)

    a = labels[:-1]
    b = labels[1:]
    same_block = block_ids[:-1] == block_ids[1:]
    both_valid = (a >= 0) & (b >= 0) & (a < n_clusters) & (b < n_clusters)
    mask = same_block & both_valid
    M = np.zeros((n_clusters, n_clusters), dtype=int)
    if int(mask.sum()) > 0:
        np.add.at(M, (a[mask], b[mask]), 1)
    return M


def _two_state_lambda2_from_count(M_count: np.ndarray) -> Dict[str, Any]:
    """λ₂ for a 2-state Markov chain plus row-norm + π.

    For 2-state row-stochastic M, λ₁ = 1 and λ₂ = trace(M) − 1. The same holds
    for reversibilized M_rev (trace invariant under reversibilization) so we
    skip the reversibilization step in the 2-state case but keep the formula
    name for consistency with the Λ_gap literature.
    """
    if M_count.shape != (2, 2):
        raise ValueError("M_count must be 2x2 for the 2-state sanity")
    row_sums = M_count.sum(axis=1)
    total = int(row_sums.sum())
    if total == 0 or (row_sums == 0).any():
        return {
            "M_row_norm": [[float("nan")] * 2 for _ in range(2)],
            "stationary_pi": [float("nan")] * 2,
            "lambda_2": float("nan"),
            "n_pairs": int(total),
            "warning": "zero_row_or_no_pairs",
        }
    M = M_count / row_sums[:, None]
    pi = (row_sums / total).tolist()
    lam2 = float(M[0, 0] + M[1, 1] - 1.0)
    return {
        "M_row_norm": M.tolist(),
        "stationary_pi": pi,
        "lambda_2": lam2,
        "stay_fracs": [float(M[0, 0]), float(M[1, 1])],
        "switch_fracs": [float(M[0, 1]), float(M[1, 0])],
        "n_pairs": int(total),
    }


def compute_pr2_label_transition_sanity(
    labels_per_event: np.ndarray,
    block_ids_per_event: np.ndarray,
    *,
    n_clusters: int = 2,
    n_perm: int = 1000,
    rng_seed: int = 0,
) -> Dict[str, Any]:
    """Coordinate-free sanity: directly probe metastability of PR-2 cluster
    labels via within-block label transitions.

    For 2-state PR-2 stable_k=2: λ₂ ∈ [-1, 1]. λ₂ → 1 = high metastability
    (long dwell, rare switches); λ₂ → 0 = no temporal structure beyond
    marginal cluster fractions; λ₂ < 0 = anti-correlated (alternating).

    Null: within-block shuffle of labels (preserves block structure and the
    marginal cluster fractions per block; destroys temporal kernel). Reports
    z and empirical p.
    """
    labels_per_event = np.asarray(labels_per_event, dtype=int)
    block_ids_per_event = np.asarray(block_ids_per_event, dtype=int)
    if labels_per_event.shape != block_ids_per_event.shape:
        raise ValueError("labels and block_ids must share shape")

    M_obs = _build_label_transition_count(
        labels_per_event, block_ids_per_event, n_clusters
    )
    obs_summary = _two_state_lambda2_from_count(M_obs) if n_clusters == 2 else None
    if obs_summary is None:
        # Generic: row-normalize and take eigenvalues
        row_sums = M_obs.sum(axis=1)
        if row_sums.sum() == 0:
            return {"error": "no_within_block_pairs"}
        M = M_obs / np.maximum(row_sums[:, None], 1)
        eigvals = np.sort(np.real(np.linalg.eigvals(M)))[::-1]
        obs_summary = {
            "M_row_norm": M.tolist(),
            "lambda_2": float(eigvals[1]) if eigvals.size > 1 else float("nan"),
            "n_pairs": int(row_sums.sum()),
        }

    # Null
    rng = np.random.default_rng(rng_seed)
    null_lam2 = np.empty(n_perm, dtype=float)
    valid_mask = (labels_per_event >= 0) & (labels_per_event < n_clusters)
    block_starts: Dict[int, np.ndarray] = {}
    for b in np.unique(block_ids_per_event):
        idx = np.where((block_ids_per_event == b) & valid_mask)[0]
        if idx.size >= 2:
            block_starts[int(b)] = idx
    for p in range(n_perm):
        shuffled = labels_per_event.copy()
        for b, idx in block_starts.items():
            perm = rng.permutation(idx.size)
            shuffled[idx] = labels_per_event[idx][perm]
        M_null = _build_label_transition_count(shuffled, block_ids_per_event, n_clusters)
        if n_clusters == 2:
            row_sums = M_null.sum(axis=1)
            if (row_sums > 0).all():
                Mn = M_null / row_sums[:, None]
                null_lam2[p] = float(Mn[0, 0] + Mn[1, 1] - 1.0)
            else:
                null_lam2[p] = np.nan
        else:
            row_sums = M_null.sum(axis=1)
            if (row_sums > 0).all():
                Mn = M_null / row_sums[:, None]
                eigvals = np.sort(np.real(np.linalg.eigvals(Mn)))[::-1]
                null_lam2[p] = float(eigvals[1]) if eigvals.size > 1 else np.nan
            else:
                null_lam2[p] = np.nan

    null_finite = null_lam2[np.isfinite(null_lam2)]
    obs_lam2 = obs_summary.get("lambda_2", float("nan"))
    if null_finite.size >= 10 and np.isfinite(obs_lam2):
        null_mean = float(np.mean(null_finite))
        null_sd = float(np.std(null_finite, ddof=1))
        z = (obs_lam2 - null_mean) / max(null_sd, 1e-12)
        p_emp = float((1 + int(np.sum(null_finite >= obs_lam2))) /
                      (1 + null_finite.size))
    else:
        null_mean = float("nan"); null_sd = float("nan")
        z = float("nan"); p_emp = float("nan")

    return {
        "n_clusters": int(n_clusters),
        "M_count": M_obs.tolist(),
        "obs": obs_summary,
        "null_n_finite": int(null_finite.size),
        "null_mean": null_mean,
        "null_sd": null_sd,
        "z_lambda_2": z,
        "p_empirical": p_emp,
        "n_perm": int(n_perm),
    }


def kmeans_centroids_from_labels(
    X: np.ndarray, labels: np.ndarray, n_clusters: int
) -> np.ndarray:
    """Mean of feature rows within each label group.

    For Topic 4 we use PR-2's labels mapped onto our eligible event subset, then
    re-compute centroids in our exact feature space (rank, NaN -> 0). This
    avoids `_legacy_hist_mean_rank` drift between PR-2's per-cluster template
    and the actual KMeans centroid in our X.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] != X.shape[0]:
        raise ValueError("labels must have one entry per row of X")
    centroids = np.zeros((n_clusters, X.shape[1]), dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if mask.any():
            centroids[k] = X[mask].mean(axis=0)
    return centroids


def run_step1_subject(
    ranks: np.ndarray,
    bools: np.ndarray,
    *,
    pr2_valid_events: Optional[np.ndarray] = None,
    pr2_labels: Optional[np.ndarray] = None,
    pr2_n_clusters: int = 2,
    min_participating: int = TOPIC4_MIN_PARTICIPATING,
    n_components: int = 3,
    gof_threshold: float = 0.6,
) -> Dict[str, Any]:
    """End-to-end Step 1 per subject: feature → PCA-k → curve → GOF + angle.

    Cluster-axis policy (post 2026-05-10 hardening):
        1. PR-2 labels MUST align to Topic 4 eligibility (length match + at
           least pr2_n_clusters clusters represented). Centroids recomputed in
           X.
        2. If PR-2 labels are unavailable / length-mismatched → exclude this
           subject (return with `excluded_reason`). NO silent template
           fallback — PR-2 template_rank may have been computed from a stale
           event ordering, which silently contaminates the H3 cluster axis.

    Returns a JSON-serialisable summary (no spline objects).
    """
    X, eligible_idx = build_rank_feature_matrix(
        ranks, bools, min_participating=min_participating
    )
    n_chan_union = int(ranks.shape[0])
    out: Dict[str, Any] = {
        "n_events_eligible": int(X.shape[0]),
        "n_chan_union": n_chan_union,
        "min_participating": int(min_participating),
        "n_components": int(n_components),
        "gof_threshold": float(gof_threshold),
    }

    if X.shape[0] < 100:
        out["skipped"] = "insufficient_eligible_events"
        return out

    pca = fit_pca(X, n_components=n_components)
    curve = fit_principal_curve(pca["scores"])
    gof = compute_gof(
        pca["explained_variance"], curve["residuals"], X.shape[0]
    )

    template_0: Optional[np.ndarray] = None
    template_1: Optional[np.ndarray] = None
    centroid_source = "none"
    n_in_cluster: List[int] = []
    excluded_reason: Optional[str] = None

    pr2_aligned = (
        pr2_valid_events is not None and pr2_labels is not None
        and len(pr2_valid_events) == len(pr2_labels)
    )
    if pr2_aligned:
        pr2_set = {int(v): int(l) for v, l in zip(pr2_valid_events, pr2_labels)}
        topic4_to_pr2_label = np.array(
            [pr2_set.get(int(ev), -1) for ev in eligible_idx], dtype=int
        )
        valid_mask = topic4_to_pr2_label >= 0
        if valid_mask.sum() >= 2 * pr2_n_clusters:
            X_lbl = X[valid_mask]
            lbls = topic4_to_pr2_label[valid_mask]
            centroids = kmeans_centroids_from_labels(X_lbl, lbls, pr2_n_clusters)
            n_in_cluster = [int((lbls == k).sum()) for k in range(pr2_n_clusters)]
            template_0 = centroids[0]
            template_1 = centroids[1] if pr2_n_clusters >= 2 else centroids[0]
            centroid_source = "pr2_labels_recomputed_in_X"
        else:
            excluded_reason = "insufficient_pr2_label_overlap_in_topic4_eligible"
    else:
        labels_n = -1 if pr2_labels is None else int(len(pr2_labels))
        valid_n = -1 if pr2_valid_events is None else int(len(pr2_valid_events))
        excluded_reason = (
            f"pr2_label_event_index_drift "
            f"(pr2_labels_n={labels_n}, pr2_valid_events_n={valid_n})"
        )

    s_median = float(np.median(curve["s"])) if curve["s"].size else 0.0
    if template_0 is not None and template_1 is not None:
        angle = compute_angle_to_kmeans_axis(
            pca["components"], template_0, template_1,
            splines=curve.get("splines"), s_eval=s_median,
        )
    else:
        angle = {
            "angle_deg": float("nan"),
            "warning": "no_kmeans_axis_available",
            "tangent_source": "n/a",
        }

    out.update({
        "pca": {
            "explained_variance": pca["explained_variance"].tolist(),
            "explained_variance_ratio": pca["explained_variance_ratio"].tolist(),
            "cumulative_top_k": float(np.sum(pca["explained_variance_ratio"])),
            "total_variance": pca["total_variance"],
        },
        "principal_curve": {
            "n_iter": curve["n_iter"],
            "converged": curve["converged"],
            "residual_mean_sq": curve["residual_mean_sq"],
            "s_min": float(curve["s"].min()) if curve["s"].size else float("nan"),
            "s_max": float(curve["s"].max()) if curve["s"].size else float("nan"),
            "s_median": s_median,
        },
        "gof": gof,
        "angle_to_kmeans_axis": angle,
        "centroid_source": centroid_source,
        "n_in_cluster": n_in_cluster,
        "gof_pass": bool(gof["var_explained_curve"] > gof_threshold),
    })
    if excluded_reason is not None:
        out["excluded_from_h3_main"] = excluded_reason
    return out
