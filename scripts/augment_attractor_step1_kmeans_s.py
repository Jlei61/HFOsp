#!/usr/bin/env python3
"""Topic 4 Step 1 augmentation — adds two diagnostics to each per-subject JSON:

1. ``s_kmeans_diagnostic``: 1D projection along the PR-2 KMeans axis (axis =
   centroid_1 − centroid_0, centroids recomputed in Topic 4 X using PR-2
   labels). Reports per-cluster s distribution + Cohen's d + midpoint
   threshold accuracy.

   ⚠ s_kmeans is a **PR-2-label-supervised** coordinate. It demonstrates that
   the two PR-2 clusters are linearly separable in rank space, **NOT** that
   they reflect bistable dynamics independently. Use only as the rank-space
   shape characterisation; H3 metastability requires temporal evidence.

2. ``label_transition_sanity``: coordinate-free metastability test. Builds the
   2×2 within-block PR-2 label transition matrix, computes λ₂ = trace(M) − 1,
   and tests against within-block label-shuffle null (n_perm=1000). z_lambda_2
   and empirical p quantify whether labels stick to themselves over time
   beyond the marginal cluster fraction. This does NOT depend on any 1D
   coordinate.

Behavior changes vs first version (post 2026-05-10 fix):
- Subjects flagged ``excluded_from_h3_main`` (e.g. PR-2 label/event drift) are
  skipped here too — both diagnostics require labels aligned with current
  events.
- Cohort CSV is refreshed with s_kmeans + label_transition columns.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.topic4_attractor_diagnostics import (  # noqa: E402
    TOPIC4_MIN_PARTICIPATING,
    build_rank_feature_matrix,
    compute_pr2_label_transition_sanity,
    kmeans_centroids_from_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("topic4_step1_augment")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
PR2_PER_SUBJECT_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
OUT_DIR = REPO_ROOT / "results" / "topic4_attractor"
PER_SUBJECT_DIR = OUT_DIR / "per_subject"
COHORT_CSV = OUT_DIR / "step1_cohort_summary.csv"

PR2_VALID_MIN_PART = 3
LABEL_TRANSITION_N_PERM = 1000


def _epilepsiae_subject_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(subject)


def _per_cluster_stats(s: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"per_cluster": []}
    for k in (0, 1):
        mask = labels == k
        sk = s[mask]
        out["per_cluster"].append({
            "cluster_id": int(k),
            "n": int(mask.sum()),
            "median": float(np.median(sk)) if sk.size else float("nan"),
            "mean": float(np.mean(sk)) if sk.size else float("nan"),
            "std": float(np.std(sk, ddof=1)) if sk.size > 1 else float("nan"),
            "iqr": (
                float(np.quantile(sk, 0.75) - np.quantile(sk, 0.25))
                if sk.size else float("nan")
            ),
            "p05": float(np.quantile(sk, 0.05)) if sk.size else float("nan"),
            "p95": float(np.quantile(sk, 0.95)) if sk.size else float("nan"),
        })
    s0 = s[labels == 0]
    s1 = s[labels == 1]
    if s0.size > 1 and s1.size > 1:
        m0, m1 = float(np.mean(s0)), float(np.mean(s1))
        v0, v1 = float(np.var(s0, ddof=1)), float(np.var(s1, ddof=1))
        n0, n1 = int(s0.size), int(s1.size)
        pooled = float(np.sqrt(((n0 - 1) * v0 + (n1 - 1) * v1) / max(n0 + n1 - 2, 1)))
        out["cohens_d"] = (m1 - m0) / pooled if pooled > 0 else float("nan")
        med0, med1 = float(np.median(s0)), float(np.median(s1))
        iqr_combined = float(np.quantile(s, 0.75) - np.quantile(s, 0.25))
        out["median_diff"] = med1 - med0
        out["sep_index_iqr"] = (
            abs(med1 - med0) / iqr_combined if iqr_combined > 0 else float("nan")
        )
        thresh = (m0 + m1) / 2.0
        if m1 >= m0:
            pred = (s >= thresh).astype(int)
        else:
            pred = (s < thresh).astype(int)
        acc = float((pred == labels).mean())
        out["midpoint_threshold_accuracy"] = acc
    else:
        out["cohens_d"] = float("nan")
        out["median_diff"] = float("nan")
        out["sep_index_iqr"] = float("nan")
        out["midpoint_threshold_accuracy"] = float("nan")
    return out


def _load_pr2_labels(json_path: Path):
    if not json_path.exists():
        return None, None
    with open(json_path) as f:
        d = json.load(f)
    ac = d.get("adaptive_cluster", {})
    labels = ac.get("labels")
    chosen_k = int(ac.get("chosen_k", -1))
    if labels is None:
        return None, chosen_k
    return np.asarray(labels, dtype=int), chosen_k


def _augment_one(per_subject_path: Path) -> Dict[str, Any]:
    with open(per_subject_path) as f:
        per = json.load(f)

    sid = per.get("sid")
    dataset = per.get("dataset")
    subject = per.get("subject")
    if not (sid and dataset and subject):
        return per
    if "skipped" in per or "error" in per:
        return per

    # New: respect Step 1 exclusion flag (e.g. PR-2 label/event drift).
    if "excluded_from_h3_main" in per:
        per.setdefault("s_kmeans_diagnostic", {
            "skipped": "excluded_from_h3_main",
            "reason": per["excluded_from_h3_main"],
        })
        per.setdefault("label_transition_sanity", {
            "skipped": "excluded_from_h3_main",
            "reason": per["excluded_from_h3_main"],
        })
        with open(per_subject_path, "w") as f:
            json.dump(per, f, indent=2, default=str)
        return per

    sub_dir = _subject_dir(dataset, subject)
    pr2_json = PR2_PER_SUBJECT_DIR / f"{sid}.json"
    labels_pr2, _chosen_k = _load_pr2_labels(pr2_json)

    loaded = load_subject_propagation_events(sub_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    block_ids_full = np.asarray(loaded.get("block_ids", []), dtype=int)

    valid_pr2 = _valid_event_indices(bools, min_participating=PR2_VALID_MIN_PART)
    if labels_pr2 is None or labels_pr2.shape[0] != valid_pr2.shape[0]:
        # Should have been caught at Step 1 already; here it means the
        # exclusion flag is missing (older Step 1 results).
        diag_err = {
            "error": "pr2_labels_unavailable_or_drift",
            "labels_n": -1 if labels_pr2 is None else int(labels_pr2.shape[0]),
            "valid_pr2_n": int(valid_pr2.shape[0]),
        }
        per["s_kmeans_diagnostic"] = diag_err
        per["label_transition_sanity"] = diag_err
        with open(per_subject_path, "w") as f:
            json.dump(per, f, indent=2, default=str)
        return per

    X, eligible_idx = build_rank_feature_matrix(
        ranks, bools, min_participating=TOPIC4_MIN_PARTICIPATING
    )

    pr2_lookup: Dict[int, int] = {int(v): int(l) for v, l in zip(valid_pr2, labels_pr2)}
    topic4_labels = np.array(
        [pr2_lookup.get(int(ev), -1) for ev in eligible_idx], dtype=int
    )
    keep = topic4_labels >= 0
    if int(keep.sum()) < 4:
        per["s_kmeans_diagnostic"] = {"error": "insufficient_overlap_with_pr2_labels"}
        per["label_transition_sanity"] = {"error": "insufficient_overlap_with_pr2_labels"}
        with open(per_subject_path, "w") as f:
            json.dump(per, f, indent=2, default=str)
        return per

    X_used = X[keep]
    labels_used = topic4_labels[keep]
    eligible_used = eligible_idx[keep]
    block_ids_used = block_ids_full[eligible_used] if block_ids_full.size else np.zeros_like(labels_used)

    # ---- s_kmeans diagnostic (PR-2-label-supervised) ----
    centroids = kmeans_centroids_from_labels(X_used, labels_used, n_clusters=2)
    axis = centroids[1] - centroids[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        per["s_kmeans_diagnostic"] = {"error": "axis_zero"}
    else:
        axis_unit = axis / axis_norm
        X_mean = X_used.mean(axis=0)
        Xc = X_used - X_mean
        s_kmeans = Xc @ axis_unit
        per_cluster_stats = _per_cluster_stats(s_kmeans, labels_used)
        midpoint = (centroids[0] @ axis_unit + centroids[1] @ axis_unit) / 2.0 - X_mean @ axis_unit
        per["s_kmeans_diagnostic"] = {
            "n_events_used": int(X_used.shape[0]),
            "axis_norm_in_X": axis_norm,
            "s_kmeans_min": float(s_kmeans.min()),
            "s_kmeans_max": float(s_kmeans.max()),
            "s_kmeans_median": float(np.median(s_kmeans)),
            "midpoint_value": float(midpoint),
            **per_cluster_stats,
            "interpretation": (
                "Supervised projection along PR-2 KMeans axis. "
                "High Cohen's d only confirms PR-2 clusters are linearly "
                "separable in rank space, NOT that they are bistable."
            ),
        }

    # ---- Coordinate-free PR-2 label transition sanity ----
    sanity = compute_pr2_label_transition_sanity(
        labels_used, block_ids_used,
        n_clusters=2, n_perm=LABEL_TRANSITION_N_PERM, rng_seed=0,
    )
    per["label_transition_sanity"] = sanity

    with open(per_subject_path, "w") as f:
        json.dump(per, f, indent=2, default=str)

    return per


def _format_value(val: Any) -> Any:
    if isinstance(val, float):
        if not np.isfinite(val):
            return ""
        return f"{val:.6f}"
    return val


def _refresh_cohort_csv(rows: List[Dict[str, Any]]) -> None:
    fields = [
        "sid", "dataset", "subject",
        "n_events_eligible", "n_chan_union",
        "pc1_ratio", "pc2_ratio", "pc3_ratio", "cumulative_top3",
        "var_explained_curve", "residual_median", "residual_p95",
        "angle_deg", "axis_explained_in_pc", "tangent_source",
        "centroid_source", "n_cluster_0", "n_cluster_1",
        "curve_n_iter", "curve_converged",
        "gof_pass",
        "s_kmeans_n_used", "s_kmeans_cohens_d",
        "s_kmeans_sep_iqr", "s_kmeans_threshold_acc",
        "label_lambda_2", "label_lambda_2_z", "label_lambda_2_p_emp",
        "label_n_pairs",
        "excluded_from_h3_main",
        "error", "warning", "warning_labels",
    ]
    expanded: List[Dict[str, Any]] = []
    for r in rows:
        pca = r.get("pca", {}) or {}
        pc_ratio = pca.get("explained_variance_ratio") or []
        gof = r.get("gof", {}) or {}
        angle = r.get("angle_to_kmeans_axis", {}) or {}
        curve = r.get("principal_curve", {}) or {}
        nc = r.get("n_in_cluster", []) or []
        diag = r.get("s_kmeans_diagnostic", {}) or {}
        sanity = r.get("label_transition_sanity", {}) or {}
        sanity_obs = sanity.get("obs", {}) or {}
        expanded.append({
            "sid": r.get("sid", ""),
            "dataset": r.get("dataset", ""),
            "subject": r.get("subject", ""),
            "n_events_eligible": r.get("n_events_eligible", 0),
            "n_chan_union": r.get("n_chan_union", 0),
            "pc1_ratio": pc_ratio[0] if len(pc_ratio) >= 1 else "",
            "pc2_ratio": pc_ratio[1] if len(pc_ratio) >= 2 else "",
            "pc3_ratio": pc_ratio[2] if len(pc_ratio) >= 3 else "",
            "cumulative_top3": pca.get("cumulative_top_k", ""),
            "var_explained_curve": gof.get("var_explained_curve", ""),
            "residual_median": gof.get("residual_median", ""),
            "residual_p95": gof.get("residual_p95", ""),
            "angle_deg": angle.get("angle_deg", ""),
            "axis_explained_in_pc": angle.get("axis_explained_in_pc", ""),
            "tangent_source": angle.get("tangent_source", ""),
            "centroid_source": r.get("centroid_source", ""),
            "n_cluster_0": nc[0] if len(nc) >= 1 else "",
            "n_cluster_1": nc[1] if len(nc) >= 2 else "",
            "curve_n_iter": curve.get("n_iter", ""),
            "curve_converged": curve.get("converged", ""),
            "gof_pass": r.get("gof_pass", ""),
            "s_kmeans_n_used": diag.get("n_events_used", ""),
            "s_kmeans_cohens_d": diag.get("cohens_d", ""),
            "s_kmeans_sep_iqr": diag.get("sep_index_iqr", ""),
            "s_kmeans_threshold_acc": diag.get("midpoint_threshold_accuracy", ""),
            "label_lambda_2": sanity_obs.get("lambda_2", ""),
            "label_lambda_2_z": sanity.get("z_lambda_2", ""),
            "label_lambda_2_p_emp": sanity.get("p_empirical", ""),
            "label_n_pairs": sanity_obs.get("n_pairs", ""),
            "excluded_from_h3_main": r.get("excluded_from_h3_main", ""),
            "error": r.get("error", "") or diag.get("error", "") or sanity.get("error", ""),
            "warning": r.get("warning", ""),
            "warning_labels": r.get("warning_labels", ""),
        })
    with open(COHORT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in expanded:
            w.writerow({k: _format_value(row.get(k, "")) for k in fields})
    logger.info("Refreshed %s (%d rows)", COHORT_CSV, len(expanded))


def main() -> int:
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    if not files:
        logger.error("No per-subject files found in %s", PER_SUBJECT_DIR)
        return 1
    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            r = _augment_one(fp)
        except Exception as exc:
            logger.exception("Failed on %s: %s", fp.name, exc)
            with open(fp) as f:
                r = json.load(f)
            r["s_kmeans_diagnostic"] = {"error": f"{type(exc).__name__}:{exc}"}
            r["label_transition_sanity"] = {"error": f"{type(exc).__name__}:{exc}"}
            with open(fp, "w") as f:
                json.dump(r, f, indent=2, default=str)
        rows.append(r)
        diag = r.get("s_kmeans_diagnostic", {}) or {}
        sanity = r.get("label_transition_sanity", {}) or {}
        sanity_obs = sanity.get("obs", {}) or {}
        excluded = r.get("excluded_from_h3_main", "")
        if excluded:
            logger.info("%-25s EXCLUDED  reason=%s", r.get("sid", "?"), excluded)
        elif "error" in diag:
            logger.info("%-25s ERROR    %s", r.get("sid", "?"), diag.get("error"))
        else:
            cd = diag.get("cohens_d", float("nan"))
            lam2 = sanity_obs.get("lambda_2", float("nan"))
            z = sanity.get("z_lambda_2", float("nan"))
            p = sanity.get("p_empirical", float("nan"))
            logger.info(
                "%-25s s_kmeans d=%.2f  λ₂=%.3f  z=%.2f  p=%.4f",
                r.get("sid", "?"),
                cd if isinstance(cd, (int, float)) and np.isfinite(cd) else float("nan"),
                lam2 if isinstance(lam2, (int, float)) and np.isfinite(lam2) else float("nan"),
                z if isinstance(z, (int, float)) and np.isfinite(z) else float("nan"),
                p if isinstance(p, (int, float)) and np.isfinite(p) else float("nan"),
            )

    _refresh_cohort_csv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
