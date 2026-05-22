#!/usr/bin/env python3
"""Topic 4 Step 1 sensitivity — max_iter sweep + grid-wide event-weighted angle.

Two purposes:

1. **max_iter sensitivity**: in the main batch only 1/35 subjects converged
   inside max_iter=15. Re-fit principal curve on the H3 main subset with
   max_iter in {15, 30, 60} and report var_explained_curve / angle stability
   to verify that the temporary (non-converged) values are still reliable
   ranges, not artefacts of stopping early.

2. **Grid-wide event-weighted angle**: replace the single-point s_median angle
   with the per-event tangent-vs-KMeans-axis distribution. Reports median,
   p25/p75, and event-weighted mean. This addresses the "angle is too
   single-point" concern.

Outputs:
    results/topic4_attractor/step1_sensitivity.csv
    results/topic4_attractor/step1_sensitivity_summary.md
"""

from __future__ import annotations

import argparse
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
    fit_pca,
    fit_principal_curve,
    kmeans_centroids_from_labels,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("topic4_step1_sensitivity")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
# Legacy (non-masked) paths. `_apply_masked_paths()` swaps these to the
# `_masked` parallel tree (Topic 0 §4 / phantom rerun roadmap §5h).
PR2_PER_SUBJECT_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
OUT_DIR = REPO_ROOT / "results" / "topic4_attractor"
PER_SUBJECT_DIR = OUT_DIR / "per_subject"
SENS_CSV = OUT_DIR / "step1_sensitivity.csv"
SENS_MD = OUT_DIR / "step1_sensitivity_summary.md"


def _apply_masked_paths() -> None:
    """Reassign module-level path globals to the `_masked` parallel tree.

    Topic 0 phantom-rank rerun roadmap §5h: feed Step 1 sensitivity (max_iter
    sweep + grid/event-wide angles) into the masked Topic 4 results dir.
    """
    global PR2_PER_SUBJECT_DIR, OUT_DIR, PER_SUBJECT_DIR, SENS_CSV, SENS_MD
    PR2_PER_SUBJECT_DIR = REPO_ROOT / "results" / "interictal_propagation_masked" / "per_subject"
    OUT_DIR = REPO_ROOT / "results" / "topic4_attractor_masked"
    PER_SUBJECT_DIR = OUT_DIR / "per_subject"
    SENS_CSV = OUT_DIR / "step1_sensitivity.csv"
    SENS_MD = OUT_DIR / "step1_sensitivity_summary.md"

PR2_VALID_MIN_PART = 3
MAX_ITER_SET = (15, 30, 60)
N_COMPONENTS = 3


def _epilepsiae_subject_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(subject)


def _grid_wide_angle_stats(
    splines, components: np.ndarray, axis_unit_pc: np.ndarray,
    s_grid: np.ndarray,
) -> Dict[str, float]:
    """Compute angle distribution along the curve.

    For each grid point, compute cosine between curve tangent (spline
    derivative) and KMeans axis direction in PC space. Return acute-angle
    distribution (deg).
    """
    if splines is None or s_grid is None or len(splines) != components.shape[0]:
        return {"median_deg": float("nan")}
    derivs = np.column_stack([
        np.asarray(spl.derivative()(s_grid), dtype=float) for spl in splines
    ])  # (G, n_comp)
    norms = np.linalg.norm(derivs, axis=1)
    valid = norms > 1e-12
    derivs = derivs[valid]
    norms = norms[valid]
    if derivs.shape[0] == 0:
        return {"median_deg": float("nan")}
    cos_signed = (derivs @ axis_unit_pc) / norms
    cos_signed = np.clip(cos_signed, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(cos_signed)))
    return {
        "median_deg": float(np.median(angles)),
        "p25_deg": float(np.quantile(angles, 0.25)),
        "p75_deg": float(np.quantile(angles, 0.75)),
        "min_deg": float(np.min(angles)),
        "max_deg": float(np.max(angles)),
        "mean_deg": float(np.mean(angles)),
        "n_grid_used": int(derivs.shape[0]),
    }


def _event_weighted_angle_stats(
    splines, components: np.ndarray, axis_unit_pc: np.ndarray, s_event: np.ndarray,
) -> Dict[str, float]:
    """Same as grid-wide but evaluated at each event's projected s."""
    if splines is None or s_event.size == 0:
        return {"median_deg": float("nan")}
    derivs = np.column_stack([
        np.asarray(spl.derivative()(s_event), dtype=float) for spl in splines
    ])
    norms = np.linalg.norm(derivs, axis=1)
    valid = norms > 1e-12
    derivs = derivs[valid]
    norms = norms[valid]
    if derivs.shape[0] == 0:
        return {"median_deg": float("nan")}
    cos_signed = (derivs @ axis_unit_pc) / norms
    cos_signed = np.clip(cos_signed, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(cos_signed)))
    return {
        "median_deg": float(np.median(angles)),
        "p25_deg": float(np.quantile(angles, 0.25)),
        "p75_deg": float(np.quantile(angles, 0.75)),
        "n_events": int(derivs.shape[0]),
    }


def _run_one(sid: str, dataset: str, subject: str, *, mask_phantom: bool = False) -> List[Dict[str, Any]]:
    out_rows: List[Dict[str, Any]] = []
    sub_dir = _subject_dir(dataset, subject)
    pr2_json = PR2_PER_SUBJECT_DIR / f"{sid}.json"
    with open(pr2_json) as f:
        pr2 = json.load(f)
    labels_pr2 = np.asarray(pr2["adaptive_cluster"]["labels"], dtype=int)

    loaded = load_subject_propagation_events(sub_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)

    valid_pr2 = _valid_event_indices(bools, min_participating=PR2_VALID_MIN_PART)
    if labels_pr2.shape[0] != valid_pr2.shape[0]:
        logger.warning("%s: PR-2 label drift, skip sensitivity", sid)
        return out_rows

    X, eligible_idx = build_rank_feature_matrix(
        ranks, bools, min_participating=TOPIC4_MIN_PARTICIPATING,
        mask_phantom=mask_phantom,
    )
    if X.shape[0] < 100:
        return out_rows

    pca = fit_pca(X, n_components=N_COMPONENTS)
    components = pca["components"]

    pr2_lookup = {int(v): int(l) for v, l in zip(valid_pr2, labels_pr2)}
    topic4_labels = np.array(
        [pr2_lookup.get(int(ev), -1) for ev in eligible_idx], dtype=int
    )
    keep = topic4_labels >= 0
    if int(keep.sum()) < 4:
        return out_rows
    X_lbl = X[keep]
    lbls = topic4_labels[keep]
    centroids = kmeans_centroids_from_labels(X_lbl, lbls, n_clusters=2)
    axis = centroids[1] - centroids[0]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return out_rows
    axis_pc = components @ axis  # in PC subspace
    axis_pc_norm = float(np.linalg.norm(axis_pc))
    if axis_pc_norm < 1e-12:
        return out_rows
    axis_unit_pc = axis_pc / axis_pc_norm

    for max_iter in MAX_ITER_SET:
        curve = fit_principal_curve(pca["scores"], max_iter=max_iter)
        residuals = curve["residuals"]
        var_curve = 1.0 - float(
            np.sum(residuals ** 2) / max(X.shape[0] - 1, 1)
        ) / max(float(np.sum(pca["explained_variance"])), 1e-12)

        s_med = float(np.median(curve["s"])) if curve["s"].size else 0.0
        # single-point angle (at s_median, matching Step 1)
        try:
            tan = np.array(
                [float(spl.derivative()(s_med)) for spl in (curve.get("splines") or [])],
                dtype=float,
            )
            tan_norm = float(np.linalg.norm(tan))
            if tan_norm > 1e-12:
                cos_signed = float(np.dot(tan, axis_unit_pc) / tan_norm)
                angle_med = float(np.degrees(np.arccos(min(1.0, max(-1.0, abs(cos_signed))))))
            else:
                angle_med = float("nan")
        except Exception:
            angle_med = float("nan")

        grid_stats = _grid_wide_angle_stats(
            curve.get("splines"), components, axis_unit_pc, curve.get("s_grid"),
        )
        s_arr = curve.get("s")
        if s_arr is None:
            s_arr = np.zeros(0)
        event_stats = _event_weighted_angle_stats(
            curve.get("splines"), components, axis_unit_pc, np.asarray(s_arr),
        )

        out_rows.append({
            "sid": sid,
            "max_iter": int(max_iter),
            "n_iter_used": int(curve["n_iter"]),
            "converged": bool(curve["converged"]),
            "var_explained_curve": float(np.clip(var_curve, -1.0, 1.0)),
            "residual_mean_sq": float(curve["residual_mean_sq"]),
            "angle_at_s_median_deg": angle_med,
            "angle_grid_median_deg": grid_stats.get("median_deg", float("nan")),
            "angle_grid_p25_deg": grid_stats.get("p25_deg", float("nan")),
            "angle_grid_p75_deg": grid_stats.get("p75_deg", float("nan")),
            "angle_event_median_deg": event_stats.get("median_deg", float("nan")),
            "angle_event_p25_deg": event_stats.get("p25_deg", float("nan")),
            "angle_event_p75_deg": event_stats.get("p75_deg", float("nan")),
            "spline_n_used": int(curve.get("spline_n_used", -1)),
        })

    return out_rows


def _format_value(v: Any) -> Any:
    if isinstance(v, float):
        return "" if not np.isfinite(v) else f"{v:.6f}"
    return v


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--masked-features", action="store_true",
        help="Use masked PR-2 JSONs + masked rank features (Topic 0 phantom "
             "rerun roadmap §5h). Reads/writes from results/topic4_attractor_masked/.",
    )
    args = parser.parse_args()
    if args.masked_features:
        _apply_masked_paths()
        logger.info("Masked-features mode: PR2_PER_SUBJECT_DIR=%s OUT_DIR=%s",
                    PR2_PER_SUBJECT_DIR, OUT_DIR)

    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    cohort: List[tuple] = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if d.get("excluded_from_h3_main") or "skipped" in d or "error" in d:
            continue
        cohort.append((d["sid"], d["dataset"], d["subject"]))
    logger.info("Sensitivity cohort: %d subjects", len(cohort))

    all_rows: List[Dict[str, Any]] = []
    for sid, dataset, subject in cohort:
        try:
            rows = _run_one(sid, dataset, subject, mask_phantom=args.masked_features)
        except Exception as exc:
            logger.exception("%s failed: %s", sid, exc)
            continue
        all_rows.extend(rows)
        for r in rows:
            logger.info(
                "%-25s max_iter=%-3d n_iter=%-3d conv=%-5s "
                "var_curve=%.3f angle_grid_med=%.1f° angle_event_med=%.1f°",
                sid, r["max_iter"], r["n_iter_used"], str(r["converged"]),
                r["var_explained_curve"], r["angle_grid_median_deg"],
                r["angle_event_median_deg"],
            )

    fields = [
        "sid", "max_iter", "n_iter_used", "converged",
        "var_explained_curve", "residual_mean_sq",
        "angle_at_s_median_deg",
        "angle_grid_median_deg", "angle_grid_p25_deg", "angle_grid_p75_deg",
        "angle_event_median_deg", "angle_event_p25_deg", "angle_event_p75_deg",
        "spline_n_used",
    ]
    SENS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(SENS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: _format_value(r.get(k, "")) for k in fields})
    logger.info("Wrote %s (%d rows)", SENS_CSV, len(all_rows))

    # Brief summary md
    if all_rows:
        by_iter: Dict[int, List[Dict[str, Any]]] = {}
        for r in all_rows:
            by_iter.setdefault(int(r["max_iter"]), []).append(r)

        lines = []
        lines.append("# Topic 4 Step 1 sensitivity — max_iter & grid/event-wide angle")
        lines.append("")
        lines.append("**目的**：")
        lines.append("- 验证 var_explained_curve 与角度在不同 max_iter 下的稳定性，"
                     "确认 main batch 的 `max_iter=15` 临时态不是 artefact。")
        lines.append("- 替换单点 `s_median` 角度为 grid-wide / event-weighted 分布。")
        lines.append("")
        lines.append("## Per-max_iter aggregate")
        lines.append("")
        lines.append("| max_iter | n_subj | converged | var_curve median (range) | angle@s_median median | angle_grid median | angle_event median |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|")
        for mi in MAX_ITER_SET:
            sub = by_iter.get(mi, [])
            if not sub:
                continue
            n_conv = sum(1 for r in sub if r["converged"])
            vcs = [r["var_explained_curve"] for r in sub if np.isfinite(r["var_explained_curve"])]
            ams = [r["angle_at_s_median_deg"] for r in sub if np.isfinite(r["angle_at_s_median_deg"])]
            ags = [r["angle_grid_median_deg"] for r in sub if np.isfinite(r["angle_grid_median_deg"])]
            aes = [r["angle_event_median_deg"] for r in sub if np.isfinite(r["angle_event_median_deg"])]
            lines.append(
                f"| {mi} | {len(sub)} | {n_conv} | "
                f"{np.median(vcs):.3f} ({min(vcs):.3f}–{max(vcs):.3f}) | "
                f"{np.median(ams):.1f}° | {np.median(ags):.1f}° | {np.median(aes):.1f}° |"
            )
        lines.append("")
        lines.append("## 解读")
        lines.append("")
        lines.append(
            "- 如果三档 max_iter 下 `var_explained_curve` 与 `angle_event_median` 接近"
            "（差 < 5%），main batch 的临时态可信。"
        )
        lines.append(
            "- 如果 max_iter=60 显著改变结果（特别是降低 var 或大幅改变 angle），"
            "需要在 main batch 重跑 max_iter ≥ 60。"
        )
        lines.append(
            "- `angle_at_s_median` vs `angle_grid_median` vs `angle_event_median` 的差距："
            "若三者一致（<10° 差），说明主曲线整体几何与单点观察一致；"
            "若 grid_median 显著高/低于 s_median，说明曲线在不同 s 区间方向变化大，"
            "需要当 caveat 写入 H3 解释。"
        )
        with open(SENS_MD, "w") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Wrote %s", SENS_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
