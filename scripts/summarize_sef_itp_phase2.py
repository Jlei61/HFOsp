"""SEF-ITP Phase 2 cohort summarizer — H3 TOST + H4 Wilcoxon verdicts.

Plan: docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md (v1.0)
Plan: docs/superpowers/plans/2026-05-23-topic4-phase2-h4-v1.1-rank-endpoint-plan.md (v1.1)
Framework: docs/topic4_sef_itp_framework.md v1.0.6

v2 schema (Stage B 2026-05-24): H4 main line = rank-based endpoint geometry drift
(rank_endpoint.I_geom_rank); supplementary = participation field drift
(supplementary_participation_field.I_geom_participation). Adds cohort summaries for
spatial radius drift (per-side: source / sink centroid_rms / mean_pairwise /
min_enclosing_radius; plus source-sink centroid distance) and decision-k drift
(applicable only to ~9 subjects with swap_class in {strict, candidate}).

Reads per-subject JSONs from results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject/,
runs cohort-level TOST equivalence (with leave-one-out per advisor catch C) for each H3
metric, then assembles the integrated H3 verdict and H4 cohort verdict. Writes:

  results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_summary.json
  results/topic4_sef_itp/phase2_temporal_x_geometry/cohort_subjects.csv

Usage:
  python scripts/summarize_sef_itp_phase2.py
  python scripts/summarize_sef_itp_phase2.py --input-dir <custom>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase2 import (
    cohort_tost_with_loo,
    compute_h3_integrated_verdict,
    compute_h4_cohort_verdict,
)


DEFAULT_INPUT_DIR = Path("results/topic4_sef_itp/phase2_temporal_x_geometry/per_subject")
DEFAULT_OUT_DIR = Path("results/topic4_sef_itp/phase2_temporal_x_geometry")

# Framework v1.0.5 lock — δ_excess = 0.05 and TOST target per metric.
DELTA_EXCESS = 0.05
H3_METRIC_TARGETS: Dict[str, float] = {
    "lag1_same_excess": 0.0,
    "window_excess_10s": 0.0,
    "window_excess_30s": 0.0,
    "window_excess_60s": 0.0,
    "window_excess_1800s": 0.0,
    "run_length_lift": 1.0,
}


def _load_per_subject_records(input_dir: Path) -> List[dict]:
    records: List[dict] = []
    for p in sorted(input_dir.glob("*.json")):
        with p.open() as f:
            records.append(json.load(f))
    return records


def _extract_h3_metric_values(
    records: List[dict],
) -> Tuple[Dict[str, np.ndarray], List[str], List[str]]:
    """For each H3 metric, return (cohort array, per-subject value list aligned to subject list).

    Excludes subjects with exit_reason != 'ok'. Returns:
      metric_arrays: {metric_name: np.ndarray of values}
      subject_ids: ['<dataset>_<sid>', ...] aligned to those arrays
      excluded_ids: subjects skipped
    """
    metric_arrays: Dict[str, list] = {k: [] for k in H3_METRIC_TARGETS}
    subject_ids: List[str] = []
    excluded: List[str] = []
    for rec in records:
        sid_full = f"{rec['dataset']}_{rec['subject_id']}"
        if rec.get("exit_reason") != "ok" or not rec.get("h3"):
            excluded.append(sid_full)
            continue
        h3 = rec["h3"]
        subject_ids.append(sid_full)
        metric_arrays["lag1_same_excess"].append(h3["lag1_same_excess_n2"])
        for w in (10.0, 30.0, 60.0, 1800.0):
            key = f"window_excess_{int(w)}s"
            metric_arrays[key].append(h3["window_excess_n2"][f"{w}"])
        metric_arrays["run_length_lift"].append(h3["run_length_lift_n2"])
    return {k: np.asarray(v, dtype=float) for k, v in metric_arrays.items()}, subject_ids, excluded


def _extract_h4_arrays(records: List[dict]) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Pull I_rate (both null methods) + I_geom (v1.1 main + v1.0 supplementary) per-subject."""
    I_rate_shuffle: List[float] = []
    I_rate_circshift: List[float] = []
    I_geom_rank: List[float] = []          # v1.1 main line
    I_geom_participation: List[float] = [] # v1.0 supplementary
    n_epochs: List[int] = []
    subject_ids: List[str] = []
    for rec in records:
        if rec.get("exit_reason") != "ok" or not rec.get("h4"):
            continue
        h4 = rec["h4"]
        subject_ids.append(f"{rec['dataset']}_{rec['subject_id']}")
        I_rate_shuffle.append(h4["I_rate_epoch_order_shuffle"]["I_rate"])
        I_rate_circshift.append(h4["I_rate_circular_shift"]["I_rate"])
        # v2 schema: rank-based endpoint is main; participation field is supplementary
        rank_block = (h4.get("rank_endpoint") or {}).get("I_geom_rank") or {}
        supp_block = (h4.get("supplementary_participation_field") or {}).get("I_geom_participation") or {}
        # backward-compat fallback to v1 flat I_geom (treats as participation field)
        legacy = (h4.get("I_geom") or {}) if not (rank_block or supp_block) else {}
        I_geom_rank.append(rank_block.get("I_geom", float("nan")))
        I_geom_participation.append(
            supp_block.get("I_geom", legacy.get("I_geom", float("nan")))
        )
        n_epochs.append(h4["n_epochs"])
    return (
        {
            "I_rate_epoch_order_shuffle": np.asarray(I_rate_shuffle, dtype=float),
            "I_rate_circular_shift": np.asarray(I_rate_circshift, dtype=float),
            "I_geom_rank": np.asarray(I_geom_rank, dtype=float),
            "I_geom_participation": np.asarray(I_geom_participation, dtype=float),
            "n_epochs": np.asarray(n_epochs, dtype=int),
        },
        subject_ids,
    )


def _extract_spatial_radius_drift_per_subject(records: List[dict]) -> Dict[str, Any]:
    """Cohort summary of spatial radius drift: per-subject std-across-epochs of each radius
    metric (source/sink centroid_rms, mean_pairwise, min_enclosing, source_sink axis)."""
    rows: List[Dict[str, Any]] = []
    for rec in records:
        if rec.get("exit_reason") != "ok" or not rec.get("h4"):
            continue
        sid = f"{rec['dataset']}_{rec['subject_id']}"
        sp = (rec["h4"].get("spatial_radius") or {}).get("per_epoch_per_cluster") or []
        if not sp:
            continue
        # Aggregate across clusters and epochs: per (cluster, side, metric) std-across-epochs
        cluster_keys = sorted({c for ep in sp for c in (ep or {}).keys()})
        per_cluster: Dict[str, Dict[str, float]] = {}
        for ck in cluster_keys:
            src_rms = np.array([
                ((ep.get(ck) or {}).get("source") or {}).get("centroid_rms", np.nan)
                for ep in sp
            ], dtype=float)
            snk_rms = np.array([
                ((ep.get(ck) or {}).get("sink") or {}).get("centroid_rms", np.nan)
                for ep in sp
            ], dtype=float)
            src_mp = np.array([
                ((ep.get(ck) or {}).get("source") or {}).get("mean_pairwise", np.nan)
                for ep in sp
            ], dtype=float)
            snk_mp = np.array([
                ((ep.get(ck) or {}).get("sink") or {}).get("mean_pairwise", np.nan)
                for ep in sp
            ], dtype=float)
            src_meb = np.array([
                ((ep.get(ck) or {}).get("source") or {}).get("min_enclosing_radius", np.nan)
                for ep in sp
            ], dtype=float)
            snk_meb = np.array([
                ((ep.get(ck) or {}).get("sink") or {}).get("min_enclosing_radius", np.nan)
                for ep in sp
            ], dtype=float)
            axis_d = np.array([
                (ep.get(ck) or {}).get("source_sink_centroid_distance", np.nan)
                for ep in sp
            ], dtype=float)

            def _stat(arr: np.ndarray) -> Dict[str, float]:
                a = arr[np.isfinite(arr)]
                return {
                    "n_finite_epochs": int(a.size),
                    "median": float(np.median(a)) if a.size else float("nan"),
                    "std": float(np.std(a)) if a.size >= 2 else float("nan"),
                    "cv": float(np.std(a) / (np.median(a) + 1e-12)) if a.size >= 2 else float("nan"),
                }

            per_cluster[ck] = {
                "source_centroid_rms": _stat(src_rms),
                "sink_centroid_rms": _stat(snk_rms),
                "source_mean_pairwise": _stat(src_mp),
                "sink_mean_pairwise": _stat(snk_mp),
                "source_min_enclosing": _stat(src_meb),
                "sink_min_enclosing": _stat(snk_meb),
                "source_sink_centroid_distance": _stat(axis_d),
            }
        rows.append({"subject": sid, "per_cluster": per_cluster})
    return {"n_subjects": len(rows), "per_subject": rows}


def _extract_decision_k_drift_per_subject(records: List[dict]) -> Dict[str, Any]:
    """Cohort summary of decision-k drift (computed for ALL subjects; stratified by swap_class).

    User-return v2 catch 2026-05-23: don't gate at subject level; compute drift for all 23,
    report swap_class as context. Summarizer stratifies. swap-positive (strict/candidate)
    interpreted with confidence; swap=none acts as noise/control baseline.
    """
    rows: List[Dict[str, Any]] = []
    swap_class_dist: Dict[str, int] = {"strict": 0, "candidate": 0, "none": 0, "unknown": 0}
    drift_std_by_swap_class: Dict[str, List[float]] = {
        "strict": [], "candidate": [], "none": [], "unknown": []
    }
    for rec in records:
        if rec.get("exit_reason") != "ok" or not rec.get("h4"):
            continue
        sid = f"{rec['dataset']}_{rec['subject_id']}"
        sc = rec["h4"].get("swap_class") or "unknown"
        swap_class_dist[sc] = swap_class_dist.get(sc, 0) + 1
        dk_blk = rec["h4"].get("decision_k_drift") or {}
        # v2 schema (Stage B v1.1.1 2026-05-24): always computed; "computed"=True
        if not dk_blk.get("computed", dk_blk.get("applicable", False)):
            continue
        result = dk_blk.get("result") or {}
        dk_std = result.get("decision_k_std")
        if dk_std is not None and not (isinstance(dk_std, float) and np.isnan(dk_std)):
            drift_std_by_swap_class.setdefault(sc, []).append(float(dk_std))
        rows.append({
            "subject": sid,
            "swap_class": sc,
            "global_decision_k": dk_blk.get("global_decision_k_context") or rec["h4"].get("swap_decision_k"),
            "n_epochs_with_decision_k": result.get("n_epochs_with_decision_k"),
            "decision_k_mean": result.get("decision_k_mean"),
            "decision_k_std": result.get("decision_k_std"),
            "decision_k_range": result.get("decision_k_range"),
            "decision_k_per_epoch": result.get("decision_k_per_epoch"),
        })
    # Stratified summary: median drift_std per swap_class
    strat_summary: Dict[str, Dict[str, float]] = {}
    for sc, vals in drift_std_by_swap_class.items():
        if vals:
            strat_summary[sc] = {
                "n_subjects": len(vals),
                "median_decision_k_std": float(np.median(vals)),
                "iqr_decision_k_std": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))],
            }
    return {
        "n_total_subjects_with_drift": len(rows),
        "swap_class_distribution": swap_class_dist,
        "stratified_summary": strat_summary,
        "per_subject": rows,
    }


def _summarize_h3(
    records: List[dict], n_boot: int, seed: int
) -> dict:
    """Cohort-level H3 summary: per-metric TOST + LOO + integrated verdict."""
    metric_arrays, subject_ids, excluded = _extract_h3_metric_values(records)
    cohort_tost: Dict[str, dict] = {}
    for metric, values in metric_arrays.items():
        if len(values) == 0:
            cohort_tost[metric] = {
                "equivalence_pass": False,
                "leave_one_out_min_pass_rate": 0.0,
                "n": 0,
                "note": "empty cohort",
            }
            continue
        target = H3_METRIC_TARGETS[metric]
        tost = cohort_tost_with_loo(
            values, target=target, delta=DELTA_EXCESS, n_boot=n_boot, seed=seed,
        )
        cohort_tost[metric] = tost

    # Endpoint stability cohort medians.
    endpoint_fh = np.array([
        r["h3"]["endpoint_jaccard_first_half"]
        for r in records
        if r.get("exit_reason") == "ok" and r.get("h3")
    ])
    endpoint_oe = np.array([
        r["h3"]["endpoint_jaccard_odd_even"]
        for r in records
        if r.get("exit_reason") == "ok" and r.get("h3")
    ])
    endpoint_fh_median = float(np.median(endpoint_fh)) if endpoint_fh.size else 0.0
    endpoint_oe_median = float(np.median(endpoint_oe)) if endpoint_oe.size else 0.0

    verdict = compute_h3_integrated_verdict(
        cohort_tost=cohort_tost,
        endpoint_jaccard_first_half_median=endpoint_fh_median,
        endpoint_jaccard_odd_even_median=endpoint_oe_median,
    )

    return {
        "n_cohort": len(subject_ids),
        "subject_ids": subject_ids,
        "excluded": excluded,
        "delta_excess": DELTA_EXCESS,
        "metric_targets": H3_METRIC_TARGETS,
        "cohort_tost": cohort_tost,
        "endpoint_jaccard_first_half_median": endpoint_fh_median,
        "endpoint_jaccard_odd_even_median": endpoint_oe_median,
        "endpoint_jaccard_first_half_n_above_07": int(np.sum(endpoint_fh >= 0.7)),
        "endpoint_jaccard_odd_even_n_above_07": int(np.sum(endpoint_oe >= 0.7)),
        "integrated_verdict": verdict,
        "wording_lock": (
            "compatible with mark-independent sampling within tested precision"
        ),
    }


def _summarize_h4(records: List[dict]) -> dict:
    """Cohort-level H4 summary (v1.1 main + v1.0 supplementary):

    Main verdict: I_rate (both null methods) vs **I_geom_rank** (rank-based endpoint).
    Supplementary verdict: I_rate vs **I_geom_participation** (participation field, v1.0).
    Plus: cohort spatial radius drift summary + cohort decision-k drift summary.
    """
    arrs, subject_ids = _extract_h4_arrays(records)
    h4_out: Dict[str, Any] = {
        "n_cohort": len(subject_ids),
        "subject_ids": subject_ids,
    }
    # v1.1 MAIN verdicts: I_rate vs I_geom_rank (rank-based endpoint)
    h4_out["main_v1_1_rank_based"] = {}
    for null_label, key in (
        ("epoch_order_shuffle_literal", "I_rate_epoch_order_shuffle"),
        ("circular_shift_within_block_proposed", "I_rate_circular_shift"),
    ):
        verdict = compute_h4_cohort_verdict(arrs[key], arrs["I_geom_rank"])
        h4_out["main_v1_1_rank_based"][null_label] = verdict
    # v1.0 SUPPLEMENTARY verdicts: I_rate vs I_geom_participation (participation field)
    h4_out["supplementary_v1_0_participation_field"] = {}
    for null_label, key in (
        ("epoch_order_shuffle_literal", "I_rate_epoch_order_shuffle"),
        ("circular_shift_within_block_proposed", "I_rate_circular_shift"),
    ):
        verdict = compute_h4_cohort_verdict(arrs[key], arrs["I_geom_participation"])
        h4_out["supplementary_v1_0_participation_field"][null_label] = verdict
    h4_out["per_subject_I_rate_circular_shift"] = arrs["I_rate_circular_shift"].tolist()
    h4_out["per_subject_I_geom_rank"] = arrs["I_geom_rank"].tolist()
    h4_out["per_subject_I_geom_participation"] = arrs["I_geom_participation"].tolist()
    h4_out["per_subject_n_epochs"] = arrs["n_epochs"].tolist()
    h4_out["spec_amendment_note"] = (
        "I_rate_epoch_order_shuffle is the framework v1.0.5 §3.4 literal null — "
        "mathematically degenerate; reported for spec-faithful audit. "
        "I_rate_circular_shift_within_block is the proposed Phase 2 v1.0.0 amendment — "
        "see docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md. "
        "User decides which enters the framework on return."
    )
    h4_out["v1_1_note"] = (
        "MAIN verdict uses I_geom_rank (rank-based endpoint via masked template_rank); "
        "SUPPLEMENTARY uses I_geom_participation (participation field top-k, v1.0). "
        "User 2026-05-23 catch: v1.0 participation-field measures 'participation field "
        "drift', not 'propagation endpoint geometry drift' — main line corrected in v1.1."
    )
    # v1.1: spatial radius drift cohort summary
    h4_out["spatial_radius_drift_cohort"] = _extract_spatial_radius_drift_per_subject(records)
    # v1.1: decision-k drift cohort summary (swap-positive subset only)
    h4_out["decision_k_drift_cohort"] = _extract_decision_k_drift_per_subject(records)
    return h4_out


def _write_csv(records: List[dict], path: Path) -> None:
    """Flat per-subject table for visual inspection."""
    rows: List[dict] = []
    for rec in records:
        sid = f"{rec['dataset']}_{rec['subject_id']}"
        row: Dict[str, Any] = {
            "subject": sid,
            "dataset": rec["dataset"],
            "exit_reason": rec.get("exit_reason", ""),
        }
        h3 = rec.get("h3") or {}
        h4 = rec.get("h4") or {}
        row["lag1_same_excess_n2"] = h3.get("lag1_same_excess_n2", "")
        for w in (10.0, 30.0, 60.0, 1800.0):
            row[f"window_excess_{int(w)}s"] = (h3.get("window_excess_n2") or {}).get(f"{w}", "")
        row["run_length_lift_n2"] = h3.get("run_length_lift_n2", "")
        row["endpoint_jaccard_first_half"] = h3.get("endpoint_jaccard_first_half", "")
        row["endpoint_jaccard_odd_even"] = h3.get("endpoint_jaccard_odd_even", "")
        row["n_epochs"] = h4.get("n_epochs", "")
        I_rate_cs = (h4.get("I_rate_circular_shift") or {}).get("I_rate", "")
        I_rate_so = (h4.get("I_rate_epoch_order_shuffle") or {}).get("I_rate", "")
        row["I_rate_circular_shift"] = I_rate_cs
        row["I_rate_epoch_order_shuffle"] = I_rate_so
        # v2 schema (v1.1 rank-based MAIN + v1.0 participation SUPP)
        rk = (h4.get("rank_endpoint") or {}).get("I_geom_rank") or {}
        supp = (h4.get("supplementary_participation_field") or {}).get("I_geom_participation") or {}
        # backward-compat fallback to old flat I_geom
        legacy = (h4.get("I_geom") or {})
        row["I_geom_rank"] = rk.get("I_geom", "")
        row["I_geom_participation"] = supp.get("I_geom", legacy.get("I_geom", ""))
        row["swap_class"] = h4.get("swap_class", "")
        row["swap_decision_k"] = h4.get("swap_decision_k", "")
        dk = (h4.get("decision_k_drift") or {})
        if dk.get("applicable"):
            dk_res = dk.get("result") or {}
            row["dk_drift_mean"] = dk_res.get("decision_k_mean", "")
            row["dk_drift_std"] = dk_res.get("decision_k_std", "")
            row["dk_drift_range"] = str(dk_res.get("decision_k_range", ""))
        else:
            row["dk_drift_mean"] = ""
            row["dk_drift_std"] = ""
            row["dk_drift_range"] = ""
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=10_000,
                    help="Bootstrap iterations for TOST equivalence + LOO (default 10k).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.input_dir.exists():
        ap.error(f"Input dir not found: {args.input_dir}")

    records = _load_per_subject_records(args.input_dir)
    print(f"[summarize] loaded {len(records)} per-subject records from {args.input_dir}")

    h3 = _summarize_h3(records, n_boot=args.n_boot, seed=args.seed)
    h4 = _summarize_h4(records)

    summary = {
        "framework_version": "v1.0.6",
        "phase2_version": "v1.1.0",
        "schema_version": "sef_itp_phase2_cohort_v2_2026_05_24",
        "n_records_in": len(records),
        "h3": h3,
        "h4": h4,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "cohort_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=float))
    print(f"[summarize] wrote {summary_path}")

    csv_path = args.output_dir / "cohort_subjects.csv"
    _write_csv(records, csv_path)
    print(f"[summarize] wrote {csv_path}")

    print()
    print(f"H3 integrated verdict: {h3['integrated_verdict']}")
    print(
        f"  n_cohort={h3['n_cohort']}; endpoint_jaccard medians "
        f"(first_half / odd_even) = {h3['endpoint_jaccard_first_half_median']:.3f} / "
        f"{h3['endpoint_jaccard_odd_even_median']:.3f}"
    )
    for metric in ("lag1_same_excess", "window_excess_10s", "window_excess_30s",
                   "window_excess_60s", "window_excess_1800s", "run_length_lift"):
        t = h3["cohort_tost"][metric]
        print(
            f"  {metric}: equiv_pass={t['equivalence_pass']}, "
            f"loo_min_pass_rate={t.get('leave_one_out_min_pass_rate', 'n/a')}"
        )
    print()
    print("H4 cohort verdicts (v1.1 MAIN = rank-based endpoint):")
    for null_label in ("circular_shift_within_block_proposed", "epoch_order_shuffle_literal"):
        v = h4["main_v1_1_rank_based"][null_label]
        print(
            f"  [{null_label}] verdict={v['verdict']}, "
            f"wilcoxon_p={v['wilcoxon_p']}, cohen_d={v['cohen_d']}, "
            f"n_subjects={v['n_subjects']}"
        )
    print()
    print("H4 cohort verdicts (v1.0 SUPPLEMENTARY = participation field):")
    for null_label in ("circular_shift_within_block_proposed", "epoch_order_shuffle_literal"):
        v = h4["supplementary_v1_0_participation_field"][null_label]
        print(
            f"  [{null_label}] verdict={v['verdict']}, "
            f"wilcoxon_p={v['wilcoxon_p']}, cohen_d={v['cohen_d']}, "
            f"n_subjects={v['n_subjects']}"
        )
    print()
    dk_summary = h4["decision_k_drift_cohort"]
    print(
        f"Decision-k drift: computed for {dk_summary['n_total_subjects_with_drift']} subject(s); "
        f"swap_class distribution: {dk_summary['swap_class_distribution']}"
    )
    if dk_summary.get("stratified_summary"):
        print("  Stratified median decision_k_std by swap_class:")
        for sc, st in dk_summary["stratified_summary"].items():
            print(f"    {sc}: n={st['n_subjects']}, median_std={st['median_decision_k_std']:.3f}, IQR={st['iqr_decision_k_std']}")


if __name__ == "__main__":
    main()
