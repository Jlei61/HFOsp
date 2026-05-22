"""SEF-ITP Phase 2 cohort summarizer — H3 TOST + H4 Wilcoxon verdicts.

Plan: docs/superpowers/plans/2026-05-23-topic4-phase2-h3-h4-plan.md
Framework: docs/topic4_sef_itp_framework.md v1.0.5

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
    """Pull I_rate (both null methods) + I_geom per-subject arrays for cohort Wilcoxon."""
    I_rate_shuffle: List[float] = []
    I_rate_circshift: List[float] = []
    I_geom: List[float] = []
    n_epochs: List[int] = []
    subject_ids: List[str] = []
    for rec in records:
        if rec.get("exit_reason") != "ok" or not rec.get("h4"):
            continue
        h4 = rec["h4"]
        subject_ids.append(f"{rec['dataset']}_{rec['subject_id']}")
        I_rate_shuffle.append(h4["I_rate_epoch_order_shuffle"]["I_rate"])
        I_rate_circshift.append(h4["I_rate_circular_shift"]["I_rate"])
        I_geom.append(h4["I_geom"]["I_geom"])
        n_epochs.append(h4["n_epochs"])
    return (
        {
            "I_rate_epoch_order_shuffle": np.asarray(I_rate_shuffle, dtype=float),
            "I_rate_circular_shift": np.asarray(I_rate_circshift, dtype=float),
            "I_geom": np.asarray(I_geom, dtype=float),
            "n_epochs": np.asarray(n_epochs, dtype=int),
        },
        subject_ids,
    )


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
    """Cohort-level H4 summary: I_rate (both null methods) vs I_geom Wilcoxon + Cohen's d."""
    arrs, subject_ids = _extract_h4_arrays(records)
    h4_out = {
        "n_cohort": len(subject_ids),
        "subject_ids": subject_ids,
    }
    # Two verdicts: one per I_rate null method.
    for null_label, key in (
        ("epoch_order_shuffle_literal", "I_rate_epoch_order_shuffle"),
        ("circular_shift_within_block_proposed", "I_rate_circular_shift"),
    ):
        verdict = compute_h4_cohort_verdict(arrs[key], arrs["I_geom"])
        h4_out[null_label] = verdict
    h4_out["per_subject_I_rate_circular_shift"] = arrs["I_rate_circular_shift"].tolist()
    h4_out["per_subject_I_geom"] = arrs["I_geom"].tolist()
    h4_out["per_subject_n_epochs"] = arrs["n_epochs"].tolist()
    h4_out["spec_amendment_note"] = (
        "I_rate_epoch_order_shuffle is the framework v1.0.5 §3.4 literal null — "
        "mathematically degenerate; reported for spec-faithful audit. "
        "I_rate_circular_shift_within_block is the proposed Phase 2 v1.0.0 amendment — "
        "see docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md. "
        "User decides which enters the framework on return."
    )
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
        row["I_geom"] = (h4.get("I_geom") or {}).get("I_geom", "")
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
        "framework_version": "v1.0.5",
        "phase2_version": "v1.0.0",
        "schema_version": "sef_itp_phase2_cohort_v1_2026_05_23",
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
    print("H4 cohort verdicts:")
    for null_label in ("circular_shift_within_block_proposed", "epoch_order_shuffle_literal"):
        v = h4[null_label]
        print(
            f"  [{null_label}] verdict={v['verdict']}, "
            f"wilcoxon_p={v['wilcoxon_p']}, cohen_d={v['cohen_d']}, "
            f"n_subjects={v['n_subjects']}"
        )


if __name__ == "__main__":
    main()
