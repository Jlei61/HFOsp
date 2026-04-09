#!/usr/bin/env python3
"""
Spatial modulation PR-1: per-channel SOZ vs non-SOZ temporal metric comparison.

Loads per-channel HFO events from *_gpu.npz with relaxed refine (k=0.0),
computes IEI serial correlation / dead-time / detrended metrics per channel,
annotates SOZ labels, and runs subject-within paired comparisons.

Usage:
    python scripts/run_spatial_modulation.py
    python scripts/run_spatial_modulation.py --refine-k 0.5 --subjects chengshuai liyouran
    python scripts/run_spatial_modulation.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.event_periodicity import (
    annotate_channels_soz,
    compute_perchannel_metrics,
    load_perchannel_events_relaxed,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("spatial_mod")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
RESULTS_DIR = Path("results/spatial_modulation")
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")

YUQUAN_SUBJECTS = [
    "chengshuai", "huangwanling", "liyouran", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    logger.info(f"Saved {path}")


def _load_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run_subject(
    subject: str,
    refine_k: float,
    min_count: int,
    min_rate: float,
    soz_channels: List[str],
) -> Optional[Dict[str, Any]]:
    """Run per-channel metrics + SOZ annotation for a single Yuquan subject."""
    subject_dir = YUQUAN_ROOT / subject

    t0 = time.time()
    loaded = load_perchannel_events_relaxed(
        subject_dir, "yuquan",
        refine_k=refine_k,
        min_count=min_count,
        min_rate=min_rate,
    )
    if loaded is None:
        logger.warning(f"  {subject}: no valid gpu data, skipping")
        return None

    ch_names = loaded["ch_names"]
    per_ch_events = loaded["per_ch_events"]
    block_ranges = loaded["block_ranges"]
    total_hours = loaded["total_hours"]
    lagpat_channels = loaded["lagpat_channels"]

    # SOZ annotation
    soz_labels = annotate_channels_soz(ch_names, soz_channels)

    # Per-channel metrics
    channel_metrics = []
    for ch in ch_names:
        events = per_ch_events[ch]
        m = compute_perchannel_metrics(
            ch, events, block_ranges, total_hours,
        )
        m["soz_label"] = soz_labels.get(ch, "unknown")
        m["is_lagpat"] = ch in lagpat_channels
        channel_metrics.append(m)

    elapsed = time.time() - t0

    n_soz = sum(1 for m in channel_metrics if m["soz_label"] == "soz" and not m["artifact_suspect"])
    n_nonsoz = sum(1 for m in channel_metrics if m["soz_label"] == "non_soz" and not m["artifact_suspect"])
    n_artifact = sum(1 for m in channel_metrics if m["artifact_suspect"])

    logger.info(
        f"  {subject}: {len(ch_names)} channels "
        f"(soz={n_soz} nonsoz={n_nonsoz} artifact={n_artifact}) "
        f"in {elapsed:.1f}s"
    )

    return {
        "subject": subject,
        "dataset": "yuquan",
        "refine_k": refine_k,
        "n_channels": len(ch_names),
        "n_soz": n_soz,
        "n_nonsoz": n_nonsoz,
        "n_artifact": n_artifact,
        "total_hours": total_hours,
        "lagpat_overlap": len([c for c in ch_names if c in lagpat_channels]),
        "channel_metrics": channel_metrics,
    }


def compute_cohort_statistics(
    all_results: List[Dict[str, Any]],
    min_group_channels: int = 3,
) -> Dict[str, Any]:
    """Compute cohort-level SOZ vs non-SOZ paired statistics."""

    paired_rows = []

    for result in all_results:
        subj = result["subject"]
        metrics = result["channel_metrics"]

        # Filter out artifact-suspect channels
        clean = [m for m in metrics if not m["artifact_suspect"]]
        soz_chns = [m for m in clean if m["soz_label"] == "soz"]
        nonsoz_chns = [m for m in clean if m["soz_label"] == "non_soz"]

        if len(soz_chns) < min_group_channels or len(nonsoz_chns) < min_group_channels:
            continue

        row = {"subject": subj, "n_soz": len(soz_chns), "n_nonsoz": len(nonsoz_chns)}

        for metric in ("iei_lag1_r", "iei_detrended_r", "detrend_fraction",
                       "iei_p02", "iei_median", "event_rate", "iei_cv"):
            soz_vals = [m[metric] for m in soz_chns if m[metric] is not None]
            nonsoz_vals = [m[metric] for m in nonsoz_chns if m[metric] is not None]

            if soz_vals and nonsoz_vals:
                row[f"soz_median_{metric}"] = float(np.median(soz_vals))
                row[f"nonsoz_median_{metric}"] = float(np.median(nonsoz_vals))
                row[f"diff_{metric}"] = row[f"soz_median_{metric}"] - row[f"nonsoz_median_{metric}"]
            else:
                row[f"soz_median_{metric}"] = None
                row[f"nonsoz_median_{metric}"] = None
                row[f"diff_{metric}"] = None

        paired_rows.append(row)

    if not paired_rows:
        return {"n_valid_subjects": 0, "tests": {}}

    df = pd.DataFrame(paired_rows)

    tests = {}
    for metric in ("iei_lag1_r", "iei_detrended_r", "detrend_fraction",
                    "iei_p02", "iei_median", "event_rate"):
        diffs = df[f"diff_{metric}"].dropna().values
        if len(diffs) < 3:
            continue

        try:
            stat, p = stats.wilcoxon(diffs, alternative="two-sided")
            n_positive = int(np.sum(diffs > 0))
            tests[metric] = {
                "wilcoxon_stat": float(stat),
                "wilcoxon_p": float(p),
                "n_subjects": len(diffs),
                "n_soz_greater": n_positive,
                "median_diff": float(np.median(diffs)),
                "mean_diff": float(np.mean(diffs)),
            }
        except Exception:
            pass

    return {
        "n_valid_subjects": len(paired_rows),
        "subjects": [r["subject"] for r in paired_rows],
        "paired_data": paired_rows,
        "tests": tests,
    }


def main():
    parser = argparse.ArgumentParser(description="Spatial modulation PR-1 analysis")
    parser.add_argument("--refine-k", type=float, default=0.0,
                        help="Refine threshold k for mean_std (default: 0.0)")
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-rate", type=float, default=5.0)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test on first 2 subjects")
    args = parser.parse_args()

    soz_yq = _load_json(SOZ_FILE_YQ)

    subjects = args.subjects or YUQUAN_SUBJECTS
    if args.smoke:
        subjects = subjects[:2]

    logger.info(f"Running spatial modulation PR-1 on {len(subjects)} subjects (k={args.refine_k})")

    all_results = []
    for subj in subjects:
        soz_chns = soz_yq.get(subj, [])
        result = run_subject(subj, args.refine_k, args.min_count, args.min_rate, soz_chns)
        if result is not None:
            # Save per-subject JSON
            out_path = RESULTS_DIR / "per_channel_metrics" / "yuquan" / f"{subj}_perchannel.json"
            _save(result, out_path)
            all_results.append(result)

    if not all_results:
        logger.error("No valid results. Exiting.")
        return

    # Cohort statistics
    cohort_stats = compute_cohort_statistics(all_results)
    _save(cohort_stats, RESULTS_DIR / "soz_comparison" / "cohort_statistics.json")

    # Build cohort CSV (one row per channel)
    csv_rows = []
    for result in all_results:
        for m in result["channel_metrics"]:
            row = dict(m)
            row["subject"] = result["subject"]
            row["dataset"] = result["dataset"]
            csv_rows.append(row)
    csv_df = pd.DataFrame(csv_rows)
    csv_path = RESULTS_DIR / "soz_comparison" / "cohort_soz_vs_nonsoz.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    # Print summary
    logger.info("\n=== COHORT SUMMARY ===")
    logger.info(f"Valid subjects for paired test: {cohort_stats['n_valid_subjects']}")
    for metric, t in cohort_stats.get("tests", {}).items():
        logger.info(
            f"  {metric}: median_diff={t['median_diff']:.4f} "
            f"Wilcoxon p={t['wilcoxon_p']:.4f} "
            f"n={t['n_subjects']} "
            f"soz>nonsoz={t['n_soz_greater']}/{t['n_subjects']}"
        )


if __name__ == "__main__":
    main()
