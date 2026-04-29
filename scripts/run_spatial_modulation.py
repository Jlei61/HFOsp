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
EPILEPSIAE_NEW_GPU_ROOT = Path("results/hfo_detection")
RESULTS_DIR = Path("results/spatial_modulation")
SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
FOCUS_REL_FILE_EP = Path("results/epilepsiae_electrode_focus_rel.json")

YUQUAN_SUBJECTS = [
    "chengshuai", "huangwanling", "liyouran", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]

# Per-metric pre-registered direction for Epilepsiae i/l/e gradient.
# "greater": expected i > l > e (Wilcoxon alternative when comparing first vs second of pair)
# "less":    expected i < l < e
# "two-sided": no pre-registered direction (still report 3-pair Wilcoxon two-sided + Bonferroni)
# event_rate is intentionally omitted: SOZ/event-rate confound is well known and reported
# in cohort medians for context but excluded from monotonicity hypothesis testing.
METRIC_DIRECTIONS: Dict[str, str] = {
    "iei_detrended_r":  "greater",   # SOZ has stronger short-range memory after detrend
    "detrend_fraction": "less",      # SOZ has less slow drift (more local short-range)
    "iei_median":       "less",      # SOZ shorter inter-event intervals
    "iei_p02":          "less",      # 2nd percentile follows iei_median direction
    "iei_lag1_r":       "two-sided", # raw serial corr confounded by drift
    "iei_cv":           "two-sided", # direction unclear
}
EVENT_RATE_METRIC = "event_rate"  # confound-only, no monotonicity test


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


def _resolve_subject_dir(subject: str, dataset: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    if dataset == "epilepsiae":
        # Stage 0 census (2026-04-27) confirmed legacy gpu_npz are corrupt stubs
        # for all 20 subjects; Stage 2 reads the new pipeline output instead.
        # See docs/archive/topic3/epilepsiae_artifact_census_2026-04-27.md.
        return EPILEPSIAE_NEW_GPU_ROOT / subject
    raise ValueError(f"Unknown dataset: {dataset}")


def run_subject(
    subject: str,
    refine_k: float,
    min_count: int,
    min_rate: float,
    dataset: str,
    soz_channels: Optional[List[str]] = None,
    focus_rel_dict: Optional[Dict[str, list]] = None,
) -> Optional[Dict[str, Any]]:
    """Run per-channel metrics + region annotation for a single subject.

    Yuquan: binary soz/non_soz via soz_channels.
    Epilepsiae: i/l/e/unknown via focus_rel_dict; soz_channels ignored.
        focus_rel_dict missing → all channels labelled 'unknown' and the
        subject is auto-excluded from cohort paired stats but still gets
        per-channel JSON output.
    """
    subject_dir = _resolve_subject_dir(subject, dataset)
    if not subject_dir.exists():
        logger.warning(f"  {subject}: subject_dir {subject_dir} missing, skipping")
        return None

    t0 = time.time()
    loaded = load_perchannel_events_relaxed(
        subject_dir, dataset,
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

    # Region annotation: dataset-specific
    if dataset == "epilepsiae":
        if focus_rel_dict is None:
            # focus_rel missing for this subject → label every channel 'unknown'
            # so paired stats auto-exclude. annotate_channels_soz with empty
            # soz_set would mislabel them 'non_soz', which is wrong.
            region_labels = {ch: "unknown" for ch in ch_names}
        else:
            region_labels = annotate_channels_soz(
                ch_names, soz_channels=[], focus_rel=focus_rel_dict,
            )
    else:
        region_labels = annotate_channels_soz(
            ch_names, soz_channels=soz_channels or [],
        )

    channel_metrics = []
    for ch in ch_names:
        events = per_ch_events[ch]
        m = compute_perchannel_metrics(
            ch, events, block_ranges, total_hours,
        )
        m["region_label"] = region_labels.get(ch, "unknown")
        m["is_lagpat"] = ch in lagpat_channels
        channel_metrics.append(m)

    elapsed = time.time() - t0

    if dataset == "epilepsiae":
        clean = [m for m in channel_metrics if not m["artifact_suspect"]]
        n_i = sum(1 for m in clean if m["region_label"] == "i")
        n_l = sum(1 for m in clean if m["region_label"] == "l")
        n_e = sum(1 for m in clean if m["region_label"] == "e")
        n_unknown = sum(1 for m in clean if m["region_label"] == "unknown")
        n_artifact = sum(1 for m in channel_metrics if m["artifact_suspect"])
        logger.info(
            f"  {subject}: {len(ch_names)} channels "
            f"(i={n_i} l={n_l} e={n_e} unknown={n_unknown} artifact={n_artifact}) "
            f"in {elapsed:.1f}s"
        )
        region_counts = {"i": n_i, "l": n_l, "e": n_e, "unknown": n_unknown}
    else:
        n_soz = sum(1 for m in channel_metrics if m["region_label"] == "soz" and not m["artifact_suspect"])
        n_nonsoz = sum(1 for m in channel_metrics if m["region_label"] == "non_soz" and not m["artifact_suspect"])
        n_artifact = sum(1 for m in channel_metrics if m["artifact_suspect"])
        logger.info(
            f"  {subject}: {len(ch_names)} channels "
            f"(soz={n_soz} nonsoz={n_nonsoz} artifact={n_artifact}) "
            f"in {elapsed:.1f}s"
        )
        region_counts = {"soz": n_soz, "non_soz": n_nonsoz}

    return {
        "subject": subject,
        "dataset": dataset,
        "refine_k": refine_k,
        "n_channels": len(ch_names),
        "n_artifact": n_artifact,
        "region_counts": region_counts,
        "total_hours": total_hours,
        "lagpat_overlap": len([c for c in ch_names if c in lagpat_channels]),
        "channel_metrics": channel_metrics,
    }


def compute_cohort_statistics(
    all_results: List[Dict[str, Any]],
    min_group_channels: int = 3,
) -> Dict[str, Any]:
    """Yuquan binary SOZ vs non-SOZ paired statistics (PR-1 backwards-compatible)."""

    paired_rows = []

    for result in all_results:
        subj = result["subject"]
        metrics = result["channel_metrics"]

        clean = [m for m in metrics if not m["artifact_suspect"]]
        soz_chns = [m for m in clean if m["region_label"] == "soz"]
        nonsoz_chns = [m for m in clean if m["region_label"] == "non_soz"]

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


def _is_monotonic(i_med: float, l_med: float, e_med: float, direction: str) -> bool:
    """Check whether (i_med, l_med, e_med) follows pre-registered monotonic order.

    direction='greater': i > l > e (strict)
    direction='less':    i < l < e (strict)
    direction='two-sided': not applicable, return False
    """
    if direction == "greater":
        return (i_med > l_med) and (l_med > e_med)
    if direction == "less":
        return (i_med < l_med) and (l_med < e_med)
    return False


def compute_cohort_statistics_three_tier(
    all_results: List[Dict[str, Any]],
    metric_directions: Dict[str, str],
    min_group_channels: int = 3,
) -> Dict[str, Any]:
    """Epilepsiae i/l/e gradient statistics.

    For each metric:
      - Subject-level paired data: (i_median, l_median, e_median) per subject
        with all three regions ≥ min_group_channels (focus_rel-missing
        subjects auto-excluded — all their channels labelled 'unknown').
      - Three-pair paired Wilcoxon (i vs l, i vs e, l vs e), per-pair
        alternative = direction map; Bonferroni p_adj = min(p * 3, 1.0).
      - Subject-level monotonicity sign test: count subjects whose
        (i, l, e) medians are strictly monotonic in pre-registered direction;
        binomtest with null=1/6 (random label permutation = 6 orderings).

    event_rate: cohort medians + 3-pair Wilcoxon two-sided only (confound,
    no monotonicity test).
    """
    paired_rows = []

    for result in all_results:
        subj = result["subject"]
        metrics_list = result["channel_metrics"]

        clean = [m for m in metrics_list if not m["artifact_suspect"]]
        i_chns = [m for m in clean if m["region_label"] == "i"]
        l_chns = [m for m in clean if m["region_label"] == "l"]
        e_chns = [m for m in clean if m["region_label"] == "e"]

        if (len(i_chns) < min_group_channels
                or len(l_chns) < min_group_channels
                or len(e_chns) < min_group_channels):
            continue

        row = {
            "subject": subj,
            "n_i": len(i_chns),
            "n_l": len(l_chns),
            "n_e": len(e_chns),
        }
        for metric in (*metric_directions.keys(), EVENT_RATE_METRIC):
            i_vals = [m[metric] for m in i_chns if m[metric] is not None]
            l_vals = [m[metric] for m in l_chns if m[metric] is not None]
            e_vals = [m[metric] for m in e_chns if m[metric] is not None]
            row[f"i_median_{metric}"] = float(np.median(i_vals)) if i_vals else None
            row[f"l_median_{metric}"] = float(np.median(l_vals)) if l_vals else None
            row[f"e_median_{metric}"] = float(np.median(e_vals)) if e_vals else None
        paired_rows.append(row)

    if not paired_rows:
        return {"n_valid_subjects": 0, "tests": {}, "paired_data": []}

    df = pd.DataFrame(paired_rows)

    tests: Dict[str, Any] = {}
    for metric, direction in metric_directions.items():
        i_arr = df[f"i_median_{metric}"].values
        l_arr = df[f"l_median_{metric}"].values
        e_arr = df[f"e_median_{metric}"].values
        valid = np.array([
            (a is not None) and (b is not None) and (c is not None)
            and np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
            for a, b, c in zip(i_arr, l_arr, e_arr)
        ])
        if valid.sum() < 3:
            tests[metric] = {"direction": direction, "n_subjects": int(valid.sum()), "skipped": "n<3"}
            continue
        i_v = i_arr[valid].astype(float)
        l_v = l_arr[valid].astype(float)
        e_v = e_arr[valid].astype(float)

        pair_specs = [
            ("i_vs_l", i_v - l_v),
            ("i_vs_e", i_v - e_v),
            ("l_vs_e", l_v - e_v),
        ]
        pair_results = {}
        for pair_name, diffs in pair_specs:
            try:
                stat, p = stats.wilcoxon(diffs, alternative=direction)
                pair_results[pair_name] = {
                    "wilcoxon_stat": float(stat),
                    "wilcoxon_p": float(p),
                    "wilcoxon_p_bonferroni": float(min(p * 3.0, 1.0)),
                    "median_diff": float(np.median(diffs)),
                    "n_positive": int(np.sum(diffs > 0)),
                    "n_subjects": int(len(diffs)),
                }
            except Exception as exc:
                pair_results[pair_name] = {"error": str(exc), "n_subjects": int(len(diffs))}

        if direction in ("greater", "less"):
            mono_mask = np.array([
                _is_monotonic(i, l, e, direction)
                for i, l, e in zip(i_v, l_v, e_v)
            ])
            n_mono = int(mono_mask.sum())
            n_total = int(len(i_v))
            try:
                bt = stats.binomtest(n_mono, n_total, p=1.0 / 6.0, alternative="greater")
                mono_p = float(bt.pvalue)
            except Exception:
                mono_p = None
            monotonicity = {
                "n_monotonic": n_mono,
                "n_subjects": n_total,
                "fraction_monotonic": n_mono / n_total if n_total else None,
                "binomial_p_one_sided": mono_p,
                "null_p": 1.0 / 6.0,
            }
        else:
            monotonicity = None

        tests[metric] = {
            "direction": direction,
            "n_subjects": int(valid.sum()),
            "i_median_cohort": float(np.median(i_v)),
            "l_median_cohort": float(np.median(l_v)),
            "e_median_cohort": float(np.median(e_v)),
            "pair_tests": pair_results,
            "monotonicity": monotonicity,
        }

    # event_rate: cohort medians + 3-pair Wilcoxon two-sided (confound report only)
    er = EVENT_RATE_METRIC
    i_arr = df[f"i_median_{er}"].values
    l_arr = df[f"l_median_{er}"].values
    e_arr = df[f"e_median_{er}"].values
    valid = np.array([
        (a is not None) and (b is not None) and (c is not None)
        and np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
        for a, b, c in zip(i_arr, l_arr, e_arr)
    ])
    if valid.sum() >= 3:
        i_v = i_arr[valid].astype(float)
        l_v = l_arr[valid].astype(float)
        e_v = e_arr[valid].astype(float)
        pair_results = {}
        for pair_name, diffs in (("i_vs_l", i_v - l_v), ("i_vs_e", i_v - e_v), ("l_vs_e", l_v - e_v)):
            try:
                stat, p = stats.wilcoxon(diffs, alternative="two-sided")
                pair_results[pair_name] = {
                    "wilcoxon_stat": float(stat),
                    "wilcoxon_p_two_sided": float(p),
                    "wilcoxon_p_bonferroni": float(min(p * 3.0, 1.0)),
                    "median_diff": float(np.median(diffs)),
                    "n_positive": int(np.sum(diffs > 0)),
                    "n_subjects": int(len(diffs)),
                }
            except Exception as exc:
                pair_results[pair_name] = {"error": str(exc), "n_subjects": int(len(diffs))}
        tests[er] = {
            "direction": "confound_report_only",
            "n_subjects": int(valid.sum()),
            "i_median_cohort": float(np.median(i_v)),
            "l_median_cohort": float(np.median(l_v)),
            "e_median_cohort": float(np.median(e_v)),
            "pair_tests": pair_results,
            "note": "event_rate is a known SOZ confound; reported for context, no monotonicity hypothesis test",
        }

    return {
        "n_valid_subjects": len(paired_rows),
        "subjects": [r["subject"] for r in paired_rows],
        "min_group_channels": min_group_channels,
        "metric_directions": dict(metric_directions),
        "paired_data": paired_rows,
        "tests": tests,
    }


def _discover_epilepsiae_subjects() -> List[str]:
    """Subject discovery for Epilepsiae: numeric subdirs of results/hfo_detection/.

    NOT focus_rel.keys() — that would silently exclude subjects with new gpu
    but no i/l/e annotation. Those subjects still get per-channel metrics
    (with region_label='unknown') and are auto-excluded from paired stats.
    """
    if not EPILEPSIAE_NEW_GPU_ROOT.exists():
        return []
    return sorted(
        d.name for d in EPILEPSIAE_NEW_GPU_ROOT.iterdir()
        if d.is_dir() and d.name.isdigit()
    )


def main():
    parser = argparse.ArgumentParser(description="Spatial modulation PR-1/PR-2 analysis")
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae"], default="yuquan",
                        help="Dataset selector (default: yuquan, PR-1 backwards-compat)")
    parser.add_argument("--refine-k", type=float, default=0.0,
                        help="Refine threshold k for mean_std (default: 0.0)")
    parser.add_argument("--min-count", type=int, default=100)
    parser.add_argument("--min-rate", type=float, default=5.0)
    parser.add_argument("--min-group-channels", type=int, default=3,
                        help="Minimum channels per region group for cohort paired stats (default: 3)")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test on first 2 subjects")
    args = parser.parse_args()

    dataset = args.dataset

    if dataset == "yuquan":
        subjects = args.subjects or YUQUAN_SUBJECTS
        soz_yq = _load_json(SOZ_FILE_YQ)
        focus_rel_all: Dict[str, Any] = {}
    else:  # epilepsiae
        subjects = args.subjects or _discover_epilepsiae_subjects()
        soz_yq = {}
        focus_rel_all = _load_json(FOCUS_REL_FILE_EP)
        if not focus_rel_all:
            logger.warning(f"focus_rel JSON missing or empty: {FOCUS_REL_FILE_EP}")

    if args.smoke:
        subjects = subjects[:2]

    logger.info(
        f"Running spatial modulation on {len(subjects)} {dataset} subjects (k={args.refine_k})"
    )

    all_results = []
    for subj in subjects:
        if dataset == "yuquan":
            soz_chns = soz_yq.get(subj, [])
            result = run_subject(
                subj, args.refine_k, args.min_count, args.min_rate,
                dataset="yuquan", soz_channels=soz_chns,
            )
        else:
            fr = focus_rel_all.get(subj)  # may be None for missing focus_rel
            result = run_subject(
                subj, args.refine_k, args.min_count, args.min_rate,
                dataset="epilepsiae", focus_rel_dict=fr,
            )
        if result is not None:
            out_path = RESULTS_DIR / "per_channel_metrics" / dataset / f"{subj}_perchannel.json"
            _save(result, out_path)
            all_results.append(result)

    if not all_results:
        logger.error("No valid results. Exiting.")
        return

    if dataset == "yuquan":
        # PR-1 path: keep original output paths for backwards compatibility
        cohort_stats = compute_cohort_statistics(all_results, min_group_channels=args.min_group_channels)
        _save(cohort_stats, RESULTS_DIR / "soz_comparison" / "cohort_statistics.json")
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
        logger.info("\n=== COHORT SUMMARY (Yuquan SOZ vs non-SOZ) ===")
        logger.info(f"Valid subjects for paired test: {cohort_stats['n_valid_subjects']}")
        for metric, t in cohort_stats.get("tests", {}).items():
            logger.info(
                f"  {metric}: median_diff={t['median_diff']:.4f} "
                f"Wilcoxon p={t['wilcoxon_p']:.4f} "
                f"n={t['n_subjects']} "
                f"soz>nonsoz={t['n_soz_greater']}/{t['n_subjects']}"
            )
    else:
        # PR-2 path: dataset-layered output to avoid Yuquan/Epilepsiae collision
        ep_dir = RESULTS_DIR / "soz_comparison" / "epilepsiae"
        ep_dir.mkdir(parents=True, exist_ok=True)
        cohort_stats = compute_cohort_statistics_three_tier(
            all_results,
            metric_directions=METRIC_DIRECTIONS,
            min_group_channels=args.min_group_channels,
        )
        _save(cohort_stats, ep_dir / "cohort_three_tier_statistics.json")

        csv_rows = []
        for result in all_results:
            for m in result["channel_metrics"]:
                row = dict(m)
                row["subject"] = result["subject"]
                row["dataset"] = result["dataset"]
                csv_rows.append(row)
        csv_df = pd.DataFrame(csv_rows)
        csv_path = ep_dir / "cohort_i_l_e.csv"
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved {csv_path}")

        logger.info("\n=== COHORT SUMMARY (Epilepsiae i/l/e gradient) ===")
        logger.info(f"Valid subjects for paired test: {cohort_stats['n_valid_subjects']}")
        for metric, t in cohort_stats.get("tests", {}).items():
            if "skipped" in t:
                logger.info(f"  {metric}: skipped ({t['skipped']})")
                continue
            mono = t.get("monotonicity")
            mono_str = (
                f" mono={mono['n_monotonic']}/{mono['n_subjects']} p={mono['binomial_p_one_sided']:.3f}"
                if mono else ""
            )
            ptests = t.get("pair_tests", {})
            pair_str = " ".join(
                f"{name}:p={pr.get('wilcoxon_p_bonferroni', pr.get('wilcoxon_p_two_sided', np.nan)):.3f}"
                for name, pr in ptests.items()
            )
            logger.info(
                f"  {metric} ({t['direction']}): "
                f"i={t['i_median_cohort']:.4f} l={t['l_median_cohort']:.4f} e={t['e_median_cohort']:.4f} "
                f"n={t['n_subjects']}{mono_str} | {pair_str}"
            )


if __name__ == "__main__":
    main()
