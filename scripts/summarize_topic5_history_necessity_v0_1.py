#!/usr/bin/env python3
"""Aggregate finite-history runs with the frozen v0.4 formal controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


NEW_CONDITIONS = ["history_1_gru", "history_2_gru", "history_3_gru"]
FROZEN_CONDITIONS = [
    "full_history_gru",
    "last_set_first_order",
    "rank_shuffle_gru",
    "unordered_prefix",
    "static_contact_hazard",
]


def _bootstrap_median(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = values[
        rng.integers(0, len(values), size=(20000, len(values)))
    ]
    median = np.median(draws, axis=1)
    return [
        float(np.quantile(median, 0.025)),
        float(np.quantile(median, 0.975)),
    ]


def _wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0)]
    return (
        float(wilcoxon(values, alternative="two-sided").pvalue)
        if len(values)
        else float("nan")
    )


def _condition_metrics(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        for fold in sorted(seed_root.iterdir()):
            if not fold.is_dir() or fold.name == "logs":
                continue
            done_path = fold / "DONE.json"
            metric_path = fold / "heldout_metrics.csv"
            if not done_path.exists() or not metric_path.exists():
                continue
            done = json.loads(done_path.read_text())
            if done.get("status") != "complete":
                continue
            frame = pd.read_csv(metric_path)
            if set(frame.condition) != set(NEW_CONDITIONS):
                raise RuntimeError(f"{fold}: incomplete finite-history conditions")
            rows.append(frame)
    if not rows:
        raise RuntimeError("no complete finite-history folds found")
    return pd.concat(rows, ignore_index=True)


def _frozen_metrics(root: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for metric_path in sorted(
            (root / f"seed_{seed}").glob("*/heldout_metrics.csv")
        ):
            frame = pd.read_csv(metric_path)
            frame = frame[frame.control.isin(FROZEN_CONDITIONS)].copy()
            frame = frame.rename(columns={"control": "condition"})
            rows.append(frame)
    if not rows:
        raise RuntimeError("no frozen formal controls found")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-root", type=Path, required=True)
    parser.add_argument("--frozen-formal-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds]

    new = _condition_metrics(args.history_root, seeds)
    frozen = _frozen_metrics(args.frozen_formal_root, seeds)
    expected = 34 * len(seeds)
    if new[["subject", "seed"]].drop_duplicates().shape[0] != expected:
        raise RuntimeError("finite-history formal folds are incomplete")
    if frozen[["subject", "seed"]].drop_duplicates().shape[0] != expected:
        raise RuntimeError("frozen formal folds are incomplete")

    columns = [
        "subject",
        "dataset",
        "condition",
        "seed",
        "heldout_event_nll",
        "n_eval_events",
        "top1_next_set_accuracy",
        "stop_brier",
        "stop_accuracy",
        "terminal_stop_probability",
        "nonterminal_stop_probability",
    ]
    combined = pd.concat(
        [new[columns], frozen[columns]], ignore_index=True
    )
    combined.to_csv(args.history_root / "all_condition_metrics.csv", index=False)
    seed_wide = combined.pivot(
        index=["subject", "dataset", "seed"],
        columns="condition",
        values="heldout_event_nll",
    ).reset_index()
    seed_wide.columns.name = None
    seed_wide.to_csv(args.history_root / "seed_level_nll.csv", index=False)

    patient = (
        seed_wide.groupby(["subject", "dataset"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed"])
    )
    patient["gain_history2_over_history1"] = (
        patient.history_1_gru - patient.history_2_gru
    )
    patient["gain_history3_over_history2"] = (
        patient.history_2_gru - patient.history_3_gru
    )
    patient["gain_full_over_history3"] = (
        patient.history_3_gru - patient.full_history_gru
    )
    patient["gain_full_over_first_order"] = (
        patient.last_set_first_order - patient.full_history_gru
    )
    patient["gain_full_over_rank_shuffle"] = (
        patient.rank_shuffle_gru - patient.full_history_gru
    )
    patient["gain_history1_over_first_order"] = (
        patient.last_set_first_order - patient.history_1_gru
    )
    patient["gain_full_over_unordered"] = (
        patient.unordered_prefix - patient.full_history_gru
    )
    patient.to_csv(
        args.history_root / "patient_seed_collapsed_nll.csv", index=False
    )

    contrasts = [
        "gain_history2_over_history1",
        "gain_history3_over_history2",
        "gain_full_over_history3",
        "gain_full_over_first_order",
        "gain_full_over_rank_shuffle",
        "gain_history1_over_first_order",
        "gain_full_over_unordered",
    ]
    contrast_summary = {}
    for index, name in enumerate(contrasts):
        values = patient[name].to_numpy(float)
        contrast_summary[name] = {
            "median": float(np.median(values)),
            "ci95": _bootstrap_median(values, 20260728 + index),
            "wilcoxon_two_sided_p": _wilcoxon(values),
            "n_positive": int(np.sum(values > 0)),
            "n_patients": int(len(values)),
        }
    necessary = [
        "gain_full_over_history3",
        "gain_full_over_first_order",
        "gain_full_over_rank_shuffle",
    ]
    short_multistep_history_supported = bool(
        contrast_summary["gain_history2_over_history1"]["ci95"][0] > 0
        and contrast_summary["gain_history3_over_history2"]["ci95"][0] > 0
    )
    history_beyond_three_supported = bool(
        contrast_summary["gain_full_over_history3"]["ci95"][0] > 0
    )
    full_history_composite_supported = bool(
        all(contrast_summary[name]["ci95"][0] > 0 for name in necessary)
    )
    coverage_failures = []
    denominator_failures = []
    fingerprint_sets = set()
    peak_allocated = []
    for seed in seeds:
        for fold in sorted((args.history_root / f"seed_{seed}").iterdir()):
            if not fold.is_dir() or fold.name == "logs":
                continue
            summary_path = fold / "run_summary.json"
            if not summary_path.exists():
                coverage_failures.append(f"{fold}: missing run summary")
                continue
            summary = json.loads(summary_path.read_text())
            fingerprint_sets.add(
                json.dumps(summary["input_fingerprints"], sort_keys=True)
            )
            peak_allocated.append(
                int(summary["resource"]["gpu_peak_allocated_bytes"])
            )
            reference_events = None
            for condition in NEW_CONDITIONS:
                coverage = json.loads(
                    (fold / condition / "coverage.json").read_text()
                )
                shared = coverage["shared"]
                if any(
                    int(value["completed_cycles"]) < 1
                    for value in shared.values()
                ):
                    coverage_failures.append(
                        f"{fold}/{condition}: shared coverage"
                    )
                if (
                    int(
                        coverage["heldout_calibration"]["completed_cycles"]
                    )
                    < 4
                ):
                    coverage_failures.append(
                        f"{fold}/{condition}: calibration coverage"
                    )
                events = pd.read_csv(
                    fold / condition / "heldout_event_nll.csv"
                )[["event_index", "event_source_index"]]
                if reference_events is None:
                    reference_events = events
                elif not events.equals(reference_events):
                    denominator_failures.append(
                        f"{fold}/{condition}: finite-history denominator"
                    )
            frozen_event_path = (
                args.frozen_formal_root
                / f"seed_{seed}"
                / fold.name
                / "heldout_event_nll.csv"
            )
            frozen_events = pd.read_csv(frozen_event_path)
            frozen_events = frozen_events[
                frozen_events.control == "full_history_gru"
            ][["event_index", "event_source_index"]].reset_index(drop=True)
            if reference_events is None or not frozen_events.equals(
                reference_events.reset_index(drop=True)
            ):
                denominator_failures.append(
                    f"{fold}: frozen/new heldout denominator"
                )

    summary = {
        "status": (
            "complete"
            if not coverage_failures and not denominator_failures
            else "audit_failed"
        ),
        "n_patients": int(len(patient)),
        "n_seeds": len(seeds),
        "n_formal_folds": int(
            new[["subject", "seed"]].drop_duplicates().shape[0]
        ),
        "n_new_models": int(len(new)),
        "contrasts": contrast_summary,
        "short_multistep_history_supported": short_multistep_history_supported,
        "history_beyond_three_supported": history_beyond_three_supported,
        "full_history_composite_supported": full_history_composite_supported,
        "coverage_failures": coverage_failures,
        "denominator_failures": denominator_failures,
        "single_input_fingerprint_set": len(fingerprint_sets) == 1,
        "maximum_gpu_peak_allocated_bytes": max(peak_allocated),
        "ictal_target_read": False,
    }
    (args.history_root / "history_necessity_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    (args.history_root / "DONE.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "n_formal_folds": summary["n_formal_folds"],
                "n_new_models": summary["n_new_models"],
                "short_multistep_history_supported": (
                    short_multistep_history_supported
                ),
                "history_beyond_three_supported": (
                    history_beyond_three_supported
                ),
                "full_history_composite_supported": (
                    full_history_composite_supported
                ),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
