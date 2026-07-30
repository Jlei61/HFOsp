#!/usr/bin/env python3
"""Aggregate the matched history-3 within-event rank-shuffle sensitivity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _bootstrap_median(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    draws = values[
        rng.integers(0, len(values), size=(20000, len(values)))
    ]
    medians = np.median(draws, axis=1)
    return [
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordered-root", type=Path, required=True)
    parser.add_argument("--shuffle-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    rows = []
    denominator_failures = []
    coverage_failures = []
    fingerprints = set()
    for seed in args.seeds:
        for shuffled_path in sorted(
            (args.shuffle_root / f"seed_{seed}").glob("*/heldout_metrics.csv")
        ):
            subject = shuffled_path.parent.name
            shuffled = pd.read_csv(shuffled_path)
            if list(shuffled.condition) != ["history_3_rank_shuffle_gru"]:
                raise RuntimeError(f"{shuffled_path}: unexpected conditions")
            ordered_path = (
                args.ordered_root
                / f"seed_{seed}"
                / subject
                / "heldout_metrics.csv"
            )
            ordered = pd.read_csv(ordered_path)
            ordered = ordered[ordered.condition == "history_3_gru"]
            if len(ordered) != 1:
                raise RuntimeError(f"{ordered_path}: missing ordered history-3")
            rows.append(
                {
                    "subject": subject,
                    "dataset": shuffled.iloc[0].dataset,
                    "seed": int(seed),
                    "ordered_history3_nll": float(
                        ordered.iloc[0].heldout_event_nll
                    ),
                    "rank_shuffle_history3_nll": float(
                        shuffled.iloc[0].heldout_event_nll
                    ),
                    "ordered_gain": float(
                        shuffled.iloc[0].heldout_event_nll
                        - ordered.iloc[0].heldout_event_nll
                    ),
                }
            )
            ordered_events = pd.read_csv(
                ordered_path.parent / "history_3_gru/heldout_event_nll.csv"
            )[["event_index", "event_source_index"]].reset_index(drop=True)
            shuffled_events = pd.read_csv(
                shuffled_path.parent
                / "history_3_rank_shuffle_gru/heldout_event_nll.csv"
            )[["event_index", "event_source_index"]].reset_index(drop=True)
            if not ordered_events.equals(shuffled_events):
                denominator_failures.append(
                    f"seed_{seed}/{subject}: heldout denominator"
                )
            fold_summary = json.loads(
                (shuffled_path.parent / "run_summary.json").read_text()
            )
            fingerprints.add(
                json.dumps(fold_summary["input_fingerprints"], sort_keys=True)
            )
            coverage = json.loads(
                (
                    shuffled_path.parent
                    / "history_3_rank_shuffle_gru/coverage.json"
                ).read_text()
            )
            if any(
                int(value["completed_cycles"]) < 1
                for value in coverage["shared"].values()
            ):
                coverage_failures.append(
                    f"seed_{seed}/{subject}: shared coverage"
                )
            if (
                int(coverage["heldout_calibration"]["completed_cycles"]) < 4
            ):
                coverage_failures.append(
                    f"seed_{seed}/{subject}: calibration coverage"
                )

    seed_frame = pd.DataFrame(rows)
    expected = 34 * len(args.seeds)
    if len(seed_frame) != expected:
        raise RuntimeError(f"expected {expected} folds, found {len(seed_frame)}")
    seed_frame.to_csv(args.shuffle_root / "seed_level_comparison.csv", index=False)
    patient = (
        seed_frame.groupby(["subject", "dataset"], as_index=False)
        .mean(numeric_only=True)
        .drop(columns=["seed"])
    )
    patient.to_csv(
        args.shuffle_root / "patient_seed_collapsed_comparison.csv", index=False
    )
    gain = patient.ordered_gain.to_numpy(float)
    nonzero = gain[gain != 0]
    ci = _bootstrap_median(gain, 20260731)
    summary = {
        "status": (
            "complete"
            if not denominator_failures and not coverage_failures
            else "audit_failed"
        ),
        "n_patients": len(patient),
        "n_seeds": len(args.seeds),
        "n_folds": len(seed_frame),
        "median_ordered_gain": float(np.median(gain)),
        "ordered_gain_ci95": ci,
        "wilcoxon_two_sided_p": (
            float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            if len(nonzero)
            else float("nan")
        ),
        "n_positive": int(np.sum(gain > 0)),
        "matched_order_sensitivity_supported": bool(ci[0] > 0),
        "denominator_failures": denominator_failures,
        "coverage_failures": coverage_failures,
        "single_input_fingerprint_set": len(fingerprints) == 1,
        "ictal_target_read": False,
    }
    (args.shuffle_root / "history3_rank_shuffle_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    (args.shuffle_root / "DONE.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
