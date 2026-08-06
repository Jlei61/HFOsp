#!/usr/bin/env python3
"""Compare direct early-ictal transfer under two target-blind training budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEVELOPMENT = "epilepsiae_1146"
# Same tie band as the direct-transfer summary: patient contrasts are Spearman
# differences on 6-16 contacts, so ties surface as exact zeros or 1e-17 residue.
TIE_TOLERANCE = 1e-9


def _p(values: np.ndarray) -> float:
    value = np.asarray(values, float)
    value = value[np.isfinite(value)]
    # Strip ties before the test so SciPy's ``auto`` selects the exact null
    # instead of the anticonservative normal approximation.
    value = value[np.abs(value) > TIE_TOLERANCE]
    if not len(value):
        return 1.0
    return float(wilcoxon(value, alternative="two-sided", method="auto").pvalue)


def _load(root: Path) -> tuple[pd.DataFrame, dict]:
    patient = pd.read_csv(root / "direct_transfer_patient_metrics.csv")
    null_path = root / "direct_transfer_channel_null_patient_metrics.csv"
    if null_path.exists():
        channel_null = pd.read_csv(null_path)
        r2_null = channel_null.loc[
            channel_null.model == "R2",
            ["subject", "margin_vs_channel_null_median"],
        ].rename(columns={
            "margin_vs_channel_null_median": "rho_R2_channel_null_margin"
        })
        patient = patient.merge(r2_null, on="subject", validate="one_to_one")
    summary = json.loads((root / "DIRECT_TRANSFER_SUMMARY.json").read_text())
    return patient, summary


def _scientific_flags(summary: dict) -> dict[str, bool | None]:
    def supported(item: dict | None) -> bool | None:
        if item is None:
            return None
        return bool(
            item["median"] > TIE_TOLERANCE
            and item["one_sided_wilcoxon_p"] < 0.05
        )

    null = summary.get("all_contact_channel_shuffle", {}).get(
        "statistics", {}
    ).get("R2")
    absolute = summary.get("absolute_patient_rho_statistics", {}).get("R2")
    return {
        "r2_increment": supported(summary.get("primary_R2_minus_M1")),
        "order_control": supported(summary.get("true_R2_minus_strict_order_shuffle")),
        "zero_state_control": supported(summary.get("true_R2_minus_zero_state")),
        "absolute_channel_null": (
            None
            if null is None or absolute is None
            else bool(absolute["median"] > 0 and supported(null))
        ),
        "seizure_pairing": supported(
            summary.get("state_seizure_correct_minus_wrong", {}).get("R2")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--short-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    short, short_summary = _load(args.short_root)
    long, long_summary = _load(args.long_root)
    requested_metrics = [
        "rho_R2", "rho_increment_R2_minus_M1",
        "rho_true_R2_minus_order_shuffle", "rho_true_R2_minus_zero_state",
        "rho_R2_channel_null_margin",
    ]
    # The original c3 run predates the exact zero-state control.  Compare every
    # endpoint that is present in both runs instead of failing the whole budget
    # audit; unavailable endpoints remain explicitly listed in the JSON output.
    metrics = [
        metric for metric in requested_metrics
        if metric in short.columns and metric in long.columns
    ]
    columns = ["subject", *metrics]
    paired = short[columns].merge(
        long[columns], on="subject", suffixes=("_short", "_long"),
        validate="one_to_one",
    )
    paired = paired.loc[paired.subject != DEVELOPMENT].copy()
    for metric in metrics:
        paired[f"delta_long_minus_short__{metric}"] = (
            paired[f"{metric}_long"] - paired[f"{metric}_short"]
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "direct_training_budget_patient_comparison.csv", index=False)
    short_flags = _scientific_flags(short_summary)
    long_flags = _scientific_flags(long_summary)
    result = {
        "status": (
            "ROBUST_SCIENTIFIC_VERDICT_ACROSS_BUDGETS"
            if short_flags == long_flags
            else "TRAINING_BUDGET_SENSITIVE_SCIENTIFIC_VERDICT"
        ),
        "contract": "topic5_history_rnn_direct_training_budget_comparison_v0_2",
        "n_primary_patients": int(len(paired)),
        "short_history_cycles": (
            int(short_summary["history_checkpoint_cycles"])
            if short_summary.get("history_checkpoint_cycles") is not None else None
        ),
        "long_history_cycles": (
            int(long_summary["history_checkpoint_cycles"])
            if long_summary.get("history_checkpoint_cycles") is not None else None
        ),
        "short_scientific_flags": short_flags,
        "long_scientific_flags": long_flags,
        "compared_metrics": metrics,
        "unavailable_metrics": [
            metric for metric in requested_metrics if metric not in metrics
        ],
        "comparisons": {},
    }
    for metric in metrics:
        delta = paired[f"delta_long_minus_short__{metric}"].to_numpy(float)
        result["comparisons"][metric] = {
            "short_median": float(paired[f"{metric}_short"].median()),
            "long_median": float(paired[f"{metric}_long"].median()),
            "median_long_minus_short": float(np.median(delta)),
            "n_long_greater": int(np.sum(delta > TIE_TOLERANCE)),
            "n_short_greater": int(np.sum(delta < -TIE_TOLERANCE)),
            "n_tied": int(np.sum(np.abs(delta) <= TIE_TOLERANCE)),
            "two_sided_p": _p(delta),
        }
    (args.output_dir / "DIRECT_TRAINING_BUDGET_COMPARISON.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
