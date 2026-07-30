#!/usr/bin/env python3
"""Summarize parameter-matched nongated architecture sensitivities patient-first."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OLD = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
SEEDS = (20260725, 20260726, 20260727)
CONTROLS = (
    "linear_state_parammatched_h64",
    "vanilla_rnn_parammatched_h48",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/topic5_ordered_history_architecture_audit/analysis",
    )
    args = parser.parse_args()
    run_root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in run_root.rglob("heldout_metrics.csv"):
        frame = pd.read_csv(path)
        if len(frame) != 1:
            raise RuntimeError(f"{path}: expected one row")
        rows.append(frame)
    new = pd.concat(rows, ignore_index=True)
    expected = 34 * len(SEEDS) * len(CONTROLS)
    if len(new) != expected or set(new.control) != set(CONTROLS):
        raise RuntimeError(f"parameter-matched ladder incomplete: {len(new)}/{expected}")

    old_rows = []
    for seed in SEEDS:
        for path in (OLD / f"seed_{seed}").glob("*/heldout_metrics.csv"):
            frame = pd.read_csv(path)
            old_rows.append(
                frame.loc[
                    frame.control.isin(["unordered_prefix", "full_history_gru"])
                ]
            )
    old = pd.concat(old_rows, ignore_index=True)
    all_seed = pd.concat([new, old], ignore_index=True)
    collapsed = (
        all_seed.groupby(["subject", "control"], as_index=False)
        .median(numeric_only=True)
    )
    nll = collapsed.pivot(
        index="subject", columns="control", values="heldout_event_nll"
    )
    rows_out = []
    status = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": "PARAMETER_MATCHED_SENSITIVITY_COMPLETE",
        "target_values_read": False,
        "reference_parameter_count": int(
            old.loc[old.control.eq("full_history_gru"), "n_parameters"].median()
        ),
        "comparisons": {},
    }
    for index, control in enumerate(CONTROLS):
        gain = nll["unordered_prefix"] - nll[control]
        rng = np.random.default_rng(20260729 + index)
        medians = np.median(
            rng.choice(gain.to_numpy(float), (10000, len(gain)), replace=True),
            axis=1,
        )
        try:
            p = float(wilcoxon(gain, alternative="greater").pvalue)
        except ValueError:
            p = 1.0
        record = {
            "control": control,
            "hidden_size": int(new.loc[new.control.eq(control), "hidden_size"].iloc[0]),
            "n_parameters": int(
                new.loc[new.control.eq(control), "n_parameters"].median()
            ),
            "median_nll_gain_vs_unordered": float(np.median(gain)),
            "ci95": np.quantile(medians, [0.025, 0.975]).tolist(),
            "n_positive_of_34": int(np.count_nonzero(gain > 0)),
            "wilcoxon_greater_p": p,
        }
        rows_out.append(record)
        status["comparisons"][control] = record
    order = np.argsort([row["wilcoxon_greater_p"] for row in rows_out])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(
            1.0,
            (len(rows_out) - rank)
            * rows_out[index]["wilcoxon_greater_p"],
        )
        running = max(running, adjusted)
        rows_out[index]["holm_p"] = running
        status["comparisons"][rows_out[index]["control"]]["holm_p"] = running
    pd.DataFrame(rows_out).to_csv(
        output / "parameter_matched_architecture_sensitivity.csv", index=False
    )
    (output / "PARAMETER_MATCHED_SENSITIVITY.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
