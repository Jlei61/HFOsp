#!/usr/bin/env python3
"""Summarize completed v0.4 pilot or cheap-screen folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, required=True)
    parser.add_argument("--output-prefix", default="cohort")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.runs_root.glob("*/run_summary.json")):
        summary = json.loads(path.read_text())
        metrics = pd.read_csv(path.parent / "heldout_metrics.csv")
        by_control = metrics.set_index("control")
        gru = by_control.loc["full_history_gru"]
        shuffle = by_control.loc["rank_shuffle_gru"]
        empirical = by_control.loc["empirical_rank_distribution"]
        rows.append(
            {
                "subject": summary["heldout_subject"],
                "dataset": summary["dataset"],
                "status": summary["status"],
                "engineering_pass": bool(summary["engineering_pass"]),
                "ordered_history_nll_gain": float(
                    summary["ordered_history_nll_gain"]
                ),
                "gru_participation_mae": float(gru["participation_mae"]),
                "shuffle_participation_mae": float(
                    shuffle["participation_mae"]
                ),
                "shuffle_minus_gru_participation_mae": float(
                    shuffle["participation_mae"] - gru["participation_mae"]
                ),
                "gru_rank_wasserstein": float(gru["rank_wasserstein"]),
                "shuffle_rank_wasserstein": float(
                    shuffle["rank_wasserstein"]
                ),
                "shuffle_minus_gru_rank_wasserstein": float(
                    shuffle["rank_wasserstein"] - gru["rank_wasserstein"]
                ),
                "empirical_rank_wasserstein": float(
                    empirical["rank_wasserstein"]
                ),
                "gru_minus_empirical_rank_wasserstein": float(
                    gru["rank_wasserstein"] - empirical["rank_wasserstein"]
                ),
                "noninferiority_margin_rank_wasserstein": float(
                    summary[
                        "distribution_noninferiority_margin_rank_wasserstein"
                    ]
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output_csv = args.runs_root / f"{args.output_prefix}_fold_summary.csv"
    frame.to_csv(output_csv, index=False)
    complete = (
        len(frame) == int(args.expected_folds)
        and bool(frame.engineering_pass.all())
        if len(frame)
        else False
    )
    screen_direction_pass = False
    if complete:
        screen_direction_pass = bool(
            frame.ordered_history_nll_gain.median() > 0
            and frame.shuffle_minus_gru_rank_wasserstein.median() > 0
            and np.median(
                frame.gru_minus_empirical_rank_wasserstein
                - frame.noninferiority_margin_rank_wasserstein
            )
            <= 0
        )
    result = {
        "status": "complete" if complete else "incomplete_or_failed",
        "n_folds_found": int(len(frame)),
        "expected_folds": int(args.expected_folds),
        "all_engineering_pass": bool(
            len(frame) and frame.engineering_pass.all()
        ),
        "median_ordered_history_nll_gain": (
            float(frame.ordered_history_nll_gain.median())
            if len(frame)
            else None
        ),
        "median_shuffle_minus_gru_rank_wasserstein": (
            float(frame.shuffle_minus_gru_rank_wasserstein.median())
            if len(frame)
            else None
        ),
        "median_gru_minus_empirical_rank_wasserstein": (
            float(frame.gru_minus_empirical_rank_wasserstein.median())
            if len(frame)
            else None
        ),
        "screen_direction_pass": screen_direction_pass,
    }
    output_json = args.runs_root / f"{args.output_prefix}_summary.json"
    output_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
