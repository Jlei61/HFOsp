#!/usr/bin/env python3
"""Freeze G1 sequential hyperparameters without opening ictal targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    rows = []
    for path in sorted(root.glob("*/*/DONE.json")):
        payload = json.loads(path.read_text())
        contrast = payload["metrics"]["contrasts"]
        rows.append(
            {
                "configuration": path.parents[1].name,
                "subject": payload["heldout_subject"],
                "dataset": payload["heldout_subject"].split("_", 1)[0],
                "static_to_matched_gain": contrast[
                    "static_minus_matched_participation_bce"
                ],
                "static_to_chronological_gain": contrast[
                    "static_minus_chronological_participation_bce"
                ],
                "chronological_increment": contrast[
                    "matched_minus_chronological_participation_bce"
                ],
                "rank_increment": contrast[
                    "matched_minus_chronological_relative_rank_huber"
                ],
                "order_shuffle_cost": contrast[
                    "shuffle_minus_chronological_participation_bce"
                ],
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 9:
        raise RuntimeError(f"sequential development grid incomplete: {len(frame)}/9")
    frame.to_csv(root / "development_patient_metrics.csv", index=False)
    summaries = []
    for configuration, group in frame.groupby("configuration", sort=True):
        summaries.append(
            {
                "configuration": configuration,
                "n_patients": int(len(group)),
                "median_static_to_matched_gain": float(
                    group.static_to_matched_gain.median()
                ),
                "n_matched_better_than_static": int(
                    np.sum(group.static_to_matched_gain > 0)
                ),
                "median_chronological_increment": float(
                    group.chronological_increment.median()
                ),
                "n_chronological_increment_positive": int(
                    np.sum(group.chronological_increment > 0)
                ),
                "median_order_shuffle_cost": float(group.order_shuffle_cost.median()),
                "n_order_shuffle_cost_positive": int(
                    np.sum(group.order_shuffle_cost > 0)
                ),
                "median_rank_increment": float(group.rank_increment.median()),
            }
        )
    summary = pd.DataFrame(summaries)
    admissible = summary.loc[
        (summary.n_matched_better_than_static >= 2)
        & (summary.n_chronological_increment_positive >= 2)
        & (summary.n_order_shuffle_cost_positive >= 2)
    ]
    if len(admissible):
        chosen = admissible.sort_values(
            ["median_order_shuffle_cost", "median_chronological_increment"],
            ascending=False,
        ).iloc[0]
        status = "FROZEN_G1_CONFIGURATION"
    else:
        chosen = summary.sort_values(
            ["median_order_shuffle_cost", "median_chronological_increment"],
            ascending=False,
        ).iloc[0]
        status = "DEVELOPMENT_WARNING_NO_FULLY_ADMISSIBLE_CONFIGURATION"
    summary["selected"] = summary.configuration.eq(chosen.configuration)
    summary.to_csv(root / "development_configuration_summary.csv", index=False)
    payload = {
        "status": status,
        "target_values_read": False,
        "n_runs": int(len(frame)),
        "selected_configuration": str(chosen.configuration),
        "selection_rule": (
            "require M1>static, M2>M1, and shuffle>M2 in at least 2/3 "
            "development patients; then maximize median order-shuffle cost"
        ),
        "selected_metrics": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in chosen.to_dict().items()
        },
    }
    (root / "DEVELOPMENT_SELECTION.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
