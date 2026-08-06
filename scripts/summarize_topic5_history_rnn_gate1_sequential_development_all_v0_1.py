#!/usr/bin/env python3
"""Freeze one G1 configuration after the complete target-sealed dev audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED = {
    "h32_half2_lr3e4",
    "h32_half0p5_lr3e4",
    "h32_half2_lr1e3",
    "h16_half2_lr3e4_c3_k256",
    "h32_half6_lr3e4_c3_k256",
    "h32_half2_lr3e4_c3_k128",
    "h32_half2_lr3e4_c6_k256",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    source_hashes = {}
    for input_dir in args.input_dir:
        root = input_dir.resolve()
        for path in sorted(root.glob("*/*/DONE.json")):
            raw = path.read_bytes()
            payload = json.loads(raw)
            if bool(payload.get("target_values_read", True)):
                raise RuntimeError(f"target seal violated: {path}")
            contrast = payload["metrics"]["contrasts"]
            configuration = path.parents[1].name
            rows.append(
                {
                    "configuration": configuration,
                    "subject": payload["heldout_subject"],
                    "dataset": payload["heldout_subject"].split("_", 1)[0],
                    "static_to_matched_gain": contrast[
                        "static_minus_matched_participation_bce"
                    ],
                    "chronological_increment": contrast[
                        "matched_minus_chronological_participation_bce"
                    ],
                    "rank_increment": contrast[
                        "matched_minus_chronological_relative_rank_huber"
                    ],
                    "engineering_global_shuffle_cost": contrast[
                        "shuffle_minus_chronological_participation_bce"
                    ],
                }
            )
            source_hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    frame = pd.DataFrame(rows)
    counts = frame.groupby("configuration").subject.nunique().to_dict()
    if set(counts) != EXPECTED or any(counts[name] != 3 for name in EXPECTED):
        raise RuntimeError(f"development audit incomplete: {counts}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "development_patient_metrics.csv", index=False)
    summary_rows = []
    for configuration, group in frame.groupby("configuration", sort=True):
        summary_rows.append(
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
                "median_engineering_global_shuffle_cost": float(
                    group.engineering_global_shuffle_cost.median()
                ),
                "n_engineering_global_shuffle_positive": int(
                    np.sum(group.engineering_global_shuffle_cost > 0)
                ),
                "median_rank_increment": float(group.rank_increment.median()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    admissible = summary.loc[
        (summary.n_matched_better_than_static >= 2)
        & (summary.n_chronological_increment_positive >= 2)
        & (summary.n_engineering_global_shuffle_positive >= 2)
    ]
    candidate = admissible if len(admissible) else summary
    chosen = candidate.sort_values(
        [
            "median_chronological_increment",
            "median_engineering_global_shuffle_cost",
        ],
        ascending=False,
    ).iloc[0]
    status = (
        "FROZEN_ADMISSIBLE_G1_CONFIGURATION"
        if len(admissible)
        else "FROZEN_BEST_AVAILABLE_FOR_INDEPENDENT_G1_TEST"
    )
    summary["selected"] = summary.configuration.eq(chosen.configuration)
    summary.to_csv(output / "development_configuration_summary.csv", index=False)
    result = {
        "status": status,
        "target_values_read": False,
        "n_runs": int(len(frame)),
        "selected_configuration": str(chosen.configuration),
        "selection_rule": (
            "first require M1>static, M2>M1, and engineering global shuffle>M2 "
            "in at least 2/3 development patients; then maximize the primary "
            "median M2-M1 increment and use shuffle cost only as tie-breaker. "
            "If none is admissible, freeze the best primary M2-M1 configuration "
            "for the independent 31-patient formal test without calling dev a gate."
        ),
        "selected_metrics": {
            key: value.item() if hasattr(value, "item") else value
            for key, value in chosen.to_dict().items()
        },
        "source_done_sha256": source_hashes,
    }
    (output / "DEVELOPMENT_SELECTION.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
