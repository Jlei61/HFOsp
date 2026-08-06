#!/usr/bin/env python3
"""Summarize 3/10/30-cycle development convergence without ictal targets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PATIENTS = ("epilepsiae_1073", "epilepsiae_1146", "yuquan_chenziyang")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--c3-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    c3 = args.c3_root.resolve()
    rows = []
    for cycles in (3, 10, 30):
        source = c3 if cycles == 3 else root / f"c{cycles}"
        for subject in PATIENTS:
            done_path = source / subject / "DONE.json"
            if not done_path.exists():
                raise RuntimeError(f"missing convergence fold {done_path}")
            done = json.loads(done_path.read_text())
            metrics = done["metrics"]
            rows.append({
                "cycles": cycles,
                "subject": subject,
                "matched_bce": metrics["matched_unordered"]["participation_bce"],
                "chronological_bce": metrics["chronological_history"]["participation_bce"],
                "chronological_increment": metrics["contrasts"][
                    "matched_minus_chronological_participation_bce"
                ],
                "rank_increment": metrics["contrasts"][
                    "matched_minus_chronological_relative_rank_huber"
                ],
                "target_values_read": bool(done.get("target_values_read", True)),
            })
    frame = pd.DataFrame(rows)
    if frame.target_values_read.any():
        raise RuntimeError("convergence audit violated target seal")
    frame.to_csv(root / "real_convergence_metrics.csv", index=False)
    pivot = frame.pivot(index="subject", columns="cycles", values="chronological_bce")
    delta_10_to_30 = pivot[10] - pivot[30]
    increment = frame.pivot(index="subject", columns="cycles", values="chronological_increment")
    stable = bool(
        float(np.median(np.abs(delta_10_to_30))) < 0.002
        and np.all(np.sign(increment[10]) == np.sign(increment[30]))
    )
    result = {
        "status": "TRAINING_SUFFICIENCY_PASS" if stable else "TRAINING_STILL_SENSITIVE",
        "contract": "topic5_history_rnn_real_convergence_v0_2",
        "target_values_read": False,
        "patients": list(PATIENTS),
        "median_chronological_bce_improvement_3_to_10": float(
            np.median(pivot[3] - pivot[10])
        ),
        "median_chronological_bce_improvement_10_to_30": float(
            np.median(delta_10_to_30)
        ),
        "median_abs_change_10_to_30": float(np.median(np.abs(delta_10_to_30))),
        "chronological_increment_by_cycles": {
            str(cycle): float(increment[cycle].median()) for cycle in (3, 10, 30)
        },
        "direction_stable_10_to_30": bool(
            np.all(np.sign(increment[10]) == np.sign(increment[30]))
        ),
    }
    (root / "REAL_CONVERGENCE_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
