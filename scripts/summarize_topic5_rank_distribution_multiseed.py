#!/usr/bin/env python3
"""Collapse formal Stage-A seed results within patient, then across patients."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_median_ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    bootstrap = np.median(values[draws], axis=1)
    return [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    frames = []
    for seed in args.seeds:
        path = args.formal_root / f"seed_{seed}" / "formal_fold_summary.csv"
        frame = pd.read_csv(path)
        frame["seed"] = int(seed)
        frames.append(frame)
    all_seed = pd.concat(frames, ignore_index=True)
    all_seed.to_csv(args.formal_root / "all_seed_fold_summary.csv", index=False)
    numeric = [
        column
        for column in all_seed.columns
        if pd.api.types.is_numeric_dtype(all_seed[column])
        and column not in {"seed"}
    ]
    patient = (
        all_seed.groupby(["subject", "dataset"], as_index=False)[numeric]
        .mean()
    )
    patient.to_csv(args.formal_root / "patient_seed_collapsed_summary.csv", index=False)
    ordered = patient.ordered_history_nll_gain.to_numpy(float)
    shuffle = patient.shuffle_minus_gru_rank_wasserstein.to_numpy(float)
    noninferiority = (
        patient.gru_minus_empirical_rank_wasserstein
        - patient.noninferiority_margin_rank_wasserstein
    ).to_numpy(float)
    result = {
        "status": "complete",
        "n_patients": int(len(patient)),
        "n_seeds": int(len(args.seeds)),
        "seeds": [int(seed) for seed in args.seeds],
        "median_ordered_history_nll_gain": float(np.median(ordered)),
        "ordered_history_nll_gain_ci95": _bootstrap_median_ci(
            ordered, 20260725
        ),
        "n_patients_ordered_gain_positive": int(np.sum(ordered > 0)),
        "median_shuffle_minus_gru_rank_wasserstein": float(
            np.median(shuffle)
        ),
        "shuffle_minus_gru_rank_wasserstein_ci95": _bootstrap_median_ci(
            shuffle, 20260726
        ),
        "n_patients_shuffle_worse": int(np.sum(shuffle > 0)),
        "median_noninferiority_excess": float(np.median(noninferiority)),
        "noninferiority_excess_ci95": _bootstrap_median_ci(
            noninferiority, 20260727
        ),
        "formal_stage_a_direction_pass": bool(
            np.quantile(
                np.median(
                    ordered[
                        np.random.default_rng(20260725).integers(
                            0, len(ordered), size=(20000, len(ordered))
                        )
                    ],
                    axis=1,
                ),
                0.025,
            )
            > 0
            and np.quantile(
                np.median(
                    shuffle[
                        np.random.default_rng(20260726).integers(
                            0, len(shuffle), size=(20000, len(shuffle))
                        )
                    ],
                    axis=1,
                ),
                0.025,
            )
            > 0
            and np.quantile(
                np.median(
                    noninferiority[
                        np.random.default_rng(20260727).integers(
                            0, len(noninferiority), size=(20000, len(noninferiority))
                        )
                    ],
                    axis=1,
                ),
                0.975,
            )
            <= 0
        ),
        "ictal_target_read": False,
    }
    (args.formal_root / "formal_stage_a_summary.json").write_text(
        json.dumps(result, indent=2)
    )
    (args.formal_root / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "formal_stage_a_direction_pass": result[
                    "formal_stage_a_direction_pass"
                ],
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
