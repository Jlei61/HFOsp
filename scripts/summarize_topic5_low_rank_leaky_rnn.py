#!/usr/bin/env python3
"""Aggregate rank 0-4 low-rank leaky RNN distribution fidelity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _ci(values: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, len(values), size=(20000, len(values)))
    bootstrap = np.median(values[draws], axis=1)
    return [
        float(np.quantile(bootstrap, 0.025)),
        float(np.quantile(bootstrap, 0.975)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--ranks", nargs="+", type=int, required=True)
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        for rank in args.ranks:
            rank_root = args.root / f"seed_{seed}" / f"rank_{rank}"
            for path in sorted(rank_root.glob("*/run_summary.json")):
                value = json.loads(path.read_text())
                model = value["distribution_errors"]
                empirical = value["empirical_distribution_errors"]
                split = value["empirical_split_half_variability"]
                rows.append(
                    {
                        "seed": int(seed),
                        "recurrent_rank": int(rank),
                        "subject": value["subject"],
                        "dataset": value["dataset"],
                        "heldout_event_nll": value["heldout_event_nll"],
                        "participation_mae": model["participation_mae"],
                        "rank_wasserstein": model["rank_wasserstein"],
                        "precedence_mae": model["precedence_mae"],
                        "participation_excess": (
                            model["participation_mae"]
                            - empirical["participation_mae"]
                            - split["participation_mae"]
                        ),
                        "rank_wasserstein_excess": value[
                            "rank_wasserstein_excess_over_empirical_variability"
                        ],
                        "precedence_excess": (
                            model["precedence_mae"]
                            - empirical["precedence_mae"]
                            - split["precedence_mae"]
                        ),
                    }
                )
    all_seed = pd.DataFrame(rows)
    expected = len(args.seeds) * len(args.ranks) * 34
    if len(all_seed) != expected:
        raise RuntimeError(f"expected {expected} completed folds, found {len(all_seed)}")
    all_seed.to_csv(args.root / "all_seed_rank_subject_summary.csv", index=False)
    numeric = [
        column
        for column in all_seed.columns
        if pd.api.types.is_numeric_dtype(all_seed[column])
        and column not in {"seed"}
    ]
    patient = (
        all_seed.groupby(
            ["recurrent_rank", "subject", "dataset"], as_index=False
        )[numeric]
        .mean()
    )
    patient.to_csv(args.root / "patient_seed_collapsed_summary.csv", index=False)
    rank_rows = []
    for rank, frame in patient.groupby("recurrent_rank"):
        participation = frame.participation_excess.to_numpy(float)
        rank_error = frame.rank_wasserstein_excess.to_numpy(float)
        precedence = frame.precedence_excess.to_numpy(float)
        participation_ci = _ci(participation, 20260725 + int(rank) * 3)
        rank_ci = _ci(rank_error, 20260726 + int(rank) * 3)
        precedence_ci = _ci(precedence, 20260727 + int(rank) * 3)
        rank_rows.append(
            {
                "recurrent_rank": int(rank),
                "n_patients": int(len(frame)),
                "median_participation_excess": float(np.median(participation)),
                "participation_excess_ci_low": participation_ci[0],
                "participation_excess_ci_high": participation_ci[1],
                "median_rank_wasserstein_excess": float(np.median(rank_error)),
                "rank_wasserstein_excess_ci_low": rank_ci[0],
                "rank_wasserstein_excess_ci_high": rank_ci[1],
                "median_precedence_excess": float(np.median(precedence)),
                "precedence_excess_ci_low": precedence_ci[0],
                "precedence_excess_ci_high": precedence_ci[1],
                "distribution_sufficient": bool(
                    participation_ci[1] <= 0
                    and rank_ci[1] <= 0
                    and precedence_ci[1] <= 0
                ),
            }
        )
    rank_summary = pd.DataFrame(rank_rows).sort_values("recurrent_rank")
    rank_summary.to_csv(args.root / "rank_summary.csv", index=False)
    sufficient = rank_summary[rank_summary.distribution_sufficient]
    minimum = (
        int(sufficient.recurrent_rank.min()) if len(sufficient) else None
    )
    result = {
        "status": "complete",
        "n_patients": 34,
        "n_seeds": len(args.seeds),
        "ranks": [int(rank) for rank in args.ranks],
        "minimum_distribution_sufficient_rank": minimum,
        "rank_zero_sufficient": bool(
            len(
                rank_summary[
                    (rank_summary.recurrent_rank == 0)
                    & rank_summary.distribution_sufficient
                ]
            )
        ),
        "ictal_target_read": False,
    }
    (args.root / "low_rank_summary.json").write_text(
        json.dumps(result, indent=2)
    )
    (args.root / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "minimum_distribution_sufficient_rank": minimum,
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
