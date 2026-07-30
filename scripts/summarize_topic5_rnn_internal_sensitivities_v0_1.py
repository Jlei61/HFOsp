#!/usr/bin/env python3
"""Summarize target-blind random-subspace and matched-order sensitivities."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_internal_state_reduction"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def summary(values: np.ndarray, seed: int, alternative: str = "greater") -> dict:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(data, size=(20_000, len(data)), replace=True)
    return {
        "n": int(len(data)),
        "median": float(np.median(data)),
        "bootstrap_ci95": np.quantile(
            np.median(sampled, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(data > 0)),
        "wilcoxon_p": (
            1.0
            if np.allclose(data, 0.0)
            else float(wilcoxon(data, alternative=alternative).pvalue)
        ),
        "alternative": alternative,
    }


def main() -> None:
    if not (BASE / "RANDOM_SUBSPACE_DONE.json").exists():
        raise SystemExit("random-subspace cells are incomplete")
    frames = []
    for path in sorted(
        (BASE / "interictal/random_subspace_cells").glob(
            "seed_*/**/random_subspace_metrics.csv"
        )
    ):
        frames.append(pd.read_csv(path))
    if len(frames) != 102:
        raise RuntimeError(f"expected 102 random-subspace cells, found {len(frames)}")
    random = pd.concat(frames, ignore_index=True)
    random.to_csv(BASE / "interictal_random_subspace_metrics.csv", index=False)
    patient_random = (
        random.groupby(["subject", "control", "k"], as_index=False)
        .agg(
            pca_advantage_nll=("pca_advantage_nll", "median"),
            pca_variance_fidelity=("pca_variance_fidelity", "median"),
            random_variance_fidelity=("random_variance_fidelity", "median"),
        )
    )
    patient_random["pca_advantage_variance"] = (
        patient_random.pca_variance_fidelity
        - patient_random.random_variance_fidelity
    )
    patient_random.to_csv(
        BASE / "interictal_random_subspace_patient_metrics.csv", index=False
    )
    metrics = {}
    for (control, k), group in patient_random.groupby(["control", "k"]):
        metrics[f"{control}__k{k}__pca_advantage_nll"] = summary(
            group.pca_advantage_nll.to_numpy(float),
            2026076000 + len(metrics),
        )
        metrics[f"{control}__k{k}__pca_advantage_variance"] = summary(
            group.pca_advantage_variance.to_numpy(float),
            2026076000 + len(metrics),
        )

    order = pd.read_csv(BASE / "interictal_order_perturbation_metrics.csv")
    order = (
        order.loc[
            (order.prefix_bin == "all")
            & (order.metric.isin(("nll_loss", "js_divergence")))
        ]
        .groupby(
            ["subject", "control", "order_perturbation", "metric"],
            as_index=False,
        )
        .value.median()
    )
    order_wide = order.pivot_table(
        index=["subject", "order_perturbation", "metric"],
        columns="control",
        values="value",
    ).reset_index()
    order_wide["full_minus_rank_shuffle_sensitivity"] = (
        order_wide.full_history_gru - order_wide.rank_shuffle_gru
    )
    order_wide.to_csv(
        BASE / "interictal_order_full_vs_rank_shuffle.csv", index=False
    )
    for (perturbation, metric), group in order_wide.groupby(
        ["order_perturbation", "metric"]
    ):
        metrics[
            f"order_{perturbation}__{metric}__full_minus_rank_shuffle"
        ] = summary(
            group.full_minus_rank_shuffle_sensitivity.to_numpy(float),
            2026077000 + len(metrics),
        )

    atomic_json(
        BASE / "INTERICTAL_SENSITIVITY_SUMMARY.json",
        {
            "contract": "topic5_rnn_internal_state_reduction_v0_1_sensitivities",
            "status": "COMPLETE",
            "n_subjects": 34,
            "n_seeds": 3,
            "random_subspaces_per_k": 8,
            "metrics": metrics,
            "target_values_used_in_these_calculations": False,
            "note": (
                "This appendix was computed only from already frozen interictal "
                "states; it does not revise direction selection."
            ),
        },
    )


if __name__ == "__main__":
    main()
