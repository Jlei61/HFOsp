#!/usr/bin/env python3
"""Patient-first summary of selected-model history interventions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]


def summarize(values: np.ndarray, seed: int) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(x, (10000, len(x)), replace=True)
    try:
        p = float(wilcoxon(x, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return {
        "n_patients": int(len(x)),
        "median_nll_cost": float(np.median(x)),
        "bootstrap_ci95": np.quantile(
            np.median(bootstrap, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(x > 0)),
        "wilcoxon_greater_p": p,
    }


def describe(values: np.ndarray, seed: int) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(x, (10000, len(x)), replace=True)
    return {
        "n_patients": int(len(x)),
        "median": float(np.median(x)),
        "bootstrap_ci95": np.quantile(
            np.median(bootstrap, axis=1), [0.025, 0.975]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    paths = list(root.glob("seed_*/*/history_intervention_metrics.csv"))
    if len(paths) != 102:
        raise RuntimeError(f"intervention cells incomplete: {len(paths)}/102")
    memory_paths = list(root.glob("seed_*/*/readout_memory_metrics.csv"))
    if len(memory_paths) != 102:
        raise RuntimeError(f"readout-memory cells incomplete: {len(memory_paths)}/102")
    all_seed = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    expected = 102 * 2 * 6
    if len(all_seed) != expected:
        raise RuntimeError(f"intervention row count drifted: {len(all_seed)}/{expected}")
    output.mkdir(parents=True, exist_ok=True)
    all_seed.to_csv(output / "all_seed_history_interventions.csv", index=False)
    memory_all_seed = pd.concat(
        [pd.read_csv(path) for path in memory_paths], ignore_index=True
    )
    memory_all_seed.to_csv(output / "all_seed_readout_memory.csv", index=False)
    memory_collapsed = (
        memory_all_seed.groupby(
            ["subject", "dataset", "selected_architecture"], as_index=False
        )
        .median(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    memory_collapsed.to_csv(
        output / "patient_seed_collapsed_readout_memory.csv", index=False
    )
    keys = [
        "subject",
        "dataset",
        "model",
        "selected_architecture",
        "intervention",
        "reset_after_rank",
    ]
    collapsed = (
        all_seed.groupby(keys, dropna=False, as_index=False)
        .median(numeric_only=True)
        .drop(columns=["seed"], errors="ignore")
    )
    collapsed.to_csv(
        output / "patient_seed_collapsed_history_interventions.csv", index=False
    )
    metrics = [
        "heldout_event_balanced_nll",
        "step0_prefix_balanced_nll",
        "step1_prefix_balanced_nll",
        "step2_prefix_balanced_nll",
        "step3_prefix_balanced_nll",
        "step4plus_prefix_balanced_nll",
    ]
    ordered = collapsed.loc[
        collapsed.intervention.eq("ordered")
    ].set_index(["subject", "model"])
    rows = []
    summaries = {}
    for intervention_index, row in enumerate(
        collapsed.loc[~collapsed.intervention.eq("ordered")].itertuples(index=False)
    ):
        base = ordered.loc[(row.subject, row.model)]
        label = (
            row.intervention
            if row.intervention != "reset_after_rank"
            else f"reset_after_rank_{int(row.reset_after_rank)}"
        )
        for metric in metrics:
            rows.append(
                {
                    "subject": row.subject,
                    "model": row.model,
                    "intervention": label,
                    "metric": metric,
                    "nll_cost_vs_ordered": float(getattr(row, metric) - base[metric]),
                }
            )
    costs = pd.DataFrame(rows)
    costs.to_csv(output / "patient_history_intervention_costs.csv", index=False)
    for index, ((model, intervention, metric), group) in enumerate(
        costs.groupby(["model", "intervention", "metric"])
    ):
        summaries[f"{model}__{intervention}__{metric}"] = summarize(
            group.nll_cost_vs_ordered.to_numpy(float),
            20261000 + index,
        )
    result = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": "TARGET_BLIND_HISTORY_INTERVENTIONS_COMPLETE",
        "n_patients": 34,
        "n_seeds": 3,
        "models": sorted(collapsed.model.unique().tolist()),
        "interventions": [
            "reverse_prefix",
            "drop_earliest",
            "reset_after_rank_0",
            "reset_after_rank_1",
            "reset_after_rank_2",
        ],
        "summaries": summaries,
        "readout_relevant_local_memory": {
            metric: describe(
                memory_collapsed[metric].to_numpy(float),
                20262000 + index,
            )
            for index, metric in enumerate(
                [
                    "readout_retention_median",
                    "readout_alignment_median",
                    "state_gain_median",
                    "local_spectral_radius_median",
                ]
            )
        },
        "target_values_read": False,
        "early_ictal_target_arrays_deserialized": False,
        "interpretation": (
            "positive cost shows that the fitted event-indexed state uses the "
            "intervened history; it is not a biological time-constant estimate"
        ),
    }
    (output / "HISTORY_INTERVENTION_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
