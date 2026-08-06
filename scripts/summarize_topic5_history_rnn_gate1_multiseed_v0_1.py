#!/usr/bin/env python3
"""Aggregate three frozen G1 seeds at the patient level."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


DEVELOPMENT = {
    "epilepsiae_1073",
    "epilepsiae_1146",
    "yuquan_chenziyang",
}


def _p(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if np.allclose(values, 0):
        return 1.0
    return float(wilcoxon(values, alternative="greater", method="auto").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.input_dir.resolve()
    frames = []
    seed_summaries = {}
    for seed in (20260725, 20260726, 20260727):
        seed_root = root / f"seed_{seed}"
        summary = json.loads((seed_root / "G1_SUMMARY.json").read_text())
        if bool(summary.get("target_values_read", True)):
            raise RuntimeError(f"target seal violated in seed {seed}")
        seed_summaries[str(seed)] = summary
        frame = pd.read_csv(seed_root / "g1_patient_metrics.csv")
        frame["seed"] = seed
        frames.append(frame)
    all_seed = pd.concat(frames, ignore_index=True)
    all_seed.to_csv(root / "g1_multiseed_patient_seed_metrics.csv", index=False)
    metrics = [
        "static_to_matched_gain",
        "chronological_increment",
        "rank_increment",
        "order_shuffle_cost",
        "prefix_matched_order_shuffle_cost",
        "within_event_rank_shuffle_cost",
    ]
    patient = (
        all_seed.groupby(["subject", "dataset", "development_patient"], as_index=False)[
            metrics
        ]
        .median()
        .sort_values("subject")
    )
    patient.to_csv(root / "g1_multiseed_patient_metrics.csv", index=False)
    primary = patient.loc[~patient.subject.isin(DEVELOPMENT)].copy()
    chronology = primary.chronological_increment.to_numpy(float)
    order = primary.prefix_matched_order_shuffle_cost.to_numpy(float)
    dataset_direction = {
        dataset: float(group.chronological_increment.median())
        for dataset, group in primary.groupby("dataset", sort=True)
    }
    # Pre-registered supportive analysis: the same contrasts on all 34 patients.
    # Reporting only — the gate below stays on the 31 development-excluded ones.
    supportive_chronology = patient.chronological_increment.to_numpy(float)
    supportive_order = patient.prefix_matched_order_shuffle_cost.to_numpy(float)
    supportive = {
        "n_patients": int(len(patient)),
        "role": "supportive_only_not_part_of_the_gate",
        "median_chronological_increment": float(np.median(supportive_chronology)),
        "chronological_increment_one_sided_wilcoxon_p": _p(supportive_chronology),
        "n_chronological_positive": int(np.sum(supportive_chronology > 0)),
        "median_prefix_matched_order_shuffle_cost": float(np.median(supportive_order)),
        "prefix_matched_order_shuffle_one_sided_wilcoxon_p": _p(supportive_order),
        "n_prefix_matched_order_shuffle_positive": int(np.sum(supportive_order > 0)),
        "dataset_median_chronological_increment": {
            dataset: float(group.chronological_increment.median())
            for dataset, group in patient.groupby("dataset", sort=True)
        },
    }
    per_seed_direction = {
        seed: {
            "median_chronological_increment": float(
                summary["primary"]["median_chronological_increment"]
            ),
            "median_prefix_matched_order_shuffle_cost": float(
                summary["primary"]["median_order_shuffle_cost"]
            ),
        }
        for seed, summary in seed_summaries.items()
    }
    pass_gate = (
        float(np.median(chronology)) > 0
        and _p(chronology) < 0.05
        and all(value > 0 for value in dataset_direction.values())
        and float(np.median(order)) > 0
        and _p(order) < 0.05
        and all(
            value["median_chronological_increment"] > 0
            and value["median_prefix_matched_order_shuffle_cost"] > 0
            for value in per_seed_direction.values()
        )
    )
    result = {
        "status": "G1_MULTI_SEED_PASS_OPEN_G2" if pass_gate else "G1_MULTI_SEED_FAIL_KEEP_ICTAL_TARGET_SEALED",
        "contract": "topic5_history_rnn_early_ictal_field_v0_1_g1_multiseed",
        "target_values_read": False,
        "n_seeds": 3,
        "n_primary_patients": int(len(primary)),
        "patient_aggregation": "median across frozen seeds before patient-first inference",
        "median_chronological_increment": float(np.median(chronology)),
        "chronological_increment_one_sided_wilcoxon_p": _p(chronology),
        "n_chronological_positive": int(np.sum(chronology > 0)),
        "median_prefix_matched_order_shuffle_cost": float(np.median(order)),
        "prefix_matched_order_shuffle_one_sided_wilcoxon_p": _p(order),
        "n_prefix_matched_order_shuffle_positive": int(np.sum(order > 0)),
        "dataset_median_chronological_increment": dataset_direction,
        "per_seed_direction": per_seed_direction,
        "supportive_all_34": supportive,
    }
    (root / "G1_MULTI_SEED_SUMMARY.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
