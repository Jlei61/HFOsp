#!/usr/bin/env python3
"""Patient-first, target-free comparison of the four full-tissue topologies."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_lbss_claims_v0_2 import holm, paired


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
)

METRICS = {
    "overall_contact_nll": ("interictal", "test_contact_nll", "lower"),
    "distal_contact_nll": ("interictal", "distal_contact_nll", "lower"),
    "free_rollout_spearman": ("interictal", "rollout_spearman", "higher"),
    "canonical_interictal_field_r": ("field", "canonical_empirical_r", "higher"),
    "seed_removed_interictal_field_r": ("field", "seed_removed_empirical_r", "higher"),
    "canonical_ab_contrast_r": ("field", "canonical_contrast_empirical_r", "higher"),
    "seed_removed_ab_contrast_r": (
        "field", "seed_removed_contrast_empirical_r", "higher"
    ),
}


def summarize(out: Path) -> dict:
    interictal = pd.read_csv(out / "interictal_per_patient.csv")
    fields = pd.read_csv(out / "model_field_patient_metrics.csv")
    sources = {
        "interictal": interictal.pivot(index="subject", columns="arm"),
        "field": fields.pivot(index="subject", columns="arm"),
    }
    result: dict[str, dict] = {}
    confirmed = []
    for name, (source, column, direction) in METRICS.items():
        wide = sources[source][column]
        if not set(ARMS).issubset(wide.columns):
            raise RuntimeError(f"{name}: incomplete arm matrix")
        rows = {}
        for left, right in combinations(ARMS, 2):
            common = wide[left].dropna().index.intersection(wide[right].dropna().index)
            raw = wide.loc[common, left] - wide.loc[common, right]
            advantage = raw if direction == "higher" else -raw
            key = f"{left}_vs_{right}"
            rows[key] = paired(advantage)
        adjusted = holm({
            key: value["wilcoxon_p_two_sided"] for key, value in rows.items()
        })
        for key, q in adjusted.items():
            rows[key]["holm_q_within_endpoint"] = float(q)
            if q < 0.05 and rows[key]["median"] > 0:
                confirmed.append({"endpoint": name, "contrast": key})
            elif q < 0.05 and rows[key]["median"] < 0:
                left, right = key.split("_vs_", 1)
                confirmed.append({
                    "endpoint": name, "contrast": f"{right}_vs_{left}"
                })
        result[name] = {
            "direction": direction,
            "n_patients": int(wide[list(ARMS)].dropna().shape[0]),
            "arm_medians": {
                arm: float(np.nanmedian(wide[arm].to_numpy(float))) for arm in ARMS
            },
            "pairwise_positive_means_left_arm_better": rows,
        }

    # A topology is called a winner only if it has at least one corrected
    # advantage and no corrected disadvantage on any prespecified endpoint.
    winners = []
    for arm in ARMS:
        wins = [item for item in confirmed if item["contrast"].startswith(arm + "_vs_")]
        losses = [item for item in confirmed if item["contrast"].endswith("_vs_" + arm)]
        if wins and not losses:
            winners.append(arm)
    decision = (
        "UNIQUE_SPATIAL_TOPOLOGY_WINNER" if len(winners) == 1
        else "NO_UNIQUE_SPATIAL_TOPOLOGY_WINNER"
    )
    return {
        "contract": "topic5_lbss_target_free_topology_plateau_v0_3",
        "n_patients": int(interictal.subject.nunique()),
        "endpoints": result,
        "confirmed_endpoint_level_pairwise_advantages": confirmed,
        "winner_arms": winners,
        "decision": decision,
        "parsimony_reference": "L0_LOCAL_ONLY",
        "parsimony_reason": (
            "When no topology has corrected multi-endpoint superiority, L0 is "
            "the minimum-edge sufficient recurrent model; this is not a claim "
            "that L0 is the patient's true connectivity."
        ),
        "early_ictal_values_used": False,
        "target_values_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root", type=Path,
        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"),
    )
    args = parser.parse_args()
    out = args.out_root.resolve()
    payload = summarize(out)
    destination = out / "SPATIAL_TOPOLOGY_PLATEAU_SUMMARY.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
