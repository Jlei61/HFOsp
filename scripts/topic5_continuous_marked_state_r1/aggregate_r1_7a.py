#!/usr/bin/env python3
"""Patient-first R1.7A aggregation with time-block scientific uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2 import load_full_design
from src.topic5_continuous_marked_state_r1.r1_7 import R1_7A_REVISION


SEEDS = tuple(range(5))
FIELDS = {
    "persistent_minus_memoryless_joint": ("persistent_minus_memoryless", "joint_nll_per_event"),
    "persistent_minus_memoryless_timing": ("persistent_minus_memoryless", "timing_nll_per_event"),
    "persistent_minus_memoryless_first_subset": ("mark_endpoints", "persistent_minus_memoryless", "first_group_subset_nll_per_event"),
    "persistent_minus_memoryless_continuation": ("mark_endpoints", "persistent_minus_memoryless", "continuation_subset_nll_per_event"),
    "persistent_minus_memoryless_same_prefix": ("mark_endpoints", "persistent_minus_memoryless", "same_prefix_continuation_nll_per_event"),
    "correct_minus_wrong_joint": ("strict_matched_wrong_time", "correct_minus_wrong_median", "joint_nll_per_event"),
    "correct_minus_wrong_first_subset": ("strict_matched_wrong_time", "endpoint_correct_minus_wrong_median", "first_group_subset_nll_per_event"),
    "correct_minus_wrong_continuation": ("strict_matched_wrong_time", "endpoint_correct_minus_wrong_median", "continuation_subset_nll_per_event"),
    "correct_minus_wrong_same_prefix": ("strict_matched_wrong_time", "endpoint_correct_minus_wrong_median", "same_prefix_continuation_nll_per_event"),
}


def get(value: dict, path: tuple[str, ...]):
    for key in path:
        value = value[key]
    return value


def bootstrap(blocks: list[dict], key: str, *, draws: int = 2000) -> dict:
    values = np.asarray([row[key] for row in blocks], dtype=np.float64)
    weights = np.asarray([row["n_events"] for row in blocks], dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"estimate": None, "ci95": [None, None], "n_blocks": len(values)}
    rng = np.random.default_rng(1701)
    sampled = []
    for _ in range(draws):
        index = rng.integers(0, len(values), len(values))
        sampled.append(float(np.average(values[index], weights=weights[index])))
    estimate = float(np.average(values, weights=weights))
    return {"estimate": estimate,
            "ci95": np.quantile(sampled, [0.025, 0.975]).tolist(),
            "n_blocks": len(values), "draws": draws, "seed": 1701}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    args = parser.parse_args()
    inventory = json.loads((args.root / "manifests/cohort_inventory.json").read_text())
    by_subject = {}; rows = []
    for subject in inventory["selected_subjects"]:
        payloads = []
        for seed in SEEDS:
            path = args.root / "fits" / subject / f"seed_{seed}/result.json"
            value = json.loads(path.read_text())
            if (value.get("status") != "COMPLETE"
                    or value.get("revision") != R1_7A_REVISION
                    or value.get("formal_test_partition_opened") is not False
                    or value.get("sealed_opened") is not False):
                raise ValueError(f"invalid R1.7A result: {path}")
            payloads.append(value)
        row = {"subject": subject, "n_seeds": 5,
               "stable_checkpoint_seeds": sum(v["stable_checkpoint"] for v in payloads)}
        for label, path in FIELDS.items():
            values = [get(value["d_state"], path) for value in payloads]
            finite = [float(value) for value in values if value is not None and np.isfinite(value)]
            row[label] = float(np.median(finite)) if finite else None
            row[label + "_favourable_seeds"] = int(sum(value < 0 for value in finite))
        keyed = {}
        for value in payloads:
            for block in value["d_state"]["nonoverlap_time_blocks"]:
                key = (float(block["start"]), float(block["stop"]))
                keyed.setdefault(key, []).append(block)
        block_medians = []
        for key, blocks in sorted(keyed.items()):
            if len(blocks) != 5:
                raise ValueError(f"{subject}: bootstrap block missing a seed")
            block_medians.append({
                "start": key[0], "stop": key[1],
                "n_events": int(np.median([v["n_events"] for v in blocks])),
                "persistent_minus_memoryless_joint": float(np.median([
                    v["persistent_minus_memoryless"]["joint_nll_per_event"] for v in blocks
                ])),
                "correct_minus_wrong_joint": float(np.median([
                    v["correct_minus_wrong"]["joint_nll_per_event"] for v in blocks
                ])),
            })
        row["bootstrap"] = {
            label: bootstrap(block_medians, label)
            for label in ("persistent_minus_memoryless_joint", "correct_minus_wrong_joint")
        }
        row["n_d_state_blocks"] = len(block_medians)
        row["patient_stable_state"] = bool(row["stable_checkpoint_seeds"] >= 3)
        manifest = json.loads((args.root / "cache" / subject / "manifest.json").read_text())
        design = load_full_design(Path(manifest["design"]))
        boundary = float(payloads[0]["d_state"]["support"]["mechanism_start"])
        d_mechanism_events = int(np.sum(
            (design.event_split == 1) & (design.event_time >= boundary)
        ))
        row["d_mechanism_events"] = d_mechanism_events
        row["d_mechanism_100_event_blocks"] = d_mechanism_events // 100
        row["t2_support_class"] = (
            "COHORT_ELIGIBLE" if d_mechanism_events >= 500
            else "CASE_ONLY" if d_mechanism_events >= 100 else "INSUFFICIENT"
        )
        row["t2_run_eligible"] = bool(
            row["patient_stable_state"] and d_mechanism_events >= 100
        )
        rows.append(row); by_subject[subject] = row
    stable = [row["subject"] for row in rows if row["patient_stable_state"]]
    t2 = [row["subject"] for row in rows if row["t2_run_eligible"]]
    summary = {
        "status": "COMPLETE", "revision": R1_7A_REVISION,
        "subjects": inventory["selected_subjects"], "n_subjects": len(rows),
        "stable_state_subjects": stable, "n_stable_state_subjects": len(stable),
        "t2_run_subjects": t2, "n_t2_run_subjects": len(t2),
        "by_subject": by_subject,
        "seed_is_optimization_uncertainty_not_scientific_replication": True,
        "scientific_uncertainty": "patient-local continuous-time block bootstrap after five-seed median",
        "ordinary_negative_results_retained": True,
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    report = args.root / "reports"; report.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(report / "r1_7a_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
