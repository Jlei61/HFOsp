#!/usr/bin/env python3
"""Patient-first aggregation of the bounded N=100/1000 T2-S1 pilot."""
from __future__ import annotations

import csv
import json

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_human import T2_HUMAN_REVISION


SUBJECTS = ("epilepsiae_620", "epilepsiae_958")
SCALES = (100, 1000)
SEEDS = (0, 1, 2)
CONTRASTS = (
    "real_minus_no_edge",
    "real_minus_state_matched_placebo",
    "current_event_minus_no_edge",
    "placebo_minus_no_edge",
)
ENDPOINTS = (
    "joint_nll_per_event", "timing_nll_per_event", "mark_nll_per_event",
    "group_size_nll_per_event", "subset_nll_per_event",
    "stop_nll_per_event", "first_group_subset_nll_per_event",
    "continuation_subset_nll_per_event",
)
# Which fitted arm each contrast depends on.  When that arm's edge selection
# returns epoch 0 the edge vector stays at its zero initialisation, so the
# contrast is identically 0.0 and can never satisfy "< 0".  Counting such a row
# as an unfavourable patient turns a structural tie into a reported negative.
CONTRAST_ARMS = {
    "real_minus_no_edge": ("real_cumulative",),
    "real_minus_state_matched_placebo": (
        "real_cumulative", "state_matched_placebo",
    ),
    "current_event_minus_no_edge": ("current_event_only",),
    "placebo_minus_no_edge": ("state_matched_placebo",),
}


def main() -> None:
    root = contract.RESULT_ROOT / "t2_s1_long_scale"
    rows = []
    for subject in SUBJECTS:
        for scale in SCALES:
            values = []
            for seed in SEEDS:
                path = root / "human" / subject / f"seed_{seed}_n_{scale}/result.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                value = json.loads(path.read_text())
                if value.get("status") != "COMPLETE":
                    raise ValueError(f"incomplete T2-S1 result: {path}")
                if value.get("revision") != T2_HUMAN_REVISION:
                    raise ValueError(f"T2-S1 revision mismatch: {path}")
                if value.get("sealed_opened") is not False:
                    raise ValueError(f"sealed partition opened: {path}")
                values.append(value)
            row = {
                "subject": subject,
                "scale_events": scale,
                "n_seeds": len(values),
                "train_pairs": int(np.median([
                    value["design"]["train_pairs"] for value in values
                ])),
                "validation_pairs": int(np.median([
                    value["design"]["validation_pairs"] for value in values
                ])),
            }
            for comparison in CONTRASTS:
                for endpoint in ENDPOINTS:
                    row[f"{comparison}_{endpoint}"] = float(np.median([
                        value["comparisons"][comparison][endpoint]
                        for value in values
                    ]))
            for arm in ("real_cumulative", "state_matched_placebo", "current_event_only"):
                row[f"{arm}_selected_epoch_median"] = float(np.median([
                    value["fits"][arm]["selected_epoch"] for value in values
                ]))
                row[f"{arm}_selected_epoch_zero"] = int(sum(
                    value["fits"][arm]["selected_epoch"] == 0 for value in values
                ))
            for comparison, arms in CONTRAST_ARMS.items():
                row[f"{comparison}_all_seeds_structural_zero"] = int(all(
                    all(value["fits"][arm]["selected_epoch"] == 0 for arm in arms)
                    for value in values
                ))
            rows.append(row)
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "t2_s1_patient_scale.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    primary = [row for row in rows if row["scale_events"] == 1000]
    primary_summary = {}
    for comparison in CONTRASTS:
        for endpoint in ENDPOINTS:
            key = f"{comparison}_{endpoint}"
            structural = [
                bool(row[f"{comparison}_all_seeds_structural_zero"])
                for row in primary
            ]
            value = [row[key] for row in primary]
            estimated = [
                item for item, dead in zip(value, structural) if not dead
            ]
            primary_summary[key] = {
                "patient_median": float(np.median(value)),
                "n_patients": len(value),
                "n_structural_zero": int(sum(structural)),
                "n_estimated": len(estimated),
                "n_favourable_negative": int(sum(item < 0 for item in estimated)),
                "denominator_note": (
                    "n_favourable_negative counts only patients whose edge was "
                    "actually fitted; a structural zero from an epoch-0 edge is "
                    "neither favourable nor unfavourable"
                ),
            }
    summary = {
        "status": "COMPLETE",
        "revision": T2_HUMAN_REVISION,
        "n_fits": 12,
        "rows": rows,
        "n_1000_primary": primary_summary,
        "n_100_role": "short-scale reference, not a gate",
        "n_1000_role": "current multi-patient long-scale primary exploration",
        "n_10000_status": (
            "unobservable in the fixed R1.3 three-patient pilot; Zhangjiaqi has "
            "recorded support but needs a target-trained T1 checkpoint before testing"
        ),
        "patient_first": True,
        "structural_zero_rule": (
            "an arm whose inner-TRAIN selection returns epoch 0 keeps its zero "
            "edge vector, so its contrast is identically 0.0; such rows are "
            "reported as n_structural_zero and excluded from the favourable "
            "denominator instead of being written as an unfavourable patient"
        ),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "two-patient development one-step screen; negative results remain "
            "instrument- and state-quality-bounded, positive prediction increments "
            "are not causal proof"
        ),
    }
    contract.atomic_json(report / "t2_s1_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
