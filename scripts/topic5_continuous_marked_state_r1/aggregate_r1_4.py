#!/usr/bin/env python3
"""Patient-first aggregation for the frozen six-patient R1.4 replication."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_3 import R1_3_REVISION


REVISION = "r1_4_six_patient_explicit_primary_raw_residual_v1"
SUBJECTS = (
    "epilepsiae_620",
    "epilepsiae_958",
    "yuquan_huanghanwen",
    "epilepsiae_922",
    "yuquan_pengzihang",
    "yuquan_hanyuxuan",
)
SEEDS = (0, 1, 2)
ARMS = ("explicit", "explicit_raw")


FIELDS = {
    "persistent_minus_memoryless_joint":
        "validation.persistent_minus_memoryless.joint_nll_per_event",
    "persistent_minus_memoryless_timing":
        "validation.persistent_minus_memoryless.timing_nll_per_event",
    "persistent_minus_memoryless_mark":
        "validation.persistent_minus_memoryless.mark_nll_per_event",
    "persistent_minus_memoryless_stop":
        "validation.mark_endpoints.persistent_minus_memoryless.stop_nll_per_event",
    "persistent_minus_memoryless_selecting_size":
        "validation.mark_endpoints.persistent_minus_memoryless.selecting_group_size_nll_per_event",
    "persistent_minus_memoryless_first_subset":
        "validation.mark_endpoints.persistent_minus_memoryless.first_group_subset_nll_per_event",
    "persistent_minus_memoryless_continuation":
        "validation.mark_endpoints.persistent_minus_memoryless.continuation_subset_nll_per_event",
    "persistent_minus_memoryless_same_prefix":
        "validation.mark_endpoints.persistent_minus_memoryless.same_prefix_continuation_nll_per_event",
    "correct_minus_wrong_joint":
        "validation.strict_matched_wrong_time.correct_minus_wrong_median.joint_nll_per_event",
    "correct_minus_wrong_timing":
        "validation.strict_matched_wrong_time.correct_minus_wrong_median.timing_nll_per_event",
    "correct_minus_wrong_mark":
        "validation.strict_matched_wrong_time.correct_minus_wrong_median.mark_nll_per_event",
    "correct_minus_wrong_stop":
        "validation.strict_matched_wrong_time.endpoint_correct_minus_wrong_median.stop_nll_per_event",
    "correct_minus_wrong_first_subset":
        "validation.strict_matched_wrong_time.endpoint_correct_minus_wrong_median.first_group_subset_nll_per_event",
    "correct_minus_wrong_continuation":
        "validation.strict_matched_wrong_time.endpoint_correct_minus_wrong_median.continuation_subset_nll_per_event",
    "correct_minus_wrong_same_prefix":
        "validation.strict_matched_wrong_time.endpoint_correct_minus_wrong_median.same_prefix_continuation_nll_per_event",
}


def value_at(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        value = value[key]
    return value


def load_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete R1.4 result: {path}")
    if value.get("r1_3_revision") != R1_3_REVISION:
        raise ValueError(f"target-observer revision mismatch: {path}")
    if value.get("experiment_label") != REVISION:
        raise ValueError(f"R1.4 experiment label mismatch: {path}")
    if value.get("initialisation_source_policy") != "r1_2_matching_seed":
        raise ValueError(f"R1.4 initialisation policy mismatch: {path}")
    swap = value["validation"]["strict_matched_wrong_time"]["audit"]
    if swap.get("same_recorded_coverage_segment") is not True:
        raise ValueError(f"R1.4 matched-swap crosses coverage segments: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed partition opened: {path}")
    return value


def median(values: list[float | None]) -> float | None:
    take = [float(value) for value in values if value is not None]
    return float(np.median(take)) if take else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "r1_4",
    )
    args = parser.parse_args()
    rows = []
    by_subject = {}
    for subject in SUBJECTS:
        arm_payloads: dict[str, list[dict]] = {}
        for arm in ARMS:
            payloads = [load_result(
                args.root / "fits" / subject / f"{arm}_seed_{seed}" / "result.json"
            ) for seed in SEEDS]
            arm_payloads[arm] = payloads
            row = {"subject": subject, "arm": arm, "n_seeds": len(payloads)}
            for label, path in FIELDS.items():
                row[label] = median([value_at(value, path) for value in payloads])
            row["selected_total_epoch_median"] = float(np.median([
                value["fit_trace"]["selected_total_epoch"] for value in payloads
            ]))
            row["n_selected_epoch_zero"] = int(sum(
                value["fit_trace"]["selected_total_epoch"] == 0
                for value in payloads
            ))
            row["validation_events"] = int(np.median([
                value["validation"]["persistent"]["n_events"]
                for value in payloads
            ]))
            row["matched_anchors"] = int(np.median([
                value["validation"]["strict_matched_wrong_time"]["audit"][
                    "n_matched_anchors"
                ] for value in payloads
            ]))
            row["all_same_recorded_segment"] = all(
                value["validation"]["strict_matched_wrong_time"]["audit"][
                    "same_recorded_coverage_segment"
                ] is True for value in payloads
            )
            if arm == "explicit_raw":
                row["raw_minus_explicit_joint"] = median([
                    value["paired_raw_minus_explicit"]["joint_nll_per_event"]
                    for value in payloads
                ])
            if arm == "explicit":
                sensitivity_paths = [
                    args.root / "sensitivity_10_donor" / subject
                    / f"explicit_seed_{seed}.json" for seed in SEEDS
                ]
                if all(path.exists() for path in sensitivity_paths):
                    sensitivity = [json.loads(path.read_text()) for path in sensitivity_paths]
                    if any(
                        value.get("same_checkpoint_as_primary_5_donor") is not True
                        or value.get("sealed_opened") is not False
                        for value in sensitivity
                    ):
                        raise ValueError(
                            f"invalid 10-donor sensitivity for {subject}"
                        )
                    row["correct_minus_wrong_joint_10_donor"] = median([
                        value["correct_minus_wrong_median"]["joint_nll_per_event"]
                        for value in sensitivity
                    ])
                    row["correct_minus_wrong_10_donor_favourable_seeds"] = int(sum(
                        value["correct_minus_wrong_median"]["joint_nll_per_event"] < 0
                        for value in sensitivity
                    ))
                else:
                    row["correct_minus_wrong_joint_10_donor"] = None
                    row["correct_minus_wrong_10_donor_favourable_seeds"] = None
            rows.append(row)
        explicit = arm_payloads["explicit"]
        raw = arm_payloads["explicit_raw"]
        by_subject[subject] = {
            "explicit_persistent_favourable_seeds": int(sum(
                value_at(value, FIELDS["persistent_minus_memoryless_joint"]) < 0
                for value in explicit
            )),
            "explicit_time_specific_favourable_seeds": int(sum(
                value_at(value, FIELDS["correct_minus_wrong_joint"]) < 0
                for value in explicit
            )),
            "explicit_first_subset_favourable_seeds": int(sum(
                value_at(value, FIELDS["persistent_minus_memoryless_first_subset"]) < 0
                for value in explicit
            )),
            "explicit_continuation_favourable_seeds": int(sum(
                value_at(value, FIELDS["persistent_minus_memoryless_continuation"]) < 0
                for value in explicit
            )),
            "raw_minus_explicit_joint_median": median([
                value["paired_raw_minus_explicit"]["joint_nll_per_event"]
                for value in raw
            ]),
            "raw_joint_favourable_seeds": int(sum(
                value["paired_raw_minus_explicit"]["joint_nll_per_event"] < 0
                for value in raw
            )),
            "stable_explicit_t1_for_t2": bool(
                sum(
                    value["fit_trace"]["selected_total_epoch"] > 0
                    and value_at(value, FIELDS["persistent_minus_memoryless_joint"]) < 0
                    and value_at(value, FIELDS["correct_minus_wrong_joint"]) < 0
                    for value in explicit
                ) >= 2
            ),
        }
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "r1_4_patient_arm.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({
            key for row in rows for key in row
        }))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETE",
        "revision": REVISION,
        "subjects": list(SUBJECTS),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "patient_arm": rows,
        "by_subject": by_subject,
        "n_stable_explicit_t1_for_t2": int(sum(
            value["stable_explicit_t1_for_t2"] for value in by_subject.values()
        )),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "six-patient development replication; patient-first predictive "
            "evidence only, not a cohort or causal mechanism conclusion"
        ),
    }
    contract.atomic_json(reports / "r1_4_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
