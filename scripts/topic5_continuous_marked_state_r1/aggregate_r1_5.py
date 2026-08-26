#!/usr/bin/env python3
"""Patient-first aggregation for the frozen R1.5 long-support extension."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_3 import R1_3_REVISION
from scripts.topic5_continuous_marked_state_r1.aggregate_r1_4 import (
    FIELDS,
    median,
    value_at,
)


REVISION = "r1_5_long_support_explicit_extension_v1"
SUBJECTS = contract.R1_5_EXTENSION_SUBJECTS
INDEPENDENT_EXTENSION_SUBJECTS = contract.R1_5_NOVEL_SUBJECTS
CALIBRATION_SUBJECTS = contract.R1_5_LONG_CARRYOVER_SUBJECTS
SEEDS = (0, 1, 2, 3, 4)


def load_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") != "COMPLETE":
        raise ValueError(f"incomplete R1.5 result: {path}")
    if value.get("r1_3_revision") != R1_3_REVISION:
        raise ValueError(f"target-observer revision mismatch: {path}")
    if value.get("experiment_label") != REVISION:
        raise ValueError(f"R1.5 experiment label mismatch: {path}")
    if value.get("initialisation_source_policy") != "r1_2_matching_seed":
        raise ValueError(f"R1.5 initialisation policy mismatch: {path}")
    swap = value["validation"]["strict_matched_wrong_time"]["audit"]
    if swap.get("same_recorded_coverage_segment") is not True:
        raise ValueError(f"R1.5 matched-swap crosses coverage segments: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed partition opened: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "r1_5",
    )
    args = parser.parse_args()
    rows = []
    by_subject = {}
    for subject in SUBJECTS:
        payloads = [load_result(
            args.root / "fits" / subject / f"explicit_seed_{seed}" / "result.json"
        ) for seed in SEEDS]
        row = {
            "subject": subject,
            "subject_role": (
                "independent_extension"
                if subject in INDEPENDENT_EXTENSION_SUBJECTS
                else "previously_seen_long_record_calibration"
            ),
            "arm": "explicit",
            "n_seeds": len(payloads),
        }
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
            value["validation"]["persistent"]["n_events"] for value in payloads
        ]))
        row["matched_anchors"] = int(np.median([
            value["validation"]["strict_matched_wrong_time"]["audit"][
                "n_matched_anchors"
            ] for value in payloads
        ]))
        rows.append(row)
        seed_stable = [
            bool(
                value["fit_trace"]["selected_total_epoch"] > 0
                and value_at(
                    value, FIELDS["persistent_minus_memoryless_joint"]
                ) < 0
                and value_at(value, FIELDS["correct_minus_wrong_joint"]) < 0
            )
            for value in payloads
        ]
        by_subject[subject] = {
            "subject_role": row["subject_role"],
            "persistent_favourable_seeds": int(sum(
                value_at(
                    value, FIELDS["persistent_minus_memoryless_joint"]
                ) < 0 for value in payloads
            )),
            "time_specific_favourable_seeds": int(sum(
                value_at(value, FIELDS["correct_minus_wrong_joint"]) < 0
                for value in payloads
            )),
            "first_subset_favourable_seeds": int(sum(
                value_at(
                    value, FIELDS["persistent_minus_memoryless_first_subset"]
                ) < 0 for value in payloads
            )),
            "continuation_favourable_seeds": int(sum(
                value_at(
                    value, FIELDS["persistent_minus_memoryless_continuation"]
                ) < 0 for value in payloads
            )),
            "joint_stable_seeds": int(sum(seed_stable)),
            "stable_explicit_t1_for_h3": bool(sum(seed_stable) >= 3),
        }
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / "r1_5_patient.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({
            key for row in rows for key in row
        }))
        writer.writeheader(); writer.writerows(rows)
    stable_new = [
        subject for subject in INDEPENDENT_EXTENSION_SUBJECTS
        if by_subject[subject]["stable_explicit_t1_for_h3"]
    ]
    stable_all = [
        subject for subject in SUBJECTS
        if by_subject[subject]["stable_explicit_t1_for_h3"]
    ]
    summary = {
        "status": "COMPLETE",
        "revision": REVISION,
        "subjects": list(SUBJECTS),
        "independent_extension_subjects": list(INDEPENDENT_EXTENSION_SUBJECTS),
        "calibration_subjects": list(CALIBRATION_SUBJECTS),
        "seeds": list(SEEDS),
        "patient_arm": rows,
        "by_subject": by_subject,
        "stable_independent_extension_subjects": stable_new,
        "stable_all_subjects": stable_all,
        "n_stable_independent_extension_subjects": len(stable_new),
        "n_stable_all_subjects": len(stable_all),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "five genuinely added development subjects plus one previously "
            "seen long-record calibration subject; predictive evidence only"
        ),
    }
    contract.atomic_json(reports / "r1_5_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
