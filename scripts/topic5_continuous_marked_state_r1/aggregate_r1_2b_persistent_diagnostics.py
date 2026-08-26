#!/usr/bin/env python3
"""Patient-first aggregation of the R1.2b closeout diagnostics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2b import R1_2B_SUBJECTS


ARMS = ("joint_explicit", "joint_explicit_raw")
SEEDS = (0, 1, 2)


def get(payload: dict, path: str):
    value = payload
    for key in path.split("."):
        value = value[key]
    return value


FIELDS = {
    "persistent_minus_memoryless_joint":
        "persistent_minus_memoryless.joint_nll_per_event",
    "persistent_minus_memoryless_timing":
        "persistent_minus_memoryless.timing_nll_per_event",
    "persistent_minus_memoryless_mark":
        "persistent_minus_memoryless.mark_nll_per_event",
    "persistent_minus_memoryless_stop":
        "mark_endpoints.persistent_minus_memoryless.stop_nll_per_event",
    "persistent_minus_memoryless_selecting_size":
        "mark_endpoints.persistent_minus_memoryless.selecting_group_size_nll_per_event",
    "persistent_minus_memoryless_first_subset":
        "mark_endpoints.persistent_minus_memoryless.first_group_subset_nll_per_event",
    "persistent_minus_memoryless_continuation":
        "mark_endpoints.persistent_minus_memoryless.continuation_subset_nll_per_event",
    "persistent_minus_memoryless_same_prefix":
        "mark_endpoints.persistent_minus_memoryless.same_prefix_continuation_nll_per_event",
    "correct_minus_wrong_joint":
        "strict_matched_wrong_time.correct_minus_wrong_median.joint_nll_per_event",
    "correct_minus_wrong_timing":
        "strict_matched_wrong_time.correct_minus_wrong_median.timing_nll_per_event",
    "correct_minus_wrong_mark":
        "strict_matched_wrong_time.correct_minus_wrong_median.mark_nll_per_event",
    "correct_minus_wrong_stop":
        "strict_matched_wrong_time.endpoint_correct_minus_wrong_median.stop_nll_per_event",
    "correct_minus_wrong_first_subset":
        "strict_matched_wrong_time.endpoint_correct_minus_wrong_median.first_group_subset_nll_per_event",
    "correct_minus_wrong_continuation":
        "strict_matched_wrong_time.endpoint_correct_minus_wrong_median.continuation_subset_nll_per_event",
    "correct_minus_wrong_same_prefix":
        "strict_matched_wrong_time.endpoint_correct_minus_wrong_median.same_prefix_continuation_nll_per_event",
}


def main() -> None:
    root = contract.RESULT_ROOT / "r1_2b"
    rows = []
    missing = []
    for subject in R1_2B_SUBJECTS:
        for arm in ARMS:
            payloads = []
            for seed in SEEDS:
                path = root / "diagnostics" / subject / f"{arm}_seed_{seed}" / "result.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                value = json.loads(path.read_text())
                if value.get("status") != "COMPLETE" or value.get("sealed_opened") is not False:
                    raise ValueError(f"invalid diagnostic result {path}")
                payloads.append(value)
            if len(payloads) != 3:
                continue
            row = {"subject": subject, "arm": arm, "n_seeds": 3}
            for label, path in FIELDS.items():
                values = [get(value, path) for value in payloads]
                values = [float(value) for value in values if value is not None]
                row[label] = float(np.median(values)) if values else None
            row["selected_epoch_median"] = float(np.median([
                value["selected_epochs"] for value in payloads
            ]))
            row["strict_swap_matched_anchors_median"] = float(np.median([
                value["strict_matched_wrong_time"]["audit"]["n_matched_anchors"]
                for value in payloads
            ]))
            rows.append(row)
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} diagnostics; first={missing[0]}")
    reports = root / "reports"
    csv_path = reports / "r1_2b_persistent_diagnostics_patient_first.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {
        "status": "COMPLETE",
        "n_results": 18,
        "n_patient_arm_rows": len(rows),
        "patient_first": rows,
        "arm_summary": {},
        "sealed_opened": False,
    }
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        summary["arm_summary"][arm] = {
            label: {
                "patient_median": float(np.median([
                    row[label] for row in selected if row[label] is not None
                ])),
                "n_favourable_negative": int(sum(
                    row[label] is not None and row[label] < 0 for row in selected
                )),
                "n_patients": len(selected),
            }
            for label in FIELDS
        }
    contract.atomic_json(
        reports / "r1_2b_persistent_diagnostics_summary.json", summary
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
