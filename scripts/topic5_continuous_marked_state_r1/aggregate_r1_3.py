#!/usr/bin/env python3
"""Patient-first aggregation for formal R1.3 paired observer fits."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_2b import R1_2B_SUBJECTS
from src.topic5_continuous_marked_state_r1.r1_3 import R1_3_REVISION


ARMS = ("explicit", "explicit_raw")
SEEDS = (0, 1, 2)


def path_value(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return value


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


def main() -> None:
    root = contract.RESULT_ROOT / "r1_3"
    rows = []
    raw_pair_rows = []
    raw_common_isolated = []
    missing = []
    for subject in R1_2B_SUBJECTS:
        by_arm = {}
        for arm in ARMS:
            payloads = []
            for seed in SEEDS:
                path = root / "fits" / subject / f"{arm}_seed_{seed}" / "result.json"
                if not path.exists():
                    missing.append(str(path)); continue
                value = json.loads(path.read_text())
                if value.get("status") != "COMPLETE":
                    raise ValueError(f"incomplete R1.3 result {path}")
                if value.get("r1_3_revision") != R1_3_REVISION:
                    raise ValueError(f"R1.3 revision mismatch {path}")
                if value.get("sealed_opened") is not False:
                    raise ValueError(f"sealed partition opened {path}")
                payloads.append(value)
            if len(payloads) != 3:
                continue
            by_arm[arm] = payloads
            row = {"subject": subject, "arm": arm, "n_seeds": 3}
            for label, path in FIELDS.items():
                value = [path_value(payload, path) for payload in payloads]
                value = [float(item) for item in value if item is not None]
                row[label] = float(np.median(value)) if value else None
            row["selected_total_epoch_median"] = float(np.median([
                payload["fit_trace"]["selected_total_epoch"] for payload in payloads
            ]))
            row["n_selected_epoch_zero"] = int(sum(
                payload["fit_trace"]["selected_total_epoch"] == 0
                for payload in payloads
            ))
            row["n_selected_final_budget"] = int(sum(
                payload["fit_trace"]["selected_total_epoch"] == 4
                for payload in payloads
            ))
            row["validation_events"] = int(np.median([
                payload["validation"]["persistent"]["n_events"]
                for payload in payloads
            ]))
            row["matched_anchors"] = int(np.median([
                payload["validation"]["strict_matched_wrong_time"]["audit"]["n_matched_anchors"]
                for payload in payloads
            ]))
            if arm == "explicit_raw":
                raw_common_isolated.extend(
                    payload.get("paired_raw_common_parameter_update_exact_zero") is True
                    for payload in payloads
                )
                row["raw_tokenizer_gradient_min"] = float(min(
                    payload["raw_patch_tokenizer_target_gradient"] for payload in payloads
                ))
                row["raw_temporal_gradient_min"] = float(min(
                    min(payload["raw_temporal_layer_target_gradients"])
                    for payload in payloads
                ))
                row["raw_tokenizer_update_median"] = float(np.median([
                    payload["parameter_update_norm"]["raw_tokenizer"]
                    for payload in payloads
                ]))
            rows.append(row)
        if set(by_arm) == set(ARMS):
            raw = by_arm["explicit_raw"]
            raw_pair_rows.append({
                "subject": subject,
                "raw_minus_explicit_joint": float(np.median([
                    payload["paired_raw_minus_explicit"]["joint_nll_per_event"]
                    for payload in raw
                ])),
                "raw_minus_explicit_timing": float(np.median([
                    payload["paired_raw_minus_explicit"]["timing_nll_per_event"]
                    for payload in raw
                ])),
                "raw_minus_explicit_mark": float(np.median([
                    payload["paired_raw_minus_explicit"]["mark_nll_per_event"]
                    for payload in raw
                ])),
                "raw_minus_explicit_group_size": float(np.median([
                    payload["paired_raw_minus_explicit"]["group_size_nll_per_event"]
                    for payload in raw
                ])),
                "raw_minus_explicit_subset": float(np.median([
                    payload["paired_raw_minus_explicit"]["subset_nll_per_event"]
                    for payload in raw
                ])),
            })
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} R1.3 fits; first={missing[0]}")
    if len(raw_common_isolated) != 9 or not all(raw_common_isolated):
        raise ValueError("a paired raw fit updated common explicit/T1 parameters")
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / "r1_3_patient_arm.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({
            key for row in rows for key in row
        }))
        writer.writeheader(); writer.writerows(rows)
    with (reports / "r1_3_raw_paired_patient.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_pair_rows[0]))
        writer.writeheader(); writer.writerows(raw_pair_rows)
    summary = {
        "status": "COMPLETE",
        "n_fits": 18,
        "patient_arm": rows,
        "raw_paired_patient": raw_pair_rows,
        "raw_paired_summary": {
            key: {
                "patient_median": float(np.median([row[key] for row in raw_pair_rows])),
                "n_favourable_negative": int(sum(row[key] < 0 for row in raw_pair_rows)),
                "n_patients": len(raw_pair_rows),
            }
            for key in raw_pair_rows[0] if key != "subject"
        },
        "all_raw_selection_gradients_nonzero": bool(all(
            row.get("raw_tokenizer_gradient_min", 1.0) > 0
            and row.get("raw_temporal_gradient_min", 1.0) > 0
            for row in rows if row["arm"] == "explicit_raw"
        )),
        "all_raw_common_parameter_updates_exact_zero": bool(
            len(raw_common_isolated) == 9 and all(raw_common_isolated)
        ),
        "raw_selected_final_budget_fits": int(sum(
            row["n_selected_final_budget"]
            for row in rows if row["arm"] == "explicit_raw"
        )),
        "sealed_opened": False,
    }
    contract.atomic_json(reports / "r1_3_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
