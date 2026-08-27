#!/usr/bin/env python3
"""Patient-first aggregation for R1.7A D_mechanism N=100 T2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.r1_7_t2 import R1_7_T2_REVISION


def nested(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return value


FIELDS = {
    "real_minus_placebo_joint": "comparisons.next_event.real_minus_state_matched_placebo.joint_nll_per_event",
    "real_minus_current_joint": "comparisons.next_event.real_minus_current_event_only.joint_nll_per_event",
    "real_minus_no_edge_joint": "comparisons.next_event.real_minus_no_edge.joint_nll_per_event",
    "real_minus_placebo_timing": "comparisons.next_event.real_minus_state_matched_placebo.timing_nll_per_event",
    "real_minus_placebo_mark": "comparisons.next_event.real_minus_state_matched_placebo.mark_nll_per_event",
    "real_minus_placebo_stop": "comparisons.next_event.real_minus_state_matched_placebo.stop_nll_per_event",
    "real_minus_placebo_first_subset": "comparisons.next_event.real_minus_state_matched_placebo.first_group_subset_nll_per_event",
    "real_minus_placebo_continuation": "comparisons.next_event.real_minus_state_matched_placebo.continuation_subset_nll_per_event",
    "H5_real_minus_placebo_mark": "comparisons.H5.real_minus_state_matched_placebo.mark_nll_per_event",
    "H10_real_minus_placebo_mark": "comparisons.H10.real_minus_state_matched_placebo.mark_nll_per_event",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    args = parser.parse_args()
    r1 = json.loads((args.root / "reports/r1_7a_summary.json").read_text())
    rows = []
    for subject in r1["t2_run_subjects"]:
        for source in ("load", "participation"):
            payloads = []
            for seed in range(5):
                t1 = json.loads((args.root / "fits" / subject / f"seed_{seed}/result.json").read_text())
                if not t1["stable_checkpoint"]:
                    continue
                path = args.root / "t2_r2" / subject / f"{source}_seed_{seed}_n_100/result.json"
                value = json.loads(path.read_text())
                if (value.get("status") != "COMPLETE"
                        or value.get("revision") != R1_7_T2_REVISION
                        or value.get("sealed_opened") is not False):
                    raise ValueError(f"invalid R1.7A T2 result: {path}")
                payloads.append(value)
            estimated = [value for value in payloads
                         if value.get("analysis_status") == "ESTIMATED"
                         and value.get("real_edge_estimable") is True]
            row = {"subject": subject, "source": source,
                   "stable_t1_seeds": len(payloads), "estimable_seeds": len(estimated),
                   "primary_increment_seeds": sum(v.get("primary_next_event_increment", False) for v in payloads),
                   "support_class": r1["by_subject"][subject]["t2_support_class"],
                   "d_mechanism_100_event_blocks": r1["by_subject"][subject]["d_mechanism_100_event_blocks"]}
            for label, path in FIELDS.items():
                values = [float(nested(value, path)) for value in estimated]
                row[label] = float(np.median(values)) if values else None
                row[label + "_favourable_seeds"] = sum(value < 0 for value in values)
            row["H5_persistence_seeds"] = sum(
                value["one_shot_persistence"]["H5"]["state_and_mark_persist"]
                for value in estimated
            )
            row["H10_persistence_seeds"] = sum(
                value["one_shot_persistence"]["H10"]["state_and_mark_persist"]
                for value in estimated
            )
            row["patient_source_support"] = bool(
                len(estimated) >= 3 and row["primary_increment_seeds"] >= 3
                and row["real_minus_placebo_joint"] is not None
                and row["real_minus_placebo_joint"] < 0
                and row["real_minus_current_joint"] < 0
                and row["real_minus_no_edge_joint"] < 0
            )
            rows.append(row)
    summary = {
        "status": "COMPLETE", "revision": R1_7_T2_REVISION,
        "patient_source": rows,
        "cohort_eligible_support": [
            {"subject": row["subject"], "source": row["source"]}
            for row in rows if row["patient_source_support"]
            and row["support_class"] == "COHORT_ELIGIBLE"
        ],
        "case_only_support": [
            {"subject": row["subject"], "source": row["source"]}
            for row in rows if row["patient_source_support"]
            and row["support_class"] == "CASE_ONLY"
        ],
        "n_ge_1000_runs": 0, "physical_clock_runs": 0,
        "free_exposure_intercept_present": False,
        "event_rows_not_treated_as_independent_patients": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    report = args.root / "reports"; report.mkdir(parents=True, exist_ok=True)
    contract.atomic_json(report / "t2_r2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
