#!/usr/bin/env python3
"""Patient-first aggregation of stable-T1 N=100 T2-R2.0 fits."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.t2_r2 import (
    T2_R2_REVISION,
    classify_one_shot_persistence,
)
from src.topic5_continuous_marked_state_r1.t2_r2_human import R1_4_REVISION, SOURCES


SEEDS = (0, 1, 2)


def median(values) -> float | None:
    take = [float(value) for value in values if value is not None]
    return float(np.median(take)) if take else None


def path_value(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return value


FIELDS = {
    "next_real_minus_placebo_joint":
        "comparisons.next_event.real_minus_state_matched_placebo.joint_nll_per_event",
    "next_real_minus_placebo_timing":
        "comparisons.next_event.real_minus_state_matched_placebo.timing_nll_per_event",
    "next_real_minus_placebo_mark":
        "comparisons.next_event.real_minus_state_matched_placebo.mark_nll_per_event",
    "next_real_minus_placebo_stop":
        "comparisons.next_event.real_minus_state_matched_placebo.stop_nll_per_event",
    "next_real_minus_placebo_first_subset":
        "comparisons.next_event.real_minus_state_matched_placebo.first_group_subset_nll_per_event",
    "next_real_minus_placebo_continuation":
        "comparisons.next_event.real_minus_state_matched_placebo.continuation_subset_nll_per_event",
    "next_real_minus_current_joint":
        "comparisons.next_event.real_minus_current_event_only.joint_nll_per_event",
    "next_real_minus_intercept_joint":
        "comparisons.next_event.real_minus_fitted_intercept_diagnostic.joint_nll_per_event",
    "next_real_minus_no_edge_joint":
        "comparisons.next_event.real_minus_no_edge.joint_nll_per_event",
    "H5_real_minus_placebo_mark":
        "comparisons.H5.real_minus_state_matched_placebo.mark_nll_per_event",
    "H10_real_minus_placebo_mark":
        "comparisons.H10.real_minus_state_matched_placebo.mark_nll_per_event",
    "H5_real_minus_placebo_state_mse":
        "comparisons.H5.real_minus_state_matched_placebo.state_mse_to_filtered_target",
    "H10_real_minus_placebo_state_mse":
        "comparisons.H10.real_minus_state_matched_placebo.state_mse_to_filtered_target",
    "H5_real_state_displacement":
        "validation.horizons.H5.real_cumulative.mean_state_displacement_from_no_edge",
    "H10_real_state_displacement":
        "validation.horizons.H10.real_cumulative.mean_state_displacement_from_no_edge",
}


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if (value.get("status") != "COMPLETE"
            or value.get("revision") != T2_R2_REVISION
            or value.get("r1_4_revision") != R1_4_REVISION
            or value.get("scale_events") != 100
            or value.get("sealed_opened") is not False):
        raise ValueError(f"invalid T2-R2.0 result: {path}")
    if value["t1"].get("r1_4_experiment_label") != R1_4_REVISION:
        raise ValueError(f"T2 did not use R1.4 T1: {path}")
    if value["design"].get("raw_correction_after_anchor") is not False:
        raise ValueError(f"T2 H5/H10 retained raw correction: {path}")
    if value["design"].get("later_t2_jumps") is not False:
        raise ValueError(f"T2 H5/H10 applied later jumps: {path}")
    if value.get("analysis_status", "ESTIMATED") not in {
        "ESTIMATED", "NOT_ESTIMABLE",
    }:
        raise ValueError(f"unknown T2 analysis status: {path}")
    return value


def corrected_persistence(value: dict, horizon: str) -> dict:
    return classify_one_shot_persistence(
        value["comparisons"][horizon][
            "real_minus_state_matched_placebo"
        ],
        value["validation"]["horizons"][horizon]["real_cumulative"],
        real_edge_estimable=bool(value["real_edge_estimable"]),
    )


def edge_estimable_payloads(payloads: list[dict]) -> list[dict]:
    """Return only fitted real-edge results eligible for effect summaries."""
    return [
        value for value in payloads
        if value.get("analysis_status", "ESTIMATED") == "ESTIMATED"
        and bool(value["real_edge_estimable"])
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--r1-4-root", type=Path, default=contract.RESULT_ROOT / "r1_4"
    )
    parser.add_argument(
        "--root", type=Path, default=contract.RESULT_ROOT / "t2_r2"
    )
    args = parser.parse_args()
    r1 = json.loads((args.r1_4_root / "reports/r1_4_summary.json").read_text())
    if r1.get("revision") != R1_4_REVISION or r1.get("sealed_opened") is not False:
        raise ValueError("invalid R1.4 summary")
    subjects = [
        subject for subject, value in r1["by_subject"].items()
        if value["stable_explicit_t1_for_t2"]
    ]
    rows = []
    by_subject = {}
    for subject in subjects:
        by_subject[subject] = {}
        for source in SOURCES:
            payloads = [load(
                args.root / "human" / subject / f"{source}_seed_{seed}_n_100/result.json"
            ) for seed in SEEDS]
            estimated = [
                value for value in payloads
                if value.get("analysis_status", "ESTIMATED") == "ESTIMATED"
            ]
            edge_estimable = edge_estimable_payloads(payloads)
            row = {"subject": subject, "source": source, "n_seeds": 3}
            for label, path in FIELDS.items():
                row[label] = median([
                    path_value(value, path) for value in edge_estimable
                ])
            row["estimable_seeds"] = int(len(edge_estimable))
            row["primary_increment_seeds"] = int(sum(
                value["primary_next_event_increment"] for value in payloads
            ))
            corrected_all = {
                horizon: [corrected_persistence(value, horizon)
                          for value in estimated]
                for horizon in ("H5", "H10")
            }
            stored_flag_mismatches = sum(
                value["one_shot_persistence"][horizon][
                    "state_and_mark_persist"
                ] != corrected_all[horizon][index]["state_and_mark_persist"]
                for index, value in enumerate(estimated)
                for horizon in ("H5", "H10")
            )
            row["H5_persistence_seeds"] = int(sum(
                corrected_persistence(value, "H5")["state_and_mark_persist"]
                for value in edge_estimable
            ))
            row["H10_persistence_seeds"] = int(sum(
                corrected_persistence(value, "H10")["state_and_mark_persist"]
                for value in edge_estimable
            ))
            row["stored_persistence_flag_mismatches"] = int(
                stored_flag_mismatches
            )
            row["structural_zero_seeds"] = int(sum(
                not value["fits"]["real_cumulative"]["edge_left_zero_initialisation"]
                for value in estimated
            ))
            row["support_ineligible_seeds"] = int(len(payloads) - len(estimated))
            row["support_ineligible_reasons"] = sorted({
                value["non_estimable_reason"] for value in payloads
                if value.get("analysis_status") == "NOT_ESTIMABLE"
            })
            train_pairs = [
                value["design"].get("train_next_event_pairs") for value in payloads
                if value["design"].get("train_next_event_pairs") is not None
            ]
            validation_pairs = [
                value["design"].get("validation_next_event_pairs") for value in payloads
                if value["design"].get("validation_next_event_pairs") is not None
            ]
            row["train_pairs"] = (
                int(np.median(train_pairs)) if train_pairs else None
            )
            row["validation_pairs"] = (
                int(np.median(validation_pairs)) if validation_pairs else None
            )
            row["eligible_for_scale_expansion"] = bool(
                row["estimable_seeds"] >= 2
                and row["primary_increment_seeds"] >= 2
                and row["next_real_minus_placebo_joint"] is not None
                and row["next_real_minus_current_joint"] is not None
                and row["next_real_minus_placebo_joint"] < 0
                and row["next_real_minus_current_joint"] < 0
            )
            rows.append(row)
            by_subject[subject][source] = {
                key: row[key] for key in (
                    "estimable_seeds", "primary_increment_seeds",
                    "H5_persistence_seeds", "H10_persistence_seeds",
                    "stored_persistence_flag_mismatches",
                    "structural_zero_seeds", "support_ineligible_seeds",
                    "support_ineligible_reasons", "eligible_for_scale_expansion",
                )
            }
    report = args.root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    with (report / "t2_r2_patient_source.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({
            key for row in rows for key in row
        }))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETE",
        "revision": T2_R2_REVISION,
        "r1_4_revision": R1_4_REVISION,
        "stable_t1_subjects": subjects,
        "n_stable_t1_subjects": len(subjects),
        "sources": list(SOURCES),
        "seeds": list(SEEDS),
        "patient_source": rows,
        "by_subject": by_subject,
        "scale_expansion_candidates": [
            {"subject": row["subject"], "source": row["source"]}
            for row in rows if row["eligible_for_scale_expansion"]
        ],
        "ordinary_negative_is_reported_not_hidden": True,
        "event_rows_are_not_treated_as_independent_patients": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "patient-first development N=100 screen; next-event increment is "
            "not called a persistent state update unless H5/H10 also retain "
            "the effect through the frozen generator"
        ),
    }
    contract.atomic_json(report / "t2_r2_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
