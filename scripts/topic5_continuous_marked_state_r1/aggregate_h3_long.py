#!/usr/bin/env python3
"""Patient-first aggregation of R1.5 exact-window H3-long results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.h3_long import (
    H3_LONG_REVISION,
    H3_LONG_SUPPORT_REVISION,
    SOURCES,
)
from src.topic5_continuous_marked_state_r1.h3_long_human import (
    R1_5_REVISION,
    cell_package_fingerprint,
)


SEEDS = (0, 1, 2, 3, 4)


def value_at(value: dict, path: str):
    for key in path.split("."):
        value = value[key]
    return value


def median(values) -> float | None:
    take = [float(value) for value in values if value is not None]
    return float(np.median(take)) if take else None


FIELDS = {
    "next_real_minus_state_joint":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.joint_nll_per_event",
    "next_real_minus_causal_joint":
        "comparisons.next_event.real_minus_causal_previous_block.joint_nll_per_event",
    "next_real_minus_current_joint":
        "comparisons.next_event.real_minus_current_event_only.joint_nll_per_event",
    "next_real_minus_chronological_joint":
        "comparisons.next_event.real_minus_chronological_trend.joint_nll_per_event",
    "next_real_minus_intercept_joint":
        "comparisons.next_event.real_minus_intercept_only.joint_nll_per_event",
    "next_real_minus_state_timing":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.timing_nll_per_event",
    "next_real_minus_state_mark":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.mark_nll_per_event",
    "next_real_minus_state_stop":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.stop_nll_per_event",
    "next_real_minus_state_first_subset":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.first_group_subset_nll_per_event",
    "next_real_minus_state_continuation":
        "comparisons.next_event.real_minus_state_matched_nonoverlap.continuation_subset_nll_per_event",
}


def optional_value(value: dict, path: str):
    try:
        return value_at(value, path)
    except KeyError:
        return None


def load(path: Path, *, expected: dict, support_path: Path,
         r1_5_root: Path) -> dict:
    value = json.loads(path.read_text())
    fingerprint, components = cell_package_fingerprint(
        expected["subject"], expected["seed"], expected["source"],
        expected["scale_events"], expected["support_role"],
        support_path=support_path, r1_5_root=r1_5_root,
        runner_path=(
            contract.REPO_ROOT
            / "scripts/topic5_continuous_marked_state_r1/run_h3_long_human.py"
        ),
    )
    if (
        value.get("status") != "COMPLETE"
        or value.get("revision") != H3_LONG_REVISION
        or value.get("r1_5_revision") != R1_5_REVISION
        or value.get("sealed_opened") is not False
        or value.get("formal_test_partition_opened") is not False
        or value.get("development_time_contract_verified") is not True
        or value.get("package_fingerprint") != fingerprint
        or value.get("package_components") != components
        or any(value.get(key) != expected[key] for key in expected)
    ):
        raise ValueError(f"invalid H3-long result: {path}")
    return value


def edge_estimable(payloads: list[dict]) -> list[dict]:
    return [
        value for value in payloads
        if value.get("analysis_status") == "ESTIMATED"
        and value.get("real_edge_estimable") is True
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "r1_5_h3_long",
    )
    parser.add_argument(
        "--r1-5-root", type=Path, default=contract.RESULT_ROOT / "r1_5",
    )
    args = parser.parse_args()
    support = json.loads((args.root / "support/summary.json").read_text())
    r1 = json.loads((args.r1_5_root / "reports/r1_5_summary.json").read_text())
    if (
        support.get("status") != "COMPLETE"
        or support.get("revision") != H3_LONG_SUPPORT_REVISION
        or support.get("sealed_opened") is not False
        or support.get("formal_test_partition_opened") is not False
        or r1.get("status") != "COMPLETE"
        or r1.get("revision") != R1_5_REVISION
        or r1.get("sealed_opened") is not False
        or r1.get("formal_test_partition_opened") is not False
    ):
        raise ValueError("invalid support or R1.5 summary")
    rows = []
    for cell in support["scheduled_cells"]:
        subject = cell["subject"]
        scale = int(cell["scale_events"])
        for source in SOURCES:
            payloads = [load(
                args.root / "human" / subject / source
                / f"seed_{seed}_n_{scale}/result.json",
                expected={
                    "subject": subject, "seed": seed, "source": source,
                    "scale_events": scale, "support_role": cell["role"],
                },
                support_path=args.root / "support/summary.json",
                r1_5_root=args.r1_5_root,
            ) for seed in SEEDS]
            estimated = [
                value for value in payloads
                if value.get("analysis_status") == "ESTIMATED"
            ]
            estimable = edge_estimable(payloads)
            row = {
                "subject": subject, "source": source,
                "scale_events": scale, "support_role": cell["role"],
                "stable_t1_patient": bool(
                    r1["by_subject"][subject]["stable_explicit_t1_for_h3"]
                ),
                "n_seeds": len(payloads), "estimated_seeds": len(estimated),
                "edge_estimable_seeds": len(estimable),
                "zero_selected_seeds": int(sum(
                    value.get("real_estimability_class") == "ZERO_SELECTED"
                    for value in payloads
                )),
                "rank_degenerate_seeds": int(sum(
                    value.get("real_estimability_class") == "RANK_DEGENERATE"
                    for value in payloads
                )),
                "zero_gradient_seeds": int(sum(
                    value.get("real_estimability_class") == "ZERO_GRADIENT"
                    for value in payloads
                )),
                "nonfinite_gradient_seeds": int(sum(
                    value.get("real_estimability_class") == "NONFINITE_GRADIENT"
                    for value in payloads
                )),
                "not_estimable_seeds": int(sum(
                    value.get("analysis_status") == "NOT_ESTIMABLE"
                    for value in payloads
                )),
                "control_numerically_invalid_seeds": int(sum(
                    value.get("analysis_status") == "ESTIMATED"
                    and value.get("control_numerically_valid") is not True
                    for value in payloads
                )),
                "primary_full_control_increment_seeds": int(sum(
                    value.get("primary_full_control_increment", False)
                    for value in payloads
                )),
                "supportive_boundary_increment_seeds": int(sum(
                    value.get("supportive_boundary_increment", False)
                    for value in payloads
                )),
                "distinct_seed_payloads": len({
                    value.get("seed_payload_sha256") for value in estimated
                    if value.get("seed_payload_sha256")
                }),
                "primary_full_control_increment_distinct_payloads": len({
                    value.get("seed_payload_sha256") for value in payloads
                    if value.get("primary_full_control_increment")
                }),
                "supportive_boundary_increment_distinct_payloads": len({
                    value.get("seed_payload_sha256") for value in payloads
                    if value.get("supportive_boundary_increment")
                }),
                "stable_t1_seed_checkpoints": int(sum(
                    value.get("stable_t1_seed", False) for value in payloads
                )),
                "train_independent_units_final_common_median": median([
                    optional_value(
                        value,
                        "design.train_independent_units_on_final_common_support",
                    ) for value in estimated
                ]),
                "validation_independent_units_final_common_median": median([
                    optional_value(
                        value,
                        "design.validation_independent_units_on_final_common_support",
                    ) for value in estimated
                ]),
                "support_audit_train_independent_blocks": cell[
                    "train_independent_blocks"
                ],
                "support_audit_validation_independent_blocks": cell[
                    "validation_independent_blocks"
                ],
                "validation_hours_median": cell["validation_hours_median"],
            }
            for label, path in FIELDS.items():
                row[label] = median([
                    optional_value(value, path) for value in estimable
                ])
            for horizon in (5, 10):
                row[f"H{horizon}_persistence_seeds"] = int(sum(
                    value.get("one_shot_persistence", {}).get(
                        f"H{horizon}", {}
                    ).get("state_and_mark_persist", False)
                    for value in payloads
                ))
                row[f"H{horizon}_eligible_stable_t1_seeds"] = int(sum(
                    value.get("stable_t1_seed", False) for value in payloads
                ))
                row[f"H{horizon}_persistence_distinct_payloads"] = len({
                    value.get("seed_payload_sha256") for value in payloads
                    if value.get("one_shot_persistence", {}).get(
                        f"H{horizon}", {}
                    ).get("state_and_mark_persist", False)
                })
                row[f"H{horizon}_real_minus_state_mark"] = median([
                    optional_value(
                        value,
                        f"comparisons.H{horizon}.real_minus_state_matched_nonoverlap.mark_nll_per_event",
                    ) for value in estimable
                ])
                row[f"H{horizon}_real_minus_state_mse"] = median([
                    optional_value(
                        value,
                        f"comparisons.H{horizon}.real_minus_state_matched_nonoverlap.state_mse_to_filtered_target",
                    ) for value in estimable
                ])
            rows.append(row)
    report = args.root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    csv_path = report / "h3_long_patient_scale_source.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({
            key for row in rows for key in row
        }))
        writer.writeheader(); writer.writerows(rows)
    scale_source = []
    for scale in sorted({row["scale_events"] for row in rows}):
        for source in SOURCES:
            take = [
                row for row in rows
                if row["scale_events"] == scale and row["source"] == source
            ]
            full = [row for row in take if row["support_role"] == "full_control"]
            boundary = [
                row for row in take
                if row["support_role"] == "boundary_incomplete_control"
            ]
            scale_source.append({
                "scale_events": scale, "source": source,
                "subjects": len(take), "full_control_subjects": len(full),
                "boundary_subjects": len(boundary),
                "full_control_patient_positive": int(sum(
                    row["primary_full_control_increment_distinct_payloads"] >= 3
                    for row in full
                )),
                "boundary_patient_supportive": int(sum(
                    row["supportive_boundary_increment_distinct_payloads"] >= 3
                    for row in boundary
                )),
                "stable_t1_subjects": int(sum(
                    row["stable_t1_patient"] for row in take
                )),
                "H5_persistent_subjects": int(sum(
                    row["H5_eligible_stable_t1_seeds"] >= 3
                    and row["H5_persistence_distinct_payloads"] >= 3
                    for row in take
                )),
                "H10_persistent_subjects": int(sum(
                    row["H10_eligible_stable_t1_seeds"] >= 3
                    and row["H10_persistence_distinct_payloads"] >= 3
                    for row in take
                )),
            })
    summary = {
        "status": "COMPLETE", "revision": H3_LONG_REVISION,
        "r1_5_revision": R1_5_REVISION,
        "patient_scale_source": rows, "scale_source": scale_source,
        "subjects": list(contract.R1_5_EXTENSION_SUBJECTS),
        "seeds": list(SEEDS), "sources": list(SOURCES),
        "scales": sorted({row["scale_events"] for row in rows}),
        "ordinary_negative_is_retained": True,
        "zero_selected_and_numerical_failures_are_separate": True,
        "duplicate_seed_payloads_do_not_count_as_seed_robustness": True,
        "boundary_incomplete_control_not_in_primary_denominator": True,
        "sliding_pairs_not_treated_as_independent_patients": True,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "claim_boundary": (
            "development exact-N antecedent screen; full controls, boundary "
            "readouts, seed-level T1 eligibility, duplicate payloads, zero "
            "selection and numerical limitations are separate"
        ),
    }
    contract.atomic_json(report / "h3_long_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
