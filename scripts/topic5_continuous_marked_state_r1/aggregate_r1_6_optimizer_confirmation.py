#!/usr/bin/env python3
"""Patient-first aggregation of the frozen R1.6 optimizer confirmation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import R1_6_REVISION
from scripts.topic5_continuous_marked_state_r1.run_r1_6_optimizer_confirmation_cell import (
    CONFIRMATION_REVISION,
    FIXED_SUBJECTS,
)
from scripts.topic5_continuous_marked_state_r1.run_r1_6_optimizer_confirmation_queue import (
    CONFIRMATION_SEEDS,
)


SUMMARY_REVISION = "r1_6_optimizer_confirmation_patient_first_v1"


def nested(value: dict, path: str):
    current = value
    for key in path.split("."):
        current = current[key]
    return current


def finite_median(values) -> float | None:
    take = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(take)) if take else None


def classify_subject(*, stable: int, stable_independent: int,
                     selected_nonzero: int, train_favourable: int,
                     overfit_pass: int) -> str:
    """Separate robust support, sensitivity, generalisation and optimisation."""
    if stable >= 3 and stable_independent >= 1:
        return "OPTIMIZATION_ROBUST_SUPPORT"
    if stable >= 1:
        return "OPTIMIZER_SENSITIVE_SUPPORT"
    if overfit_pass >= 2 and train_favourable >= 3:
        return "GENERALISATION_FAILURE_OR_CURRENT_MODEL_NONIDENTIFIABLE"
    if overfit_pass >= 2 and selected_nonzero == 0:
        return "INNER_SELECTION_NO_UPDATE_AFTER_OVERFIT_PASS"
    return "OPTIMIZATION_FAILURE_OR_INSUFFICIENT_DIAGNOSTIC"


def validate_result(value: dict, *, subject: str, seed: int,
                    prefix: str, config: str) -> None:
    required = {
        "status": "COMPLETE",
        "revision": R1_6_REVISION,
        "confirmation_revision": CONFIRMATION_REVISION,
        "subject": subject,
        "seed": seed,
        "selected_prefix_config": prefix,
        "selected_config": config,
        "development_validation_scored": True,
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(
                f"confirmation field mismatch {subject} seed {seed}: "
                f"{key}={value.get(key)!r}, expected {expected!r}"
            )
    checkpoint = Path(value["checkpoint"])
    if contract.sha256_file(checkpoint) != value["checkpoint_sha256"]:
        raise ValueError(f"confirmation checkpoint hash mismatch: {checkpoint}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    root = args.root
    status = json.loads((root / "CONFIRMATION_STATUS.json").read_text())
    if status.get("status") != "COMPLETE":
        raise ValueError("R1.6 confirmation queue is not complete")
    prefix = str(status["selected_prefix_config"])
    config = str(status["selected_config"])
    overfit = json.loads((root / "reports/tuning_summary.json").read_text())

    metrics = {
        "persistent_minus_memoryless_joint": (
            "validation.persistent_minus_memoryless.joint_nll_per_event"
        ),
        "persistent_minus_memoryless_timing": (
            "validation.persistent_minus_memoryless.timing_nll_per_event"
        ),
        "persistent_minus_memoryless_mark": (
            "validation.persistent_minus_memoryless.mark_nll_per_event"
        ),
        "correct_minus_wrong_joint": (
            "validation.strict_matched_wrong_time.correct_minus_wrong_median."
            "joint_nll_per_event"
        ),
        "correct_minus_wrong_timing": (
            "validation.strict_matched_wrong_time.correct_minus_wrong_median."
            "timing_nll_per_event"
        ),
        "correct_minus_wrong_mark": (
            "validation.strict_matched_wrong_time.correct_minus_wrong_median."
            "mark_nll_per_event"
        ),
        "persistent_minus_memoryless_stop": (
            "validation.mark_endpoints.persistent_minus_memoryless."
            "stop_nll_per_event"
        ),
        "persistent_minus_memoryless_first_subset": (
            "validation.mark_endpoints.persistent_minus_memoryless."
            "first_group_subset_nll_per_event"
        ),
        "persistent_minus_memoryless_continuation": (
            "validation.mark_endpoints.persistent_minus_memoryless."
            "continuation_subset_nll_per_event"
        ),
        "correct_minus_wrong_stop": (
            "validation.strict_matched_wrong_time."
            "endpoint_correct_minus_wrong_median.stop_nll_per_event"
        ),
        "correct_minus_wrong_first_subset": (
            "validation.strict_matched_wrong_time."
            "endpoint_correct_minus_wrong_median.first_group_subset_nll_per_event"
        ),
        "correct_minus_wrong_continuation": (
            "validation.strict_matched_wrong_time."
            "endpoint_correct_minus_wrong_median.continuation_subset_nll_per_event"
        ),
    }
    seed_rows = []
    by_subject = {}
    for subject in FIXED_SUBJECTS:
        local = []
        for seed in CONFIRMATION_SEEDS:
            path = (
                root / "confirmation" / prefix / config
                / subject / f"seed_{seed}/result.json"
            )
            value = json.loads(path.read_text())
            validate_result(
                value, subject=subject, seed=seed,
                prefix=prefix, config=config,
            )
            trace = value["fit_trace"]
            trajectory = trace["trajectory"]
            train_values = [
                float(row["evaluated_train_joint_nll"])
                for row in trajectory
            ]
            row = {
                "subject": subject,
                "seed": int(seed),
                "seed_role": value["seed_role"],
                "selected_epoch": int(trace["selected_total_epoch"]),
                "best_train_improvement": float(
                    train_values[0] - min(train_values)
                ),
                "terminal_train_improvement": float(
                    train_values[0] - train_values[-1]
                ),
                "stable_checkpoint": bool(value["stable_checkpoint"]),
                "matched_anchors": int(nested(
                    value,
                    "validation.strict_matched_wrong_time.audit.n_matched_anchors",
                )),
                "checkpoint_sha256": value["checkpoint_sha256"],
                "result": str(path),
                "result_sha256": contract.sha256_file(path),
            }
            for label, metric_path in metrics.items():
                try:
                    row[label] = nested(value, metric_path)
                except KeyError:
                    row[label] = None
            seed_rows.append(row)
            local.append(row)

        stable = sum(row["stable_checkpoint"] for row in local)
        stable_independent = sum(
            row["stable_checkpoint"] and row["seed"] in (3, 4)
            for row in local
        )
        selected_nonzero = sum(row["selected_epoch"] > 0 for row in local)
        train_favourable = sum(
            row["best_train_improvement"] > 0 for row in local
        )
        overfit_pass = int(overfit["overfit_patient_pass"].get(subject, 0))
        summary = {
            "stable_checkpoints": int(stable),
            "stable_independent_seeds": int(stable_independent),
            "selected_nonzero_seeds": int(selected_nonzero),
            "train_favourable_seeds": int(train_favourable),
            "overfit_pass_seeds": overfit_pass,
            "classification": classify_subject(
                stable=stable, stable_independent=stable_independent,
                selected_nonzero=selected_nonzero,
                train_favourable=train_favourable,
                overfit_pass=overfit_pass,
            ),
        }
        for label in metrics:
            values = [row[label] for row in local]
            summary[f"median_{label}"] = finite_median(values)
            summary[f"favourable_{label}_seeds"] = int(sum(
                value is not None and np.isfinite(value) and value < 0
                for value in values
            ))
        by_subject[subject] = summary

    h3_eligible = [
        subject for subject, value in by_subject.items()
        if value["stable_checkpoints"] >= 3
    ]
    result = {
        "status": "COMPLETE",
        "revision": SUMMARY_REVISION,
        "r1_6_revision": R1_6_REVISION,
        "confirmation_revision": CONFIRMATION_REVISION,
        "selected_prefix_config": prefix,
        "selected_config": config,
        "subjects": list(FIXED_SUBJECTS),
        "seeds": list(CONFIRMATION_SEEDS),
        "seed_rows": seed_rows,
        "by_subject": by_subject,
        "stable_t1_subjects_for_minimal_h3": h3_eligible,
        "n_stable_t1_subjects_for_minimal_h3": len(h3_eligible),
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "claim_boundary": (
            "patient-first development optimizer confirmation; five seeds "
            "measure optimisation stability and are not five patients"
        ),
    }
    report = root / "reports"
    contract.atomic_json(report / "optimizer_confirmation_summary.json", result)
    csv_path = report / "optimizer_confirmation_seed_rows.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seed_rows[0]))
        writer.writeheader()
        writer.writerows(seed_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
