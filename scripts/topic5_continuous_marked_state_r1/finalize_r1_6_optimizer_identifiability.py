#!/usr/bin/env python3
"""Build the machine closeout for the R1.6 optimizer diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import R1_6_REVISION
from src.topic5_continuous_marked_state_r1.optimizer_h3 import (
    R1_6_MINIMAL_H3_REVISION,
)


CLOSEOUT_REVISION = "r1_6_optimizer_identifiability_closeout_v1"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def repo_relative(path: Path) -> str:
    """Persist a path without tying the audit to its isolated worktree."""
    # Keep this lexical: some read-only upstream result trees are symlinked
    # from the isolated worktree to the canonical workspace.
    return str(path.absolute().relative_to(contract.REPO_ROOT.absolute()))


def digest_manifest(paths: list[Path], root: Path) -> dict:
    rows = [
        {
            "path": str(path.relative_to(root)),
            "sha256": contract.sha256_file(path),
        }
        for path in sorted(paths)
    ]
    digest = hashlib.sha256(json.dumps(
        rows, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return {"count": len(rows), "combined_sha256": digest, "files": rows}


def assert_closed(value: dict, path: Path) -> None:
    if value.get("formal_test_partition_opened") is not False:
        raise ValueError(f"formal partition flag is not false: {path}")
    if value.get("sealed_opened") is not False:
        raise ValueError(f"sealed flag is not false: {path}")


def synthetic_summary(root: Path) -> dict:
    selected = {
        "ample_positive": root / "synthetic/syn_scale_600.json",
        "high_lr_positive": root / "synthetic/syn_lr1e-2_e80.json",
        "high_lr_zero_reversed": (
            root / "synthetic/syn_selected_lr1e-2_zero_reversed.json"
        ),
        "short_patience": root / "synthetic/syn_early_lr3e-3_p10.json",
    }
    result = {}
    for label, path in selected.items():
        value = read(path)
        assert_closed(value, path)
        by_truth = {}
        for truth in sorted({row["truth"] for row in value["rows"]}):
            rows = [row for row in value["rows"] if row["truth"] == truth]
            by_truth[truth] = {
                "recovered": int(sum(row["recovered"] for row in rows)),
                "total": len(rows),
                "selected_nonzero": int(sum(
                    int(row["selected_epoch"]) > 0 for row in rows
                )),
                "test_better_than_baseline": int(sum(
                    float(row["test_minus_baseline"]) < 0 for row in rows
                )),
                "correct_better_than_wrong_time": int(sum(
                    float(row["test_minus_wrong_time"]) < 0 for row in rows
                )),
                "median_test_minus_baseline": float(np.median([
                    row["test_minus_baseline"] for row in rows
                ])),
            }
        result[label] = {
            "path": repo_relative(path),
            "sha256": contract.sha256_file(path),
            "by_truth": by_truth,
        }
    return result


def h3_summary(paths: list[Path]) -> dict:
    rows = []
    for path in paths:
        value = read(path)
        assert_closed(value, path)
        if value.get("revision") != R1_6_MINIMAL_H3_REVISION:
            raise ValueError(f"minimal H3 revision mismatch: {path}")
        row = {
            "subject": value["subject"],
            "seed": int(value["seed"]),
            "source": value["source"],
            "estimable": bool(value["real_edge_estimable"]),
            "independent_validation_units": int(
                value["independent_block_analysis"]["n_units"]
            ),
            "primary": bool(value["primary_full_control_increment"]),
            "result": repo_relative(path),
            "result_sha256": contract.sha256_file(path),
        }
        for comparator in (
            "intercept_only", "state_matched_nonoverlap",
            "causal_previous_block", "current_event_only",
            "chronological_trend",
        ):
            key = f"real_minus_{comparator}"
            row[f"{key}_joint"] = float(
                value["comparisons"]["next_event"][key][
                    "joint_nll_per_event"
                ]
            )
            row[f"{key}_block_median"] = float(
                value["independent_block_analysis"]["comparisons"][key][
                    "median"
                ]
            )
        rows.append(row)
    by_source = {}
    for source in sorted({row["source"] for row in rows}):
        local = [row for row in rows if row["source"] == source]
        by_source[source] = {
            "cells": len(local),
            "estimable_cells": int(sum(row["estimable"] for row in local)),
            "primary_cells": int(sum(row["primary"] for row in local)),
            "independent_validation_units": sorted({
                row["independent_validation_units"] for row in local
            }),
            "favourable_next_event_vs_each_control": {
                comparator: int(sum(
                    row[f"real_minus_{comparator}_joint"] < 0
                    for row in local
                ))
                for comparator in (
                    "intercept_only", "state_matched_nonoverlap",
                    "causal_previous_block", "current_event_only",
                    "chronological_trend",
                )
            },
            "favourable_independent_block_vs_each_control": {
                comparator: int(sum(
                    row[f"real_minus_{comparator}_block_median"] < 0
                    for row in local
                ))
                for comparator in (
                    "intercept_only", "state_matched_nonoverlap",
                    "causal_previous_block", "current_event_only",
                    "chronological_trend",
                )
            },
        }
    return {"rows": rows, "by_source": by_source}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    root = args.root
    reports = root / "reports"
    prefix_status_path = root / "PREFIX_TUNING_STATUS.json"
    alignment_status_path = root / "ALIGNMENT_TUNING_STATUS.json"
    confirmation_status_path = root / "CONFIRMATION_STATUS.json"
    h3_status_path = root / "MINIMAL_H3_STATUS.json"
    prefix_summary_path = reports / "prefix_tuning_summary.json"
    tuning_summary_path = reports / "tuning_summary.json"
    confirmation_summary_path = reports / "optimizer_confirmation_summary.json"
    top_paths = [
        prefix_status_path, alignment_status_path, confirmation_status_path,
        h3_status_path, prefix_summary_path, tuning_summary_path,
        confirmation_summary_path,
    ]
    for path in top_paths:
        value = read(path)
        if value.get("status") != "COMPLETE":
            raise ValueError(f"incomplete R1.6 closeout input: {path}")
        assert_closed(value, path)

    prefix_status = read(prefix_status_path)
    alignment_status = read(alignment_status_path)
    confirmation_status = read(confirmation_status_path)
    h3_status = read(h3_status_path)
    prefix_summary = read(prefix_summary_path)
    tuning = read(tuning_summary_path)
    confirmation = read(confirmation_summary_path)

    prefix_paths = list((root / "prefix_initialisation").glob(
        "*/*/seed_*/result.json"
    ))
    selected_prefix_paths = list((root / "prefix_initialisation").glob(
        f"{prefix_summary['selected_prefix_config']}/*/seed_*/result.json"
    ))
    alignment_result_paths = list((root / "selection_cells").glob(
        "*/*/seed_*/result.json"
    ))
    alignment_failure_paths = list((root / "selection_cells").glob(
        "*/*/seed_*/failure.json"
    ))
    overfit_paths = list((root / "overfit").glob("*/*/seed_*/result.json"))
    selected_overfit_paths = list((root / "overfit").glob(
        f"overfit__prefix__{prefix_summary['selected_prefix_config']}"
        "/*/seed_*/result.json"
    ))
    confirmation_paths = list((root / "confirmation").glob(
        "*/*/*/seed_*/result.json"
    ))
    h3_paths = list((root / "minimal_h3").glob("*/*/*/result.json"))
    synthetic_paths = list((root / "synthetic").glob("*.json"))
    for path in (
        prefix_paths + alignment_result_paths + alignment_failure_paths
        + overfit_paths + confirmation_paths + h3_paths + synthetic_paths
    ):
        assert_closed(read(path), path)

    if (
        prefix_status["completed"] != prefix_status["expected"]
        or prefix_status["expected"] != 108
    ):
        raise ValueError("prefix status count mismatch")
    if (
        alignment_status["completed_selection"]
        != alignment_status["expected_selection"]
    ):
        raise ValueError("alignment status count mismatch")
    if confirmation_status["completed_confirmation"] != 30:
        raise ValueError("confirmation count is not 30")
    if h3_status["completed"] != h3_status["expected"]:
        raise ValueError("minimal H3 count mismatch")
    if len(selected_prefix_paths) != 30:
        raise ValueError("selected-prefix confirmation count is not 30")
    if len(selected_overfit_paths) != 18:
        raise ValueError("selected-config overfit count is not 18")
    if len(tuning["overfit_rows"]) != 18 or not all(
        row["joint_nll_improvement"] > 0 for row in tuning["overfit_rows"]
    ):
        raise ValueError("selected-config overfit did not pass 18/18")

    old_t1_path = (
        contract.RESULT_ROOT / "r1_5/fits/yuquan_zhangjiaqi"
        / "explicit_seed_0/result.json"
    )
    old_h3_path = (
        contract.RESULT_ROOT / "r1_5_h3_long/human/yuquan_zhangjiaqi"
        / "load/seed_0_n_1000/result.json"
    )
    old_t1 = read(old_t1_path)
    old_h3 = read(old_h3_path)
    recommended = {
        "status": "COMPLETE",
        "revision": CLOSEOUT_REVISION,
        "prefix_core": {
            "config_id": prefix_summary["selected_prefix_config"],
            **prefix_status["configs"][prefix_summary["selected_prefix_config"]],
            "selection": "base_train 0-60% -> base_select 60-80%",
        },
        "target_alignment": {
            "config_id": tuning["selected_config"],
            **alignment_status["configs"][tuning["selected_config"]],
            "selection": "fit TRAIN first 80%; select on TRAIN last 20%; refit full TRAIN",
        },
        "early_stopping": {
            "default": "no short global patience",
            "reason": (
                "synthetic positive seeds can first improve near epoch 100; "
                "use frozen module budgets and chronological selection"
            ),
        },
        "h3_policy": {
            "run_only_if_stable_t1_checkpoints_at_least": 3,
            "minimal_scale_events": 1000,
            "sources": ["load", "participation"],
            "note": "H3 learning rate cannot repair an exact-zero downstream derivative",
        },
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    recommended_path = reports / "recommended_optimizer_config.json"
    contract.atomic_json(recommended_path, recommended)

    h3 = h3_summary(h3_paths)
    machine = {
        "status": "COMPLETE",
        "revision": CLOSEOUT_REVISION,
        "r1_6_revision": R1_6_REVISION,
        "scope": (
            "development optimizer/trainability diagnosis; not a cohort H1-H3 "
            "acceptance and not a formal-test result"
        ),
        "counts": {
            "prefix_tuning_cells": prefix_status["completed"],
            "prefix_tuning_expected": prefix_status["expected"],
            "prefix_artifacts_total": len(prefix_paths),
            "selected_prefix_confirmation_cells": len(selected_prefix_paths),
            "alignment_cells_including_expected_failures": alignment_status[
                "completed_selection"
            ],
            "alignment_results": len(alignment_result_paths),
            "alignment_expected_nonfinite_failures": len(alignment_failure_paths),
            "alignment_admissible_configs": int(sum(
                value["admissible"] for value in tuning["by_config"].values()
            )),
            "overfit_selected_config_cells": len(selected_overfit_paths),
            "overfit_artifacts_total": len(overfit_paths),
            "overfit_selected_config_pass_cells": int(sum(
                row["joint_nll_improvement"] > 0
                for row in tuning["overfit_rows"]
            )),
            "confirmation_cells": len(confirmation_paths),
            "confirmation_patients": len(confirmation["subjects"]),
            "confirmation_seeds_per_patient": len(confirmation["seeds"]),
            "minimal_h3_cells": len(h3_paths),
            "synthetic_packages": len(synthetic_paths),
        },
        "selected_config": recommended,
        "synthetic": synthetic_summary(root),
        "confirmation": {
            "stable_t1_subjects_for_minimal_h3": confirmation[
                "stable_t1_subjects_for_minimal_h3"
            ],
            "by_subject": confirmation["by_subject"],
        },
        "minimal_h3": h3,
        "diagnostic_explanations": {
            "epoch_zero": (
                "R1.5 compared alignment updates with a checkpoint already "
                "refit on the selection tail; R1.6 uses disjoint 0-60%, "
                "60-80%, and 80-100% chronological TRAIN partitions"
            ),
            "zero_selected": (
                "gradients and training improvement are nonzero, but the "
                "chronologically later selection block chooses epoch 0"
            ),
            "zero_gradient": (
                "the old epoch-0 T1 has exactly zero state_timing, "
                "state_contact, and state_size weights, so the H3 edge "
                "derivative at zero is exactly zero"
            ),
        },
        "old_structural_example": {
            "t1_result": repo_relative(old_t1_path),
            "t1_result_sha256": contract.sha256_file(old_t1_path),
            "t1_selected_total_epoch": old_t1["fit_trace"]["selected_total_epoch"],
            "t1_selection_gradient_max": old_t1["fit_trace"][
                "selection_gradient_max"
            ],
            "h3_result": repo_relative(old_h3_path),
            "h3_result_sha256": contract.sha256_file(old_h3_path),
            "h3_estimability_class": old_h3["real_estimability_class"],
            "h3_matrix_gradient_at_zero_norm": old_h3["estimability"][
                "real_cumulative"
            ]["matrix_gradient_at_zero_norm"],
            "h3_intercept_gradient_at_zero_norm": old_h3["estimability"][
                "real_cumulative"
            ]["intercept_gradient_at_zero_norm"],
        },
        "manifests": {
            "prefix": digest_manifest(prefix_paths, root),
            "alignment_results": digest_manifest(alignment_result_paths, root),
            "alignment_failures": digest_manifest(alignment_failure_paths, root),
            "overfit": digest_manifest(overfit_paths, root),
            "confirmation": digest_manifest(confirmation_paths, root),
            "minimal_h3": digest_manifest(h3_paths, root),
            "synthetic": digest_manifest(synthetic_paths, root),
        },
        "top_artifact_hashes": {
            str(path.relative_to(root)): contract.sha256_file(path)
            for path in top_paths + [recommended_path]
        },
        "source_hashes": {
            str(path.relative_to(contract.REPO_ROOT)): contract.sha256_file(path)
            for path in [
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/optimizer_audit.py",
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/optimizer_runtime.py",
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/optimizer_synthetic.py",
                contract.REPO_ROOT / "src/topic5_continuous_marked_state_r1/optimizer_h3.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_6_prefix_optimizer_queue.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_6_alignment_optimizer_queue.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_confirmation_queue.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_6_minimal_h3_cell.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/run_r1_6_minimal_h3_queue.py",
                contract.REPO_ROOT / "scripts/topic5_continuous_marked_state_r1/finalize_r1_6_optimizer_identifiability.py",
                contract.SPLIT_MANIFEST,
            ]
        },
        "correction_boundary": {
            "r1_5_epoch0_no_update": (
                "superseded as a scientific negative; retained as evidence of "
                "selection and optimization failure"
            ),
            "r1_5_h3_zero_gradient": (
                "superseded as a biological H3 result; it is a structural-zero "
                "instrument failure"
            ),
            "r1_6_t1": (
                "E384 is development robust support; Chengshuai and Chenziyang "
                "are optimizer-sensitive; E1096, Zhangjiaqi, and Zhangkexuan "
                "are current-model generalization/nonidentifiability results"
            ),
            "r1_6_minimal_h3": (
                "no support in six E384 cells; only two independent validation "
                "units, so this is not a biological absence claim"
            ),
        },
        "development_validation_used_for_configuration_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    machine_path = reports / "optimizer_identifiability_machine_audit.json"
    contract.atomic_json(machine_path, machine)
    print(json.dumps({
        "status": "COMPLETE",
        "machine_audit": str(machine_path),
        "machine_audit_sha256": contract.sha256_file(machine_path),
        "recommended_config": str(recommended_path),
        "recommended_config_sha256": contract.sha256_file(recommended_path),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
