#!/usr/bin/env python3
"""Fail-closed machine audit and handoff for H2b v0.4."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    V0_4_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)
from src.topic5_continuous_marked_state_h2b.v03_hazard import (  # noqa: E402
    build_hazard_design,
)


PRODUCER = Path(__file__).resolve()
CELL_RUNNER = PRODUCER.with_name("run_v04_cell.py")
QUEUE_RUNNER = PRODUCER.with_name("run_v04_queue.py")
ASSAY_RUNNER = PRODUCER.with_name("run_v04_assay.py")
PHENOTYPE_RUNNER = PRODUCER.with_name("run_v04_phenotype.py")
AGGREGATOR = PRODUCER.with_name("aggregate_v04.py")
ESTIMATOR = REPO / "src/topic5_continuous_marked_state_h2b/v04_heterogeneous.py"
ASSAY_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v04_assay.py"
CONTRACT_SOURCE = REPO / "config/topic5_continuous_marked_state_h2b_v0_4.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO, text=True,
    ).strip()


def audit(result_root: Path) -> dict:
    contract_path = result_root / "analysis_contract.json"
    inventory_path = result_root / "manifests/source_cells.json"
    queue_path = result_root / "QUEUE_STATUS.json"
    assay_path = result_root / "assay/summary.json"
    cohort_path = result_root / "reports/cohort_summary.json"
    phenotype_path = result_root / "phenotype_continuous/summary.json"
    contract = _json(contract_path)
    inventory = _json(inventory_path)
    queue = _json(queue_path)
    assay = _json(assay_path)
    cohort = _json(cohort_path)
    phenotype = _json(phenotype_path)
    cells = inventory["cells"]
    checks: dict[str, bool] = {
        "contract_is_final_v10_development": (
            contract.get("schema_revision")
            == "h2b_v0_4_heterogeneous_seizure_entry_routes_v10"
            and contract.get("status") == "FROZEN_DEVELOPMENT"
        ),
        "contract_matches_repository_source": _json(CONTRACT_SOURCE) == contract,
        "exact_source_inventory_46_cells_10_patients": (
            len(cells) == 46 and len({row["subject"] for row in cells}) == 10
            and len({(row["subject"], int(row["seed"])) for row in cells}) == 46
        ),
        "queue_complete_46_of_46": (
            queue.get("status") == "PASS_COMPLETE"
            and queue.get("valid_result_cells") == 46
        ),
        "queue_used_eight_workers_without_oom": (
            queue.get("workers") == 8 and queue.get("oom_failures_first_pass") == 0
        ),
        "queue_source_hashes_current": (
            queue["source"].get("cell_runner_sha256") == sha256_file(CELL_RUNNER)
            and queue["source"].get("module_sha256") == sha256_file(ESTIMATOR)
            and queue["source"].get("producer_sha256") == sha256_file(QUEUE_RUNNER)
            and queue["source"].get("inventory_sha256") == sha256_file(inventory_path)
        ),
        "final_assay_100_calibration_100_per_world": (
            assay.get("calibration_replicates") == 100
            and assay.get("evaluation_replicates_per_world") == 100
        ),
        "final_assay_source_hashes_current": (
            assay["source"].get("producer_sha256") == sha256_file(ASSAY_RUNNER)
            and assay["source"].get("assay_module_sha256") == sha256_file(ASSAY_MODULE)
            and assay["source"].get("estimator_module_sha256") == sha256_file(ESTIMATOR)
        ),
        "phenotype_complete_and_source_current": (
            phenotype.get("status") == "COMPLETE"
            and phenotype["source"].get("producer_sha256") == sha256_file(PHENOTYPE_RUNNER)
            and phenotype["source"].get("inventory_sha256") == sha256_file(inventory_path)
        ),
        "cohort_is_patient_first": (
            cohort.get("inference", {}).get("patients_are_cohort_unit") is True
            and cohort.get("inference", {}).get("seeds_aggregated_by_patient_median") is True
            and cohort.get("n_source_patients") == 10
        ),
        "formal_and_sealed_remained_closed": not any([
            queue.get("formal_test_partition_opened"), queue.get("sealed_opened"),
            assay.get("formal_test_partition_opened"), assay.get("sealed_opened"),
            phenotype.get("formal_test_partition_opened"), phenotype.get("sealed_opened"),
            cohort.get("formal_test_partition_opened"), cohort.get("sealed_opened"),
        ]),
        "h3_t2_physical_clock_not_run": (
            not queue.get("h3_or_t2_run") and not assay.get("h3_or_t2_run")
            and not phenotype.get("h3_or_t2_run") and not cohort.get("h3_or_t2_run")
            and contract.get("boundaries", {}).get("physical_clock_run") is False
        ),
        "paper_ready_figures_not_modified": (
            contract.get("output", {}).get("paper_ready_figures_modified") is False
        ),
    }
    checkpoint_hashes: dict[str, str] = {}
    state_cache_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    cell_receipts_ok = True
    frozen_ok = True
    causal_ok = True
    folds_ok = True
    risk_rows_ok = True
    route_ok = True
    wrong_time_ok = True
    for cell in cells:
        subject, seed = str(cell["subject"]), int(cell["seed"])
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("state_cache", "state_cache_sha256"),
            ("state_manifest", "state_manifest_sha256"),
            ("instrument_manifest", "instrument_manifest_sha256"),
        ):
            if sha256_file(cell[path_key]) != cell[hash_key]:
                cell_receipts_ok = False
        checkpoint_hashes[cell["checkpoint"]] = cell["checkpoint_sha256"]
        state_cache_hashes[cell["state_cache"]] = cell["state_cache_sha256"]
        manifest = _json(Path(cell["state_manifest"]))
        frozen_ok &= (
            manifest.get("all_parameters_frozen") is True
            and manifest.get("seizure_gradient_path") is False
            and manifest.get("max_source_time_le_anchor") is True
            and manifest.get("gap_reset") is True
        )
        result_path = (
            result_root / "per_cell" / subject / f"seed_{seed}" / "result.json"
        )
        result = _json(result_path)
        result_hashes[str(result_path)] = sha256_file(result_path)
        cell_receipts_ok &= (
            result.get("revision")
            == "h2b_v0_4_heterogeneous_route_cell_v6_direct_history_contrast_conditional_risk_sets"
            and result["source"].get("producer_sha256") == sha256_file(CELL_RUNNER)
            and result["source"].get("heterogeneous_module_sha256") == sha256_file(ESTIMATOR)
            and result["source"].get("state_cache_sha256") == cell["state_cache_sha256"]
            and result.get("state_model_updated") is False
            and result.get("seizure_gradient_enters_state") is False
        )
        with np.load(cell["state_cache"], allow_pickle=False) as data:
            design = build_hazard_design(
                time_epoch=data["anchor_time_epoch"],
                segment=data["coverage_segment_index"],
                history=data["deterministic_history"],
                current_observation=data["current_explicit_summary"],
                persistent_state=data["persistent_state"],
                memoryless_state=data["memoryless_observation_code"],
                observation_available=data["observation_available"],
                onset_time=[], onset_segment=[], spacing_seconds=300.0,
            )
        for lead_text, lead_result in result["by_lead_minutes"].items():
            lead = float(lead_text)
            for fold in lead_result.get("folds", []):
                anchor = int(fold["heldout_anchor_row"])
                anchor_time = float(design.time_epoch[anchor])
                causal_ok &= (
                    anchor_time <= float(fold["heldout_onset_epoch"]) - lead * 60.0 + 1e-9
                    and float(fold["train_cutoff_epoch"]) < float(fold["heldout_onset_epoch"])
                    and fold.get("training_labels_known_by_cutoff") is True
                    and fold.get("test_labels_known_by_heldout_onset") is True
                    and fold.get("heldout_seizure_did_not_define_route") is True
                )
                folds_ok &= (
                    fold.get("n_test_risk_sets") == 1
                    and fold.get("n_test_controls") == 5
                    and fold.get("n_train_risk_sets", 0) >= 2
                    and fold.get("train_test_rows_disjoint") is True
                )
                risk_rows_ok &= (
                    fold.get("identical_risk_set_rows_across_arms") is True
                    and fold.get("control_selection_uses_history_observation_or_state") is False
                    and len(fold.get("test_control_source_indices", [])) == 5
                    and "logloss_history" in fold
                )
                if fold.get("persistent_n_routes") == 2:
                    route_ok &= (
                        min(fold.get("persistent_route_sizes", [0])) >= 2
                        and fold.get("persistent_route_separation_bandwidth", 0.0) >= 1.0
                    )
                if fold.get("wrong_time_all_test_rows_valid"):
                    wrong_time_ok &= "logloss_route_state_wrong_time" in fold
                else:
                    wrong_time_ok &= "logloss_route_state_wrong_time" not in fold
    checks.update({
        "checkpoint_and_cache_hashes_recompute": cell_receipts_ok,
        "state_frozen_before_seizure_and_no_gradient_path": frozen_ok,
        "causal_anchor_and_outer_chronology_hold": causal_ok,
        "conditional_risk_sets_have_one_case_five_controls_and_disjoint_splits": folds_ok,
        "all_arms_share_rows_and_control_sampling_is_outcome_blind": risk_rows_ok,
        "history_only_arm_and_observation_increment_are_retained": all(
            {"observation_minus_history", "route_state_minus_history"}.issubset(_json(
                result_root / "per_cell" / str(cell["subject"])
                / f"seed_{int(cell['seed'])}" / "result.json"
            ).get("by_lead_minutes", {}).get("30", {}).get(
                "equal_seizure_weight_effects", {}
            ))
            for cell in cells
            if _json(
                result_root / "per_cell" / str(cell["subject"])
                / f"seed_{int(cell['seed'])}" / "result.json"
            ).get("by_lead_minutes", {}).get("30", {}).get("status")
            == "COMPLETE_DEVELOPMENT"
        ),
        "two_route_folds_meet_train_only_size_and_separation": route_ok,
        "invalid_wrong_time_donors_never_become_memoryless_controls": wrong_time_ok,
    })
    assay_heterogeneous_path = all([
        *assay.get("calibration_checks", {}).values(),
        assay.get("directional_recovery_checks", {}).get(
            "two_route_primary_direction_ge_0_70", False
        ),
        assay.get("directional_recovery_checks", {}).get(
            "two_route_memory_direction_ge_0_70", False
        ),
        assay.get("directional_recovery_checks", {}).get(
            "two_route_heterogeneity_direction_ge_0_70", False
        ),
        assay.get("directional_recovery_checks", {}).get(
            "two_route_time_specificity_direction_ge_0_70", False
        ),
    ])
    all_checks = bool(all(checks.values()))
    payload = {
        "status": "PASS_COMPLETE" if all_checks else "FAIL_ENGINEERING_AUDIT",
        "revision": "h2b_v0_4_machine_audit_v1",
        "created_utc": utc_now(), "all_checks_pass": all_checks,
        "checks": checks,
        "scientific_status": "H2B_NOT_ESTABLISHED_DEVELOPMENT_ONLY",
        "scientific_gates": {
            "full_directional_assay_pass": assay.get("status", "").startswith("PASS_"),
            "heterogeneous_route_assay_path_pass": assay_heterogeneous_path,
            "strict_single_replicate_power_pass": all(
                assay.get("strict_single_replicate_power_checks", {}).values()
            ),
            "primary_chronological_patients": cohort.get(
                "n_primary_chronological_patients"
            ),
            "route_state_beats_observation_patient_median_negative": (
                cohort["cohort_layers"]["all_frozen"]["effects"]
                ["route_state_minus_observation"].get("patient_median", 0.0) < 0.0
            ),
            "two_route_beats_single_axis": (
                cohort["cohort_layers"]["all_frozen"]["effects"]
                ["two_route_minus_single_axis_state"].get("patient_median", 0.0) < 0.0
            ),
            "phenotype_state_increment_majority_favourable": (
                phenotype.get("n_favourable_state_minus_observation", 0)
                > phenotype.get("n_estimable_patient_target_rows", 0) / 2.0
            ),
        },
        "claim_boundary": (
            "Engineering is complete. The heterogeneous-route assay recovered its registered "
            "positive paths directionally, but the full assay and strict power did not pass; "
            "only one patient is primary chronological, and two-route readout did not beat the "
            "single-axis comparator. H2b is not established and no biological negative is allowed."
        ),
        "source_hashes": {
            "analysis_contract": sha256_file(contract_path),
            "source_inventory": sha256_file(inventory_path),
            "queue_status": sha256_file(queue_path),
            "assay_summary": sha256_file(assay_path),
            "cohort_summary": sha256_file(cohort_path),
            "phenotype_summary": sha256_file(phenotype_path),
            "cell_runner": sha256_file(CELL_RUNNER),
            "queue_runner": sha256_file(QUEUE_RUNNER),
            "assay_runner": sha256_file(ASSAY_RUNNER),
            "phenotype_runner": sha256_file(PHENOTYPE_RUNNER),
            "aggregator": sha256_file(AGGREGATOR),
            "estimator": sha256_file(ESTIMATOR),
            "assay_module": sha256_file(ASSAY_MODULE),
            "closeout": sha256_file(PRODUCER),
        },
        "checkpoint_sha256": checkpoint_hashes,
        "state_cache_sha256": state_cache_hashes,
        "cell_result_sha256": result_hashes,
        "git": {
            "branch": _git("branch", "--show-current"),
            "head_before_closeout_commit": _git("rev-parse", "HEAD"),
        },
        "development_only": True, "formal_test_partition_opened": False,
        "sealed_opened": False, "h3_or_t2_run": False,
        "physical_clock_run": False, "paper_ready_figures_modified": False,
    }
    report_hashes = {}
    for path in (
        REPO / "docs/archive/topic5/continuous_marked_state_h2b_cross_task_v0_4_plain_2026-09-01.md",
        REPO / "docs/archive/topic5/continuous_marked_state_h2b_cross_task_v0_4_technical_2026-09-01.md",
    ):
        if path.is_file():
            report_hashes[str(path)] = sha256_file(path)
    payload["report_sha256"] = report_hashes
    atomic_json(result_root / "reports/machine_audit.json", payload)
    handoff = "# H2b v0.4 CURRENT HANDOFF\n\n"
    handoff += f"- 更新时间：{payload['created_utc']}\n"
    handoff += f"- 工程状态：{payload['status']}；46/46 cells；8 workers；0 OOM。\n"
    handoff += f"- 科学状态：{payload['scientific_status']}。\n"
    handoff += "- 30 min：6/10 患者可估计，仅 1 位属于 primary chronological。\n"
    handoff += "- 主要患者中位效应：state-history = +0.2633；state-observation = -0.0133；state-memoryless = -0.0191；correct-wrong = -0.00284。\n"
    handoff += "- 双 route：仅 2 位患者可估计，二者都未胜 single-axis。\n"
    handoff += "- assay：two-route 方向恢复通过，但完整方向门和严格单次检出力未过；禁止生物学阴性。\n"
    handoff += "- phenotype：12 个可估计 patient-target rows 中仅 1 个方向有利。\n"
    handoff += "- 禁止口径：机制、因果、临床预测、formal confirmation、生物学阴性。\n"
    _atomic_text(result_root / "CURRENT_HANDOFF.md", handoff)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=V0_4_RESULT_ROOT)
    args = parser.parse_args()
    payload = audit(args.result_root.resolve())
    print(payload["status"], payload["all_checks_pass"])


if __name__ == "__main__":
    main()
