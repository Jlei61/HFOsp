#!/usr/bin/env python3
"""Fail-closed audit of the active continuous-marked-state result package."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.exposure import EXPOSURE_REVISION
from src.topic5_continuous_marked_state.regular_t1 import REGULAR_T1_REVISION


def load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def source_hash() -> tuple[str, list[str]]:
    files = sorted(
        list((contract.REPO_ROOT / "src/topic5_continuous_marked_state").glob("*.py"))
        + list((contract.REPO_ROOT / "scripts/topic5_continuous_marked_state").glob("*.py"))
        + list((contract.REPO_ROOT / "tests/topic5_continuous_marked_state").glob("*.py"))
        + [
            contract.REPO_ROOT / "docs/archive/topic5/continuous_marked_state_scientific_spec_2026-08-24.md",
            contract.REPO_ROOT / "docs/archive/topic5/continuous_marked_state_execution_plan_2026-08-24.md",
        ]
    )
    digest = hashlib.sha256()
    relative = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        token = str(path.relative_to(contract.REPO_ROOT))
        relative.append(token)
        digest.update(token.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative


def main() -> None:
    root = contract.RESULT_ROOT
    checks: dict[str, dict] = {}

    bridge = load(root / "bridge/BRIDGE_E0_SUMMARY.json")
    checks["bridge"] = {
        "pass": (
            bridge.get("fit_revision") == contract.FIT_REVISION
            and bridge.get("n_subjects_complete") == len(contract.PILOT_SUBJECTS)
            and bridge.get("n_runs") == 4 * len(contract.PILOT_SUBJECTS)
            and bridge.get("sealed_opened") is False
        ),
        "fit_revision": bridge.get("fit_revision"),
        "n_subjects_complete": bridge.get("n_subjects_complete"),
        "n_runs": bridge.get("n_runs"),
    }

    observation_rows = []
    for variant, folder in (("spectral", "features"), ("raw", "features_raw")):
        for subject in contract.PILOT_SUBJECTS:
            row = load(root / f"regular_observation/{folder}/{subject}.manifest.json")
            observation_rows.append({
                "variant": variant, "subject": subject,
                "contract": row.get("contract"),
                "feature_kind": row.get("feature_kind", "spectral"),
                "sealed_opened": row.get("sealed_opened"),
            })
    checks["regular_observations"] = {
        "pass": all(
            row["contract"] == contract.REVISION
            and row["feature_kind"] == row["variant"]
            and row["sealed_opened"] is False
            for row in observation_rows
        ),
        "n_manifests": len(observation_rows),
    }

    t1_rows = []
    for variant, folder in (("spectral", "regular_t1"), ("raw", "regular_t1/raw_e0")):
        summary = load(root / folder / "REGULAR_T1_SUMMARY.json")
        runs = []
        for path in sorted((root / folder / "runs").glob("*.json")):
            row = load(path)
            if row.get("regular_t1_revision") == REGULAR_T1_REVISION:
                runs.append(row)
        t1_rows.append({
            "variant": variant,
            "summary_revision": summary.get("regular_t1_revision"),
            "summary_n_subjects": summary.get("n_subjects"),
            "summary_n_runs": summary.get("n_runs"),
            "summary_n_paired": summary.get("n_paired"),
            "seed_pairs_per_subject": sorted(
                row.get("n_seed_pairs") for row in summary.get("per_subject", [])
            ),
            "run_state_dims": sorted({row.get("state_dim") for row in runs}),
            "sealed_values": sorted({row.get("sealed_opened") for row in runs}),
        })
    checks["regular_t1"] = {
        "pass": all(
            row["summary_revision"] == REGULAR_T1_REVISION
            and row["summary_n_subjects"] == len(contract.PILOT_SUBJECTS)
            and row["summary_n_runs"] == 36
            and row["summary_n_paired"] == 18
            and row["seed_pairs_per_subject"] == [3] * len(contract.PILOT_SUBJECTS)
            and row["run_state_dims"] == [8]
            and row["sealed_values"] == [False]
            for row in t1_rows
        ),
        "variants": t1_rows,
    }

    synthetic = load(root / "state_smoke/REGULAR_T1_SYNTHETIC_RECOVERY.json")
    recovery = synthetic.get("recovery_summary", {})
    checks["regular_t1_synthetic_recovery"] = {
        "pass": (
            synthetic.get("regular_t1_revision") == REGULAR_T1_REVISION
            and synthetic.get("state_dim") == 8
            and recovery.get("n_filtered_joint_better") == 3
            and recovery.get("n_swap_correct_better") == 3
            and recovery.get("post_anchor_n_better") == {
                "5": 3, "10": 3, "20": 3,
            }
            and synthetic.get("sealed_opened") is False
        ),
        "regular_t1_revision": synthetic.get("regular_t1_revision"),
        "state_dim": synthetic.get("state_dim"),
        "recovery_summary": recovery,
    }

    exposure = load(root / "exposure_screen/EXPOSURE_SCREEN_SUMMARY.json")
    checks["exposure_core"] = {
        "pass": (
            exposure.get("fit_revision") == contract.FIT_REVISION
            and exposure.get("exposure_revision") == EXPOSURE_REVISION
            and exposure.get("core_grid_complete") is True
            and exposure.get("n_core_runs") == exposure.get("expected_n_core_runs") == 272
            and exposure.get("n_posthoc_refinement_runs") == 272
            and exposure.get("n_fast_control_runs") == 204
            and exposure.get("n_runs") == 748
            and exposure.get("sealed_opened") is False
        ),
        "fit_revision": exposure.get("fit_revision"),
        "n_core_runs": exposure.get("n_core_runs"),
        "expected_n_core_runs": exposure.get("expected_n_core_runs"),
        "core_grid_complete": exposure.get("core_grid_complete"),
        "n_posthoc_refinement_runs": exposure.get("n_posthoc_refinement_runs"),
        "n_fast_control_runs": exposure.get("n_fast_control_runs"),
    }

    cumulative = load(
        root / "exposure_screen/CUMULATIVE_EXPOSURE_IDENTIFIABILITY.json"
    )
    checks["cumulative_exposure_identifiability"] = {
        "pass": (
            cumulative.get("analysis_revision")
            == "current_event_vs_cumulative_patient_paired_v1"
            and cumulative.get("exposure_revision") == EXPOSURE_REVISION
            and cumulative.get("n_source_runs") == 748
            and cumulative.get("n_patients") == 34
            and len(cumulative.get("cells", [])) == 20
            and cumulative.get("current_event_limit_tau_minutes") == 1e-6
            and cumulative.get("sealed_opened") is False
        ),
        "analysis_revision": cumulative.get("analysis_revision"),
        "n_source_runs": cumulative.get("n_source_runs"),
        "n_patients": cumulative.get("n_patients"),
        "n_cells": len(cumulative.get("cells", [])),
    }

    heterogeneity = load(root / "exposure_screen/TIME_SCALE_HETEROGENEITY.json")
    checks["exposure_heterogeneity"] = {
        "pass": (
            heterogeneity.get("heterogeneity_revision")
            == "h3_s0_patient_time_scale_heterogeneity_v1"
            and heterogeneity.get("exposure_revision") == EXPOSURE_REVISION
            and heterogeneity.get("n_runs") == 544
            and heterogeneity.get("n_source_runs") == 748
            and heterogeneity.get("n_patients") == 34
            and heterogeneity.get("sealed_opened") is False
        ),
        "heterogeneity_revision": heterogeneity.get("heterogeneity_revision"),
        "n_runs": heterogeneity.get("n_runs"),
        "n_patients": heterogeneity.get("n_patients"),
    }

    alignment = load(root / "exposure_screen/H2A_H3_ALIGNMENT.json")
    checks["h2a_h3_alignment"] = {
        "pass": (
            alignment.get("alignment_revision")
            == "h2a_graph_vs_h3_exposure_patient_alignment_v1"
            and alignment.get("h3_exposure_revision") == EXPOSURE_REVISION
            and alignment.get("h2a_graph_package_hash")
            == "8fd11957dceec1c2a81b4b87ca9687fa5d8ab93557f5bc20715e4b4f38048087"
            and alignment.get("n_cells") == 60
            and alignment.get("sealed_opened") is False
        ),
        "alignment_revision": alignment.get("alignment_revision"),
        "n_cells": alignment.get("n_cells"),
        "h2a_graph_package_hash": alignment.get("h2a_graph_package_hash"),
    }

    state16 = load(
        root / "regular_t1/sensitivities/state16/STATE16_SENSITIVITY_SUMMARY.json"
    )
    state16_rows = state16.get("rows", [])
    checks["regular_t1_state16_sensitivity"] = {
        "pass": (
            state16.get("sensitivity_revision")
            == "state16_capacity_sensitivity_v1_on_regular_t1_v10"
            and state16.get("summary_revision")
            == "state16_capacity_patient_unit_three_seed_v2"
            and state16.get("regular_t1_revision") == REGULAR_T1_REVISION
            and state16.get("n_runs") == 36
            and len(state16_rows) == 36
            and state16.get("state_dim") == 16
            and state16.get("seeds") == [0, 1, 2]
            and sorted({row.get("observation_variant") for row in state16_rows})
            == ["raw", "spectral"]
            and {row.get("subject") for row in state16_rows}
            == set(contract.PILOT_SUBJECTS)
            and {row.get("seed") for row in state16_rows} == {0, 1, 2}
            and state16.get("sealed_opened") is False
        ),
        "sensitivity_revision": state16.get("sensitivity_revision"),
        "summary_revision": state16.get("summary_revision"),
        "n_runs": state16.get("n_runs"),
        "state_dim": state16.get("state_dim"),
        "seeds": state16.get("seeds"),
    }

    clock = load(
        root / "exposure_clock_control/PHYSICAL_VS_EVENT_COUNT_CLOCK.json"
    )
    pairing = clock.get("pairing_audit", {})
    checks["physical_vs_event_count_clock"] = {
        "pass": (
            clock.get("analysis_revision")
            == "physical_time_vs_train_median_iei_event_clock_v1"
            and clock.get("n_physical_source_runs") == 408
            and clock.get("n_event_count_source_runs") == 408
            and clock.get("n_current_event_source_runs") == 68
            and len(clock.get("cells", [])) == 12
            and pairing.get("n_cells") == 408
            and pairing.get("all_sample_counts_exact") is True
            and pairing.get("all_history_baselines_exact") is True
            and pairing.get("all_current_event_sample_counts_exact") is True
            and pairing.get("all_current_event_history_baselines_exact") is True
            and pairing.get("maximum_history_endpoint_difference") == 0.0
            and pairing.get("maximum_current_event_history_endpoint_difference") == 0.0
            and pairing.get("all_sealed_partitions_closed") is True
            and clock.get("sealed_opened") is False
        ),
        "analysis_revision": clock.get("analysis_revision"),
        "n_physical_source_runs": clock.get("n_physical_source_runs"),
        "n_event_count_source_runs": clock.get("n_event_count_source_runs"),
        "n_cells": len(clock.get("cells", [])),
        "pairing_audit_n_cells": pairing.get("n_cells"),
    }

    clock_synthetic = load(
        root / "exposure_clock_control/CLOCK_IDENTIFIABILITY_SYNTHETIC.json"
    )
    checks["clock_synthetic_recovery"] = {
        "pass": (
            clock_synthetic.get("analysis_revision")
            == "real_timeline_clock_truth_recovery_v1"
            and clock_synthetic.get("n_patients") == 34
            and len(clock_synthetic.get("aggregate", [])) == 8
            and clock_synthetic.get("sealed_opened") is False
        ),
        "analysis_revision": clock_synthetic.get("analysis_revision"),
        "n_patients": clock_synthetic.get("n_patients"),
        "n_rows": len(clock_synthetic.get("aggregate", [])),
    }

    clock_separability = load(
        root / "exposure_clock_control/CLOCK_SEPARABILITY_STRATA.json"
    )
    checks["clock_separability_sensitivity"] = {
        "pass": (
            clock_separability.get("analysis_revision")
            == "clock_separability_patient_strata_v1"
            and clock_separability.get("n_more_separable") == 17
            and clock_separability.get("n_less_separable") == 17
            and len(clock_separability.get("human_rows", [])) == 48
            and clock_separability.get("sealed_opened") is False
        ),
        "analysis_revision": clock_separability.get("analysis_revision"),
        "n_more_separable": clock_separability.get("n_more_separable"),
        "n_less_separable": clock_separability.get("n_less_separable"),
        "n_rows": len(clock_separability.get("human_rows", [])),
    }

    fixed_count = load(
        root / "exposure_event_count_grid/FIXED_MEMORY_CLOCK_GRID_SUMMARY.json"
    )
    count_status = load(root / "EVENT_COUNT_GRID_STATUS.json")
    physical_status = load(root / "FIXED_MEMORY_CLOCK_GRID_STATUS_physical.json")
    chain_status = load(root / "FIXED_MEMORY_CLOCK_CHAIN_STATUS.json")
    checks["fixed_grid_execution_status"] = {
        "pass": (
            count_status.get("stage") == "COMPLETE"
            and count_status.get("n_completed") == count_status.get("n_jobs") == 340
            and count_status.get("failures") == []
            and count_status.get("sealed_opened") is False
            and physical_status.get("stage") == "COMPLETE"
            and physical_status.get("n_completed")
            == physical_status.get("n_jobs") == 340
            and physical_status.get("failures") == []
            and physical_status.get("sealed_opened") is False
            and chain_status.get("stage") == "COMPLETE"
            and chain_status.get("sealed_opened") is False
        ),
        "count_stage": count_status.get("stage"),
        "count_completed": count_status.get("n_completed"),
        "physical_stage": physical_status.get("stage"),
        "physical_completed": physical_status.get("n_completed"),
        "chain_stage": chain_status.get("stage"),
    }
    fixed_pairing = fixed_count.get("pairing_audit", {})
    fixed_producer = fixed_count.get("producer_package_audit", {})
    fixed_parity = fixed_count.get("superseded_rerun_parity", {})
    checks["fixed_event_count_grid"] = {
        "pass": (
            fixed_count.get("analysis_revision")
            == "fixed_event_count_with_rate_matched_physical_v2"
            and fixed_count.get("n_event_count_source_runs") == 340
            and fixed_count.get("n_physical_source_runs") == 340
            and fixed_count.get("n_patients") == 34
            and fixed_count.get("memories_events")
            == [25.0, 50.0, 100.0, 200.0, 400.0]
            and len(fixed_count.get("cells", [])) == 10
            and fixed_pairing.get("n_cells") == 340
            and fixed_pairing.get("all_sample_counts_exact") is True
            and fixed_pairing.get("all_history_baselines_exact") is True
            and fixed_pairing.get("maximum_history_endpoint_difference") == 0.0
            and fixed_pairing.get("all_sealed_partitions_closed") is True
            and fixed_producer.get("all_cells_postdate_active_producer") is True
            and len(fixed_producer.get("producer_files", [])) == 5
            and len(fixed_producer.get("producer_source_sha256", "")) == 64
            and fixed_parity.get("n_archived_cells") == 219
            and fixed_parity.get("n_exact_reruns") == 219
            and fixed_parity.get("all_json_fields_exact") is True
            and fixed_count.get("sealed_opened") is False
        ),
        "analysis_revision": fixed_count.get("analysis_revision"),
        "n_event_count_source_runs": fixed_count.get("n_event_count_source_runs"),
        "n_physical_source_runs": fixed_count.get("n_physical_source_runs"),
        "n_cells": len(fixed_count.get("cells", [])),
        "pairing_audit_n_cells": fixed_pairing.get("n_cells"),
        "producer_source_sha256": fixed_producer.get("producer_source_sha256"),
        "n_superseded_exact_reruns": fixed_parity.get("n_exact_reruns"),
    }

    evidence = load(root / "manifests/HYPOTHESIS_EVIDENCE_CARD.json")
    checks["hypothesis_evidence_card"] = {
        "pass": (
            evidence.get("contract")
            == "continuous_marked_state_h1_h3_evidence_ledger_v1"
            and set(evidence.get("hypotheses", {}))
            == {"H1", "H2a", "H2b", "H3a", "H3b"}
            and evidence.get("sealed_formal_partition_opened") is False
        ),
        "contract": evidence.get("contract"),
        "hypotheses": sorted(evidence.get("hypotheses", {})),
    }

    package_sha256, source_files = source_hash()
    output = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "checks": checks,
        "all_pass": all(row["pass"] for row in checks.values()),
        "source_package_sha256": package_sha256,
        "source_files": source_files,
        "sealed_opened": False,
    }
    path = root / "manifests/FINAL_PACKAGE_AUDIT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(temporary, path)
    print(json.dumps({
        "all_pass": output["all_pass"],
        "source_package_sha256": package_sha256,
        "path": str(path),
    }, sort_keys=True))
    if not output["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
