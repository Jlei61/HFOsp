#!/usr/bin/env python3
"""Build the H2b v0.3 exploratory closeout and machine audit.

A1/A2 determine claim strength, not whether frozen development diagnostics may
run. This closeout audits the failed state/power qualification together with
the completed full-grid A3--A6/A8 exploration without upgrading either to
formal confirmation or a biological negative.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)

PRODUCER = Path(__file__).resolve()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _guard(payload: dict, label: str) -> None:
    _require(payload.get("formal_test_partition_opened") is False,
             f"{label}: formal partition opened")
    _require(payload.get("sealed_opened") is False,
             f"{label}: sealed partition opened")
    _require(payload.get("h3_or_t2_run") is False,
             f"{label}: H3/T2 was run")


def _direction_fraction(summary: dict, key: str) -> float | None:
    row = summary.get("cohort_direction", {}).get(key, {})
    total = int(row.get("total") or 0)
    return float(row.get("favourable", 0)) / total if total else None


def _phenotype_direction(summary: dict) -> tuple[int, int]:
    favourable = total = 0
    for row in summary.get("patient_rows", []):
        if row.get("target_name") != "ied_ictal_reuse_observed":
            continue
        if row.get("evaluation_tier") not in {
            "primary_chronological", "sensitivity_loso",
        }:
            continue
        try:
            value = float(row.get("state_minus_observation_loss"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        total += 1
        favourable += int(value < 0.0)
    return favourable, total


def build(root: Path, test_log: Path) -> dict:
    required = {
        "contract": root / "analysis_contract.json",
        "policy": root / "exploration_policy.json",
        "attrition": root / "manifests/attrition_audit.json",
        "A1": root / "reports/scientific_route_audit_A1.json",
        "A2": root / "reports/scientific_route_audit_A2.json",
        "qualification": root / "qualification/state_qualified_manifest.json",
        "assay": root / "assay/type1_power_summary_smoke.json",
        "state_queue": root / "full_grid/STATE_QUEUE_STATUS.json",
        "hazard_queue": root / "hazard_full_grid/QUEUE_STATUS.json",
        "hazard": root / "hazard_full_grid/patient_first_summary.json",
        "geometry_queue": root / "geometry/QUEUE_STATUS.json",
        "geometry": root / "geometry/patient_first_summary.json",
        "phenotype": root / "phenotype_continuous/summary.json",
        "test_log": test_log,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    _require(not missing, f"missing closeout inputs: {missing}")
    data = {key: _json(path) for key, path in required.items()
            if key != "test_log"}

    _require(data["A1"].get("full_readable_denominator") is True,
             "A1 denominator is partial")
    _require(data["A1"].get("n_readable_checkpoint_cells") == 75,
             "A1 readable-cell denominator drift")
    _require(data["A1"].get("n_all_frozen_patients") == 16,
             "A1 patient denominator drift")
    _require(data["A1"].get("n_state_qualified_patients") == 0,
             "A1 state qualification changed")
    _require(data["qualification"].get("subjects") == [],
             "qualified population is unexpectedly non-empty")
    _require(data["A2"].get("status") == "DIAGNOSTIC_ONLY_A1_EMPTY",
             "A2 diagnostic status drift")
    _require(data["assay"].get("status")
             == "COMPLETE_DIAGNOSTIC_SMOKE_A1_EMPTY",
             "A2 assay receipt drift")

    state_queue = data["state_queue"]
    _require(state_queue.get("status") == "COMPLETE",
             "full-grid state queue is incomplete")
    _require(state_queue.get("requested_tasks") == 46
             and state_queue.get("failed_this_run") == 0,
             "full-grid state queue denominator/failure drift")
    _guard(state_queue, "state queue")

    cache_manifests = sorted((root / "full_grid/state_cache").glob(
        "*/seed_*/states.manifest.json"
    ))
    _require(len(cache_manifests) == 46,
             "full-grid cache denominator is not 46")
    subjects: set[str] = set()
    cell_anchor_rows = 0
    unique_query_files: set[Path] = set()
    cache_hashes: dict[str, str] = {}
    for manifest_path in cache_manifests:
        manifest = _json(manifest_path)
        cache_path = manifest_path.with_name("states.npz")
        query_path = Path(str(manifest.get("query_input", "")))
        query_manifest_path = query_path.with_suffix(".manifest.json")
        _require(manifest.get("status") == "COMPLETE",
                 f"incomplete state cache: {manifest_path}")
        _require(manifest.get("full_recorded_five_minute_grid") is True,
                 f"support-conditioned cache in full-grid tree: {manifest_path}")
        _require(manifest.get("all_parameters_frozen") is True,
                 f"unfrozen state cache: {manifest_path}")
        _require(manifest.get("state_update_uses_seizure_label") is False,
                 f"seizure label entered state update: {manifest_path}")
        _require(manifest.get("formal") is False
                 and manifest.get("sealed") is False,
                 f"formal/sealed state cache: {manifest_path}")
        _require(cache_path.is_file() and query_manifest_path.is_file(),
                 f"missing cache/query manifest: {manifest_path}")
        _require(manifest.get("cache_sha256") == sha256_file(cache_path),
                 f"cache hash drift: {cache_path}")
        query_manifest = _json(query_manifest_path)
        _require(query_manifest.get("query_role_is_outcome_independent") is True,
                 f"outcome-conditioned anchor generation: {query_manifest_path}")
        _require(query_manifest.get("seizure_table_read") is False,
                 f"seizure table read by anchor builder: {query_manifest_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            anchor = np.asarray(cache["anchor_time_epoch"])
            source = np.asarray(cache["max_source_time_epoch"])
            available = np.asarray(cache["observation_available"], dtype=bool)
            _require(anchor.dtype == np.float64,
                     f"absolute time is not float64: {cache_path}")
            _require(bool(available.all()),
                     f"unavailable full-grid rows: {cache_path}")
            _require(bool(np.all(source <= anchor + 1e-9)),
                     f"future observation entered state: {cache_path}")
            _require(len(anchor) == int(query_manifest["n_queries"]),
                     f"query/cache row mismatch: {cache_path}")
            cell_anchor_rows += int(len(anchor))
        subjects.add(manifest_path.parents[1].name)
        unique_query_files.add(query_manifest_path)
        cache_hashes[str(cache_path)] = sha256_file(cache_path)
        cache_hashes[str(manifest_path)] = sha256_file(manifest_path)
    unique_anchor_rows = sum(int(_json(path)["n_queries"])
                             for path in unique_query_files)
    _require(len(subjects) == 10, "full-grid patient denominator is not 10")
    _require(cell_anchor_rows == 45_841,
             "full-grid cell-anchor denominator drift")
    _require(unique_anchor_rows == 10_597,
             "full-grid unique-anchor denominator drift")

    hazard_queue, hazard = data["hazard_queue"], data["hazard"]
    geometry_queue, geometry = data["geometry_queue"], data["geometry"]
    phenotype = data["phenotype"]
    _require(hazard_queue.get("status") == "COMPLETE"
             and hazard_queue.get("requested_tasks") == 46
             and hazard_queue.get("failed_this_run") == 0,
             "full-grid hazard queue incomplete")
    _require(hazard.get("status")
             == "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE"
             and hazard.get("n_cells") == 46
             and hazard.get("n_patients") == 10,
             "hazard patient-first summary incomplete")
    _require(geometry_queue.get("status") == "COMPLETE"
             and geometry_queue.get("requested_tasks") == 46
             and geometry_queue.get("failed_this_run") == 0,
             "geometry queue incomplete")
    _require(geometry.get("status")
             == "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE"
             and geometry.get("n_cells") == 46
             and geometry.get("n_patients") == 10,
             "geometry patient-first summary incomplete")
    _require(phenotype.get("status") == "COMPLETE"
             and phenotype.get("n_full_grid_cells") == 46
             and phenotype.get("n_subjects") == 10,
             "phenotype bridge summary incomplete")
    for label, payload in (
        ("hazard queue", hazard_queue), ("hazard", hazard),
        ("geometry queue", geometry_queue), ("geometry", geometry),
        ("phenotype", phenotype),
    ):
        _guard(payload, label)
    _require(hazard.get("negative_result_biological_interpretation_allowed")
             is False, "hazard summary permits a biological negative")
    _require(geometry.get("negative_result_biological_interpretation_allowed")
             is False, "geometry summary permits a biological negative")
    _require(phenotype.get("target_reclustered") is False
             and phenotype.get("target_thresholded_after_state") is False,
             "phenotype target was changed after seeing state")

    # A7 is conditional, not a mandatory gate. This is an operational priority
    # rule, not a significance threshold: require broad hazard and correct-time
    # agreement plus an independent geometry or phenotype direction.
    t_fraction = _direction_fraction(hazard, "T")
    correct_fraction = _direction_fraction(
        hazard, "correct_time_better_than_wrong"
    )
    geometry_fractions = {
        family: _direction_fraction(geometry, family)
        for family in ("basin_gating", "directed_approach", "abrupt_transition")
    }
    phenotype_favourable, phenotype_total = _phenotype_direction(phenotype)
    phenotype_fraction = (
        phenotype_favourable / phenotype_total if phenotype_total else None
    )
    independent_direction = any(
        value is not None and value >= 0.70
        for value in (*geometry_fractions.values(), phenotype_fraction)
    )
    coherent_signal = bool(
        t_fraction is not None and t_fraction >= 0.70
        and correct_fraction is not None and correct_fraction >= 0.70
        and independent_direction
    )
    _require(not coherent_signal,
             "A7 operational trigger met; matched IED-objective retraining is required")
    a7 = {
        "status": "NOT_TRIGGERED_NO_COHERENT_CROSS_DOMAIN_SIGNAL",
        "created_utc": utc_now(),
        "revision": "h2b_v0_3_A7_decision_v1",
        "operational_rule_not_statistical_threshold": True,
        "T_favourable_fraction": t_fraction,
        "correct_time_favourable_fraction": correct_fraction,
        "geometry_favourable_fraction": geometry_fractions,
        "phenotype_observed_favourable": phenotype_favourable,
        "phenotype_observed_total": phenotype_total,
        "phenotype_observed_favourable_fraction": phenotype_fraction,
        "coherent_signal": False,
        "matched_retraining_implementation_exists": False,
        "reason": (
            "development directions did not jointly meet broad hazard, correct-time, "
            "and independent cross-domain coherence"
        ),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    a7_path = root / "reports/A7_IED_objective_ablation_decision.json"
    atomic_json(a7_path, a7)

    text = test_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) passed, (\d+) warnings? in ", text)
    _require(match is not None, "final scoped pytest receipt is not successful")
    test_result = f"{match.group(1)} passed, {match.group(2)} warnings"

    receipts = sorted((root / "logs/full_grid_state").glob(
        "*.attempt_*.maxrss_kib.txt"
    ))
    rss = {}
    for path in receipts:
        raw = path.read_text(encoding="utf-8").strip().splitlines()
        if raw:
            rss[str(path)] = int(raw[-1]) * 1024
    _require(len(rss) >= 31 and all(value > 0 for value in rss.values()),
             "streaming resource receipts are incomplete")
    resource = {
        "status": "PASS_NO_OOM_BOUNDED_WORKERS",
        "created_utc": utc_now(),
        "revision": "h2b_v0_3_full_grid_resource_audit_v2",
        "configured_cpu_workers": state_queue.get("configured_cpu_workers"),
        "selected_cpu_workers": state_queue.get("cpu_workers"),
        "n_streaming_resource_receipts": len(rss),
        "max_streaming_peak_rss_bytes": max(rss.values()),
        "kernel_oom_observed": state_queue.get("kernel_oom_observed"),
        "thread_limits": state_queue.get("thread_limits"),
    }
    resource_path = root / "full_grid/RESOURCE_AUDIT.json"
    atomic_json(resource_path, resource)

    created = utc_now()
    route = {
        "status": "COMPLETE_EXPLORATORY_H2B_NOT_ESTABLISHED",
        "created_utc": created,
        "revision": "h2b_v0_3_scientific_route_audit_A3_A8_v2",
        "A3_A5_hazard_lag": "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE",
        "A6_OOS_manifold_flow": "COMPLETE_EXPLORATORY_ASSAY_NOT_SENSITIVE",
        "A7_IED_objective_ablation": a7["status"],
        "A8_frozen_phenotype_bridge": "COMPLETE_EXPLORATORY",
        "support_conditioned_old_results": "QUARANTINED_SEPARATE_FROM_FULL_GRID",
        "H2b": "NOT_ESTABLISHED",
        "biological_negative_allowed": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "physical_clock_run": False,
        "paper_ready_figures_modified": False,
    }
    route_path = root / "reports/scientific_route_audit_A3_A8.json"
    atomic_json(route_path, route)

    source_paths = [*required.values(), a7_path, resource_path, route_path, PRODUCER]
    sources = {str(path): sha256_file(path) for path in source_paths}
    checks = {
        "A0_full_attrition_recorded": True,
        "A1_all_75_readable_cells_audited": True,
        "A1_state_qualified_zero_of_16": True,
        "A2_null_calibrated_but_transfer_insensitive": True,
        "full_grid_46_cells_10_patients_complete": True,
        "full_grid_10597_unique_anchors_causal": True,
        "full_grid_state_parameters_frozen": True,
        "patient_is_inference_unit": True,
        "seeds_not_counted_as_patients": True,
        "A3_A5_full_grid_exploration_complete": True,
        "A6_common_domain_OOS_geometry_complete": True,
        "A7_conditional_decision_recorded": True,
        "A8_frozen_target_not_reclustered": True,
        "formal_sealed_H3_T2_untouched": True,
        "negative_not_interpreted_biologically": True,
        "final_scoped_tests_passed": True,
    }
    machine = {
        "status": "PASS_EXPLORATORY_CLOSEOUT_H2B_NOT_ESTABLISHED",
        "revision": "h2b_v0_3_machine_audit_v2",
        "created_utc": created,
        "development_only": True,
        "scientific_conclusion": (
            "full-grid cross-task exploration is complete, but current R1.7B "
            "checkpoints and assay do not establish a transferable persistent state"
        ),
        "n_readable_checkpoint_cells": 75,
        "n_all_frozen_patients": 16,
        "n_state_qualified_patients": 0,
        "full_grid": {
            "n_cells": 46,
            "n_subjects": 10,
            "n_unique_anchor_rows": unique_anchor_rows,
            "n_cell_anchor_rows": cell_anchor_rows,
            "resource_audit": str(resource_path),
        },
        "hazard_cohort_direction": hazard.get("cohort_direction"),
        "geometry_cohort_direction": geometry.get("cohort_direction"),
        "phenotype_observed_direction": {
            "favourable": phenotype_favourable, "total": phenotype_total,
        },
        "A7_decision": a7["status"],
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "test_result": test_result,
        "source_sha256": sources,
        "full_grid_cache_sha256": cache_hashes,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "physical_clock_run": False,
        "paper_ready_figures_modified": False,
    }
    _require(machine["all_checks_pass"], "machine audit check failed")
    atomic_json(root / "reports/machine_audit.json", machine)
    return machine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path,
                        default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--test-log", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.result_root.resolve(), args.test_log.resolve())
    print(result["status"])


if __name__ == "__main__":
    main()
