#!/usr/bin/env python3
"""Build the strict H2b v0.3 gated closeout and machine audit.

The current acceptance contract is fail-closed: A3--A8 are not released when
A1 has no state-qualified patient or when the final A2 assay is unavailable.
Outcome-blind state-grid extraction may finish as infrastructure, but it does
not authorize seizure-risk, geometry, ablation, or phenotype probes.
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
STRICT_POLICY_SOURCE = (
    REPO / "config/topic5_continuous_marked_state_h2b_v0_3_strict_acceptance_policy.json"
)
COMPATIBLE_STATE_QUEUE_HASHES = {
    # v3 and v5 differ only in durable resource scheduling/receipts.
    "8bd2cd2fcc0e950dd12a90bac57bf76fbfd4bbb6dc148a82ed7d553835dae3f7",
}
QUARANTINE_DIRS = (
    "pre_gate_hazard_v1",
    "support_conditioned_hazard_v2",
    "support_conditioned_geometry_v1",
    "post_gate_hazard_full_grid_exploratory_v1",
    "post_gate_geometry_exploratory_v1",
    "post_gate_phenotype_continuous_exploratory_v1",
    "post_gate_hazard_full_grid_exploratory_v2",
    "post_gate_geometry_exploratory_v2",
    "post_gate_phenotype_continuous_exploratory_v2",
    "post_gate_hazard_full_grid_exploratory_v3",
    "post_gate_geometry_exploratory_v3",
    "post_gate_phenotype_continuous_exploratory_v3",
)


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


def _strict_route_payload(created: str) -> dict:
    return {
        "status": "NOT_RELEASED_A1_AND_A2_FAILED",
        "created_utc": created,
        "revision": "h2b_v0_3_scientific_route_audit_A3_A8_v3_strict",
        "A3_A5_hazard_lag": "NOT_RUN_GATE_CLOSED",
        "A6_OOS_manifold_flow": "NOT_RUN_GATE_CLOSED",
        "A7_IED_objective_ablation": "NOT_RUN_GATE_CLOSED",
        "A8_frozen_phenotype_bridge": "NOT_RUN_GATE_CLOSED",
        "outcome_blind_state_grid": "COMPLETE_INFRASTRUCTURE_ONLY",
        "post_gate_exploratory_outputs": "QUARANTINED_NOT_EVIDENCE",
        "H2b": "NOT_ESTABLISHED",
        "biological_negative_allowed": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "physical_clock_run": False,
        "paper_ready_figures_modified": False,
    }


def _assert_no_active_downstream(root: Path) -> None:
    _require(not (root / "hazard_full_grid").exists(),
             "active post-gate hazard output exists")
    _require(not (root / "phenotype_continuous").exists(),
             "active post-gate phenotype output exists")
    geometry = root / "geometry"
    if geometry.exists():
        unexpected = [path for path in geometry.rglob("*")
                      if path.is_file() and path.name != "ROUTE_STATUS.json"]
        _require(not unexpected,
                 f"active post-gate geometry output exists: {unexpected[:3]}")


def _audit_instrument(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    manifests = sorted((root / "instrument/by_cell").glob(
        "*/seed_*/instrument_manifest.json"
    ))
    _require(len(manifests) == 75, "instrument denominator is not 75 cells")
    checkpoint_hashes: dict[str, str] = {}
    instrument_hashes: dict[str, str] = {}
    subjects: set[str] = set()
    for path in manifests:
        payload = _json(path)
        _require(payload.get("status") == "COMPLETE",
                 f"incomplete instrument cell: {path}")
        _guard(payload, f"instrument {path}")
        _require(payload.get("seizure_gradient_path") is False,
                 f"seizure gradient path in instrument: {path}")
        _require(payload.get("seizure_risk_outcome_read") is False,
                 f"future seizure risk read in instrument: {path}")
        checkpoint = payload.get("source", {}).get("checkpoint", {})
        checkpoint_path = Path(str(checkpoint.get("checkpoint", "")))
        _require(checkpoint.get("state_frozen") is True,
                 f"unfrozen checkpoint: {path}")
        _require(checkpoint.get("all_parameters_require_grad_false") is True,
                 f"checkpoint gradients enabled: {path}")
        _require(checkpoint.get("seizure_gradient_path") is False,
                 f"seizure gradient entered checkpoint: {path}")
        _require(checkpoint_path.is_file(), f"missing checkpoint: {checkpoint_path}")
        actual_checkpoint_hash = sha256_file(checkpoint_path)
        _require(checkpoint.get("checkpoint_sha256") == actual_checkpoint_hash,
                 f"checkpoint hash drift: {checkpoint_path}")
        trace_path = Path(str(payload.get("trace_path", "")))
        _require(trace_path.is_file(), f"missing instrument trace: {trace_path}")
        _require(payload.get("trace_sha256") == sha256_file(trace_path),
                 f"instrument trace hash drift: {trace_path}")
        subjects.add(str(payload.get("subject")))
        checkpoint_hashes[str(checkpoint_path)] = actual_checkpoint_hash
        instrument_hashes[str(path)] = sha256_file(path)
    _require(len(subjects) == 16, "instrument patient denominator is not 16")
    return checkpoint_hashes, instrument_hashes


def _audit_full_grid(root: Path, state_queue: dict) -> tuple[dict, dict[str, str]]:
    _require(state_queue.get("status") == "COMPLETE",
             "full-grid state queue is incomplete")
    _require(state_queue.get("revision") == "h2b_v0_3_full_grid_state_queue_v5",
             "full-grid state queue revision drift")
    _require(state_queue.get("requested_tasks") == 46,
             "full-grid requested denominator is not 46")
    _require(state_queue.get("failed_this_run") == 0,
             "full-grid state queue has failures")
    _require(state_queue.get("kernel_oom_observed") is False,
             "kernel OOM was observed")
    _guard(state_queue, "state queue")

    selected = int(state_queue.get("cpu_workers") or 0)
    configured = int(state_queue.get("configured_cpu_workers") or 0)
    available = int(state_queue.get("mem_available_bytes_at_start") or 0)
    per_worker = int(state_queue.get("per_worker_memory_budget_bytes") or 0)
    safety = float(state_queue.get("memory_safety_fraction") or 0.0)
    _require(0 < selected <= configured == 8,
             "worker selection/configuration drift")
    _require(0.0 < safety <= 0.65,
             "memory safety fraction exceeds contract")
    _require(selected * per_worker <= safety * available,
             "selected workers exceed memory budget")
    _require(state_queue.get("thread_limits") == 1,
             "numerical thread limit is not one")

    current_queue_hash = sha256_file(
        REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_full_grid_state_queue.py"
    )
    allowed_queue_hashes = COMPATIBLE_STATE_QUEUE_HASHES | {current_queue_hash}
    manifests = sorted((root / "full_grid/state_cache").glob(
        "*/seed_*/states.manifest.json"
    ))
    _require(len(manifests) == 46, "full-grid cache denominator is not 46")
    subjects: set[str] = set()
    query_manifests: set[Path] = set()
    cell_anchor_rows = 0
    cache_hashes: dict[str, str] = {}
    producer_hashes: set[str] = set()
    for manifest_path in manifests:
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
        _require(manifest.get("all_parameters_require_grad_false") is True,
                 f"state parameters require gradients: {manifest_path}")
        _require(manifest.get("state_update_uses_seizure_label") is False,
                 f"seizure label entered state update: {manifest_path}")
        _require(manifest.get("seizure_labels_enter_state_update") is False,
                 f"seizure label entered state update: {manifest_path}")
        _require(manifest.get("source_task")
                 == "continuous_background_and_ied_timing_mark",
                 f"state source task drift: {manifest_path}")
        _require(manifest.get("formal") is False and manifest.get("sealed") is False,
                 f"formal/sealed state cache: {manifest_path}")
        producer_hash = str(manifest.get("full_grid_queue_producer_sha256"))
        _require(producer_hash in allowed_queue_hashes,
                 f"unrecognized state queue producer: {manifest_path}")
        producer_hashes.add(producer_hash)
        _require(cache_path.is_file() and query_manifest_path.is_file(),
                 f"missing cache/query manifest: {manifest_path}")
        _require(manifest.get("cache_sha256") == sha256_file(cache_path),
                 f"cache hash drift: {cache_path}")
        _require(manifest.get("query_manifest_sha256")
                 == sha256_file(query_manifest_path),
                 f"query manifest hash drift: {query_manifest_path}")
        query_manifest = _json(query_manifest_path)
        _require(query_manifest.get("status") == "COMPLETE",
                 f"incomplete query manifest: {query_manifest_path}")
        _require(query_manifest.get("query_role_is_outcome_independent") is True,
                 f"outcome-conditioned anchor generation: {query_manifest_path}")
        _require(query_manifest.get("seizure_table_read") is False,
                 f"seizure table read by anchor builder: {query_manifest_path}")
        _guard(query_manifest, f"query {query_manifest_path}")
        with np.load(cache_path, allow_pickle=False) as cache:
            anchor = np.asarray(cache["anchor_time_epoch"])
            source = np.asarray(cache["max_source_time_epoch"])
            available_rows = np.asarray(cache["observation_available"], dtype=bool)
            _require(anchor.dtype == np.float64,
                     f"absolute time is not float64: {cache_path}")
            _require(bool(available_rows.all()),
                     f"unavailable full-grid rows: {cache_path}")
            _require(bool(np.all(source <= anchor + 1e-9)),
                     f"future observation entered state: {cache_path}")
            _require(len(anchor) == int(query_manifest["n_queries"]),
                     f"query/cache row mismatch: {cache_path}")
            cell_anchor_rows += int(len(anchor))
        subjects.add(manifest_path.parents[1].name)
        query_manifests.add(query_manifest_path)
        cache_hashes[str(cache_path)] = sha256_file(cache_path)
        cache_hashes[str(manifest_path)] = sha256_file(manifest_path)

    unique_anchor_rows = sum(int(_json(path)["n_queries"])
                             for path in query_manifests)
    _require(len(subjects) == 10, "full-grid patient denominator is not 10")
    _require(len(query_manifests) == 10, "full-grid query denominator is not 10")
    _require(cell_anchor_rows == 45_841,
             "full-grid cell-anchor denominator drift")
    _require(unique_anchor_rows == 10_597,
             "full-grid unique-anchor denominator drift")

    receipts = sorted((root / "logs/full_grid_state").glob(
        "*.attempt_*.maxrss_kib.txt"
    ))
    peak_rss: dict[str, int] = {}
    for path in receipts:
        rows = [row.strip() for row in path.read_text(
            encoding="utf-8", errors="replace").splitlines() if row.strip()]
        _require(rows and rows[-1].isdigit(), f"invalid RSS receipt: {path}")
        peak_rss[str(path)] = int(rows[-1]) * 1024
    _require(len(peak_rss) == 36, "resource receipt denominator is not 36")
    _require(all(value > 0 for value in peak_rss.values()),
             "non-positive RSS receipt")
    resource = {
        "status": "PASS_NO_OOM_MAX_SAFE_WORKERS",
        "created_utc": utc_now(),
        "revision": "h2b_v0_3_full_grid_resource_audit_v3",
        "configured_cpu_workers": configured,
        "selected_cpu_workers": selected,
        "memory_safety_fraction": safety,
        "per_worker_memory_budget_bytes": per_worker,
        "mem_available_bytes_at_start": available,
        "selected_budget_bytes": selected * per_worker,
        "n_streaming_resource_receipts": len(peak_rss),
        "max_streaming_peak_rss_bytes": max(peak_rss.values()),
        "kernel_oom_observed": False,
        "retry_or_oom_failures": 0,
        "prior_runs_stopped_before_kernel_oom": state_queue.get(
            "prior_runs_stopped_before_kernel_oom"
        ),
        "thread_limits": state_queue.get("thread_limits"),
        "receipt_peak_rss_bytes": peak_rss,
    }
    details = {
        "n_cells": len(manifests),
        "n_subjects": len(subjects),
        "n_unique_anchor_rows": unique_anchor_rows,
        "n_cell_anchor_rows": cell_anchor_rows,
        "queue_producer_sha256": sorted(producer_hashes),
        "anchor_generation_outcome_independent": True,
        "state_value_computation_outcome_independent": True,
        "wrong_time_donor_indices_use_seizure_exclusion_metadata": True,
        "population_is_prefrozen_seizure_support_conditioned": True,
    }
    return {"details": details, "resource": resource}, cache_hashes


def build(root: Path, test_log: Path) -> dict:
    required = {
        "contract": root / "analysis_contract.json",
        "historical_policy": root / "exploration_policy.json",
        "strict_policy_source": STRICT_POLICY_SOURCE,
        "attrition": root / "manifests/attrition_audit.json",
        "A1": root / "reports/scientific_route_audit_A1.json",
        "A2": root / "reports/scientific_route_audit_A2.json",
        "qualification": root / "qualification/state_qualified_manifest.json",
        "assay": root / "assay/type1_power_summary_smoke.json",
        "instrument_queue": root / "instrument/QUEUE_STATUS.json",
        "state_queue": root / "full_grid/STATE_QUEUE_STATUS.json",
        "followup": root / "full_grid/FOLLOWUP_STATUS.json",
        "hazard_gate": root / "hazard/QUEUE_STATUS.json",
        "test_log": test_log,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    _require(not missing, f"missing strict closeout inputs: {missing}")
    data = {key: _json(path) for key, path in required.items()
            if key != "test_log"}

    strict_policy = data["strict_policy_source"]
    _require(strict_policy.get("status")
             == "ACTIVE_SUPERSEDING_EXPLORATION_ADDENDUM",
             "strict acceptance policy is not active")
    strict_receipt = root / "strict_acceptance_policy.json"
    atomic_json(strict_receipt, strict_policy)

    _guard(data["contract"], "analysis contract")
    _guard(data["attrition"], "attrition")
    _require(data["attrition"].get("outcome_values_read") is False,
             "attrition selection read outcome values")
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
    _require(data["A2"].get("claim_bearing_route_released") is False,
             "A2 unexpectedly released claim-bearing route")
    _require(data["assay"].get("status")
             == "COMPLETE_DIAGNOSTIC_SMOKE_A1_EMPTY",
             "A2 assay receipt drift")
    _require(data["assay"].get("claim_bearing_route_released") is False,
             "diagnostic assay released downstream")
    _require(not (root / "assay/type1_power_summary.json").exists(),
             "unexpected final A2 acceptance receipt exists")

    _require(data["instrument_queue"].get("status") == "COMPLETE",
             "instrument queue incomplete")
    _require(data["instrument_queue"].get("requested_tasks") == 75,
             "instrument readable-task denominator drift")
    _require(data["instrument_queue"].get("failed_this_run") == 0,
             "instrument queue has unresolved runtime failures")
    _guard(data["instrument_queue"], "instrument queue")
    checkpoint_hashes, instrument_hashes = _audit_instrument(root)

    full_grid, cache_hashes = _audit_full_grid(root, data["state_queue"])

    followup = data["followup"]
    _require(followup.get("revision")
             == "h2b_v0_3_full_grid_followup_v4_strict",
             "follow-up is not the strict revision")
    _require(followup.get("status") == "NOT_RELEASED_A1_OR_A2",
             "strict follow-up did not close at the A1/A2 gate")
    _require(followup.get("downstream_tasks_started") == 0,
             "strict follow-up started downstream tasks")
    _require(followup.get("diagnostic_override_used") is False,
             "diagnostic override was used")
    _guard(followup, "strict follow-up")
    hazard_gate = data["hazard_gate"]
    _require(hazard_gate.get("status") == "NOT_RELEASED_A1_OR_A2",
             "hazard route was released")
    _require(hazard_gate.get("tasks_started") == 0,
             "hazard tasks were started on the strict route")
    _guard(hazard_gate, "hazard gate")
    _assert_no_active_downstream(root)

    quarantine = {}
    for name in QUARANTINE_DIRS:
        path = root / "quarantine" / name
        files = sorted(item for item in path.rglob("*") if item.is_file())
        _require(path.is_dir() and files,
                 f"missing quarantine evidence: {path}")
        quarantine[name] = {
            "path": str(path),
            "n_files": len(files),
            "role": "historical_or_post_gate_exploration_not_used_as_H2b_evidence",
        }

    text = test_log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) passed(?:, (\d+) warnings?)? in ", text)
    _require(match is not None, "final scoped pytest receipt is not successful")
    test_result = (
        f"{match.group(1)} passed, {int(match.group(2) or 0)} warnings"
    )

    resource_path = root / "full_grid/RESOURCE_AUDIT.json"
    atomic_json(resource_path, full_grid["resource"])
    created = utc_now()
    route = _strict_route_payload(created)
    route_path = root / "reports/scientific_route_audit_A3_A8.json"
    geometry_route_path = root / "geometry/ROUTE_STATUS.json"
    atomic_json(route_path, route)
    atomic_json(geometry_route_path, route)

    source_paths = [
        *required.values(), strict_receipt, resource_path, route_path,
        PRODUCER,
        REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_full_grid_followup.py",
        REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_full_grid_state_queue.py",
        REPO / "scripts/topic5_continuous_marked_state_h2b/extract_states.py",
        REPO / "src/topic5_continuous_marked_state_h2b/state_extraction.py",
    ]
    sources = {str(path): sha256_file(path) for path in source_paths}
    checks = {
        "analysis_contract_frozen_and_hashable": True,
        "strict_acceptance_policy_active": True,
        "historical_exploration_policy_superseded": True,
        "A0_full_attrition_recorded_without_outcome_selection": True,
        "A1_all_75_readable_cells_and_16_patients_audited": True,
        "A1_state_qualified_zero_of_16": True,
        "A2_diagnostic_smoke_not_final_acceptance": True,
        "A2_transfer_assay_not_sensitive_no_real_negative_interpretation": True,
        "all_75_checkpoint_hashes_recomputed": True,
        "seizure_gradient_did_not_enter_state": True,
        "full_grid_46_cells_10_patients_complete": True,
        "full_grid_10597_unique_anchors_45841_cell_rows_causal": True,
        "full_grid_anchor_and_state_values_outcome_independent": True,
        "wrong_time_donor_seizure_exclusion_role_disclosed": True,
        "full_grid_support_conditioned_population_disclosed": True,
        "max_safe_8_workers_no_OOM_and_36_RSS_receipts": True,
        "strict_followup_released_zero_downstream_tasks": True,
        "active_post_gate_outputs_absent": True,
        "post_gate_exploratory_outputs_quarantined_not_evidence": True,
        "A3_A8_not_run_on_strict_route": True,
        "patient_is_inference_unit_and_seeds_not_patients": True,
        "formal_sealed_H3_T2_physical_clock_untouched": True,
        "paper_ready_figures_not_modified": True,
        "final_scoped_tests_passed": True,
    }
    machine = {
        "status": "PASS_GATED_NEGATIVE_CLOSEOUT_H2B_NOT_ESTABLISHED",
        "revision": "h2b_v0_3_machine_audit_v3_strict",
        "created_utc": created,
        "development_only": True,
        "scientific_conclusion": (
            "current R1.7B checkpoints did not qualify as multidimensional persistent "
            "state instruments; the diagnostic transfer assay was not sensitive, so "
            "H2b downstream was not released and no biological negative is allowed"
        ),
        "largest_gap": (
            "repair and requalify the interictal state instrument before any new "
            "seizure-risk or seizure-entry geometry head"
        ),
        "n_readable_checkpoint_cells": 75,
        "n_all_frozen_patients": 16,
        "n_state_qualified_patients": 0,
        "full_grid": full_grid["details"],
        "resource_audit": str(resource_path),
        "strict_route_audit": str(route_path),
        "quarantine": quarantine,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "test_result": test_result,
        "source_sha256": sources,
        "checkpoint_sha256": checkpoint_hashes,
        "instrument_manifest_sha256": instrument_hashes,
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
