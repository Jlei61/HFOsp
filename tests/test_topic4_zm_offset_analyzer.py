"""Fail-closed contracts for the Phase-2B offset analyzer."""

from __future__ import annotations

import copy

import scripts.adjudicate_topic4_zm_branch_decision as J
import scripts.analyze_topic4_zm_offset_boundary as A


def _base_rows(seed):
    rows = []
    for family in A.FAMILIES:
        for level in A.BASE_LEVELS:
            for initial_kind in A.BASE_INITIAL_KINDS:
                rows.append(
                    {
                        "key": (
                            f"{family}|lambda={level:g}|"
                            f"{initial_kind}|noise_replay"
                        ),
                        "seed": seed,
                        "family": family,
                        "lambda": level,
                        "initial_kind": initial_kind,
                        "replicate": A.BASE_REPLICATE,
                        "bank_sha": f"bank-{seed}-{family}-{level}-{initial_kind}",
                        "completed": True,
                        "response_valid": True,
                        "boundary_version": A.BD.BOUNDARY_VERSION,
                        "remained_carrier": True,
                        "low_basin_persisted": False,
                    }
                )
    return rows


def _manifest(seed):
    return {
        "seed": seed,
        "complete": True,
        "pending_cells": [],
        "config_sha": f"config-{seed}",
        "boundary_version": A.BD.BOUNDARY_VERSION,
        "metrics_version": A.MC.METRICS_VERSION,
        "response_ms": 8000.0,
        "dt": 0.1,
        "resolution": "dt",
        "state_schema": "zm_sim_state_v1",
        "families": list(A.FAMILIES),
        "levels": list(A.BASE_LEVELS),
        "engine_sha256": {"engine.py": "abc"},
        "source_state_hashes": {
            label: f"{seed}-{label}" for label in A.REQUIRED_STATE_LABELS
        },
        "rows": _base_rows(seed),
    }


def _anchors():
    return {
        seed: {"seed": seed, "config_sha": f"config-{seed}"}
        for seed in A.PRIMARY_SEEDS
    }


def test_complete_three_seed_manifest_contract_passes():
    manifests = [_manifest(seed) for seed in A.PRIMARY_SEEDS]
    out = A.validate_offset_manifest_contract(manifests, _anchors())
    assert out["passed"]
    assert out["reasons"] == []
    assert out["required_base_cells_per_seed"] == 24


def test_incomplete_pending_or_missing_base_manifest_fails_closed():
    manifests = [_manifest(seed) for seed in A.PRIMARY_SEEDS]
    manifests[0]["complete"] = False
    manifests[1]["pending_cells"] = [{"key": "x"}]
    manifests[2]["rows"].pop()
    out = A.validate_offset_manifest_contract(manifests, _anchors())
    assert not out["passed"]
    assert any("manifest_not_complete" in reason for reason in out["reasons"])
    assert any("pending_cells_nonempty" in reason for reason in out["reasons"])
    assert any("missing_required_base_cells" in reason for reason in out["reasons"])


def test_duplicate_key_config_or_engine_drift_fails_closed():
    manifests = [_manifest(seed) for seed in A.PRIMARY_SEEDS]
    manifests[0]["rows"].append(copy.deepcopy(manifests[0]["rows"][0]))
    manifests[1]["config_sha"] = "wrong"
    manifests[2]["engine_sha256"] = {"engine.py": "drift"}
    out = A.validate_offset_manifest_contract(manifests, _anchors())
    assert not out["passed"]
    assert any("row_keys_missing_or_duplicate" in reason for reason in out["reasons"])
    assert any("anchor_config_sha_mismatch" in reason for reason in out["reasons"])
    assert any("engine_sha256_mismatch" in reason for reason in out["reasons"])


def test_physical_bound_invalid_extension_does_not_invalidate_required_base():
    manifests = [_manifest(seed) for seed in A.PRIMARY_SEEDS]
    manifests[0]["rows"].append(
        {
            "key": "M_alone|lambda=1.25|active|noise_replay",
            "seed": 1,
            "family": "M_alone",
            "lambda": 1.25,
            "initial_kind": "active",
            "replicate": "noise_replay",
            "completed": True,
            "response_valid": False,
            "invalid_reason": "physical_bound_violation_without_clipping",
        }
    )
    out = A.validate_offset_manifest_contract(manifests, _anchors())
    assert out["passed"]


def test_phase_execution_status_does_not_list_completed_fail_closed_analyses():
    status, not_run = J.phase_execution_status(
        {"status": "class_disagreement"},
        {"verdict": "no_evidence_incomplete_central_pairs"},
        {"status": "insufficient_seeds", "n_complete_seeds": 0},
        {"verdict": "conditional_Z_entry_boundary_unresolved"},
        {"verdict": "no_evidence"},
    )
    assert status == {
        "functional_rank": "completed_no_evidence",
        "modal_operator": "skipped_source_class_disagreement",
        "entry_boundary": "completed_unresolved",
        "offset_boundary": "completed_no_evidence",
        "exit_driver_selection": "not_authorized",
    }
    assert not any("Task 9A" in item for item in not_run)
    assert not any("Task 10" in item for item in not_run)
    assert not any("Task 11" in item for item in not_run)
    assert any("Task 9B" in item and "skipped" in item for item in not_run)
    assert any("Task 12" in item and "not authorized" in item for item in not_run)
