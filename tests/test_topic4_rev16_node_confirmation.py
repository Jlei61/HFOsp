from __future__ import annotations

import hashlib
import json

import pytest

from scripts import freeze_topic4_rev16_joint_m3_m4_candidates as selection_freezer
from scripts import audit_topic4_rev16_node_confirmation as audit
from scripts import freeze_topic4_rev16_node_confirmation as freezer
from scripts import monitor_topic4_rev16_node_confirmation as monitor
from scripts import prepare_topic4_rev16_node_confirmation_config as prepare
from scripts import run_topic4_rev14_m3_canary_worker as shared_worker
from scripts import run_topic4_rev16_node_confirmation_worker as worker


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    artifact = tmp_path / "artifact"
    monkeypatch.setattr(selection_freezer, "_validate_config", lambda config: None)
    inputs = {}
    for name in (
        "rev13_config", "rev13_exact_off_manifest", "j14_config",
        "patient_support_config",
    ):
        path = _write(artifact / f"inputs/{name}.json", {"name": name})
        inputs[name] = {"path": str(path.relative_to(artifact)), "sha256": _sha(path)}
    selection_config_path = repository / "config/selection.json"
    selection_manifest_path = artifact / "results/selection/manifest.json"
    selection_aggregate_path = artifact / "results/selection/aggregate.json"
    selection_config = {
        "schema_id": selection_freezer.EXPECTED_SCHEMA,
        "candidate_manifest": str(selection_manifest_path.relative_to(artifact)),
        "network_cache": "results/cache",
        "inputs": inputs,
        "field_design": {
            "basis_family": "absolute_paired_phase_whole_sheet_fourier",
            "maximum_order": 4, "expected_modes": 24,
            "expected_real_coefficients": 48, "sheet_length_mm": 20.0,
            "quadrature_per_axis": 128, "coordinate_decimal_places": 13,
        },
        "node_mapping": {"signed_depth_contract": {"sha256": "d" * 64}},
        "pathways": selection_freezer.EXPECTED_PATHWAYS,
    }
    _write(selection_config_path, selection_config)
    candidates = [
        {
            "candidate_id": "exact_off", "selection_eligible": False,
            "field_kind": "stage_ak_exact_off_benchmark", "pathways": selection_freezer.EXPECTED_PATHWAYS,
            "fourier_coordinate": None,
        },
        {
            "candidate_id": "joint", "selection_eligible": True,
            "field_kind": "absolute_paired_phase_fourier_joint_m3_m4",
            "pathways": selection_freezer.EXPECTED_PATHWAYS,
            "fourier_coordinate": {"coefficients_sha256": "c" * 64},
        },
    ]
    _write(selection_manifest_path, {
        "status": selection_freezer.STATUS,
        "config_sha256": _sha(selection_config_path), "candidates": candidates,
        "event_unit": {"unit": "causal"}, "source_topology": {"bin_mm": 1.0},
    })
    _write(selection_aggregate_path, {
        "schema_id": prepare.EXPECTED_AGGREGATE_SCHEMA, "status": "COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "ranking_contract": {
            "J14_improvement": "3/3 fresh networks",
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "best_usable_anchor": "joint",
        "candidate_summaries": [{
            "candidate_id": "joint", "usable_two_mode_anchor": True,
            "fresh_J14_improvement_count": 3,
            "fresh_A_improvement_count": 3,
            "fresh_B_protection_count": 3,
            "fresh_A_support_count": 3,
            "fresh_B_support_count": 3,
        }],
    })
    return repository, artifact, selection_config_path, selection_aggregate_path


def test_confirmation_config_separates_selection_and_confirmation_networks(
    tmp_path, monkeypatch,
):
    repository, artifact, config_path, aggregate_path = _tree(tmp_path, monkeypatch)
    payload = prepare.build_config(
        selection_config_path=config_path,
        selection_aggregate_path=aggregate_path,
        artifact_root=artifact, repository_root=repository,
    )
    assert payload["field_design"]["candidate_ids"] == ["exact_off", "joint"]
    assert payload["search"]["selection_network_seeds"] == [2351, 2352, 2353]
    assert payload["search"]["confirmation_network_seeds"] == [2361, 2362, 2363]
    assert payload["boundaries"]["field_reranking_allowed"] is False
    assert payload["boundaries"]["EE_EtoI_ZM"] == "off"
    assert payload["confirmation_acceptance"] == {
        "J14_improvement_required_networks": 3,
        "A_improvement_required_networks": 3,
        "B_protection_required_networks": 3,
        "B_protection_ratio": 1.10,
        "equal_network_effective_support_minimum_per_mode": 6.0,
    }


def test_confirmation_rejects_selection_without_full_j14(tmp_path, monkeypatch):
    repository, artifact, config_path, aggregate_path = _tree(tmp_path, monkeypatch)
    data = json.loads(aggregate_path.read_text())
    del data["ranking_contract"]["J14_improvement"]
    _write(aggregate_path, data)
    with pytest.raises(RuntimeError, match="full-J14"):
        prepare.build_config(
            selection_config_path=config_path,
            selection_aggregate_path=aggregate_path,
            artifact_root=artifact, repository_root=repository,
        )


def test_confirmation_freezer_rejects_relaxed_acceptance(tmp_path, monkeypatch):
    repository, artifact, config_path, aggregate_path = _tree(tmp_path, monkeypatch)
    payload = prepare.build_config(
        selection_config_path=config_path,
        selection_aggregate_path=aggregate_path,
        artifact_root=artifact, repository_root=repository,
    )
    payload["confirmation_acceptance"][
        "J14_improvement_required_networks"
    ] = 2
    with pytest.raises(RuntimeError, match="acceptance contract"):
        freezer._validate_config(payload)


def test_confirmation_worker_switches_only_its_freezer():
    old = (
        shared_worker.freezer, shared_worker.WORKER_STATUS,
        shared_worker.PREPARE_STATUS, shared_worker.EXPECTED_PATHWAYS,
    )
    worker.configure_base()
    try:
        assert shared_worker.freezer is freezer
        assert shared_worker.WORKER_STATUS == worker.WORKER_STATUS
    finally:
        (
            shared_worker.freezer, shared_worker.WORKER_STATUS,
            shared_worker.PREPARE_STATUS, shared_worker.EXPECTED_PATHWAYS,
        ) = old


def test_confirmation_monitor_has_disjoint_unit_prefix():
    assert monitor._validate_unit_prefix(monitor.DEFAULT_UNIT_PREFIX) == (
        monitor.DEFAULT_UNIT_PREFIX
    )


def _scored_rows(*, failing_j14_seed=None, failing_support_seed=None):
    rows = []
    for seed in prepare.CONFIRMATION_SEEDS:
        rows.extend([
            {
                "candidate_id": "exact_off", "seed": seed,
                "mode_0_mean": 1.0, "mode_1_mean": 1.0, "j14": 2.0,
                "mode_0_effective_events": 8.0,
                "mode_1_effective_events": 8.0,
            },
            {
                "candidate_id": "joint", "seed": seed,
                "mode_0_mean": 0.8, "mode_1_mean": 1.05,
                "j14": 2.1 if seed == failing_j14_seed else 1.8,
                "mode_0_effective_events": (
                    4.0 if seed == failing_support_seed else 7.0
                ),
                "mode_1_effective_events": 7.0,
                "family": "mean_j14", "target_rms": 0.6,
                "m3_l2_fraction": 0.8, "m4_shell_l2_fraction": 0.6,
            },
        ])
    return rows


def test_confirmation_decision_requires_all_three_unseen_networks():
    manifest = {
        "candidates": [
            {"candidate_id": "exact_off"}, {"candidate_id": "joint"},
        ],
    }
    acceptance = {
        "J14_improvement_required_networks": 3,
        "A_improvement_required_networks": 3,
        "B_protection_required_networks": 3,
        "B_protection_ratio": 1.10,
        "equal_network_effective_support_minimum_per_mode": 6.0,
    }
    passed = audit.confirmation_decision(
        _scored_rows(), manifest, acceptance,
    )
    assert passed["accepted"] is True
    assert passed["J14_improvement_count"] == 3
    failed = audit.confirmation_decision(
        _scored_rows(failing_j14_seed=2363), manifest, acceptance,
    )
    assert failed["accepted"] is False
    assert failed["J14_improvement_count"] == 2
    assert failed["failure_does_not_trigger_reranking"] is True
    support_failed = audit.confirmation_decision(
        _scored_rows(failing_support_seed=2363), manifest, acceptance,
    )
    assert support_failed["accepted"] is False
    assert support_failed["A_support_count"] == 2
