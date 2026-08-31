from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import aggregate_topic4_rev15_node_intervention as aggregate_base
from scripts import aggregate_topic4_rev17_node_intervention as aggregate
from scripts import prepare_topic4_rev17_node_intervention as prepare
from scripts import run_topic4_rev15_node_intervention_worker as worker_base
from scripts import run_topic4_rev17_node_intervention_worker as worker
from scripts import topic4_rev17_node_substrate_adapter as adapter
from scripts import monitor_topic4_rev17_node_intervention as monitor


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _record(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": prepare._sha256(path)}


def test_intervention_config_freezes_loo_hotspots_and_dual_mapping(tmp_path):
    repository = tmp_path / "repo"
    artifact = tmp_path / "artifact"
    cohort = _write(repository / "cohort.json", {})
    classifier = _write(repository / "classifier.json", {})
    confirmation_path = repository / "confirmation.json"
    confirmation = {
        "schema_id": "topic4_rev17_node_confirmation_v1",
        "candidate_manifest": "manifest.json",
        "selected_candidate": {
            "candidate_id": "winner", "mapping_sha256": "b" * 64,
        },
    }
    _write(confirmation_path, confirmation)
    manifest = _write(artifact / "manifest.json", {
        "status": "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN",
        "config_sha256": prepare._sha256(confirmation_path),
        "candidates": [{
            "candidate_id": "winner",
            "node_mapping": {"mapping_sha256": "b" * 64},
        }],
    })
    final_config_path = repository / "final.json"
    final_config = {
        "schema_id": "topic4_rev17_node_final_science_v1",
        "selected_candidate": {
            "candidate_id": "winner", "mapping_sha256": "b" * 64,
        },
        "network_seeds": [2381, 2382, 2383],
        "inputs": {
            "confirmation_config": _record(confirmation_path, repository),
            "confirmation_manifest": _record(manifest, artifact),
            "cohort_config": _record(cohort, repository),
            "classifier_config": _record(classifier, repository),
        },
    }
    _write(final_config_path, final_config)
    final_audit = _write(artifact / "final_audit.json", {
        "schema_id": "topic4_rev17_node_final_science_audit_v1",
        "status": "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION",
        "candidate_id": "winner",
        "decision": {
            "accepted_for_same_checkpoint_intervention": True,
            "node_freeze_permitted": False,
        },
        "inputs": {"config": {"sha256": prepare._sha256(final_config_path)}},
    })
    config = prepare.build_config(
        final_config_path=final_config_path, final_audit_path=final_audit,
        repository_root=repository, artifact_root=artifact,
    )
    assert config["candidate_mapping_sha256"] == "b" * 64
    assert config["network_seeds"] == [2381, 2382, 2383]
    assert config["hotspot_construction"]["leave_one_network_out"] is True
    assert config["hotspot_construction"]["mode_discriminative_contrast"] is True
    assert config["hotspot_construction"]["uses_patient_heldout"] is False
    assert config["hotspot_construction"]["matching_covariate_keys"] == [
        "h_mean", "delta_vtheta_mean", "e_density", "baseline_rate_hz",
    ]
    assert config["intervention"]["common_checkpoint_state_and_random_stream"] is True
    assert config["mechanism_freeze"] == {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }


def test_dual_adapter_builds_both_fields_and_checks_stored_arrays(
    monkeypatch, tmp_path,
):
    artifact = tmp_path / "artifact"
    transition = _write(artifact / "transition.json", {})
    config_path = tmp_path / "confirmation.json"
    config = {
        "schema_id": "topic4_rev17_node_confirmation_v1",
        "candidate_manifest": "manifest.json", "network_cache": "cache",
        "inputs": {"transition_config": {
            "path": "transition.json", "sha256": adapter._sha256(transition),
        }},
        "selected_candidate": {
            "candidate_id": "winner", "mapping_sha256": "b" * 64,
        },
        "search": {"confirmation_network_seeds": [2381, 2382, 2383]},
    }
    _write(config_path, config)
    mean_field = {"field_sha256": "m" * 64}
    dispersion_field = {"field_sha256": "d" * 64}
    _write(artifact / "manifest.json", {
        "status": "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN",
        "config_sha256": adapter._sha256(config_path),
        "provenance": {"formal_ready": True},
        "candidates": [{
            "candidate_id": "winner", "node_field": mean_field,
            "node_dispersion_field": dispersion_field,
            "node_mapping": {
                "mapping_type": "dual_continuous_mean_dispersion",
                "mapping_sha256": "b" * 64,
            },
        }],
    })
    captured = {}
    substrate = SimpleNamespace(
        h_e=np.array([0.1, 0.2]), vtheta=np.array([-50.1, -49.8]),
        delta_vtheta=np.array([-0.1, 0.2]), edge_coefficients=np.zeros(2),
        extras={"node_mapping_audit": {
            "mapping_type": "dual_continuous_mean_dispersion",
        }},
    )

    def fake_build(round_config, arm, seed, **kwargs):
        captured.update(kwargs)
        return substrate

    monkeypatch.setattr(adapter, "load_round_config", lambda path: {"round": True})
    monkeypatch.setattr(adapter, "build_substrate", fake_build)
    observed, projection, _ = adapter.build_projected_node_substrate(
        robust_config_path=config_path, candidate_id="winner", seed=2381,
        artifact_root=artifact,
    )
    assert observed is substrate
    assert captured["node_candidate_override"] == mean_field
    assert captured["node_dispersion_candidate_override"] == dispersion_field
    assert captured["ee_dose"] == captured["etoi_dose"] == 0.0
    arrays = tmp_path / "worker.npz"
    np.savez_compressed(
        arrays, h=projection["h"].astype(np.float32),
        delta_vtheta=projection["delta_vtheta"].astype(np.float32),
    )
    parity = adapter.verify_projection_against_worker(projection, arrays)
    assert parity["exact_array_parity"] == {"h": True, "delta_vtheta": True}

    np.savez_compressed(
        arrays, h=(projection["h"] + 1).astype(np.float32),
        delta_vtheta=projection["delta_vtheta"].astype(np.float32),
    )
    with pytest.raises(RuntimeError, match="differs from confirmation worker"):
        adapter.verify_projection_against_worker(projection, arrays)


def test_worker_wrapper_switches_and_restores_rev17_contract(monkeypatch):
    seen = {}

    def fake_run_worker(**kwargs):
        seen.update({
            "config": worker_base.EXPECTED_CONFIG_SCHEMA,
            "status": worker_base.WORKER_STATUS,
            "final": worker_base.FINAL_AUDIT_ADVANCE_STATUS,
            "builder": worker_base.build_projected_node_substrate,
        })
        return {"status": "ok"}

    old = (
        worker_base.EXPECTED_CONFIG_SCHEMA, worker_base.WORKER_STATUS,
        worker_base.FINAL_AUDIT_ADVANCE_STATUS,
        worker_base.build_projected_node_substrate,
    )
    monkeypatch.setattr(worker_base, "run_worker", fake_run_worker)
    assert worker.run_worker()["status"] == "ok"
    assert seen == {
        "config": worker.EXPECTED_CONFIG_SCHEMA,
        "status": worker.WORKER_STATUS,
        "final": worker.FINAL_AUDIT_ADVANCE_STATUS,
        "builder": adapter.build_projected_node_substrate,
    }
    assert (
        worker_base.EXPECTED_CONFIG_SCHEMA, worker_base.WORKER_STATUS,
        worker_base.FINAL_AUDIT_ADVANCE_STATUS,
        worker_base.build_projected_node_substrate,
    ) == old


def test_aggregate_uses_only_stored_rev17_projection_keys(monkeypatch, tmp_path):
    seen = {}

    def fake_aggregate(**kwargs):
        seen["keys"] = aggregate_base.EXPECTED_PROJECTION_PARITY_KEYS
        return {"status": "x"}

    original = aggregate_base.EXPECTED_PROJECTION_PARITY_KEYS
    monkeypatch.setattr(aggregate_base, "aggregate", fake_aggregate)
    aggregate.aggregate(config_path=tmp_path / "config.json", artifact_root=tmp_path)
    assert seen["keys"] == {"h", "delta_vtheta"}
    assert aggregate_base.EXPECTED_PROJECTION_PARITY_KEYS == original


def test_rev17_monitor_switches_controller_contract_and_restores(monkeypatch):
    seen = {}

    def fake_tick(**kwargs):
        seen.update({
            "worker": monitor.base.worker,
            "aggregate": monitor.base.aggregate,
            "seeds": monitor.base.EXPECTED_NETWORK_SEEDS,
            "final": monitor.base.FINAL_AUDIT_ADVANCE_STATUS,
            "parity": monitor.base.EXPECTED_PROJECTION_PARITY_KEYS,
        })
        return {"status": "x"}

    old = (
        monitor.base.worker, monitor.base.aggregate,
        monitor.base.EXPECTED_NETWORK_SEEDS,
        monitor.base.FINAL_AUDIT_ADVANCE_STATUS,
        monitor.base.EXPECTED_PROJECTION_PARITY_KEYS,
    )
    monkeypatch.setattr(monitor.base, "tick", fake_tick)
    assert monitor.tick()["status"] == "x"
    assert seen == {
        "worker": worker, "aggregate": aggregate,
        "seeds": [2381, 2382, 2383],
        "final": monitor.FINAL_AUDIT_ADVANCE_STATUS,
        "parity": {"h", "delta_vtheta"},
    }
    assert (
        monitor.base.worker, monitor.base.aggregate,
        monitor.base.EXPECTED_NETWORK_SEEDS,
        monitor.base.FINAL_AUDIT_ADVANCE_STATUS,
        monitor.base.EXPECTED_PROJECTION_PARITY_KEYS,
    ) == old
