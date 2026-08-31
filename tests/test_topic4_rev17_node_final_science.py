from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_topic4_rev15_node_final_science as base
from scripts import audit_topic4_rev17_node_final_science as audit
from scripts import prepare_topic4_rev17_node_final_science as prepare


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_final_science_config_freezes_dual_mapping_and_exact_anchor(tmp_path):
    repository = tmp_path / "repo"
    artifact = tmp_path / "artifact"
    cohort = _write(repository / "config/cohort.json", {"cohort": True})
    classifier = _write(
        repository / "config/classifier.json", {"classifier": True}
    )
    rev12 = _write(repository / "config/rev12.json", {
        "inputs": {
            "cohort_config": {
                "path": "config/cohort.json", "sha256": _sha(cohort),
            },
            "classifier_config": {
                "path": "config/classifier.json", "sha256": _sha(classifier),
            },
        },
    })
    confirmation_path = repository / "config/confirmation.json"
    confirmation = {
        "schema_id": "topic4_rev17_node_confirmation_v1",
        "candidate_manifest": "results/confirmation_manifest.json",
        "selected_candidate": {"candidate_id": "winner"},
        "search": {"confirmation_network_seeds": [2381, 2382, 2383]},
    }
    _write(confirmation_path, confirmation)
    manifest = _write(artifact / "results/confirmation_manifest.json", {
        "status": "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN",
        "candidates": [
            {
                "candidate_id": "exact_dual_anchor",
                "node_mapping": {"mapping_sha256": "a" * 64},
            },
            {
                "candidate_id": "winner",
                "node_mapping": {"mapping_sha256": "b" * 64},
            },
        ],
    })
    confirmation_audit = _write(artifact / "results/confirmation_audit.json", {
        "candidate_id": "winner",
        "scientific_confirmation": {"accepted": True},
    })
    postselection_config = _write(repository / "config/postselection.json", {
        "selected_candidate": {"candidate_id": "winner"},
    })
    postselection_audit = _write(artifact / "results/postselection_audit.json", {
        "status": "REV17_NODE_POSTSELECTION_ACCEPTED",
        "candidate_id": "winner", "acceptance": {"accepted": True},
    })
    config = prepare.build_config(
        confirmation_config_path=confirmation_path,
        confirmation_audit_path=confirmation_audit,
        postselection_config_path=postselection_config,
        postselection_audit_path=postselection_audit,
        rev12_config_path=rev12,
        repository_root=repository, artifact_root=artifact,
    )
    assert config["selected_candidate"] == {
        "candidate_id": "winner",
        "mapping_sha256": "b" * 64,
        "paired_reference_candidate_id": "exact_dual_anchor",
        "reference_mapping_sha256": "a" * 64,
    }
    assert config["network_seeds"] == [2381, 2382, 2383]
    assert config["source_topology"]["must_improve_paired_exact_anchor"] is True
    assert config["boundaries"]["field_reranking_allowed"] is False
    assert config["boundaries"]["EE_EtoI_ZM"] == "off"
    assert config["inputs"]["confirmation_manifest"]["sha256"] == _sha(manifest)


def test_dual_worker_contract_uses_rev17_status_and_mapping(monkeypatch, tmp_path):
    worker_root = tmp_path / "workers"
    npz_path = worker_root / "winner_seed_2381.npz"
    npz_path.parent.mkdir(parents=True)
    npz_path.write_bytes(b"arrays")
    payload = {
        "status": audit.WORKER_STATUS,
        "candidate_id": "winner", "seed": 2381,
        "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        "simulation": {"duration_ms": 20000.0, "runaway_early_stop_ms": None},
        "provenance": {
            "runtime_modules_dirty": 0,
            "runtime_modules_match_expected_commit": 1,
        },
        "arrays": {"sha256": _sha(npz_path)},
        "node_mapping": {"mapping_sha256": "b" * 64},
    }
    _write(npz_path.with_suffix(".json"), payload)
    monkeypatch.setattr(
        base, "_load_network_worker",
        lambda *args, **kwargs: {
            "seed": 2381, "ranks": np.zeros((2, 2)),
            "labels": np.array([0, 1]),
        },
    )
    monkeypatch.setattr(
        base, "_source_bundle",
        lambda *args, **kwargs: (
            np.zeros((2, 2, 2)), np.array([0, 1]),
        ),
    )
    workers, _, _, _ = base._candidate_workers(
        "winner", robust_config={"output_root": "."}, seeds=[2381],
        patient={"contact_names": np.array(["A", "B"])}, classifier={},
        label_map=np.array([0, 1]), artifact_root=tmp_path,
        expected_coefficients_sha256=None,
        expected_mapping_sha256="b" * 64,
        expected_worker_status=audit.WORKER_STATUS,
    )
    assert workers[0]["seed"] == 2381

    payload["node_mapping"]["mapping_sha256"] = "c" * 64
    _write(npz_path.with_suffix(".json"), payload)
    with pytest.raises(RuntimeError, match="dual mapping changed"):
        base._candidate_workers(
            "winner", robust_config={"output_root": "."}, seeds=[2381],
            patient={"contact_names": np.array(["A", "B"])}, classifier={},
            label_map=np.array([0, 1]), artifact_root=tmp_path,
            expected_coefficients_sha256=None,
            expected_mapping_sha256="b" * 64,
            expected_worker_status=audit.WORKER_STATUS,
        )
