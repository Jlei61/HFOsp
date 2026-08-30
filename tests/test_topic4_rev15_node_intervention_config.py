from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_topic4_rev15_node_intervention_config as prepare


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repo"
    artifact = tmp_path / "artifact"
    manifest = _write(artifact / "results/manifest.json", {"frozen": True})
    robust = _write(repository / "config/robust.json", {
        "candidate_manifest": "results/manifest.json",
    })
    cohort = _write(repository / "config/cohort.json", {"cohort": True})
    classifier = _write(repository / "config/classifier.json", {"classifier": True})
    final_config = _write(repository / "config/final.json", {
        "selected_candidate": {"candidate_id": "selected"},
        "network_seeds": [2341, 2342, 2343],
        "inputs": {"robust_config": {
            "path": "config/robust.json", "sha256": _sha(robust),
        }, "cohort_config": {
            "path": "config/cohort.json", "sha256": _sha(cohort),
        }, "classifier_config": {
            "path": "config/classifier.json", "sha256": _sha(classifier),
        }},
    })
    final_audit = _write(artifact / "results/final.json", {
        "status": "NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION",
        "candidate_id": "selected",
        "decision": {
            "accepted_for_same_checkpoint_intervention": True,
            "node_freeze_permitted": False,
        },
    })
    return {
        "repository": repository, "artifact": artifact,
        "manifest": manifest, "robust": robust,
        "final_config": final_config, "final_audit": final_audit,
    }


def test_intervention_preparer_freezes_crossed_three_network_design(tmp_path):
    tree = _tree(tmp_path)
    config = prepare.build_config(
        final_config_path=tree["final_config"],
        final_audit_path=tree["final_audit"],
        repository_root=tree["repository"], artifact_root=tree["artifact"],
    )
    assert config["candidate_id"] == "selected"
    assert config["network_seeds"] == [2341, 2342, 2343]
    assert config["intervention"]["crossed_design"] is True
    assert config["intervention"]["arms_per_native_mode"] == [
        "sham", "MTA_hotspot", "MTA_matched_off_template",
        "MTB_hotspot", "MTB_matched_off_template",
    ]
    assert config["decision"]["required_selective_networks"] == 2
    assert config["intervention"]["maximum_event_shift_ms"] == 200.0
    assert set(config["inputs"]) >= {"cohort_config", "classifier_config"}
    assert config["mechanism_freeze"] == {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }


def test_intervention_preparer_rejects_zero_simulation_failure(tmp_path):
    tree = _tree(tmp_path)
    payload = json.loads(tree["final_audit"].read_text())
    payload["status"] = "NODE_FINAL_SCIENCE_ZERO_SIMULATION_REJECTED"
    tree["final_audit"].write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not advanced"):
        prepare.build_config(
            final_config_path=tree["final_config"],
            final_audit_path=tree["final_audit"],
            repository_root=tree["repository"], artifact_root=tree["artifact"],
        )
