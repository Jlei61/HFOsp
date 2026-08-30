from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_topic4_rev15_node_final_science as audit
from scripts import prepare_topic4_rev15_node_final_science_config as prepare


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree(tmp_path: Path) -> dict[str, Path]:
    repository = tmp_path / "repo"
    artifact = tmp_path / "artifact"
    cohort = _write(repository / "config/cohort.json", {"cohort": True})
    classifier = _write(repository / "config/classifier.json", {"classifier": True})
    rev12 = _write(repository / "config/rev12.json", {
        "inputs": {
            "cohort_config": {"path": "config/cohort.json", "sha256": _sha(cohort)},
            "classifier_config": {"path": "config/classifier.json", "sha256": _sha(classifier)},
        },
    })
    manifest = _write(artifact / "results/manifest.json", {
        "candidates": [
            {"candidate_id": "exact_off", "fourier_coordinate": None},
            {"candidate_id": "selected", "fourier_coordinate": {
                "coefficients_sha256": "a" * 64,
            }},
        ],
    })
    robust_config = _write(repository / "config/robust.json", {
        "candidate_manifest": "results/manifest.json",
    })
    robust_aggregate = _write(artifact / "results/aggregate.json", {
        "status": "COMPLETE", "best_usable_anchor": "selected",
    })
    postselection_config = _write(repository / "config/post.json", {
        "selected_candidate": {"candidate_id": "selected"},
    })
    postselection_audit = _write(artifact / "results/post.json", {
        "status": "NODE_POSTSELECTION_ACCEPTED",
        "candidate_id": "selected",
        "acceptance": {"accepted": True},
    })
    return {
        "repository": repository, "artifact": artifact,
        "cohort": cohort, "classifier": classifier, "rev12": rev12,
        "manifest": manifest, "robust_config": robust_config,
        "robust_aggregate": robust_aggregate,
        "postselection_config": postselection_config,
        "postselection_audit": postselection_audit,
    }


def test_preparer_freezes_one_selected_candidate_without_opening_heldout(tmp_path):
    tree = _tree(tmp_path)
    config = prepare.build_config(
        robust_config_path=tree["robust_config"],
        robust_aggregate_path=tree["robust_aggregate"],
        postselection_config_path=tree["postselection_config"],
        postselection_audit_path=tree["postselection_audit"],
        rev12_config_path=tree["rev12"],
        repository_root=tree["repository"], artifact_root=tree["artifact"],
    )
    assert config["selected_candidate"]["candidate_id"] == "selected"
    assert config["selected_candidate"]["paired_reference_candidate_id"] == "exact_off"
    assert config["network_seeds"] == [2341, 2342, 2343]
    assert "transition_config" not in config["inputs"]
    assert config["boundaries"]["field_reranking_allowed"] is False
    assert config["boundaries"]["SNN_simulation_run"] is False


def test_preparer_rejects_unaccepted_postselection(tmp_path):
    tree = _tree(tmp_path)
    payload = json.loads(tree["postselection_audit"].read_text())
    payload["status"] = "NODE_POSTSELECTION_REJECTED"
    tree["postselection_audit"].write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="not been accepted"):
        prepare.build_config(
            robust_config_path=tree["robust_config"],
            robust_aggregate_path=tree["robust_aggregate"],
            postselection_config_path=tree["postselection_config"],
            postselection_audit_path=tree["postselection_audit"],
            rev12_config_path=tree["rev12"],
            repository_root=tree["repository"], artifact_root=tree["artifact"],
        )


def test_paired_deltas_keep_network_identity_and_both_modes():
    def score(offset: float) -> dict:
        return {
            "network_scores": [{
                "seed": seed,
                "weakest_mode_lse": seed + offset,
                "modes": {
                    "0": {"mean": 0.5 * seed + offset},
                    "1": {"mean": 0.25 * seed + offset},
                },
                "mode_counts": [3, 4],
            } for seed in (2341, 2342, 2343)]
        }
    rows = audit._paired_deltas(score(-0.1), score(0.0))
    assert [row["network_seed"] for row in rows] == [2341, 2342, 2343]
    assert all(row["delta_weakest_mode_loss"] < 0 for row in rows)
    assert all(row["delta_mode_0_loss"] < 0 for row in rows)
    assert all(row["delta_mode_1_loss"] < 0 for row in rows)
