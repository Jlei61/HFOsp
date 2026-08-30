from __future__ import annotations

import hashlib
import json

import pytest

from scripts import audit_topic4_rev15_node_postselection as rev15_audit
from scripts import audit_topic4_rev16_node_postselection as audit
from scripts import prepare_topic4_rev16_node_postselection_config as prepare
from scripts import run_topic4_rev16_joint_m3_m4_candidate_worker as worker
from scripts.paper_figures import (
    plot_topic4_rev15_node_postselection_fig4 as rev15_figure,
)
from scripts.paper_figures import (
    plot_topic4_rev16_node_postselection_fig4 as figure,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _inputs(tmp_path):
    repository = tmp_path / "repository"
    artifact = tmp_path / "artifact"
    manifest_path = artifact / "results/joint/candidate_manifest.json"
    aggregate_path = artifact / "results/joint/aggregate.json"
    config_path = repository / "config/joint.json"
    j14_path = repository / "config/j14.json"
    patient_inputs = {}
    for name in (
        "patient_training_target", "frozen_direction_classifier_manifest",
        "contact_contract",
    ):
        path = artifact / f"inputs/{name}.json"
        _write(path, {"name": name})
        patient_inputs[name] = {
            "path": str(path.relative_to(artifact)), "sha256": _sha(path),
        }
    _write(j14_path, {"inputs": patient_inputs})
    _write(manifest_path, {
        "candidates": [{
            "candidate_id": "joint", "selection_eligible": True,
            "fourier_coordinate": {"coefficients_sha256": "c" * 64},
        }],
    })
    _write(config_path, {
        "schema_id": prepare.EXPECTED_CANDIDATE_SCHEMA,
        "candidate_manifest": str(manifest_path.relative_to(artifact)),
        "inputs": {"j14_config": {
            "path": str(j14_path.relative_to(repository)),
            "sha256": _sha(j14_path),
        }},
        "search": {"active_network_seeds": prepare.NETWORK_SEEDS},
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        },
    })
    _write(aggregate_path, {
        "schema_id": prepare.EXPECTED_AGGREGATE_SCHEMA,
        "status": "COMPLETE",
        "inventory": {"complete_cartesian_product": True},
        "ranking_contract": {
            "natural_kmeans_used": False, "patient_heldout_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "best_usable_anchor": "joint",
        "usable_two_mode_anchor_ids": ["joint"],
    })
    figure2 = artifact / "inputs/figure2.json"
    _write(figure2, {"interictal_field": {}})
    return repository, artifact, config_path, aggregate_path, figure2


def test_rev16_postselection_uses_fresh_pool_without_reranking(tmp_path):
    repository, artifact, config, aggregate, figure2 = _inputs(tmp_path)
    payload = prepare.build_config(
        candidate_config_path=config, candidate_aggregate_path=aggregate,
        artifact_root=artifact, repository_root=repository,
        figure2_field_path=figure2,
    )
    assert payload["schema_id"] == prepare.OUTPUT_SCHEMA
    assert payload["selected_candidate"]["candidate_id"] == "joint"
    assert payload["network_seeds"] == [2351, 2352, 2353]
    assert payload["boundaries"]["natural_kmeans_used_for_field_selection"] is False
    assert payload["boundaries"]["patient_heldout_used"] is False
    assert payload["boundaries"]["EE_EtoI_ZM"] == "off"


def test_rev16_postselection_rejects_absent_anchor(tmp_path):
    repository, artifact, config, aggregate, figure2 = _inputs(tmp_path)
    data = json.loads(aggregate.read_text())
    data["best_usable_anchor"] = None
    data["usable_two_mode_anchor_ids"] = []
    _write(aggregate, data)
    with pytest.raises(RuntimeError, match="no training-qualified"):
        prepare.build_config(
            candidate_config_path=config, candidate_aggregate_path=aggregate,
            artifact_root=artifact, repository_root=repository,
            figure2_field_path=figure2,
        )


def test_rev16_postselection_rejects_kmeans_in_candidate_ranking(tmp_path):
    repository, artifact, config, aggregate, figure2 = _inputs(tmp_path)
    data = json.loads(aggregate.read_text())
    data["ranking_contract"]["natural_kmeans_used"] = True
    _write(aggregate, data)
    with pytest.raises(RuntimeError, match="forbidden boundary"):
        prepare.build_config(
            candidate_config_path=config, candidate_aggregate_path=aggregate,
            artifact_root=artifact, repository_root=repository,
            figure2_field_path=figure2,
        )


def test_rev16_audit_wrapper_switches_schema_and_worker_status(monkeypatch, tmp_path):
    seen = {}

    def fake_audit(*, config_path, artifact_root):
        seen["schema"] = rev15_audit.EXPECTED_CONFIG_SCHEMA
        seen["worker_status"] = rev15_audit.WORKER_STATUS
        return {
            "status": "NODE_POSTSELECTION_REJECTED", "candidate_id": "joint",
            "acceptance": {"accepted": False},
        }

    original_schema = rev15_audit.EXPECTED_CONFIG_SCHEMA
    original_status = rev15_audit.WORKER_STATUS
    monkeypatch.setattr(rev15_audit, "audit", fake_audit)
    payload = audit.audit(config_path=tmp_path / "x", artifact_root=tmp_path)
    assert seen == {
        "schema": prepare.OUTPUT_SCHEMA, "worker_status": worker.WORKER_STATUS,
    }
    assert payload["status"] == "NODE_POSTSELECTION_REJECTED"
    assert rev15_audit.EXPECTED_CONFIG_SCHEMA == original_schema
    assert rev15_audit.WORKER_STATUS == original_status


def test_rev16_figure_wrapper_uses_rev16_identity_and_worker(monkeypatch, tmp_path):
    seen = {}

    def fake_render(**kwargs):
        seen.update({
            "revision": rev15_figure.REVISION_ID,
            "description": rev15_figure.NODE_DESCRIPTION,
            "role": rev15_figure.SCIENTIFIC_ROLE,
            "status": rev15_figure.RENDER_STATUS,
            "worker": rev15_figure.post.WORKER_STATUS,
        })
        return {"status": rev15_figure.RENDER_STATUS}

    old = (
        rev15_figure.REVISION_ID, rev15_figure.NODE_DESCRIPTION,
        rev15_figure.SCIENTIFIC_ROLE, rev15_figure.RENDER_STATUS,
        rev15_figure.post.WORKER_STATUS,
    )
    monkeypatch.setattr(rev15_figure, "render", fake_render)
    payload = figure.render(
        config_path=tmp_path / "config.json", audit_path=tmp_path / "audit.json",
        artifact_root=tmp_path,
    )
    assert payload["status"] == figure.RENDER_STATUS
    assert seen == {
        "revision": "rev16", "description": figure.NODE_DESCRIPTION,
        "role": figure.SCIENTIFIC_ROLE, "status": figure.RENDER_STATUS,
        "worker": worker.WORKER_STATUS,
    }
    assert (
        rev15_figure.REVISION_ID, rev15_figure.NODE_DESCRIPTION,
        rev15_figure.SCIENTIFIC_ROLE, rev15_figure.RENDER_STATUS,
        rev15_figure.post.WORKER_STATUS,
    ) == old
