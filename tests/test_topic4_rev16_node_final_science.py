from __future__ import annotations

import json

from scripts import audit_topic4_rev15_node_final_science as rev15_audit
from scripts import audit_topic4_rev16_node_final_science as audit
from scripts import prepare_topic4_rev15_node_final_science_config as rev15_prepare
from scripts import prepare_topic4_rev16_node_final_science_config as prepare
from scripts import prepare_topic4_rev16_node_postselection_config as post
from scripts import run_topic4_rev16_joint_m3_m4_candidate_worker as worker


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_rev16_final_config_preserves_one_time_boundary(monkeypatch, tmp_path):
    candidate_config = tmp_path / "candidate.json"
    aggregate = tmp_path / "aggregate.json"
    postselection_config = tmp_path / "postselection.json"
    postselection_audit = tmp_path / "postselection_audit.json"
    rev12 = tmp_path / "rev12.json"
    _write(candidate_config, {
        "schema_id": post.EXPECTED_CANDIDATE_SCHEMA,
        "search": {"active_network_seeds": post.NETWORK_SEEDS},
    })
    _write(aggregate, {"schema_id": post.EXPECTED_AGGREGATE_SCHEMA})
    _write(postselection_config, {"schema_id": post.OUTPUT_SCHEMA})
    _write(postselection_audit, {})
    _write(rev12, {})
    seen = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {
            "schema_id": "old", "scientific_role": "old", "output_root": "old",
            "network_seeds": [2341, 2342, 2343],
            "selected_candidate": {"candidate_id": "joint"},
            "boundaries": {
                "field_reranking_allowed": False,
                "patient_heldout_used_for_field_selection": False,
                "patient_heldout_opened_by_final_audit": True,
                "natural_kmeans_must_already_be_accepted": True,
                "SNN_simulation_run": False, "EE_EtoI_ZM": "off",
            },
        }

    monkeypatch.setattr(rev15_prepare, "build_config", fake_build)
    payload = prepare.build_config(
        candidate_config_path=candidate_config,
        candidate_aggregate_path=aggregate,
        postselection_config_path=postselection_config,
        postselection_audit_path=postselection_audit,
        rev12_config_path=rev12, repository_root=tmp_path,
        artifact_root=tmp_path,
    )
    assert payload["schema_id"] == prepare.OUTPUT_SCHEMA
    assert payload["network_seeds"] == [2351, 2352, 2353]
    assert payload["boundaries"]["field_reranking_allowed"] is False
    assert payload["boundaries"]["patient_heldout_used_for_field_selection"] is False
    assert payload["boundaries"]["EE_EtoI_ZM"] == "off"
    assert seen["robust_config_path"] == candidate_config


def test_rev16_final_audit_switches_worker_contract(monkeypatch, tmp_path):
    seen = {}

    def fake_audit(config_path, artifact_root):
        seen["schema"] = rev15_audit.EXPECTED_CONFIG_SCHEMA
        seen["worker"] = rev15_audit.EXPECTED_WORKER_STATUS
        seen["output"] = rev15_audit.OUTPUT_SCHEMA
        return {"status": "X", "candidate_id": "joint", "decision": {}}

    old = (
        rev15_audit.EXPECTED_CONFIG_SCHEMA,
        rev15_audit.EXPECTED_WORKER_STATUS,
        rev15_audit.OUTPUT_SCHEMA,
    )
    monkeypatch.setattr(rev15_audit, "audit", fake_audit)
    payload = audit.audit(tmp_path / "config.json", tmp_path)
    assert payload["candidate_id"] == "joint"
    assert seen == {
        "schema": prepare.OUTPUT_SCHEMA,
        "worker": worker.WORKER_STATUS,
        "output": audit.OUTPUT_SCHEMA,
    }
    assert (
        rev15_audit.EXPECTED_CONFIG_SCHEMA,
        rev15_audit.EXPECTED_WORKER_STATUS,
        rev15_audit.OUTPUT_SCHEMA,
    ) == old
