from __future__ import annotations

from scripts import aggregate_topic4_rev15_node_intervention as rev15_aggregate
from scripts import aggregate_topic4_rev16_node_intervention as aggregate
from scripts import prepare_topic4_rev15_node_intervention_config as rev15_prepare
from scripts import prepare_topic4_rev16_node_intervention_config as prepare
from scripts import run_topic4_rev15_node_intervention_worker as rev15_worker
from scripts import run_topic4_rev16_node_intervention_worker as worker
from scripts import topic4_rev16_node_substrate_adapter as adapter


def test_rev16_intervention_config_uses_fresh_pool(monkeypatch, tmp_path):
    seen = {}

    def fake_build(**kwargs):
        seen["seeds_during_build"] = rev15_prepare.EXPECTED_NETWORK_SEEDS
        return {
            "schema_id": "old", "scientific_role": "old", "output_root": "old",
            "candidate_id": "joint", "network_seeds": [2351, 2352, 2353],
            "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        }

    old = rev15_prepare.EXPECTED_NETWORK_SEEDS
    monkeypatch.setattr(rev15_prepare, "build_config", fake_build)
    payload = prepare.build_config(
        final_config_path=tmp_path / "final.json",
        final_audit_path=tmp_path / "audit.json",
        repository_root=tmp_path, artifact_root=tmp_path,
    )
    assert seen["seeds_during_build"] == [2351, 2352, 2353]
    assert rev15_prepare.EXPECTED_NETWORK_SEEDS == old
    assert payload["schema_id"] == prepare.OUTPUT_SCHEMA
    assert payload["mechanism_freeze"] == {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }


def test_rev16_worker_wraps_schema_status_and_projection_adapter(monkeypatch):
    seen = {}

    def fake_run(**kwargs):
        seen.update({
            "schema": rev15_worker.EXPECTED_CONFIG_SCHEMA,
            "output": rev15_worker.OUTPUT_SCHEMA,
            "status": rev15_worker.WORKER_STATUS,
            "builder": rev15_worker.build_projected_node_substrate,
            "parity": rev15_worker.verify_projection_against_worker,
        })
        return {"status": rev15_worker.WORKER_STATUS}

    old = (
        rev15_worker.EXPECTED_CONFIG_SCHEMA, rev15_worker.OUTPUT_SCHEMA,
        rev15_worker.WORKER_STATUS,
        rev15_worker.build_projected_node_substrate,
        rev15_worker.verify_projection_against_worker,
    )
    monkeypatch.setattr(rev15_worker, "run_worker", fake_run)
    result = worker.run_worker(x=1)
    assert result["status"] == worker.WORKER_STATUS
    assert seen == {
        "schema": worker.EXPECTED_CONFIG_SCHEMA,
        "output": worker.OUTPUT_SCHEMA, "status": worker.WORKER_STATUS,
        "builder": adapter.build_projected_node_substrate,
        "parity": adapter.verify_projection_against_worker,
    }
    assert (
        rev15_worker.EXPECTED_CONFIG_SCHEMA, rev15_worker.OUTPUT_SCHEMA,
        rev15_worker.WORKER_STATUS,
        rev15_worker.build_projected_node_substrate,
        rev15_worker.verify_projection_against_worker,
    ) == old


def test_rev16_aggregate_wraps_freeze_identity(monkeypatch, tmp_path):
    seen = {}

    def fake_aggregate(**kwargs):
        seen.update({
            "schema": rev15_aggregate.EXPECTED_CONFIG_SCHEMA,
            "output": rev15_aggregate.OUTPUT_SCHEMA,
            "freeze": rev15_aggregate.FREEZE_SCHEMA,
            "frozen_status": rev15_aggregate.FROZEN_STATUS,
            "worker": rev15_aggregate.WORKER_STATUS,
        })
        return {"status": rev15_aggregate.FROZEN_STATUS}

    old = (
        rev15_aggregate.EXPECTED_CONFIG_SCHEMA, rev15_aggregate.OUTPUT_SCHEMA,
        rev15_aggregate.FREEZE_SCHEMA, rev15_aggregate.FROZEN_STATUS,
        rev15_aggregate.NOT_SELECTIVE_STATUS, rev15_aggregate.WORKER_STATUS,
    )
    monkeypatch.setattr(rev15_aggregate, "aggregate", fake_aggregate)
    result = aggregate.aggregate(config_path=tmp_path / "x", artifact_root=tmp_path)
    assert result["status"] == aggregate.FROZEN_STATUS
    assert seen == {
        "schema": aggregate.EXPECTED_CONFIG_SCHEMA,
        "output": aggregate.OUTPUT_SCHEMA, "freeze": aggregate.FREEZE_SCHEMA,
        "frozen_status": aggregate.FROZEN_STATUS,
        "worker": worker.WORKER_STATUS,
    }
    assert (
        rev15_aggregate.EXPECTED_CONFIG_SCHEMA, rev15_aggregate.OUTPUT_SCHEMA,
        rev15_aggregate.FREEZE_SCHEMA, rev15_aggregate.FROZEN_STATUS,
        rev15_aggregate.NOT_SELECTIVE_STATUS, rev15_aggregate.WORKER_STATUS,
    ) == old
