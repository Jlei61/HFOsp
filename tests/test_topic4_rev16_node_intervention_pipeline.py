from __future__ import annotations

import pytest

from scripts import aggregate_topic4_rev15_node_intervention as rev15_aggregate
from scripts import aggregate_topic4_rev16_node_intervention as aggregate
from scripts import prepare_topic4_rev15_node_intervention_config as rev15_prepare
from scripts import prepare_topic4_rev16_node_intervention_config as prepare
from scripts import audit_topic4_rev16_node_final_science as final_audit
from scripts import prepare_topic4_rev16_node_final_science_config as final_config
from scripts import run_topic4_rev15_node_intervention_worker as rev15_worker
from scripts import run_topic4_rev16_node_intervention_worker as worker
from scripts import topic4_rev16_node_substrate_adapter as adapter
from scripts import monitor_topic4_rev16_node_intervention as monitor


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
    (tmp_path / "final.json").write_text(
        '{"schema_id":"' + final_config.OUTPUT_SCHEMA + '"}\n'
    )
    (tmp_path / "audit.json").write_text(
        '{"schema_id":"' + final_audit.OUTPUT_SCHEMA + '"}\n'
    )
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
    assert payload["resources"] == {
        "maximum_workers": 3, "numerical_threads_per_worker": 1,
        "safe_peak_rss_gib_per_worker": 16.0,
        "stop_launching_below_available_memory_gib": 80.0,
        "minimum_free_disk_gib": 40.0, "monitor_interval_seconds": 600,
        "long_run_launcher": "systemd-run --user plus nohup",
    }


def test_rev16_intervention_rejects_old_final_audit_schema(tmp_path):
    (tmp_path / "final.json").write_text(
        '{"schema_id":"' + final_config.OUTPUT_SCHEMA + '"}\n'
    )
    (tmp_path / "audit.json").write_text(
        '{"schema_id":"topic4_rev16_node_final_science_audit_v1"}\n'
    )
    with pytest.raises(RuntimeError, match="audit schema"):
        prepare.build_config(
            final_config_path=tmp_path / "final.json",
            final_audit_path=tmp_path / "audit.json",
            repository_root=tmp_path, artifact_root=tmp_path,
        )


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


def _monitor_config(tmp_path):
    return {
        "schema_id": worker.EXPECTED_CONFIG_SCHEMA,
        "output_root": "results/intervention", "candidate_id": "joint",
        "network_seeds": [2351, 2352, 2353],
        "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        "resources": {
            "maximum_workers": 3, "numerical_threads_per_worker": 1,
            "safe_peak_rss_gib_per_worker": 16.0,
            "stop_launching_below_available_memory_gib": 80.0,
            "minimum_free_disk_gib": 40.0, "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
    }


def test_intervention_monitor_preserves_memory_reserve(monkeypatch, tmp_path):
    config = _monitor_config(tmp_path)
    monkeypatch.setattr(monitor, "load_contract", lambda *args, **kwargs: config)
    monkeypatch.setattr(monitor, "classify", lambda **kwargs: {
        "states": {"2351": "pending", "2352": "pending", "2353": "pending"},
        "invalid": [],
    })
    monkeypatch.setattr(
        monitor.psutil, "virtual_memory",
        lambda: type("M", (), {"available": 90 * 2**30})(),
    )
    monkeypatch.setattr(
        monitor.shutil, "disk_usage",
        lambda path: type("D", (), {"free": 100 * 2**30})(),
    )
    monkeypatch.setattr(monitor, "_atomic_json", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        monitor, "_launch",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not launch")),
    )
    result = monitor.tick(
        config_path=tmp_path / "x.json", artifact_root=tmp_path,
        expected_commit="a" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
        worker_cap=3, execute=True,
    )
    assert result["status"] == monitor.RESOURCE_WAIT
    assert result["n_active"] == 0


def test_intervention_monitor_launches_only_frozen_three(monkeypatch, tmp_path):
    config = _monitor_config(tmp_path)
    launched = []
    monkeypatch.setattr(monitor, "load_contract", lambda *args, **kwargs: config)
    monkeypatch.setattr(monitor, "classify", lambda **kwargs: {
        "states": {"2351": "pending", "2352": "pending", "2353": "pending"},
        "invalid": [],
    })
    monkeypatch.setattr(
        monitor.psutil, "virtual_memory",
        lambda: type("M", (), {"available": 200 * 2**30})(),
    )
    monkeypatch.setattr(
        monitor.shutil, "disk_usage",
        lambda path: type("D", (), {"free": 100 * 2**30})(),
    )
    monkeypatch.setattr(monitor, "_atomic_json", lambda *args, **kwargs: None)

    def fake_launch(**kwargs):
        launched.append(kwargs["seed"])
        return f"unit-{kwargs['seed']}"

    monkeypatch.setattr(monitor, "_launch", fake_launch)
    result = monitor.tick(
        config_path=tmp_path / "x.json", artifact_root=tmp_path,
        expected_commit="a" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
        worker_cap=3, execute=True,
    )
    assert launched == [2351, 2352, 2353]
    assert result["n_active"] == 3
    assert result["status"] == monitor.RUNNING


def test_intervention_monitor_rejects_success_status_without_artifact(
    monkeypatch, tmp_path,
):
    config = _monitor_config(tmp_path)
    status = (
        tmp_path / config["output_root"]
        / "run_logs/workers/intervention_seed_2351.status"
    )
    status.parent.mkdir(parents=True)
    status.write_text("SUCCESS exit_code=0\n")
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n")
    monkeypatch.setattr(monitor, "_is_active", lambda unit: False)
    snapshot = monitor.classify(
        config=config, config_path=config_path,
        artifact_root=tmp_path, expected_commit="a" * 40,
        unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
    )
    assert snapshot["states"]["2351"] == "invalid"
    assert "stale status" in snapshot["invalid"][0]
