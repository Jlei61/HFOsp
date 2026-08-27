import hashlib
import json
from pathlib import Path

import pytest

from scripts import launch_topic4_rev13_node_zero_sum_workers as launcher
from scripts import monitor_topic4_rev13_node_zero_sum_workers as monitor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev13_node_zero_sum_recovery.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _inputs():
    config = json.loads(CONFIG.read_text())
    manifest = json.loads((ARTIFACT_ROOT / config["candidate_manifest"]).read_text())
    return config, manifest


def test_frozen_phase_jobs_use_one_sentinel_six_canary_and_twelve_fit_jobs():
    config, manifest = _inputs()
    sentinel = monitor._phase_jobs(config, manifest, "sentinel", ARTIFACT_ROOT)
    canary = monitor._phase_jobs(config, manifest, "canary", ARTIFACT_ROOT)
    fit = monitor._phase_jobs(config, manifest, "fit", ARTIFACT_ROOT)
    assert [(row["candidate_id"], row["seed"]) for row in sentinel] == [
        ("exact_off", 2311)
    ]
    assert sentinel[0]["duration_ms"] == 2000.0
    assert sentinel[0]["engineering_run_kind"] == "sentinel"
    assert sentinel[0]["json"].parent.name == "sentinel_workers"
    assert sentinel[0]["status"].parent.name == "sentinel"
    assert sentinel[0]["log"].parent.name == "sentinel"
    assert all(row["duration_ms"] is None for row in canary + fit)
    assert all(row["engineering_run_kind"] is None for row in canary + fit)
    assert all(row["json"].parent.name == "workers" for row in canary + fit)
    assert all(row["status"].parent.name == "workers" for row in canary + fit)
    assert all(row["log"].parent.name == "workers" for row in canary + fit)
    assert len(canary) == 6
    assert {row["seed"] for row in canary} == {2311}
    assert len(fit) == 12
    assert {row["seed"] for row in fit} == {2312, 2313}
    assert {row["candidate_id"] for row in canary} == {
        row["candidate_id"] for row in manifest["candidates"]
    }


def test_config_and_manifest_are_hash_and_order_locked(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["candidate_manifest"] = "candidate_manifest.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    manifest = {
        "status": "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN",
        "schema_id": "topic4_rev13_node_zero_sum_recovery_manifest_v2",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidates": [
            {"candidate_id": row["arm_id"]} for row in config["arms"]
        ],
        "provenance": {"git_commit": "f" * 40},
    }
    (tmp_path / config["candidate_manifest"]).write_text(json.dumps(manifest))
    config, manifest = monitor._validate_inputs(
        config_path, tmp_path, "f" * 40
    )
    assert [row["arm_id"] for row in config["arms"]] == [
        row["candidate_id"] for row in manifest["candidates"]
    ]


def test_peak_rss_parser_uses_last_time_v_record(tmp_path):
    path = tmp_path / "sentinel.log"
    path.write_text(
        "Maximum resident set size (kbytes): 100\n"
        "Maximum resident set size (kbytes): 7340032\n"
    )
    assert monitor._peak_rss_kib(path) == 7340032


def test_worker_capacity_reserves_48_gib_and_honours_cap_14():
    peak = 8 * 1024**2
    assert monitor._worker_capacity(
        available_gib=160.0, peak_rss_kib=peak, reserve_gib=48.0, cap=14,
    ) == 14
    assert monitor._worker_capacity(
        available_gib=100.0, peak_rss_kib=peak, reserve_gib=48.0, cap=14,
    ) == 6
    assert monitor._worker_capacity(
        available_gib=47.0, peak_rss_kib=peak, reserve_gib=48.0, cap=14,
    ) == 0
    assert monitor._launch_slots(
        additional_capacity=6, active=6, cap=14,
    ) == 6
    assert monitor._launch_slots(
        additional_capacity=14, active=8, cap=14,
    ) == 6


def test_worker_command_is_systemd_nohup_time_v_and_single_threaded(tmp_path):
    job = {
        "candidate_id": "zero_sum_c020", "seed": 2312,
        "json": tmp_path / "worker.json", "npz": tmp_path / "worker.npz",
        "status": tmp_path / "worker.status", "log": tmp_path / "worker.log",
        "expected_duration_ms": 20000.0,
    }
    unit, command = monitor._worker_command(
        job, config_path=CONFIG, artifact_root=ARTIFACT_ROOT,
        expected_commit="a" * 40, unit_prefix="codex-test",
    )
    assert unit.startswith("codex-test-zero_sum_c020-s2312-")
    assert command[:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert str(monitor.TIME) in command and "-v" in command
    assert str(monitor.WORKER) in command
    for name in monitor.NUMERIC_ENV:
        assert f"--setenv={name}=1" in command
    assert command[command.index("--candidate-id") + 1] == "zero_sum_c020"
    assert command[command.index("--seed") + 1] == "2312"
    assert "--duration-ms" not in command
    assert "--engineering-run-kind" not in command


def test_sentinel_worker_command_has_frozen_two_second_override(tmp_path):
    job = {
        "candidate_id": "exact_off", "seed": 2311,
        "json": tmp_path / "worker.json", "npz": tmp_path / "worker.npz",
        "status": tmp_path / "worker.status", "log": tmp_path / "worker.log",
        "duration_ms": 2000.0,
        "engineering_run_kind": "sentinel",
        "expected_duration_ms": 20000.0,
    }
    _, command = monitor._worker_command(
        job, config_path=CONFIG, artifact_root=ARTIFACT_ROOT,
        expected_commit="a" * 40, unit_prefix="codex-test",
    )
    assert command[command.index("--duration-ms") + 1] == "2000.0"
    assert command[command.index("--engineering-run-kind") + 1] == "sentinel"


def test_sentinel_success_files_cannot_complete_formal_canary(tmp_path):
    config, manifest = _inputs()
    sentinel = monitor._phase_jobs(config, manifest, "sentinel", tmp_path)[0]
    canary = next(
        row for row in monitor._phase_jobs(config, manifest, "canary", tmp_path)
        if row["candidate_id"] == sentinel["candidate_id"]
        and row["seed"] == sentinel["seed"]
    )
    assert sentinel["json"] != canary["json"]
    assert sentinel["npz"] != canary["npz"]
    assert sentinel["status"] != canary["status"]
    assert sentinel["log"] != canary["log"]

    commit = "f" * 40
    sentinel["json"].parent.mkdir(parents=True)
    sentinel["status"].parent.mkdir(parents=True)
    sentinel["npz"].write_bytes(b"sentinel")
    sentinel["json"].write_text(json.dumps({
        "status": monitor.COMPLETE_STATUS,
        "candidate_id": sentinel["candidate_id"],
        "seed": sentinel["seed"],
        "provenance": {
            "expected_git_commit": commit,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
        },
        "arrays": {
            "path": str(sentinel["npz"]),
            "sha256": monitor._sha256(sentinel["npz"]),
        },
        "simulation": {"duration_ms": 2000.0},
        "execution_duration": {
            "frozen_duration_ms": 20000.0,
            "requested_duration_ms": 2000.0,
            "engineering_run_kind": "sentinel",
            "duration_override_used": True,
        },
    }))
    sentinel["status"].write_text("SUCCESS exit_code=0\n")

    assert monitor._job_state(sentinel, commit) == "complete"
    assert monitor._job_state(canary, commit) == "pending"


def test_launcher_wraps_monitor_in_systemd_and_nohup(tmp_path):
    unit, command = launcher._monitor_command(
        config_path=CONFIG, phase="canary", expected_commit="b" * 40,
        artifact_root=ARTIFACT_ROOT, unit_prefix="codex-test",
        log_path=tmp_path / "monitor.log",
    )
    assert unit == "codex-test-monitor-canary-bbbbbbbb"
    assert command[:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert str(launcher.MONITOR) in command
    assert command[command.index("--phase") + 1] == "canary"
    for name in launcher.NUMERIC_ENV:
        assert f"--setenv={name}=1" in command


def test_monitor_logs_are_isolated_by_run_kind(tmp_path):
    sentinel = launcher._phase_monitor_log_path(tmp_path, "sentinel")
    canary = launcher._phase_monitor_log_path(tmp_path, "canary")
    fit = launcher._phase_monitor_log_path(tmp_path, "fit")
    assert sentinel == tmp_path / "run_logs/sentinel/sentinel_monitor.log"
    assert canary == tmp_path / "run_logs/workers/canary_monitor.log"
    assert fit == tmp_path / "run_logs/workers/fit_monitor.log"


def test_completed_worker_requires_matching_clean_provenance(tmp_path):
    job = {
        "candidate_id": "exact_off", "seed": 2311,
        "json": tmp_path / "worker.json", "npz": tmp_path / "worker.npz",
        "status": tmp_path / "worker.status", "log": tmp_path / "worker.log",
        "expected_duration_ms": 20000.0,
    }
    job["npz"].write_bytes(b"npz")
    job["json"].write_text(json.dumps({
        "status": monitor.COMPLETE_STATUS,
        "candidate_id": "exact_off", "seed": 2311,
        "provenance": {
            "expected_git_commit": "c" * 40,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
        },
        "arrays": {
            "path": str(job["npz"]),
            "sha256": monitor._sha256(job["npz"]),
        },
        "simulation": {"duration_ms": 20000.0},
        "execution_duration": {
            "frozen_duration_ms": 20000.0,
            "requested_duration_ms": None,
            "engineering_run_kind": None,
            "duration_override_used": False,
        },
    }))
    assert monitor._worker_complete(job, "c" * 40)
    payload = json.loads(job["json"].read_text())
    payload["provenance"]["runtime_modules_dirty"] = True
    job["json"].write_text(json.dumps(payload))
    assert not monitor._worker_complete(job, "c" * 40)


def test_status_tokens_do_not_treat_success_without_artifact_as_complete(tmp_path):
    job = {
        "candidate_id": "exact_off", "seed": 2311,
        "json": tmp_path / "worker.json", "npz": tmp_path / "worker.npz",
        "status": tmp_path / "worker.status", "log": tmp_path / "worker.log",
        "expected_duration_ms": 20000.0,
    }
    assert monitor._job_state(job, "d" * 40) == "pending"
    job["status"].write_text("RUNNING pid=1\n")
    assert monitor._job_state(job, "d" * 40) == "active"
    job["status"].write_text("SUCCESS exit_code=0\n")
    assert monitor._job_state(job, "d" * 40) == "failed"


def test_stale_running_status_fails_when_systemd_unit_is_dead(tmp_path, monkeypatch):
    job = {
        "candidate_id": "exact_off", "seed": 2311,
        "json": tmp_path / "worker.json", "npz": tmp_path / "worker.npz",
        "status": tmp_path / "worker.status", "log": tmp_path / "worker.log",
        "expected_duration_ms": 20000.0,
    }
    job["status"].write_text("RUNNING pid=1\n")
    monkeypatch.setattr(monitor, "_systemd_unit_active", lambda unit: False)
    assert monitor._job_state(
        job, "d" * 40, expected_unit="dead.service"
    ) == "failed"


def test_monitor_interval_and_resource_floors_are_frozen():
    config, _ = _inputs()
    resources = config["resources"]
    assert resources["monitor_interval_seconds"] == 600
    assert resources["reserved_available_memory_gib"] == 48
    assert resources["maximum_workers"] == 14
    assert resources["minimum_free_disk_gib"] == 40


def test_parity_pass_is_a_commit_bound_hard_precondition(tmp_path):
    output_root = tmp_path / "output"
    rev13_json = output_root / "parity/exact_off_seed_2291.json"
    rev13_json.parent.mkdir(parents=True)
    commit = "1" * 40
    rev13_json.write_text(json.dumps({
        "provenance": {"expected_git_commit": commit},
    }))
    audit_path = monitor._parity_audit_path(output_root)
    audit_path.write_text(json.dumps({
        "schema_id": monitor.PARITY_SCHEMA,
        "status": "PASS",
        "inputs": {
            "rev13_json": {
                "path": str(rev13_json),
                "sha256": monitor._sha256(rev13_json),
            },
        },
    }))
    assert monitor._require_parity_pass(output_root, commit)["status"] == "PASS"
    with pytest.raises(RuntimeError, match="another commit"):
        monitor._require_parity_pass(output_root, "2" * 40)


def test_fit_requires_complete_noncatastrophic_seed2311_aggregate(tmp_path):
    output_root = tmp_path / "output"
    (output_root / "status").mkdir(parents=True)
    (output_root / "analysis").mkdir()
    (output_root / "aggregate").mkdir()
    commit = "3" * 40
    (output_root / "status/canary_controller.json").write_text(json.dumps({
        "status": "REV13_NODE_ZERO_SUM_QUEUE_COMPLETE",
        "expected_git_commit": commit,
        "n_complete": 6,
    }))
    (output_root / "analysis/model_internal_decision.json").write_text(json.dumps({
        "schema_id": monitor.DECISION_SCHEMA,
        "canary_complete": True,
    }))
    rows = [
        {"runaway": True, "n_returned_evaluable_causal_families": 1}
        for _ in range(6)
    ]
    per_run = output_root / "aggregate/model_internal_per_run.json"
    per_run.write_text(json.dumps({"rows": rows}))
    with pytest.raises(RuntimeError, match="every arm ran away"):
        monitor._require_fit_precondition(output_root, commit)
    rows[0]["runaway"] = False
    per_run.write_text(json.dumps({"rows": rows}))
    monitor._require_fit_precondition(output_root, commit)


def test_capacity_uses_full_duration_parity_peak(tmp_path):
    output_root = tmp_path / "output"
    (output_root / "status").mkdir(parents=True)
    (output_root / "run_logs/parity").mkdir(parents=True)
    (output_root / "status/sentinel_memory_audit.json").write_text(json.dumps({
        "status": "REV13_SENTINEL_MEMORY_COMPLETE",
        "peak_rss_kib": 100,
    }))
    monitor._parity_log_path(output_root).write_text(
        "Maximum resident set size (kbytes): 700\n"
    )
    assert monitor._load_peak_rss(output_root) == 700


def test_low_disk_waits_without_launching_a_new_worker(tmp_path, monkeypatch):
    config, manifest = _inputs()
    config = dict(config)
    config["output_root"] = "rev13-test-output"
    launched = []
    monkeypatch.setattr(
        monitor, "_validate_inputs",
        lambda config_path, artifact_root, expected_commit: (config, manifest)
    )
    monkeypatch.setattr(monitor, "_require_parity_pass", lambda *args: {})
    monkeypatch.setattr(
        monitor, "_job_state", lambda job, commit, **kwargs: "pending"
    )
    monkeypatch.setattr(monitor, "_available_memory_gib", lambda: 200.0)
    monkeypatch.setattr(monitor, "_free_disk_gib", lambda path: 39.0)
    monkeypatch.setattr(
        monitor, "_launch_worker", lambda *args, **kwargs: launched.append(kwargs)
    )
    snapshot = monitor.run_monitor(
        config_path=CONFIG, phase="sentinel", expected_commit="e" * 40,
        artifact_root=tmp_path, unit_prefix="codex-test", once=True,
    )
    assert snapshot["status"] == "REV13_WAITING_FOR_DISK"
    assert launched == []


def test_unknown_phase_is_rejected():
    config, manifest = _inputs()
    with pytest.raises(ValueError, match="unknown phase"):
        monitor._phase_jobs(config, manifest, "confirmation", ARTIFACT_ROOT)
