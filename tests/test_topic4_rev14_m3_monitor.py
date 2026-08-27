import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import monitor_topic4_rev14_m3_canary as monitor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev14_m3_canary.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _fake_contract(tmp_path, commit="a" * 40):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = "output"
    config["candidate_manifest"] = "candidate_manifest.json"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    candidates = [
        {
            "candidate_id": "exact_off" if index == 0 else (
                "uniform_node" if index == 1 else f"m3_{index:02d}"
            ),
            "selection_eligible": index >= 2,
        }
        for index in range(34)
    ]
    manifest = {
        "schema_id": monitor.MANIFEST_SCHEMA,
        "status": monitor.MANIFEST_STATUS,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "candidates": candidates,
        "provenance": {
            "formal_ready": True,
            "git_commit": commit,
            "all_explicit_paths_clean": True,
            "all_explicit_paths_match_expected_commit": True,
        },
    }
    (tmp_path / "candidate_manifest.json").write_text(json.dumps(manifest))
    return config_path, config, manifest


def _job(tmp_path, candidate_id="uniform_node", seed=2321):
    return {
        "candidate_id": candidate_id,
        "seed": seed,
        "selection_eligible": False,
        "json": tmp_path / f"{candidate_id}.json",
        "npz": tmp_path / f"{candidate_id}.npz",
        "log": tmp_path / f"{candidate_id}.log",
        "status": tmp_path / f"{candidate_id}.status",
        "expected_duration_ms": 20000.0,
        "expected_config_sha256": "config-sha",
        "expected_manifest_sha256": "manifest-sha",
    }


def _write_complete(job, commit="b" * 40):
    job["npz"].write_bytes(b"frozen-npz")
    job["json"].write_text(json.dumps({
        "status": monitor.COMPLETE_STATUS,
        "candidate_id": job["candidate_id"],
        "seed": job["seed"],
        "simulation": {
            "duration_ms": 20000.0,
            "runaway_early_stop_ms": None,
        },
        "arrays": {
            "path": str(job["npz"]),
            "sha256": monitor._sha256(job["npz"]),
        },
        "provenance": {
            "git_commit": commit,
            "expected_git_commit": commit,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
            "config_sha256": job["expected_config_sha256"],
            "rev14_explicit_runtime_freeze": {
                "formal_ready": True,
                "git_commit": commit,
            },
            "rev14_manifest_audit": {
                "manifest_read": True,
                "manifest_sha256": job["expected_manifest_sha256"],
            },
        },
    }))


def test_formal_contract_and_all_34_jobs_are_frozen(tmp_path):
    config_path, expected_config, expected_manifest = _fake_contract(tmp_path)
    config, manifest = monitor._load_contract(
        config_path, tmp_path, "a" * 40,
    )
    assert config == expected_config
    assert manifest == expected_manifest
    jobs = monitor._jobs(config, manifest, tmp_path)
    assert len(jobs) == 34
    assert monitor._prewarm_job(jobs)["candidate_id"] == "uniform_node"
    assert all(job["seed"] == 2321 for job in jobs)
    assert all(job["expected_duration_ms"] == 20000.0 for job in jobs)
    assert all(job["json"].parent.name == "workers" for job in jobs)
    assert all(job["log"].parent.name == "workers" for job in jobs)


def test_formal_contract_rejects_manifest_from_another_commit(tmp_path):
    config_path, _, _ = _fake_contract(tmp_path)
    with pytest.raises(RuntimeError, match="another commit"):
        monitor._load_contract(config_path, tmp_path, "f" * 40)


def test_worker_command_is_full_duration_systemd_nohup_and_single_threaded(tmp_path):
    job = _job(tmp_path, "m3_02")
    unit, command = monitor._worker_command(
        job, config_path=CONFIG, artifact_root=ARTIFACT_ROOT,
        expected_commit="c" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
    )
    assert unit.startswith("codex-t4-r14-m3-m3_02-s2321-")
    assert command[:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert str(monitor.MANAGER) in command
    assert str(monitor.TIME) in command and "-v" in command
    assert str(monitor.WORKER) in command
    assert "--duration-ms" not in command
    for key in monitor.NUMERIC_ENV:
        assert f"--setenv={key}=1" in command
    assert command[command.index("--out-json") + 1] == str(job["json"])
    assert command[command.index("--out-npz") + 1] == str(job["npz"])
    assert not any("topic4-ps-cohort-v2p1" in token for token in command)


def test_complete_result_requires_hash_duration_and_clean_rev14_provenance(tmp_path):
    job = _job(tmp_path)
    commit = "b" * 40
    _write_complete(job, commit)
    assert monitor._worker_complete(job, commit)
    assert monitor._full_prewarm_complete(job, commit)
    payload = json.loads(job["json"].read_text())
    payload["provenance"]["rev14_explicit_runtime_freeze"]["formal_ready"] = False
    job["json"].write_text(json.dumps(payload))
    assert not monitor._worker_complete(job, commit)


def test_early_stopped_worker_cannot_supply_full_prewarm_rss(tmp_path):
    job = _job(tmp_path)
    commit = "b" * 40
    _write_complete(job, commit)
    payload = json.loads(job["json"].read_text())
    payload["simulation"]["runaway_early_stop_ms"] = 6123.0
    job["json"].write_text(json.dumps(payload))
    assert monitor._worker_complete(job, commit)
    assert not monitor._full_prewarm_complete(job, commit)


def test_complete_result_is_skipped_but_partial_artifact_fails_closed(
    tmp_path, monkeypatch,
):
    complete = _job(tmp_path, "uniform_node")
    _write_complete(complete)
    assert monitor._job_state(
        complete, "b" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
    ) == "complete"
    partial = _job(tmp_path, "m3_partial")
    partial["npz"].write_bytes(b"partial")
    assert monitor._job_state(
        partial, "b" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
    ) == "invalid_artifact"
    partial["status"].write_text("SUCCESS exit_code=0\n")
    assert monitor._job_state(
        partial, "b" * 40, unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
    ) == "failed"


def test_full_prewarm_rss_is_inflated_and_default_9_can_explicitly_raise_to_10(
    tmp_path,
):
    log = tmp_path / "uniform.log"
    log.write_text(
        "Maximum resident set size (kbytes): 100\n"
        "Maximum resident set size (kbytes): 14680064\n"
    )
    peak = monitor._peak_rss_kib(log)
    safe = monitor._safe_peak_rss_kib(peak, 1.2)
    assert peak == 14680064
    assert safe == 17616077
    assert monitor._worker_capacity(
        available_gib=230.0, safe_peak_rss_kib=safe,
        reserve_gib=64.0, requested_cap=9,
        recommended_cap=9, hard_cap=10,
    ) == 9
    assert monitor._worker_capacity(
        available_gib=240.0, safe_peak_rss_kib=safe,
        reserve_gib=64.0, requested_cap=10,
        recommended_cap=9, hard_cap=10,
    ) == 10
    assert monitor._worker_capacity(
        available_gib=63.0, safe_peak_rss_kib=safe,
        reserve_gib=64.0, requested_cap=9,
        recommended_cap=9, hard_cap=10,
    ) == 0
    with pytest.raises(ValueError):
        monitor._worker_capacity(
            available_gib=230.0, safe_peak_rss_kib=safe,
            reserve_gib=64.0, requested_cap=11,
            recommended_cap=9, hard_cap=10,
        )


def test_oom_patterns_are_detected_from_log(tmp_path):
    job = _job(tmp_path)
    job["log"].write_text("Command terminated by signal 9\n")
    assert monitor._contains_oom(job)


def test_resource_thresholds_hold_then_emergency_stop_without_crossing_units():
    resources = json.loads(CONFIG.read_text())["resources"]
    assert monitor._resource_decision(
        available_gib=63.9, disk_gib=50.0, oom=False, resources=resources,
    ) == "hold"
    assert monitor._resource_decision(
        available_gib=80.0, disk_gib=39.9, oom=False, resources=resources,
    ) == "hold"
    assert monitor._resource_decision(
        available_gib=47.9, disk_gib=50.0, oom=False, resources=resources,
    ) == "emergency_stop"
    assert monitor._resource_decision(
        available_gib=80.0, disk_gib=34.9, oom=False, resources=resources,
    ) == "emergency_stop"
    assert monitor._resource_decision(
        available_gib=80.0, disk_gib=50.0, oom=True, resources=resources,
    ) == "emergency_stop"
    assert monitor._resource_decision(
        available_gib=64.0, disk_gib=40.0, oom=False, resources=resources,
    ) == "launch"


def test_emergency_stop_only_targets_exact_rev14_prefix(monkeypatch):
    listing = (
        "codex-t4-r14-m3-a.service loaded active running test\n"
        "codex-t4-r14-m3-controller.service loaded active running test\n"
        "topic4-ps-cohort-v2p1-1c90a66d.service loaded active running protected\n"
    )
    monkeypatch.setattr(
        monitor.subprocess, "check_output", lambda *args, **kwargs: listing,
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(monitor.subprocess, "run", fake_run)
    stopped = monitor._stop_rev14_units(monitor.DEFAULT_UNIT_PREFIX)
    assert stopped == [
        "codex-t4-r14-m3-a.service",
    ]
    assert all(command[:3] == ["systemctl", "--user", "stop"] for command in calls)
    assert not any(
        "topic4-ps-cohort-v2p1" in token
        for command in calls for token in command
    )


def test_monitor_is_dry_run_by_default_and_interval_is_600_seconds():
    args = monitor._parse_args([
        "--config", str(CONFIG), "--expected-commit", "HEAD",
    ])
    config = json.loads(CONFIG.read_text())
    assert args.execute is False
    assert args.worker_cap == 9
    assert config["resources"]["monitor_interval_seconds"] == 600
    assert config["resources"]["maximum_workers"] == 10
    assert config["resources"]["recommended_workers_after_full_prewarm"] == 9
    assert config["resources"]["stop_launching_below_available_memory_gib"] == 64
    assert config["resources"]["emergency_stop_below_available_memory_gib"] == 48
    assert config["resources"]["minimum_free_disk_gib"] == 40


def test_dry_run_controller_plans_only_full_prewarm(monkeypatch, tmp_path):
    config_path, config, manifest = _fake_contract(tmp_path)
    monkeypatch.setattr(monitor, "_resolve_commit", lambda value: "a" * 40)
    monkeypatch.setattr(monitor, "_require_clean_commit", lambda value: None)
    monkeypatch.setattr(
        monitor, "_load_contract", lambda *args: (config, manifest),
    )
    monkeypatch.setattr(monitor, "_available_memory_gib", lambda: 230.0)
    monkeypatch.setattr(monitor, "_free_disk_gib", lambda path: 60.0)
    calls = []
    monkeypatch.setattr(
        monitor.subprocess, "run", lambda *args, **kwargs: calls.append(args),
    )
    result = monitor.run_controller(
        config_path=config_path, artifact_root=tmp_path,
        expected_commit="HEAD", unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
        execute=False, worker_cap=9,
    )
    assert result["status"] == "REV14_M3_FULL_PREWARM_RUNNING"
    assert len(result["launched_this_cycle"]) == 1
    assert "uniform_node" in result["launched_this_cycle"][0]
    assert result["commands"][0][:2] == ["systemd-run", "--user"]
    assert calls == []
