#!/usr/bin/env python3
"""Resource-bounded controller for the frozen rev14 M3 replication queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev14_m3_replication as freezer  # noqa: E402
from scripts import monitor_topic4_rev14_m3_canary as base  # noqa: E402
from scripts.run_topic4_rev14_m3_replication_worker import (  # noqa: E402
    WORKER_STATUS,
)


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
WORKER = ROOT / "scripts/run_topic4_rev14_m3_replication_worker.py"
DEFAULT_UNIT_PREFIX = "codex-t4-r14-m3-rep"
CONTROLLER_SCHEMA = "topic4_rev14_m3_replication_controller_v1"
QUEUE_EMERGENCY_STATUS = "REV14_M3_REPLICATION_EMERGENCY_STOPPED"
QUEUE_COMPLETE_STATUS = "REV14_M3_REPLICATION_COMPLETE"
QUEUE_DRAINING_STATUS = "REV14_M3_REPLICATION_DRAINING_AFTER_FAILURE"
QUEUE_FAILED_STATUS = "REV14_M3_REPLICATION_FAILED"
QUEUE_WAIT_STATUS = "REV14_M3_REPLICATION_RESOURCE_WAIT"
QUEUE_RUNNING_STATUS = "REV14_M3_REPLICATION_RUNNING"


def _validate_unit_prefix(prefix: str) -> str:
    token = base._validate_unit_prefix(prefix)
    if not token.startswith(DEFAULT_UNIT_PREFIX):
        raise ValueError(f"replication unit prefix must start with {DEFAULT_UNIT_PREFIX!r}")
    return token


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    resources = config["resources"]
    frozen_resources = {
        "maximum_workers": 10,
        "recommended_workers": 9,
        "measured_canary_peak_rss_kib": 13356224,
        "worker_rss_safety_multiplier": 1.2,
        "stop_launching_below_available_memory_gib": 64,
        "emergency_stop_below_available_memory_gib": 48,
        "minimum_free_disk_gib": 40,
        "monitor_interval_seconds": 600,
        "numerical_threads_per_worker": 1,
    }
    for key, expected in frozen_resources.items():
        if resources.get(key) != expected:
            raise RuntimeError(f"M3 replication resource contract changed: {key}")
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("M3 replication manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("M3 replication manifest is not formally frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("M3 replication manifest schema changed")
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("M3 replication manifest/config mismatch")
    provenance = manifest.get("provenance", {})
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("expected_git_commit") != expected_commit
        or provenance.get("formal_ready") is not True
        or provenance.get("all_explicit_paths_clean") is not True
        or provenance.get("all_explicit_paths_match_expected_commit") is not True
    ):
        raise RuntimeError("M3 replication manifest provenance is not formal")
    selected = config["selection"]["selected_candidate_ids"]
    expected_ids = ["exact_off", *selected]
    observed_ids = [row.get("candidate_id") for row in manifest["candidates"]]
    if observed_ids != expected_ids:
        raise RuntimeError("M3 replication manifest shortlist changed")
    if manifest.get("search", {}).get("active_network_seeds") != [2322, 2323]:
        raise RuntimeError("M3 replication manifest seed pool changed")
    return config, manifest


def _worker_complete(job: dict[str, Any], expected_commit: str) -> bool:
    if not job["json"].is_file() or not job["npz"].is_file():
        return False
    try:
        payload = json.loads(job["json"].read_text())
        arrays = payload.get("arrays", {})
        provenance = payload.get("provenance", {})
        explicit = provenance.get("rev14_explicit_runtime_freeze", {})
        manifest_audit = provenance.get("rev14_manifest_audit", {})
        return bool(
            payload.get("status") == WORKER_STATUS
            and payload.get("candidate_id") == job["candidate_id"]
            and int(payload.get("seed", -1)) == int(job["seed"])
            and float(payload.get("simulation", {}).get("duration_ms"))
            == float(job["expected_duration_ms"])
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("git_commit") == expected_commit
            and int(provenance.get("runtime_modules_match_expected_commit", 0)) == 1
            and int(provenance.get("runtime_modules_dirty", 1)) == 0
            and provenance.get("config_sha256") == job["expected_config_sha256"]
            and explicit.get("formal_ready") is True
            and explicit.get("git_commit") == expected_commit
            and manifest_audit.get("manifest_read") is True
            and manifest_audit.get("manifest_sha256")
            == job["expected_manifest_sha256"]
            and Path(arrays.get("path", "")).resolve() == job["npz"].resolve()
            and arrays.get("sha256") == _sha256(job["npz"])
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _worker_unit(
    job: dict[str, Any], *, expected_commit: str, unit_prefix: str,
) -> str:
    prefix = _validate_unit_prefix(unit_prefix)
    return (
        f"{prefix}-{base._unit_token(job['candidate_id'])}-"
        f"s{int(job['seed'])}-{expected_commit[:8]}"
    )


def _job_state(
    job: dict[str, Any], expected_commit: str, *, unit_prefix: str,
) -> str:
    if _worker_complete(job, expected_commit):
        return "complete"
    unit = _worker_unit(job, expected_commit=expected_commit, unit_prefix=unit_prefix)
    token = base._status_token(job["status"])
    if token == "RUNNING":
        return "active" if base._systemd_unit_active(unit) else "failed"
    if token in {"FAILED", "SUCCESS"}:
        return "failed"
    if job["json"].exists() or job["npz"].exists():
        return "invalid_artifact"
    return "pending"


def _state_counts(
    jobs: list[dict[str, Any]], expected_commit: str, unit_prefix: str,
) -> tuple[dict[str, int], dict[str, str]]:
    states = {
        f"{job['candidate_id']}_seed_{job['seed']}": _job_state(
            job, expected_commit, unit_prefix=unit_prefix,
        )
        for job in jobs
    }
    counts = {
        state: sum(value == state for value in states.values())
        for state in ("complete", "active", "pending", "failed", "invalid_artifact")
    }
    return counts, states


def _worker_command(
    job: dict[str, Any], *, config_path: Path, artifact_root: Path,
    expected_commit: str, unit_prefix: str,
) -> tuple[str, list[str]]:
    unit = _worker_unit(job, expected_commit=expected_commit, unit_prefix=unit_prefix)
    command = [
        "systemd-run", "--user", "--collect", "--quiet", f"--unit={unit}",
        "--property=Type=exec", "--property=Nice=5", "--property=CPUWeight=50",
        f"--working-directory={ROOT}",
        f"--property=StandardOutput=append:{job['log']}",
        f"--property=StandardError=append:{job['log']}",
        f"--setenv=REV14_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in base.NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(base.MANAGER), str(job["status"]), str(job["log"]),
        f"rev14 M3 replication {job['candidate_id']} seed={job['seed']}",
        expected_commit[:8], str(base.TIME), "-v", str(base.PYTHON), str(WORKER),
        "--config", str(config_path), "--candidate-id", str(job["candidate_id"]),
        "--seed", str(job["seed"]), "--expected-commit", expected_commit,
        "--artifact-root", str(artifact_root), "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
    return unit, command


def _launch_worker(
    job: dict[str, Any], *, config_path: Path, artifact_root: Path,
    expected_commit: str, unit_prefix: str,
) -> str:
    job["log"].parent.mkdir(parents=True, exist_ok=True)
    unit, command = _worker_command(
        job, config_path=config_path, artifact_root=artifact_root,
        expected_commit=expected_commit, unit_prefix=unit_prefix,
    )
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def _snapshot(
    *, status: str, jobs: list[dict[str, Any]], expected_commit: str,
    unit_prefix: str, available_gib: float, disk_gib: float,
    safe_peak_rss_kib: int, concurrency: int, launched: list[str],
    stopped: list[str],
) -> dict[str, Any]:
    counts, states = _state_counts(jobs, expected_commit, unit_prefix)
    return {
        "schema_id": CONTROLLER_SCHEMA,
        "status": status,
        "expected_git_commit": expected_commit,
        "unit_prefix": _validate_unit_prefix(unit_prefix),
        "monitor_interval_seconds": 600,
        "resources": {
            "available_memory_gib": available_gib,
            "free_disk_gib": disk_gib,
            "measured_canary_peak_rss_kib": 13356224,
            "safe_peak_rss_kib": safe_peak_rss_kib,
            "worker_concurrency": concurrency,
        },
        "n_jobs": len(jobs),
        **{f"n_{key}": value for key, value in counts.items()},
        "states": states,
        "launched_this_cycle": launched,
        "stopped_this_cycle": stopped,
        "updated_at_epoch": time.time(),
    }


def run_controller(
    *, config_path: Path, artifact_root: Path, expected_commit: str,
    unit_prefix: str, execute: bool, worker_cap: int,
    sleep_fn=time.sleep,
) -> dict[str, Any]:
    expected_commit = base._resolve_commit(expected_commit)
    prefix = _validate_unit_prefix(unit_prefix)
    base._require_clean_commit(expected_commit)
    config, manifest = _load_contract(config_path, artifact_root, expected_commit)
    jobs = base._jobs(config, manifest, artifact_root)
    expected_jobs = len(manifest["candidates"]) * len(
        config["search"]["active_network_seeds"]
    )
    if len(jobs) != expected_jobs:
        raise RuntimeError("M3 replication Cartesian product is incomplete")
    resources = config["resources"]
    hard_cap = int(resources["maximum_workers"])
    recommended = int(resources["recommended_workers"])
    if not 1 <= worker_cap <= hard_cap:
        raise ValueError("worker-cap violates the M3 replication hard cap")
    safe_peak = int(math.ceil(
        float(resources["measured_canary_peak_rss_kib"])
        * float(resources["worker_rss_safety_multiplier"])
    ))
    output_root = artifact_root / str(config["output_root"])
    controller_path = output_root / "status" / "m3_replication_controller.json"

    while True:
        base._require_clean_commit(expected_commit)
        _load_contract(config_path, artifact_root, expected_commit)
        available_gib = base._available_memory_gib()
        disk_gib = base._free_disk_gib(artifact_root / "results")
        counts, states = _state_counts(jobs, expected_commit, prefix)
        failed_jobs = [
            job for job in jobs
            if states[f"{job['candidate_id']}_seed_{job['seed']}"]
            in {"failed", "invalid_artifact"}
        ]
        decision = base._resource_decision(
            available_gib=available_gib, disk_gib=disk_gib,
            oom=any(base._contains_oom(job) for job in failed_jobs),
            resources=resources,
        )
        launched: list[str] = []
        stopped: list[str] = []
        if decision == "emergency_stop":
            stopped = base._stop_rev14_units(prefix) if execute else []
            snapshot = _snapshot(
                status=QUEUE_EMERGENCY_STATUS, jobs=jobs,
                expected_commit=expected_commit, unit_prefix=prefix,
                available_gib=available_gib, disk_gib=disk_gib,
                safe_peak_rss_kib=safe_peak, concurrency=0,
                launched=launched, stopped=stopped,
            )
            if execute:
                base._atomic_json(controller_path, snapshot)
                base._notify("M3 replication stopped on resource/OOM emergency")
            return snapshot
        if counts["complete"] == len(jobs):
            snapshot = _snapshot(
                status=QUEUE_COMPLETE_STATUS, jobs=jobs,
                expected_commit=expected_commit, unit_prefix=prefix,
                available_gib=available_gib, disk_gib=disk_gib,
                safe_peak_rss_kib=safe_peak, concurrency=0,
                launched=launched, stopped=stopped,
            )
            if execute:
                base._atomic_json(controller_path, snapshot)
                base._notify("M3 replication complete: 18/18")
            return snapshot
        if failed_jobs:
            status = (
                QUEUE_DRAINING_STATUS if counts["active"] else QUEUE_FAILED_STATUS
            )
            snapshot = _snapshot(
                status=status, jobs=jobs, expected_commit=expected_commit,
                unit_prefix=prefix, available_gib=available_gib,
                disk_gib=disk_gib, safe_peak_rss_kib=safe_peak,
                concurrency=0, launched=launched, stopped=stopped,
            )
            if execute:
                base._atomic_json(controller_path, snapshot)
            if not counts["active"] or not execute:
                if execute:
                    base._notify("M3 replication failed closed")
                return snapshot
            sleep_fn(float(resources["monitor_interval_seconds"]))
            continue
        concurrency = base._worker_capacity(
            available_gib=available_gib, safe_peak_rss_kib=safe_peak,
            reserve_gib=float(resources["stop_launching_below_available_memory_gib"]),
            requested_cap=worker_cap, recommended_cap=recommended,
            hard_cap=hard_cap,
        )
        slots = max(0, concurrency - counts["active"])
        if decision != "hold":
            pending = [
                job for job in jobs
                if states[f"{job['candidate_id']}_seed_{job['seed']}"] == "pending"
            ]
            for job in pending[:slots]:
                unit = _worker_unit(
                    job, expected_commit=expected_commit, unit_prefix=prefix,
                )
                if execute:
                    _launch_worker(
                        job, config_path=config_path, artifact_root=artifact_root,
                        expected_commit=expected_commit, unit_prefix=prefix,
                    )
                launched.append(unit)
        status = (
            QUEUE_WAIT_STATUS
            if decision == "hold" or concurrency == 0
            else QUEUE_RUNNING_STATUS
        )
        snapshot = _snapshot(
            status=status, jobs=jobs, expected_commit=expected_commit,
            unit_prefix=prefix, available_gib=available_gib,
            disk_gib=disk_gib, safe_peak_rss_kib=safe_peak,
            concurrency=concurrency, launched=launched, stopped=stopped,
        )
        if execute:
            base._atomic_json(controller_path, snapshot)
        else:
            snapshot["commands"] = [
                _worker_command(
                    job, config_path=config_path, artifact_root=artifact_root,
                    expected_commit=expected_commit, unit_prefix=prefix,
                )[1]
                for job in jobs
                if _worker_unit(
                    job, expected_commit=expected_commit, unit_prefix=prefix,
                ) in launched
            ]
            return snapshot
        sleep_fn(float(resources["monitor_interval_seconds"]))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=9)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_controller(
        config_path=args.config.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_commit=args.expected_commit,
        unit_prefix=args.unit_prefix,
        execute=bool(args.execute), worker_cap=int(args.worker_cap),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
