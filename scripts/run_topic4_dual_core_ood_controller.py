#!/usr/bin/env python3
"""Run the dual-core OOD experiment as a resource-bounded phased queue."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
PHASES = ("fit", "selection", "confirmation", "pathway")
MANIFEST_STATUS = "REV16_DUAL_CORE_OOD_PHASE_FROZEN"
WORKER_STATUS = "REV10R_EDGE_FLOW_WORKER_COMPLETE"
CONTROLLER_SCHEMA = "topic4_dual_core_ood_controller_v1"
COMPLETE_STATUS = "DUAL_CORE_OOD_ALL_PHASES_COMPLETE"
FAILED_STATUS = "DUAL_CORE_OOD_QUEUE_FAILED"
RUNNING_STATUS = "DUAL_CORE_OOD_QUEUE_RUNNING"
WAIT_STATUS = "DUAL_CORE_OOD_RESOURCE_WAIT"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def available_memory_gib(meminfo: Path = Path("/proc/meminfo")) -> float:
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024.0 ** 2)
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def launch_capacity(
    available_gib: float, *, reserve_gib: float, peak_gib: float,
    worker_cap: int, active_workers: int,
) -> int:
    memory_slots = max(
        0, math.floor((available_gib - reserve_gib) / peak_gib)
    )
    return max(0, min(worker_cap - active_workers, memory_slots))


def _unit_token(phase: str, candidate_id: str, seed: int, commit: str) -> str:
    digest = hashlib.sha256(candidate_id.encode()).hexdigest()[:8]
    return f"codex-t4-dualcore-{phase}-{digest}-s{seed}-{commit[:8]}"


def _is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"{unit}.service"],
        check=False,
    )
    return result.returncode == 0


def _artifact_complete(json_path: Path, npz_path: Path) -> bool:
    if not json_path.is_file() or not npz_path.is_file():
        return False
    try:
        payload = json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == WORKER_STATUS
        and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
    )


def _job_inventory(
    root: Path, manifest: dict[str, Any], phase: str, commit: str,
) -> list[dict[str, Any]]:
    seeds = list(map(int, manifest["fixed_contract"]["network_seeds"]))
    jobs = []
    for candidate in manifest["candidate_set"]["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        for seed in seeds:
            stem = root / "workers" / f"{candidate_id}_seed_{seed}"
            jobs.append({
                "phase": phase,
                "candidate_id": candidate_id,
                "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": root / "run_logs/workers" / (
                    f"{candidate_id}_seed_{seed}.status"
                ),
                "log": root / "run_logs/workers" / (
                    f"{candidate_id}_seed_{seed}.log"
                ),
                "unit": _unit_token(phase, candidate_id, seed, commit),
            })
    return jobs


def _job_state(job: dict[str, Any]) -> str:
    if _artifact_complete(job["json"], job["npz"]):
        return "complete"
    status_path = job["status"]
    status = status_path.read_text().strip() if status_path.is_file() else ""
    if status.startswith("FAILED"):
        return "failed"
    if _is_active(job["unit"]):
        return "running"
    if status.startswith("SUCCESS"):
        return "invalid_artifact"
    if status.startswith("RUNNING"):
        return "orphaned"
    return "pending"


def _launch_job(
    job: dict[str, Any], *, config_path: Path, manifest_path: Path,
    commit: str,
) -> None:
    job["status"].parent.mkdir(parents=True, exist_ok=True)
    job["json"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", "--collect", "--unit", job["unit"],
        "--property=OOMPolicy=stop",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1", "--setenv=NUMEXPR_NUM_THREADS=1",
        "/usr/bin/nohup", str(ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"),
        str(job["status"]), str(job["log"]),
        f"dual-core {job['phase']} {job['candidate_id']} seed={job['seed']}",
        commit,
        "/usr/bin/time", "-v", sys.executable,
        str(ROOT / "scripts/run_topic4_rev10_r_edge_flow_worker.py"),
        "--config", str(config_path),
        "--manifest", str(manifest_path),
        "--candidate-id", job["candidate_id"],
        "--seed", str(job["seed"]),
        "--expected-commit", commit,
        "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _freeze_phase(config_path: Path, phase: str, commit: str) -> Path:
    config = json.loads(config_path.read_text())
    path = ROOT / config["output_root"] / phase / "candidate_manifest.json"
    if not path.is_file():
        subprocess.run([
            sys.executable, str(ROOT / "scripts/freeze_topic4_dual_core_ood_phase.py"),
            "--config", str(config_path), "--phase", phase,
            "--expected-commit", commit,
        ], cwd=ROOT, check=True)
    manifest = json.loads(path.read_text())
    if manifest.get("status") != MANIFEST_STATUS:
        raise RuntimeError(f"{phase} manifest is not frozen")
    if manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError(f"{phase} manifest/config hash mismatch")
    provenance = manifest.get("provenance", {})
    if provenance.get("git_commit") != commit:
        raise RuntimeError(f"{phase} manifest commit mismatch")
    return path


def _aggregate_phase(config_path: Path, phase: str) -> Path:
    config = json.loads(config_path.read_text())
    path = ROOT / config["output_root"] / phase / "aggregate.json"
    if not path.is_file():
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/aggregate_topic4_dual_core_ood_phase.py"),
            "--config", str(config_path), "--phase", phase,
        ], cwd=ROOT, check=True)
    payload = json.loads(path.read_text())
    if payload.get("status") != "DUAL_CORE_OOD_PHASE_COMPLETE":
        raise RuntimeError(f"{phase} aggregate is incomplete")
    return path


def _status_payload(
    *, state: str, phase: str, jobs: list[dict[str, Any]],
    states: list[str], available_gib: float, free_disk_gib: float,
    commit: str, message: str,
) -> dict[str, Any]:
    return {
        "schema_id": CONTROLLER_SCHEMA,
        "status": state,
        "phase": phase,
        "message": message,
        "expected_commit": commit,
        "updated_unix_seconds": time.time(),
        "available_memory_gib": available_gib,
        "free_disk_gib": free_disk_gib,
        "n_jobs": len(jobs),
        "n_complete": states.count("complete"),
        "n_running": states.count("running"),
        "n_pending": states.count("pending"),
        "n_failed": sum(state in {
            "failed", "invalid_artifact", "orphaned",
        } for state in states),
        "state_counts": {
            state: states.count(state) for state in sorted(set(states))
        },
    }


def run_controller(
    config_path: Path, commit: str, *, execute: bool,
    stop_after_phase: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    execution = config["execution"]
    monitor_seconds = int(execution["monitor_interval_seconds"])
    worker_cap = int(execution["worker_cap"])
    reserve_gib = float(execution["minimum_reserved_memory_gib"])
    peak_gib = float(execution["safe_peak_rss_gib"])
    min_disk_gib = float(execution["minimum_free_disk_gib"])
    output_root = ROOT / config["output_root"]
    status_path = output_root / "status/controller.json"
    for phase in PHASES:
        manifest_path = _freeze_phase(config_path, phase, commit)
        manifest = json.loads(manifest_path.read_text())
        phase_root = output_root / phase
        jobs = _job_inventory(phase_root, manifest, phase, commit)
        while True:
            states = [_job_state(job) for job in jobs]
            available_gib = available_memory_gib()
            free_disk_gib = shutil.disk_usage(ROOT).free / (1024.0 ** 3)
            failed = [
                (job, state) for job, state in zip(jobs, states)
                if state in {"failed", "invalid_artifact", "orphaned"}
            ]
            if failed:
                message = "; ".join(
                    f"{job['candidate_id']}:{job['seed']}={state}"
                    for job, state in failed[:8]
                )
                payload = _status_payload(
                    state=FAILED_STATUS, phase=phase, jobs=jobs,
                    states=states, available_gib=available_gib,
                    free_disk_gib=free_disk_gib, commit=commit,
                    message=message,
                )
                _atomic_json(status_path, payload)
                return payload
            if all(state == "complete" for state in states):
                _aggregate_phase(config_path, phase)
                if stop_after_phase == phase:
                    payload = _status_payload(
                        state=COMPLETE_STATUS, phase=phase, jobs=jobs,
                        states=states, available_gib=available_gib,
                        free_disk_gib=free_disk_gib, commit=commit,
                        message=f"stopped after completed phase {phase}",
                    )
                    _atomic_json(status_path, payload)
                    return payload
                break
            active = states.count("running")
            slots = launch_capacity(
                available_gib, reserve_gib=reserve_gib, peak_gib=peak_gib,
                worker_cap=worker_cap, active_workers=active,
            )
            state = RUNNING_STATUS
            message = "queue active"
            if free_disk_gib < min_disk_gib:
                slots = 0
                state = WAIT_STATUS
                message = "waiting for free disk"
            elif slots == 0 and states.count("pending"):
                state = WAIT_STATUS
                message = "waiting for memory or worker slot"
            if execute and slots:
                for job, job_state in zip(jobs, states):
                    if job_state != "pending" or slots <= 0:
                        continue
                    _launch_job(
                        job, config_path=config_path,
                        manifest_path=manifest_path, commit=commit,
                    )
                    slots -= 1
                states = [_job_state(job) for job in jobs]
                state = RUNNING_STATUS
                message = "launched pending workers"
            payload = _status_payload(
                state=state, phase=phase, jobs=jobs, states=states,
                available_gib=available_gib, free_disk_gib=free_disk_gib,
                commit=commit, message=message,
            )
            _atomic_json(status_path, payload)
            if not execute:
                return payload
            time.sleep(monitor_seconds)
    final_states = [_job_state(job) for job in jobs]
    payload = _status_payload(
        state=COMPLETE_STATUS, phase=PHASES[-1], jobs=jobs,
        states=final_states, available_gib=available_memory_gib(),
        free_disk_gib=shutil.disk_usage(ROOT).free / (1024.0 ** 3),
        commit=commit, message="all phases aggregated",
    )
    _atomic_json(status_path, payload)
    subprocess.run(
        ["notify-send", "Topic 4 dual-core OOD", "all phases completed"],
        check=False,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after-phase", choices=PHASES)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    payload = run_controller(
        args.config, commit, execute=bool(args.execute),
        stop_after_phase=args.stop_after_phase,
    )
    print(json.dumps(payload, indent=2))
    if payload["status"] == FAILED_STATUS:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
