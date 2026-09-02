#!/usr/bin/env python3
"""Persistent resource-aware controller for the rev21 3x4 seed audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
COMPLETE_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable unavailable")


def _active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"{unit}.service"],
        check=False,
    ).returncode == 0


def _complete(job: dict) -> bool:
    if not job["json"].is_file() or not job["npz"].is_file():
        return False
    try:
        payload = json.loads(job["json"].read_text())
        return (payload.get("status") == COMPLETE_STATUS
                and payload.get("topology_seed") == job["topology_seed"]
                and payload.get("dynamics_seed") == job["dynamics_seed"]
                and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"]))
    except (OSError, ValueError, KeyError):
        return False


def _state(job: dict, commit: str) -> str:
    if _complete(job):
        return "complete"
    if _active(job["unit"]):
        return "running"
    if job["status_path"].is_file():
        status = job["status_path"].read_text().strip()
        if f"commit={commit}" not in status:
            return "pending"
        if status.startswith("FAILED"):
            return "failed"
        if status.startswith("RUNNING"):
            return "orphaned"
        if status.startswith("SUCCESS"):
            return "invalid_artifact"
    return "pending"


def _jobs(config: dict, output_root: Path, commit: str) -> list[dict]:
    rows = []
    for topology in config["search"]["canary_network_seeds"]:
        for dynamics in config["search"]["seed_audit_dynamics_seeds"]:
            stem = f"rev21_zm_off_topology_{topology}_dynamics_{dynamics}"
            unit = f"codex-t4-r21-seed-t{topology}-d{dynamics}-{commit[:8]}"
            rows.append({
                "topology_seed": int(topology), "dynamics_seed": int(dynamics),
                "unit": unit,
                "json": output_root / "seed_audit" / "workers" / f"{stem}.json",
                "npz": output_root / "seed_audit" / "workers" / f"{stem}.npz",
                "status_path": output_root / "seed_audit" / "run_logs" / f"{stem}.status",
                "log": output_root / "seed_audit" / "run_logs" / f"{stem}.log",
            })
    return rows


def _launch(job: dict, *, config_path: Path, artifact_root: Path,
            commit: str) -> None:
    job["status_path"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", f"--unit={job['unit']}", "--collect",
        f"--working-directory={ROOT}", "--property=OOMPolicy=stop",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1", "--setenv=NUMEXPR_NUM_THREADS=1",
        f"--setenv=REV12ND_SYSTEMD_UNIT={job['unit']}.service",
        "/usr/bin/nohup", str(ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"),
        str(job["status_path"]), str(job["log"]),
        (f"rev21 seed audit topology={job['topology_seed']} "
         f"dynamics={job['dynamics_seed']}"), commit,
        "/usr/bin/time", "-v", str(PYTHON),
        str(ROOT / "scripts/run_topic4_rev12_node_worker.py"),
        "--config", str(config_path), "--candidate-id", "rev21_zm_off",
        "--seed", str(job["topology_seed"]),
        "--dynamics-seed", str(job["dynamics_seed"]),
        "--expected-commit", commit, "--artifact-root", str(artifact_root),
        "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != commit:
        raise RuntimeError("controller HEAD differs from expected commit")
    parity = artifact_root / config["output_root"] / "parity" / "parity_audit.json"
    if not parity.is_file() or json.loads(parity.read_text()).get("status") != "PASS":
        raise RuntimeError("seed parity audit must PASS before the 3x4 run")
    manifest = json.loads((artifact_root / config["candidate_manifest"]).read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("rev21 candidate manifest is stale")

    output_root = artifact_root / config["output_root"]
    jobs = _jobs(config, output_root, commit)
    resources = config["resources"]
    maximum_workers = int(resources["maximum_workers_seed_audit"])
    worker_gib = float(resources["initial_worker_gib"])
    reserve = float(resources["reserved_available_memory_gib"])
    interval = int(resources["monitor_interval_seconds"])
    status_path = output_root / "seed_audit" / "status" / "controller.json"

    while True:
        states = [_state(job, commit) for job in jobs]
        failed = [(job, state) for job, state in zip(jobs, states)
                  if state in {"failed", "orphaned", "invalid_artifact"}]
        available = _available_gib()
        free_disk = shutil.disk_usage(artifact_root).free / 1024 ** 3
        payload = {
            "schema_id": "topic4_rev21_seed_audit_controller_v1",
            "status": "RUNNING", "git_commit": commit,
            "updated_unix": time.time(), "job_count": len(jobs),
            "state_counts": {name: states.count(name) for name in set(states)},
            "available_memory_gib": available, "free_disk_gib": free_disk,
            "maximum_workers": maximum_workers, "worker_gib": worker_gib,
        }
        if failed:
            payload["status"] = "FAILED"
            payload["failed_jobs"] = [
                {"topology_seed": job["topology_seed"],
                 "dynamics_seed": job["dynamics_seed"], "state": state}
                for job, state in failed
            ]
            _atomic_json(status_path, payload)
            subprocess.run(["notify-send", "Topic 4 rev21",
                            f"seed audit failed ({len(failed)} jobs)"], check=False)
            raise RuntimeError("rev21 seed audit contains failed jobs")
        if all(state == "complete" for state in states):
            payload["status"] = "COMPLETE"
            _atomic_json(status_path, payload)
            subprocess.run(["notify-send", "Topic 4 rev21",
                            "3x4 seed audit complete"], check=False)
            return
        if free_disk < float(resources["minimum_free_disk_gib"]):
            payload["status"] = "PAUSED_LOW_DISK"
            _atomic_json(status_path, payload)
            time.sleep(interval)
            continue
        running = states.count("running")
        slots = min(maximum_workers - running,
                    max(0, math.floor((available - reserve) / worker_gib)))
        for job, state in zip(jobs, states):
            if slots <= 0:
                break
            if state == "pending":
                _launch(job, config_path=config_path, artifact_root=artifact_root,
                        commit=commit)
                slots -= 1
        _atomic_json(status_path, payload)
        time.sleep(interval)


if __name__ == "__main__":
    main()
