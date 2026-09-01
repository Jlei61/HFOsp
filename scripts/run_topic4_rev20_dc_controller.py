#!/usr/bin/env python3
"""Resource-aware persistent controller for rev20-DC worker phases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
WORKER_STATUS = "REV12ND_NODE_WORKER_COMPLETE"


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
    raise RuntimeError("MemAvailable is unavailable")


def _active(unit: str) -> bool:
    return subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", f"{unit}.service"],
        check=False,
    ).returncode == 0


def _artifact_complete(job: dict) -> bool:
    if not job["json"].is_file() or not job["npz"].is_file():
        return False
    try:
        payload = json.loads(job["json"].read_text())
        return (
            payload.get("status") == WORKER_STATUS
            and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
        )
    except (OSError, ValueError, KeyError):
        return False


def _state(job: dict, commit: str) -> str:
    if _artifact_complete(job):
        return "complete"
    if _active(job["unit"]):
        return "running"
    if job["status"].is_file():
        status = job["status"].read_text().strip()
        if f"commit={commit}" not in status:
            return "pending"
        if status.startswith("FAILED"):
            return "failed"
        if status.startswith("SUCCESS"):
            return "invalid_artifact"
        if status.startswith("RUNNING"):
            return "orphaned"
    return "pending"


def _unit(phase: str, candidate_id: str, seed: int, commit: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:8]
    return f"codex-t4-r20dc-{phase}-{digest}-s{seed}-{commit[:8]}"


def _phase_candidates(phase: str, manifest: dict, output_root: Path) -> list[dict]:
    if phase == "canary":
        return [row for row in manifest["candidates"] if row["is_reference"]]
    if phase == "screen":
        return list(manifest["candidates"])
    if phase == "confirmation":
        selection = json.loads((output_root / "selected_candidates.json").read_text())
        selected = set(selection["candidate_ids"])
        rows = [row for row in manifest["candidates"] if row["candidate_id"] in selected]
        if {row["candidate_id"] for row in rows} != selected:
            raise RuntimeError("confirmation selection is outside frozen manifest")
        return rows
    raise ValueError(f"unknown phase {phase}")


def _jobs(phase: str, config: dict, manifest: dict, output_root: Path,
          commit: str) -> list[dict]:
    seed_key = {
        "canary": "canary_network_seeds",
        "screen": "fit_network_seeds",
        "confirmation": "confirmation_network_seeds",
    }[phase]
    seeds = [int(seed) for seed in config["search"][seed_key]]
    rows = []
    for candidate in _phase_candidates(phase, manifest, output_root):
        candidate_id = str(candidate["candidate_id"])
        for seed in seeds:
            stem = output_root / phase / "workers" / f"{candidate_id}_seed_{seed}"
            log_root = output_root / phase / "run_logs" / "workers"
            rows.append({
                "candidate_id": candidate_id,
                "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": log_root / f"{candidate_id}_seed_{seed}.status",
                "log": log_root / f"{candidate_id}_seed_{seed}.log",
                "unit": _unit(phase, candidate_id, seed, commit),
            })
    return rows


def _launch(job: dict, *, phase: str, config_path: Path, artifact_root: Path,
            commit: str) -> None:
    job["status"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", f"--unit={job['unit']}", "--collect",
        f"--working-directory={ROOT}",
        "--property=OOMPolicy=stop",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1", "--setenv=NUMEXPR_NUM_THREADS=1",
        f"--setenv=REV12ND_SYSTEMD_UNIT={job['unit']}.service",
        "/usr/bin/nohup",
        str(ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"),
        str(job["status"]), str(job["log"]),
        f"rev20-DC {phase} {job['candidate_id']} seed={job['seed']}", commit,
        "/usr/bin/time", "-v", str(PYTHON),
        str(ROOT / "scripts/run_topic4_rev12_node_worker.py"),
        "--config", str(config_path),
        "--candidate-id", job["candidate_id"],
        "--seed", str(job["seed"]),
        "--expected-commit", commit,
        "--artifact-root", str(artifact_root),
        "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--phase", choices=("canary", "screen", "confirmation"),
                        required=True)
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
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != commit:
        raise RuntimeError("controller HEAD differs from expected commit")
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("candidate manifest is stale")
    output_root = artifact_root / config["output_root"]
    jobs = _jobs(args.phase, config, manifest, output_root, commit)
    resources = config["resources"]
    interval = int(resources["monitor_interval_seconds"])
    status_path = output_root / args.phase / "status" / "controller.json"

    while True:
        states = [_state(job, commit) for job in jobs]
        failed = [
            (job, state) for job, state in zip(jobs, states)
            if state in {"failed", "invalid_artifact", "orphaned"}
        ]
        available = _available_gib()
        free_disk = shutil.disk_usage(artifact_root).free / 1024 ** 3
        counts = {name: states.count(name) for name in sorted(set(states))}
        payload = {
            "schema_id": "topic4_rev20_dc_controller_v1",
            "phase": args.phase,
            "git_commit": commit,
            "updated_unix": time.time(),
            "job_count": len(jobs),
            "state_counts": counts,
            "available_memory_gib": available,
            "free_disk_gib": free_disk,
            "status": "RUNNING",
        }
        if failed:
            payload["status"] = "FAILED"
            payload["failed_jobs"] = [
                {"candidate_id": job["candidate_id"], "seed": job["seed"],
                 "state": state} for job, state in failed
            ]
            _atomic_json(status_path, payload)
            subprocess.run([
                "notify-send", "Topic 4 rev20-DC",
                f"{args.phase} failed: {len(failed)} job(s)",
            ], check=False)
            raise RuntimeError("rev20-DC phase contains failed jobs")
        if all(state == "complete" for state in states):
            payload["status"] = "COMPLETE"
            _atomic_json(status_path, payload)
            subprocess.run([
                "notify-send", "Topic 4 rev20-DC",
                f"{args.phase} complete ({len(jobs)} jobs)",
            ], check=False)
            return
        if free_disk < float(resources["minimum_free_disk_gib"]):
            payload["status"] = "PAUSED_LOW_DISK"
            _atomic_json(status_path, payload)
            time.sleep(interval)
            continue
        running = states.count("running")
        memory_slots = max(0, math.floor(
            (available - float(resources["reserved_available_memory_gib"]))
            / float(resources["estimated_worker_gib"])
        ))
        slots = max(0, min(
            int(resources["maximum_workers"]) - running, memory_slots,
        ))
        for job, state in zip(jobs, states):
            if slots <= 0:
                break
            if state == "pending":
                _launch(
                    job, phase=args.phase, config_path=config_path,
                    artifact_root=artifact_root, commit=commit,
                )
                slots -= 1
        _atomic_json(status_path, payload)
        time.sleep(interval)


if __name__ == "__main__":
    main()
