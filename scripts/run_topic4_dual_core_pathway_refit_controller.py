#!/usr/bin/env python3
"""Resource-bounded controller for the dual-core pathway response surface."""
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_pathway_refit.json"
COMPLETE = "DUAL_CORE_PATHWAY_REFIT_SCREEN_COMPLETE"
RUNNING = "DUAL_CORE_PATHWAY_REFIT_SCREEN_RUNNING"
FAILED = "DUAL_CORE_PATHWAY_REFIT_SCREEN_FAILED"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _available_memory_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 ** 2
    raise RuntimeError("MemAvailable is missing")


def _unit(candidate_id: str, seed: int, commit: str) -> str:
    return f"codex-t4-prefit-{candidate_id}-s{seed}-{commit[:8]}"


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
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "REV10R_EDGE_FLOW_WORKER_COMPLETE"
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
    )


def _state(job: dict) -> str:
    if _complete(job):
        return "complete"
    text = job["status"].read_text().strip() if job["status"].is_file() else ""
    if text.startswith("FAILED"):
        return "failed"
    if _active(job["unit"]):
        return "running"
    if text.startswith("RUNNING"):
        return "orphaned"
    return "pending"


def _launch(job: dict, config_path: Path, manifest: Path, commit: str) -> None:
    job["status"].parent.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", "--collect", "--unit", job["unit"],
        "--working-directory", str(ROOT), "--property=OOMPolicy=stop",
        "--setenv=OMP_NUM_THREADS=1", "--setenv=OPENBLAS_NUM_THREADS=1",
        "--setenv=MKL_NUM_THREADS=1", "--setenv=NUMEXPR_NUM_THREADS=1",
        f"--setenv=REV10R_SYSTEMD_UNIT={job['unit']}.service",
        "/usr/bin/nohup", str(ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"),
        str(job["status"]), str(job["log"]),
        f"dual-core pathway refit {job['candidate_id']} seed={job['seed']}",
        commit, "/usr/bin/time", "-v", sys.executable,
        str(ROOT / "scripts/run_topic4_rev10_r_edge_flow_worker.py"),
        "--config", str(config_path), "--manifest", str(manifest),
        "--candidate-id", job["candidate_id"], "--seed", str(job["seed"]),
        "--expected-commit", commit, "--out-json", str(job["json"]),
        "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def run(config_path: Path, commit: str) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("pathway-refit manifest is stale")
    jobs = []
    for candidate in manifest["candidate_set"]["candidates"]:
        for seed in manifest["fixed_contract"]["network_seeds"]:
            stem = output_root / "workers" / f"{candidate['candidate_id']}_seed_{seed}"
            jobs.append({
                "candidate_id": candidate["candidate_id"], "seed": int(seed),
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": output_root / "run_logs" / f"{stem.name}.status",
                "log": output_root / "run_logs" / f"{stem.name}.log",
                "unit": _unit(candidate["candidate_id"], int(seed), commit),
            })
    execution = config["execution"]
    status_path = output_root / "controller_status.json"
    while True:
        states = [_state(job) for job in jobs]
        failed = [
            (job, state) for job, state in zip(jobs, states)
            if state in {"failed", "orphaned"}
        ]
        available = _available_memory_gib()
        free_disk = shutil.disk_usage(ROOT).free / 1024.0 ** 3
        payload = {
            "status": FAILED if failed else RUNNING,
            "expected_commit": commit,
            "n_jobs": len(jobs), "n_complete": states.count("complete"),
            "n_running": states.count("running"),
            "n_pending": states.count("pending"), "n_failed": len(failed),
            "available_memory_gib": available, "free_disk_gib": free_disk,
            "updated_unix_seconds": time.time(),
        }
        if failed:
            payload["failed_jobs"] = [
                f"{job['candidate_id']}:{job['seed']}={state}"
                for job, state in failed
            ]
            _atomic_json(status_path, payload)
            return payload
        if states.count("complete") == len(jobs):
            subprocess.run([
                sys.executable,
                str(ROOT / "scripts/aggregate_topic4_dual_core_pathway_refit.py"),
                "--config", str(config_path),
            ], cwd=ROOT, check=True)
            payload["status"] = COMPLETE
            _atomic_json(status_path, payload)
            subprocess.run(
                ["notify-send", "Topic 4 pathway refit", "screen complete"],
                check=False,
            )
            return payload

        slots_by_memory = max(0, math.floor(
            (available - float(execution["minimum_reserved_memory_gib"]))
            / float(execution["safe_peak_rss_gib"])
        ))
        slots = min(
            int(execution["worker_cap"]) - states.count("running"),
            slots_by_memory,
        )
        if free_disk < float(execution["minimum_free_disk_gib"]):
            slots = 0
        warmed_seeds = {
            job["seed"] for job, state in zip(jobs, states) if state == "complete"
        }
        active_seeds = {
            job["seed"] for job, state in zip(jobs, states) if state == "running"
        }
        for job, state in zip(jobs, states):
            if state != "pending" or slots <= 0:
                continue
            if job["seed"] not in warmed_seeds and job["seed"] in active_seeds:
                continue
            _launch(job, config_path, manifest_path, commit)
            active_seeds.add(job["seed"])
            slots -= 1
        _atomic_json(status_path, payload)
        time.sleep(int(execution["monitor_interval_seconds"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    payload = run(args.config, commit)
    print(json.dumps(payload, indent=2))
    if payload["status"] == FAILED:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
