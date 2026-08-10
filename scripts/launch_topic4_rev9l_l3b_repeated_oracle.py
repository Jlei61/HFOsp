"""Launch the repeated-event rev9-L L3b fit oracle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9l_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev9l_forced_source_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev9l_l3b_repeated_oracle.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_l3b_repeated_oracle.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING", None
    fields = path.read_text().strip().split()
    commit = next((field.split("=", 1)[1] for field in fields
                   if field.startswith("commit=")), None)
    return (fields[0] if fields else "MISSING"), commit


def _complete(job, *, base_sha, commit):
    state, _ = _state(job["status"])
    if state != "SUCCESS" or not job["json"].exists() or not job["npz"].exists():
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("candidate_id") == job["candidate_id"]
        and int(payload.get("network_seed", -1)) == job["network_seed"]
        and int(payload.get("dynamics_seed", -1)) == job["dynamics_seed"]
        and payload.get("config", {}).get("sha256") == base_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L3b input changed: {name}")
    if (config["inputs"]["parity_canary_npz"]["sha256"]
            != config["inputs"]["parity_reference_npz"]["sha256"]):
        raise RuntimeError("network/dynamics seed parity canary did not pass")
    base_path = Path(config["inputs"]["component_pair_config"]["path"]).resolve()
    base = json.loads(base_path.read_text())
    base_sha = _sha256(base_path)
    fit = json.loads(Path(config["inputs"]["l2_fit_summary"]["path"]).read_text())
    candidates = [
        {"candidate_id": row["candidate_id"], "gamma": row["gamma"]}
        for row in fit["candidates"] if row["score"].get("eligible")
    ]
    if len(candidates) != fit["n_eligible"]:
        raise RuntimeError("L3b eligible candidate count changed")
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")

    output_root = ROOT / base["output_root"] / config["output_stage"]
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        for network_seed in config["network_seeds"]:
            for dynamics_seed in config["dynamics_seeds"]:
                stem = worker_dir / (
                    f"{candidate['candidate_id']}_net{network_seed}_dyn{dynamics_seed}")
                jobs.append({
                    "candidate_id": candidate["candidate_id"],
                    "gamma": candidate["gamma"],
                    "network_seed": int(network_seed),
                    "dynamics_seed": int(dynamics_seed),
                    "json": stem.with_suffix(".json"),
                    "npz": stem.with_suffix(".npz"),
                    "status": run_dir / f"{stem.name}.status",
                    "log": run_dir / f"{stem.name}.log",
                })
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        if _complete(job, base_sha=base_sha, commit=commit):
            completed.append(job)
        else:
            state, status_commit = _state(job["status"])
            if state == "RUNNING":
                if status_commit != commit[:8]:
                    raise RuntimeError("active L3b worker belongs to another commit")
                active.append(job)
            else:
                pending.append(job)

    wait_seconds = float(config["execution"]["wait_seconds"])
    max_workers = int(config["execution"]["max_workers"])
    while pending or active:
        remaining = []
        for job in active:
            if _complete(job, base_sha=base_sha, commit=commit):
                completed.append(job)
                continue
            state, status_commit = _state(job["status"])
            if state == "RUNNING" and status_commit == commit[:8]:
                remaining.append(job)
            elif state == "FAILED" and status_commit == commit[:8]:
                failed.append(job)
            else:
                pending.append(job)
        active = remaining
        while pending and len(active) < max_workers:
            job = pending.pop(0)
            unit = (
                f"topic4-rev9l-l3b-{job['candidate_id']}-"
                f"n{job['network_seed']}-d{job['dynamics_seed']}-{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9L_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                (f"L3b {job['candidate_id']} net={job['network_seed']} "
                 f"dyn={job['dynamics_seed']}"),
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(base_path), "--arm", "Edge",
                "--seed", str(job["network_seed"]),
                "--dynamics-seed", str(job["dynamics_seed"]),
                "--sources", *base["packet"]["formal_sources"],
                "--packet-fractions", str(base["packet"]["frozen_fraction_of_E"]),
                "--component-pair-gamma", *map(str, job["gamma"]),
                "--candidate-id", job["candidate_id"],
                "--expected-commit", commit,
                "--out-json", str(job["json"]),
                "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "completed": len(completed),
                "active": len(active), "pending": len(pending),
                "candidate_id": job["candidate_id"],
                "network_seed": job["network_seed"],
                "dynamics_seed": job["dynamics_seed"],
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    if failed:
        raise RuntimeError(f"{len(failed)} L3b worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9-L",
        f"L3b repeated oracle completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": "L3B_REPEATED_ORACLE_LAUNCH_COMPLETE",
        "n_success": len(completed), "n_failed": 0,
        "n_candidates": len(candidates),
        "max_workers": max_workers, "wait_seconds": wait_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
