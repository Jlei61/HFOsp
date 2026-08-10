"""Launch the frozen L3a surrogate candidate on three selection networks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.launch_topic4_rev9l_component_pair_sobol import (  # noqa: E402
    _complete,
    _state,
    _state_commit,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9l_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev9l_forced_source_worker.py"
REVIEW = ROOT / "scripts/review_topic4_rev9l_l3a_selection_sanity.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_l3a_selection_sanity.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    sanity_path = Path(args.config).resolve()
    sanity = json.loads(sanity_path.read_text())
    for name, record in sanity["inputs"].items():
        if _sha256(record["path"]) != record["sha256"]:
            raise RuntimeError(f"L3a selection input changed: {name}")
    base_path = Path(sanity["inputs"]["component_pair_config"]["path"]).resolve()
    base = json.loads(base_path.read_text())
    base_sha = _sha256(base_path)
    l3a = json.loads(Path(sanity["inputs"]["l3a_fit_surrogate"]["path"]).read_text())
    candidate = sanity["candidate"]
    frozen = l3a["shared_tie_break_candidate"]
    if (frozen["candidate_id"] != candidate["candidate_id"]
            or not all(abs(left - right) <= 1e-12 for left, right in zip(
                frozen["gamma"], candidate["gamma"]))):
        raise RuntimeError("L3a selection candidate differs from frozen fit result")
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")

    output_root = ROOT / base["output_root"] / sanity["output_stage"]
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in sanity["network_seeds"]:
        stem = worker_dir / f"{candidate['candidate_id']}_seed{seed}"
        jobs.append({
            "candidate_id": candidate["candidate_id"],
            "gamma": candidate["gamma"], "seed": int(seed),
            "json": stem.with_suffix(".json"),
            "npz": stem.with_suffix(".npz"),
            "status": run_dir / f"{candidate['candidate_id']}_seed{seed}.status",
            "log": run_dir / f"{candidate['candidate_id']}_seed{seed}.log",
        })
    pending, active, complete, failed = [], [], [], []
    for job in jobs:
        if _complete(job, config_sha=base_sha, commit=commit):
            complete.append(job)
        elif _state(job["status"]) == "RUNNING":
            state, status_commit = _state_commit(job["status"])
            if state != "RUNNING" or status_commit != commit[:8]:
                raise RuntimeError("active L3a worker belongs to another commit")
            active.append(job)
        else:
            pending.append(job)

    wait_seconds = float(sanity["execution"]["wait_seconds"])
    max_workers = int(sanity["execution"]["max_workers"])
    while pending or active:
        remaining = []
        for job in active:
            if _complete(job, config_sha=base_sha, commit=commit):
                complete.append(job)
                continue
            state, status_commit = _state_commit(job["status"])
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
                f"topic4-rev9l-l3a-selection-{job['candidate_id']}-"
                f"s{job['seed']}-{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9L_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                f"L3a selection {job['candidate_id']} seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(base_path), "--arm", "Edge",
                "--seed", str(job["seed"]), "--sources",
                *base["packet"]["formal_sources"], "--packet-fractions",
                str(base["packet"]["frozen_fraction_of_E"]),
                "--component-pair-gamma", *map(str, job["gamma"]),
                "--candidate-id", job["candidate_id"],
                "--expected-commit", commit,
                "--out-json", str(job["json"]),
                "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "candidate_id": job["candidate_id"],
                "seed": job["seed"], "active": len(active),
                "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    if failed:
        raise RuntimeError(f"{len(failed)} L3a selection worker(s) failed")
    subprocess.run([
        str(PYTHON), str(REVIEW), "--config", str(sanity_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True)
    print(json.dumps({
        "status": "L3A_SELECTION_SANITY_LAUNCH_COMPLETE",
        "candidate_id": candidate["candidate_id"],
        "n_success": len(complete), "n_failed": 0,
        "wait_seconds": wait_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
