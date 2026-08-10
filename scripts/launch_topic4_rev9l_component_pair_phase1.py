"""Launch bounded L2 component-pair finite-difference workers."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9l_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev9l_forced_source_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev9l_component_pair_phase1.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_component_pair_edge.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def _complete(job, *, config_sha, commit):
    if (_state(job["status"]) != "SUCCESS" or not job["json"].exists()
            or not job["npz"].exists()):
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("config", {}).get("sha256") == config_sha
        and payload.get("candidate_id") == job["candidate_id"]
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--unit-prefix", default="topic4-rev9l-l2-phase1")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")
    max_concurrent = int(args.max_concurrent or config["execution"]["max_workers"])
    if max_concurrent < 1 or max_concurrent > config["execution"]["max_workers"]:
        raise ValueError("max-concurrent exceeds the frozen L2 budget")
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    output_root = ROOT / config["output_root"] / "phase1"
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in config["component_pair_family"]["phase1_candidates"]:
        for seed in config["network_seeds"]["fit"]:
            stem = worker_dir / f"{candidate['candidate_id']}_seed{seed}"
            jobs.append({
                "candidate_id": candidate["candidate_id"],
                "gamma": candidate["gamma"], "seed": int(seed),
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": run_dir / f"{candidate['candidate_id']}_seed{seed}.status",
                "log": run_dir / f"{candidate['candidate_id']}_seed{seed}.log",
            })
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        if _complete(job, config_sha=config_sha, commit=commit):
            completed.append(job)
        elif _state(job["status"]) == "RUNNING":
            active.append(job)
        else:
            pending.append(job)

    def refresh():
        nonlocal active
        remaining = []
        for job in active:
            if _complete(job, config_sha=config_sha, commit=commit):
                completed.append(job)
            elif _state(job["status"]) == "FAILED":
                failed.append(job)
            else:
                remaining.append(job)
        active = remaining

    while pending or active:
        refresh()
        while pending and len(active) < max_concurrent:
            job = pending.pop(0)
            unit = (f"{args.unit_prefix}-{job['candidate_id'].replace('_', '-')}-"
                    f"s{job['seed']}-{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9L_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                f"L2 {job['candidate_id']} seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(config_path), "--arm", "Edge",
                "--seed", str(job["seed"]), "--sources",
                *config["packet"]["formal_sources"], "--packet-fractions",
                str(config["packet"]["frozen_fraction_of_E"]),
                "--component-pair-gamma", *[str(value) for value in job["gamma"]],
                "--candidate-id", job["candidate_id"],
                "--expected-commit", commit,
                "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
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
    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} L2 phase1 worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9-L",
        f"L2 component-pair phase1 completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": "REV9L_L2_PHASE1_LAUNCH_COMPLETE",
        "commit": commit, "n_success": len(completed), "n_failed": 0,
        "max_concurrent": max_concurrent, "wait_seconds": wait_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
