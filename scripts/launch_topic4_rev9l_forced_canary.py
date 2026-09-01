"""Launch bounded rev9-L forced-packet canary workers and aggregate once."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9l_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev9l_forced_source_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev9l_packet_canary.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_forced_source.json"


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--unit-prefix", default="topic4-rev9l-forced-canary")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")
    max_concurrent = int(args.max_concurrent or min(
        len(config["network_seeds"]["canary"]),
        config["execution"]["max_workers"]))
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    output_root = ROOT / config["output_root"] / "canary"
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in config["network_seeds"]["canary"]:
        stem = worker_dir / f"node_seed{int(seed)}"
        jobs.append({
            "seed": int(seed),
            "json": stem.with_suffix(".json"),
            "npz": stem.with_suffix(".npz"),
            "status": run_dir / f"node_seed{int(seed)}.status",
            "log": run_dir / f"node_seed{int(seed)}.log",
        })
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        state = _state(job["status"])
        if state == "SUCCESS" and job["json"].exists() and job["npz"].exists():
            completed.append(job)
        elif state == "RUNNING":
            active.append(job)
        else:
            pending.append(job)

    def refresh():
        nonlocal active
        still_active = []
        for job in active:
            state = _state(job["status"])
            if state == "SUCCESS":
                completed.append(job)
            elif state == "FAILED":
                failed.append(job)
            else:
                still_active.append(job)
        active = still_active

    while pending or active:
        refresh()
        while pending and len(active) < max_concurrent:
            job = pending.pop(0)
            unit = f"{args.unit_prefix}-s{job['seed']}-{commit[:8]}"
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9L_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]), f"forced canary Node seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(config_path), "--arm", "Node",
                "--seed", str(job["seed"]), "--sources",
                *config["packet"]["canary_sources"], "--packet-fractions",
                *[str(value) for value in config["packet"]["canary_fractions_of_E"]],
                "--expected-commit", commit,
                "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            job["unit"] = unit
            active.append(job)
            print(json.dumps({
                "progress": "launched", "seed": job["seed"], "unit": unit,
                "active": len(active), "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)

    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} forced canary worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
    ], cwd=ROOT, check=True)
    print(json.dumps({
        "status": "REV9L_FORCED_CANARY_COMPLETE",
        "commit": commit, "n_success": len(completed), "n_failed": 0,
        "wait_seconds": wait_seconds,
    }, indent=2), flush=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9-L",
        f"forced packet canary completed: {len(completed)}/{len(jobs)}",
    ], check=False)


if __name__ == "__main__":
    main()
