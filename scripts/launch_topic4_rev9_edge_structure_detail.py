"""Launch bounded edge-structure workers and aggregate after a coarse wait."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev9_edge_structure_detail_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev9_edge_structure_detail.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9_factorial.json"


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--wait-seconds", type=float, default=120.0)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    output = ROOT / (
        "results/topic4_sef_hfo/data_driven_core_field_rev9/edge_structure_detail")
    worker_dir, run_dir = output / "workers", output / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = [{
        "seed": int(seed),
        "json": worker_dir / f"seed{int(seed)}.json",
        "status": run_dir / f"seed{int(seed)}.status",
        "log": run_dir / f"seed{int(seed)}.log",
    } for seed in config["seeds"]]
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        state = _state(job["status"])
        if state == "SUCCESS" and job["json"].exists():
            completed.append(job)
        elif state == "RUNNING":
            active.append(job)
        else:
            pending.append(job)

    def refresh():
        nonlocal active
        remaining = []
        for job in active:
            state = _state(job["status"])
            if state == "SUCCESS" and job["json"].exists():
                completed.append(job)
            elif state == "FAILED":
                failed.append(job)
            else:
                remaining.append(job)
        active = remaining

    while pending or active:
        refresh()
        while pending and len(active) < int(args.max_concurrent):
            job = pending.pop(0)
            unit = f"topic4-rev9-edge-structure-s{job['seed']}-{commit[:8]}"
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]), f"edge structure seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", "config/topic4_rev9_exploratory.json",
                "--factorial-config", str(config_path),
                "--seed", str(job["seed"]), "--alpha", "0.75",
                "--out", str(job["json"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            job["unit"] = unit
            active.append(job)
            print(json.dumps({
                "progress": "launched", "seed": job["seed"], "unit": unit,
                "active": len(active), "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(float(args.wait_seconds))

    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} edge structure worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--input-dir", str(worker_dir),
        "--out", str(output / "edge_structure_detail_summary.json"),
    ], cwd=ROOT, check=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9",
        f"edge structure detail completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": "REV9_EDGE_STRUCTURE_DETAIL_LAUNCH_COMPLETE",
        "n_success": len(completed), "n_failed": 0,
        "wait_seconds": float(args.wait_seconds),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
