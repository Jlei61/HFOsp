"""Launch bounded rev9 four-arm workers through systemd and nohup."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev9_managed_command.sh"
PRODUCER = ROOT / "scripts/run_topic4_rev9_factorial_worker.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9_factorial.json"


def _slug(arm):
    return arm.lower().replace("+", "_")


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--arms", nargs="+",
                        choices=("Null", "Node", "Edge", "Node+Edge"))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--wait-seconds", type=float, default=60.0)
    parser.add_argument("--unit-prefix", default="topic4-rev9-factorial")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    arms = args.arms or config["arms"]
    seeds = args.seeds or config["seeds"]
    max_concurrent = int(args.max_concurrent or config["max_concurrent_workers"])
    if max_concurrent < 1:
        raise ValueError("max-concurrent must be positive")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")

    output_root = (ROOT / config["output_root"]).resolve()
    worker_dir = output_root / "workers"
    run_dir = output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in arms:
        for seed in seeds:
            tag = f"{_slug(arm)}_seed{int(seed)}"
            jobs.append(dict(
                arm=arm, seed=int(seed), tag=tag,
                status=run_dir / f"{tag}.status",
                log=run_dir / f"{tag}.log",
                json=worker_dir / f"{tag}.json",
                npz=worker_dir / f"{tag}.npz"))

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
        nonlocal active, completed, failed
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
            arm, seed, tag = job["arm"], job["seed"], job["tag"]
            unit = f"{args.unit_prefix}-{tag.replace('_', '-')}-{commit[:8]}"
            title = f"rev9 factorial {arm} seed={seed}"
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]), title, commit[:8], str(PYTHON),
                str(PRODUCER), "--config", str(config_path),
                "--arm", arm, "--seed", str(seed),
                "--out-json", str(job["json"]),
                "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            job["unit"] = unit
            active.append(job)
            print(json.dumps(dict(
                progress="launched", arm=arm, seed=seed, unit=unit,
                active=len(active), pending=len(pending))), flush=True)
        if active:
            time.sleep(float(args.wait_seconds))

    refresh()
    summary = dict(
        status=("REV9_FACTORIAL_WORKERS_COMPLETE" if not failed
                else "REV9_FACTORIAL_WORKERS_FAILED"),
        commit=commit, max_concurrent=max_concurrent,
        n_jobs=len(jobs), n_success=len(completed), n_failed=len(failed),
        failed=[dict(arm=row["arm"], seed=row["seed"],
                     status=str(row["status"]), log=str(row["log"]))
                for row in failed])
    print(json.dumps(summary, indent=2), flush=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9",
        (f"factorial workers completed: {len(completed)}/{len(jobs)}"
         if not failed else f"factorial workers failed: {len(failed)}")],
        check=False)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
