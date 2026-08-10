"""Launch frozen rev9-L forced-source four-arm workers with coarse waits."""
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
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev9l_forced_source_formal.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev9l_forced_source_formal.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def _slug(arm):
    return arm.lower().replace("+", "_")


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
    parser.add_argument("--unit-prefix", default="topic4-rev9l-forced-formal")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    config_sha = _sha256(config_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")
    seeds = [int(seed) for seed in config["network_seeds"]["fit"]]
    if set(seeds) & {1001, 1002, 1003}:
        raise RuntimeError("formal fit seeds overlap packet-size canary seeds")
    max_concurrent = int(args.max_concurrent or min(
        len(seeds) * len(config["arms"]), config["execution"]["max_workers"]))
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    formal_dir = ROOT / config["output_root"] / "formal_fit"
    worker_dir, run_dir = formal_dir / "workers", formal_dir / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in config["arms"]:
        for seed in seeds:
            stem = worker_dir / f"{_slug(arm)}_seed{seed}"
            jobs.append({
                "arm": arm, "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": run_dir / f"{_slug(arm)}_seed{seed}.status",
                "log": run_dir / f"{_slug(arm)}_seed{seed}.log",
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
            unit = (
                f"{args.unit_prefix}-{_slug(job['arm'])}-s{job['seed']}-{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV9L_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                f"forced formal {job['arm']} seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(config_path), "--arm", job["arm"],
                "--seed", str(job["seed"]), "--sources",
                *config["packet"]["formal_sources"], "--packet-fractions",
                str(config["packet"]["frozen_fraction_of_E"]),
                "--expected-commit", commit,
                "--out-json", str(job["json"]),
                "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "arm": job["arm"],
                "seed": job["seed"], "unit": unit,
                "active": len(active), "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)

    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} forced formal worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True)
    subprocess.run([
        "notify-send", "Topic 4 rev9-L",
        f"forced formal completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": "REV9L_FORCED_FORMAL_LAUNCH_COMPLETE",
        "commit": commit, "n_success": len(completed), "n_failed": 0,
        "max_concurrent": max_concurrent, "wait_seconds": wait_seconds,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
