"""Launch continuous-field candidate/seed workers with bounded concurrency."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev10_sa_dual_shaft_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev10_sa_continuous_field_capacity.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_continuous_field_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def _complete(job, *, config_sha, commit, worker_status):
    if (_state(job["status"]) != "SUCCESS" or not job["json"].exists()
            or not job["npz"].exists()):
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == worker_status
        and payload.get("config", {}).get("sha256") == config_sha
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
    parser.add_argument("--unit-prefix", default="topic4-rev10-sa-continuous")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    support_control = "sa6g_continuous_support" in config
    if support_control:
        assay = config["sa6g_continuous_support"]
        freezer = ROOT / "scripts/freeze_topic4_rev10_sa_continuous_support_candidates.py"
        output_subdir = "continuous_support_capacity"
        worker_status = "SA6G_CONTINUOUS_SUPPORT_WORKER_COMPLETE"
        phase = "SA6G"
    else:
        assay = config["sa6f_continuous_field"]
        freezer = ROOT / "scripts/freeze_topic4_rev10_sa_continuous_field_candidates.py"
        output_subdir = "continuous_field_capacity"
        worker_status = "SA6F_CONTINUOUS_FIELD_WORKER_COMPLETE"
        phase = "SA6F"
    config_sha = _sha256(config_path)
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True,
    ).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not current HEAD {head}")

    output_root = ROOT / config["output_root"] / output_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "candidate_manifest.json"
    subprocess.run([
        str(PYTHON), str(freezer), "--config", str(config_path),
        "--expected-commit", commit, "--out", str(manifest_path),
    ], cwd=ROOT, check=True)
    manifest = json.loads(manifest_path.read_text())
    candidates = [row["candidate_id"]
                  for row in manifest["candidate_set"]["candidates"]]
    seeds = [int(value) for value in assay["network_seeds"]]
    maximum = int(args.max_concurrent or min(
        len(candidates) * len(seeds), config["execution"]["max_workers"],
    ))
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        for seed in seeds:
            stem = worker_dir / f"{candidate}_seed_{seed}"
            jobs.append({
                "candidate": candidate,
                "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": run_dir / f"{candidate}_seed_{seed}.status",
                "log": run_dir / f"{candidate}_seed_{seed}.log",
            })
    pending, active, completed, failed = [], [], [], []
    for job in jobs:
        if _complete(
                job, config_sha=config_sha, commit=commit,
                worker_status=worker_status):
            completed.append(job)
        elif _state(job["status"]) == "RUNNING":
            active.append(job)
        else:
            pending.append(job)

    def refresh():
        nonlocal active
        remaining = []
        for job in active:
            if _complete(
                    job, config_sha=config_sha, commit=commit,
                    worker_status=worker_status):
                completed.append(job)
            elif _state(job["status"]) == "FAILED":
                failed.append(job)
            else:
                remaining.append(job)
        active = remaining

    while pending or active:
        refresh()
        while pending and len(active) < maximum:
            job = pending.pop(0)
            unit = (f"{args.unit_prefix}-{job['candidate']}-s{job['seed']}-"
                    f"{commit[:8]}")
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", f"--working-directory={ROOT}",
                f"--setenv=REV10SA_SYSTEMD_UNIT={unit}",
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]),
                f"{phase} {job['candidate']} seed={job['seed']}", commit[:8],
                str(PYTHON), str(WORKER), "--config", str(config_path),
                "--candidate-id", job["candidate"], "--seed", str(job["seed"]),
                "--expected-commit", commit,
                "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "candidate": job["candidate"],
                "seed": job["seed"], "active": len(active),
                "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} {phase} worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True)
    subprocess.run([
        "notify-send", "Topic 4 rev10-SA",
        f"{phase} continuous-field canary completed: {len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": f"{phase}_CONTINUOUS_FIELD_LAUNCH_COMPLETE",
        "commit": commit, "n_success": len(completed), "n_failed": 0,
        "n_candidates": len(candidates), "n_seeds": len(seeds),
        "max_concurrent": maximum, "wait_seconds": wait_seconds,
    }, indent=2))


if __name__ == "__main__":
    main()
