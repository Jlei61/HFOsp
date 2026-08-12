"""Launch bounded rev10-R2 spatial edge capacity audits."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER = ROOT / "scripts/build_topic4_rev10_r2_spatial_edge_audit.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path):
    import hashlib
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
        payload.get("worker_status") == "REV10R2_SPATIAL_EDGE_AUDIT_COMPLETE"
        and payload.get("seed") == job["seed"]
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
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--unit-prefix", default="topic4-rev10r2-audit")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True,
    ).strip()
    if head != commit:
        raise RuntimeError(f"launcher commit {commit} is not HEAD {head}")
    config_sha = _sha256(config_path)
    output_root = ROOT / config["output_root"]
    audit_dir, run_dir = output_root / "feature_audit", output_root / "run_logs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in map(int, config["search"]["fit_network_seeds"]):
        jobs.append({
            "seed": seed,
            "json": audit_dir / f"seed_{seed}.json",
            "npz": audit_dir / f"seed_{seed}.npz",
            "status": run_dir / f"feature_audit_seed_{seed}.status",
            "log": run_dir / f"feature_audit_seed_{seed}.log",
        })
    maximum = min(len(jobs), int(config["execution"]["audit_max_workers"]))
    wait_seconds = float(
        args.wait_seconds or config["execution"]["wait_seconds"]
    )
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
        while pending and len(active) < maximum:
            job = pending.pop(0)
            unit = f"{args.unit_prefix}-s{job['seed']}-{commit[:8]}"
            command = [
                "systemd-run", "--user", "--collect", f"--unit={unit}",
                "--property=Type=exec", "--property=MemoryMax=24G",
                "--property=MemoryHigh=20G", f"--working-directory={ROOT}",
                f"--setenv=REV10R2_SYSTEMD_UNIT={unit}",
                *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
                "/usr/bin/nohup", str(MANAGER), str(job["status"]),
                str(job["log"]), f"rev10-R2 feature audit seed={job['seed']}",
                commit[:8], str(PYTHON), str(WORKER),
                "--config", str(config_path), "--seed", str(job["seed"]),
                "--expected-commit", commit,
                "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
            ]
            subprocess.run(command, cwd=ROOT, check=True)
            active.append(job)
            print(json.dumps({
                "progress": "launched", "seed": job["seed"],
                "active": len(active), "pending": len(pending),
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} spatial feature audit worker(s) failed")
    records = [json.loads(job["json"].read_text()) for job in completed]
    scientific_pass = all(
        record["status"] == "REV10R2_SPATIAL_FEATURE_CAPACITY_PASS"
        for record in records
    )
    summary = {
        "status": (
            "REV10R2_SPATIAL_FEATURE_CAPACITY_PASS"
            if scientific_pass else "REV10R2_SPATIAL_FEATURE_CAPACITY_FAIL"
        ),
        "scientific_role": config["scientific_role"],
        "network_seeds": [record["seed"] for record in records],
        "effective_rank_by_seed": {
            str(record["seed"]): record["feature_audit"]["effective_rank"]
            for record in records
        },
        "condition_number_by_seed": {
            str(record["seed"]): record["feature_audit"][
                "covariance_condition_number"
            ] for record in records
        },
        "worker_inputs": [{
            "seed": job["seed"], "json_sha256": _sha256(job["json"]),
            "npz_sha256": _sha256(job["npz"]),
        } for job in completed],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "commit": commit,
    }
    summary_path = output_root / "feature_capacity_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    subprocess.run([
        "notify-send", "Topic 4 rev10-R2",
        f"Spatial edge capacity audit: {summary['status']}",
    ], check=False)
    print(json.dumps(summary, indent=2), flush=True)
    if not scientific_pass:
        raise RuntimeError("spatial edge family lacks frozen effective capacity")


if __name__ == "__main__":
    main()
