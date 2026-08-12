"""Managed D4 launcher with a measured-RSS sentinel and long waits."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
TIME = Path("/usr/bin/time")
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"
WORKER = ROOT / "scripts/run_topic4_rev10_d4_uniform_source_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev10_d4_uniform_source_map.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_uniform_forced_source_map.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _state(path):
    if not path.exists():
        return "MISSING"
    return path.read_text().strip().split(maxsplit=1)[0]


def _complete(job, *, config_sha, manifest_sha, commit):
    if (_state(job["status"]) != "SUCCESS" or not job["json"].exists()
            or not job["npz"].exists()):
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") == "REV10D4_UNIFORM_SOURCE_WORKER_COMPLETE"
        and payload.get("seed") == job["seed"]
        and payload.get("config", {}).get("sha256") == config_sha
        and payload.get("manifest", {}).get("sha256") == manifest_sha
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
    )


def _available_memory_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent")


def _peak_rss_kib(path):
    matches = re.findall(
        r"Maximum resident set size \(kbytes\):\s*(\d+)",
        Path(path).read_text(),
    )
    if not matches:
        raise RuntimeError("D4 sentinel RSS is absent")
    return int(matches[-1])


def _launch(job, *, config_path, commit, prefix):
    unit = f"{prefix}-s{job['seed']}-{commit[:8]}"
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}",
        "--property=Type=exec", "--property=MemoryHigh=20G",
        "--property=MemoryMax=24G", f"--working-directory={ROOT}",
        f"--setenv=REV10D4_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"rev10-D4 uniform source seed={job['seed']}", commit[:8],
        str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path), "--seed", str(job["seed"]),
        "--expected-commit", commit,
        "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _wait(job, *, config_sha, manifest_sha, commit, wait_seconds):
    while True:
        if _complete(
            job, config_sha=config_sha, manifest_sha=manifest_sha,
            commit=commit,
        ):
            return
        if _state(job["status"]) == "FAILED":
            raise RuntimeError(f"D4 worker failed: seed {job['seed']}")
        time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--unit-prefix", default="topic4-rev10d4")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True,
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if commit != head:
        raise RuntimeError("D4 launcher commit must be HEAD")
    root = ROOT / config["output_root"]
    manifest_path = root / "source_grid_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("D4 source grid is not frozen")
    config_sha, manifest_sha = _sha256(config_path), _sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["sha256"] != config_sha:
        raise RuntimeError("D4 manifest uses another config")
    worker_dir, log_dir = root / "workers", root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in map(int, config["network_seeds"]):
        stem = worker_dir / f"uniform_source_seed_{seed}"
        jobs.append({
            "seed": seed,
            "json": stem.with_suffix(".json"),
            "npz": stem.with_suffix(".npz"),
            "status": log_dir / f"uniform_source_seed_{seed}.status",
            "log": log_dir / f"uniform_source_seed_{seed}.log",
        })
    pending = [
        job for job in jobs if not _complete(
            job, config_sha=config_sha, manifest_sha=manifest_sha,
            commit=commit,
        )
    ]
    wait_seconds = float(config["execution"]["wait_seconds"])
    audit_path = root / "memory_audit.json"
    if pending:
        sentinel = pending.pop(0)
        _launch(
            sentinel, config_path=config_path, commit=commit,
            prefix=f"{args.unit_prefix}-sentinel",
        )
        print(json.dumps({
            "progress": "sentinel_launched", "seed": sentinel["seed"],
        }), flush=True)
        _wait(
            sentinel, config_sha=config_sha, manifest_sha=manifest_sha,
            commit=commit, wait_seconds=wait_seconds,
        )
        peak_rss = _peak_rss_kib(sentinel["log"])
        available = _available_memory_kib()
        per_worker = max(
            1.5 * peak_rss,
            float(config["execution"]["minimum_available_memory_gib_per_worker"])
            * 1024 ** 2,
        )
        maximum = min(
            int(config["execution"]["max_workers"]),
            max(1, int(math.floor(0.5 * available / per_worker))),
        )
        audit_path.write_text(json.dumps({
            "status": "REV10D4_MEMORY_SENTINEL_COMPLETE",
            "sentinel_seed": sentinel["seed"],
            "peak_rss_kib": peak_rss,
            "mem_available_kib_after_sentinel": available,
            "selected_max_workers": maximum,
            "available_memory_fraction_budgeted": 0.5,
            "cgroup_memory_high_gib": 20,
            "cgroup_memory_max_gib": 24,
            "config_sha256": config_sha,
            "manifest_sha256": manifest_sha,
            "commit": commit,
        }, indent=2))
    else:
        maximum = int(json.loads(audit_path.read_text())["selected_max_workers"])

    active = []
    while pending or active:
        remaining = []
        for job in active:
            if _complete(
                job, config_sha=config_sha, manifest_sha=manifest_sha,
                commit=commit,
            ):
                continue
            if _state(job["status"]) == "FAILED":
                raise RuntimeError(f"D4 worker failed: seed {job['seed']}")
            remaining.append(job)
        active = remaining
        while pending and len(active) < maximum:
            job = pending.pop(0)
            _launch(
                job, config_path=config_path, commit=commit,
                prefix=args.unit_prefix,
            )
            active.append(job)
            print(json.dumps({
                "progress": "launched", "seed": job["seed"],
                "active": len(active), "pending": len(pending),
                "max_workers": maximum,
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV})
    verdict = json.loads((root / "uniform_source_verdict.json").read_text())
    subprocess.run([
        "notify-send", "Topic 4 rev10-D4",
        f"{verdict['status']}; workers={maximum}",
    ], check=False)
    print(json.dumps({
        "status": verdict["status"],
        "completed": len(jobs), "total": len(jobs),
        "max_workers": maximum,
    }, indent=2))


if __name__ == "__main__":
    main()
