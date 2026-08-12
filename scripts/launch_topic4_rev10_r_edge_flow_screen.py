"""Launch the rev10-R fit screen after a measured-RSS sentinel."""
from __future__ import annotations

import argparse
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
WORKER = ROOT / "scripts/run_topic4_rev10_r_edge_flow_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_rev10_r_edge_flow_screen.py"
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path):
    import hashlib
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
        payload.get("status") == "REV10R_EDGE_FLOW_WORKER_COMPLETE"
        and payload.get("candidate", {}).get("candidate_id") == job["candidate"]
        and payload.get("seed") == job["seed"]
        and payload.get("config", {}).get("sha256") == config_sha
        and payload.get("manifest", {}).get("sha256") == manifest_sha
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
        and payload.get("arrays", {}).get("sha256") == _sha256(job["npz"])
    )


def _available_memory_kib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _peak_rss_kib(log_path):
    text = Path(log_path).read_text()
    matches = re.findall(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if not matches:
        raise RuntimeError(f"sentinel RSS is absent from {log_path}")
    return int(matches[-1])


def _memory_bounded_workers(config, peak_rss_kib, available_kib):
    minimum = float(config["execution"][
        "minimum_available_memory_gib_per_screen_worker"
    ]) * 1024 ** 2
    budget_per_worker = max(1.5 * int(peak_rss_kib), minimum)
    # Use at most half of currently available RAM; the rest remains for the
    # OS cache and unrelated user work. Every worker also has a 24 GiB cgroup.
    by_memory = max(1, int(math.floor(0.5 * available_kib / budget_per_worker)))
    return min(int(config["execution"]["screen_max_workers"]), by_memory)


def _launch(job, *, config_path, commit, unit_prefix):
    unit = (
        f"{unit_prefix}-{job['candidate']}-s{job['seed']}-{commit[:8]}"
    )
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}",
        "--property=Type=exec", "--property=MemoryMax=24G",
        "--property=MemoryHigh=20G", f"--working-directory={ROOT}",
        f"--setenv=REV10R_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"rev10-R {job['candidate']} seed={job['seed']}", commit[:8],
        str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path), "--candidate-id", job["candidate"],
        "--seed", str(job["seed"]), "--expected-commit", commit,
        "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def _wait_one(job, *, config_sha, manifest_sha, commit, wait_seconds):
    while True:
        if _complete(
                job, config_sha=config_sha, manifest_sha=manifest_sha,
                commit=commit):
            return
        if _state(job["status"]) == "FAILED":
            raise RuntimeError(
                f"sentinel failed: {job['candidate']} seed {job['seed']}"
            )
        time.sleep(wait_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--commit", required=True)
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--unit-prefix", default="topic4-rev10r-screen")
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
    output_root = ROOT / config["output_root"]
    manifest_path = output_root / "candidate_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError("graph-basis controller has not frozen the library")
    manifest = json.loads(manifest_path.read_text())
    config_sha, manifest_sha = _sha256(config_path), _sha256(manifest_path)
    if manifest.get("config", {}).get("sha256") != config_sha:
        raise RuntimeError("candidate manifest uses another config")
    candidates = [
        row["candidate_id"] for row in manifest["candidate_set"]["candidates"]
    ]
    seeds = list(map(int, config["search"]["fit_network_seeds"]))
    worker_dir, run_dir = output_root / "workers", output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        for seed in seeds:
            stem = worker_dir / f"{candidate}_seed_{seed}"
            jobs.append({
                "candidate": candidate, "seed": seed,
                "json": stem.with_suffix(".json"),
                "npz": stem.with_suffix(".npz"),
                "status": run_dir / f"{candidate}_seed_{seed}.status",
                "log": run_dir / f"{candidate}_seed_{seed}.log",
            })
    completed = [
        job for job in jobs if _complete(
            job, config_sha=config_sha, manifest_sha=manifest_sha, commit=commit,
        )
    ]
    pending = [job for job in jobs if job not in completed]
    wait_seconds = float(
        args.wait_seconds or config["execution"]["wait_seconds"]
    )

    memory_audit_path = output_root / "screen_memory_audit.json"
    if pending:
        sentinel = next(
            (job for job in pending if job["candidate"] != "edge_noop"),
            pending[0],
        )
        _launch(
            sentinel, config_path=config_path, commit=commit,
            unit_prefix=f"{args.unit_prefix}-sentinel",
        )
        print(json.dumps({
            "progress": "sentinel_launched",
            "candidate": sentinel["candidate"], "seed": sentinel["seed"],
        }), flush=True)
        _wait_one(
            sentinel, config_sha=config_sha, manifest_sha=manifest_sha,
            commit=commit, wait_seconds=wait_seconds,
        )
        completed.append(sentinel)
        pending.remove(sentinel)
        peak_rss = _peak_rss_kib(sentinel["log"])
        available = _available_memory_kib()
        maximum = _memory_bounded_workers(config, peak_rss, available)
        memory_audit = {
            "status": "REV10R_SCREEN_MEMORY_SENTINEL_COMPLETE",
            "sentinel": {
                "candidate_id": sentinel["candidate"], "seed": sentinel["seed"],
                "peak_rss_kib": peak_rss,
                "log": str(sentinel["log"]), "log_sha256": _sha256(sentinel["log"]),
            },
            "mem_available_kib_after_sentinel": available,
            "available_memory_fraction_budgeted": 0.5,
            "minimum_budget_per_worker_gib": config["execution"][
                "minimum_available_memory_gib_per_screen_worker"
            ],
            "selected_max_workers": maximum,
            "cgroup_memory_high_gib": 20,
            "cgroup_memory_max_gib": 24,
            "config_sha256": config_sha, "manifest_sha256": manifest_sha,
            "commit": commit,
        }
        memory_audit_path.write_text(json.dumps(memory_audit, indent=2))
    else:
        memory_audit = json.loads(memory_audit_path.read_text())
        maximum = int(memory_audit["selected_max_workers"])

    active, failed = [], []

    def refresh():
        nonlocal active
        remaining = []
        for job in active:
            if _complete(
                    job, config_sha=config_sha, manifest_sha=manifest_sha,
                    commit=commit):
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
            _launch(
                job, config_path=config_path, commit=commit,
                unit_prefix=args.unit_prefix,
            )
            active.append(job)
            print(json.dumps({
                "progress": "launched", "candidate": job["candidate"],
                "seed": job["seed"], "active": len(active),
                "pending": len(pending), "max_workers": maximum,
            }), flush=True)
        if active:
            time.sleep(wait_seconds)
    refresh()
    if failed:
        raise RuntimeError(f"{len(failed)} rev10-R screen worker(s) failed")
    subprocess.run([
        str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
        "--expected-commit", commit,
    ], cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV})
    subprocess.run([
        "notify-send", "Topic 4 rev10-R",
        f"Fit screen completed: {len(completed)}/{len(jobs)}; workers={maximum}",
    ], check=False)
    print(json.dumps({
        "status": "REV10R_EDGE_FLOW_FIT_SCREEN_COMPLETE",
        "completed": len(completed), "total": len(jobs),
        "max_workers": maximum,
        "memory_audit": str(memory_audit_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
