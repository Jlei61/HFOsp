#!/usr/bin/env python3
"""Wait for rev11 pathway work, then run the memory-bounded cohort canary."""
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
WORKER = ROOT / "scripts/run_topic4_data_driven_snn_cohort_worker.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_data_driven_snn_cohort_canary.py"
DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_canary_v1.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def _sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    text = path.read_text().strip()
    return text.split(maxsplit=1)[0] if text else "EMPTY"


def _complete(job: dict, *, commit: str) -> bool:
    if (_state(job["status"]) != "SUCCESS" or not job["json"].exists()
            or not job["npz"].exists()):
        return False
    try:
        payload = json.loads(job["json"].read_text())
    except (OSError, json.JSONDecodeError):
        return False
    provenance = payload.get("provenance", {})
    return bool(
        payload.get("status") in {"COMPLETE", "INVALID_RUNAWAY"}
        and payload.get("candidate_id") == job["candidate"]
        and payload.get("seed") == job["seed"]
        and payload.get("output_npz_sha256") == _sha256(job["npz"])
        and provenance.get("expected_git_commit") == commit
        and provenance.get("runtime_modules_match_expected_commit") is True
        and not provenance.get("runtime_modules_dirty")
    )


def _available_memory_kib() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])
    raise RuntimeError("MemAvailable is absent from /proc/meminfo")


def _peak_rss_kib(path: Path) -> int:
    matches = re.findall(
        r"Maximum resident set size \(kbytes\):\s*(\d+)", path.read_text(),
    )
    if not matches:
        raise RuntimeError(f"sentinel RSS is absent from {path}")
    return int(matches[-1])


def _max_workers(config: dict, peak_rss_kib: int) -> tuple[int, int]:
    available = _available_memory_kib()
    minimum = float(config["execution"][
        "minimum_available_memory_gib_per_worker"
    ]) * 1024 ** 2
    budget = max(1.5 * int(peak_rss_kib), minimum)
    by_memory = max(1, int(math.floor(0.5 * available / budget)))
    return min(int(config["execution"]["max_workers"]), by_memory), available


def _launch(job: dict, *, config_path: Path, commit: str, unit_prefix: str) -> str:
    unit = f"{unit_prefix}-{job['candidate']}-s{job['seed']}-{commit[:8]}"
    command = [
        "systemd-run", "--user", "--collect", f"--unit={unit}",
        "--property=Type=exec", "--property=MemoryMax=24G",
        "--property=MemoryHigh=20G", f"--working-directory={ROOT}",
        f"--setenv=TOPIC4_COHORT_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"topic4-cohort {job['candidate']} seed={job['seed']}", commit[:8],
        str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path), "--candidate-id", job["candidate"],
        "--seed", str(job["seed"]), "--expected-commit", commit,
        "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def _wait_for_upstream(config: dict, wait_seconds: float, controller_status: Path) -> None:
    upstream = ROOT / config["execution"]["upstream_wait_status"]
    if not upstream.exists():
        raise RuntimeError(f"upstream pathway status is absent: {upstream}")
    while _state(upstream) == "RUNNING":
        controller_status.write_text(
            f"WAITING_FOR_UPSTREAM path={upstream} checked_at={time.time()}\n"
        )
        time.sleep(wait_seconds)
    if _state(upstream) not in {"COMPLETE", "SUCCESS"}:
        raise RuntimeError(f"upstream pathway did not complete cleanly: {upstream.read_text()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--unit-prefix", default="topic4-ddcohort")
    parser.add_argument("--wait-seconds", type=float)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    commit = subprocess.check_output(
        ["git", "rev-parse", args.commit], cwd=ROOT, text=True,
    ).strip()
    if head != commit:
        raise RuntimeError(f"cohort launcher commit {commit} is not HEAD {head}")
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    controller_status = output_root / "controller.status"
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    _wait_for_upstream(config, wait_seconds, controller_status)

    manifest_path = output_root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("cohort candidate manifest uses another config")
    candidates = [
        row["candidate_id"] for row in manifest["candidate_set"]["candidates"]
    ]
    seeds = [int(seed) for seed in config["search"]["fit_network_seeds"]]
    worker_dir = output_root / "workers"
    run_dir = output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for candidate in candidates:
        for seed in seeds:
            stem = f"{candidate}_seed_{seed}"
            jobs.append({
                "candidate": candidate,
                "seed": seed,
                "json": worker_dir / f"{stem}.json",
                "npz": worker_dir / f"{stem}.npz",
                "status": run_dir / f"{stem}.status",
                "log": run_dir / f"{stem}.log",
            })
    completed = [job for job in jobs if _complete(job, commit=commit)]
    pending = [job for job in jobs if job not in completed]
    controller_status.write_text(
        f"RUNNING commit={commit} total={len(jobs)} completed={len(completed)}\n"
    )

    memory_path = output_root / "screen_memory_audit.json"
    if pending:
        sentinel = pending.pop(0)
        _launch(
            sentinel, config_path=config_path, commit=commit,
            unit_prefix=f"{args.unit_prefix}-sentinel",
        )
        while not _complete(sentinel, commit=commit):
            if _state(sentinel["status"]) == "FAILED":
                raise RuntimeError("cohort memory sentinel failed")
            time.sleep(wait_seconds)
        completed.append(sentinel)
        peak = _peak_rss_kib(sentinel["log"])
        maximum, available = _max_workers(config, peak)
        memory_path.write_text(json.dumps({
            "status": "COHORT_MEMORY_SENTINEL_COMPLETE",
            "sentinel": {
                "candidate_id": sentinel["candidate"],
                "seed": sentinel["seed"],
                "peak_rss_kib": peak,
                "log": str(sentinel["log"].relative_to(ROOT)),
                "log_sha256": _sha256(sentinel["log"]),
            },
            "mem_available_kib_after_sentinel": available,
            "selected_max_workers": maximum,
            "available_memory_fraction_budgeted": 0.5,
            "cgroup_memory_high_gib": 20,
            "cgroup_memory_max_gib": 24,
            "commit": commit,
        }, indent=2))
    else:
        maximum = int(json.loads(memory_path.read_text())["selected_max_workers"])

    active = []
    failed = []
    while pending or active:
        remaining = []
        for job in active:
            if _complete(job, commit=commit):
                completed.append(job)
            elif _state(job["status"]) == "FAILED":
                failed.append(job)
            else:
                remaining.append(job)
        active = remaining
        if failed:
            raise RuntimeError(f"{len(failed)} cohort canary workers failed")
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
        controller_status.write_text(
            f"RUNNING commit={commit} completed={len(completed)} total={len(jobs)} active={len(active)} pending={len(pending)}\n"
        )
        if active:
            time.sleep(wait_seconds)

    subprocess.run(
        [
            str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
            "--expected-commit", commit,
        ],
        cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV},
    )
    selection = json.loads((output_root / "fit_selection.json").read_text())
    controller_status.write_text(
        f"COMPLETE status={selection['status']} workers={len(completed)}/{len(jobs)} commit={commit}\n"
    )
    subprocess.run([
        "notify-send", "Topic 4 data-driven SNN cohort",
        f"{selection['status']}; workers={len(completed)}/{len(jobs)}",
    ], check=False)
    print(json.dumps({
        "status": selection["status"],
        "completed": len(completed),
        "total": len(jobs),
        "max_workers": maximum,
    }, indent=2))


if __name__ == "__main__":
    main()
