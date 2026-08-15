#!/usr/bin/env python3
"""Drive the staged formal cohort run under measured-RSS memory admission."""
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
WORKER = ROOT / "scripts/run_topic4_data_driven_snn_cohort_formal_worker.py"
FREEZER = ROOT / "scripts/freeze_topic4_data_driven_snn_cohort_formal_candidates.py"
AGGREGATOR = ROOT / "scripts/aggregate_topic4_data_driven_snn_cohort_formal.py"
FIGURE = ROOT / "scripts/paper_figures/plot_topic4_data_driven_snn_cohort.py"
DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
MINIMUM_FREE_GIB = 6.0


def _sha256(path: Path) -> str:
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


def _free_disk_gib() -> float:
    usage = os.statvfs(ROOT)
    return usage.f_bavail * usage.f_frsize / 1024 ** 3


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
        "--property=Type=exec", "--property=MemoryMax=26G",
        "--property=MemoryHigh=22G", f"--working-directory={ROOT}",
        f"--setenv=TOPIC4_COHORT_SYSTEMD_UNIT={unit}",
        *[f"--setenv={key}={value}" for key, value in NUMERIC_ENV.items()],
        "/usr/bin/nohup", str(MANAGER), str(job["status"]), str(job["log"]),
        f"topic4-formal {job['candidate']} seed={job['seed']}", commit[:8],
        str(TIME), "-v", str(PYTHON), str(WORKER),
        "--config", str(config_path), "--candidate-id", job["candidate"],
        "--seed", str(job["seed"]), "--expected-commit", commit,
        "--out-json", str(job["json"]), "--out-npz", str(job["npz"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return unit


def _jobs(output_root: Path, candidates: list[str], seeds: list[int]) -> list[dict]:
    worker_dir = output_root / "workers"
    run_dir = output_root / "run_logs"
    worker_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in candidates:
        for seed in seeds:
            stem = f"{candidate}_seed_{seed}"
            rows.append({
                "candidate": candidate,
                "seed": seed,
                "json": worker_dir / f"{stem}.json",
                "npz": worker_dir / f"{stem}.npz",
                "status": run_dir / f"{stem}.status",
                "log": run_dir / f"{stem}.log",
            })
    return rows


def _wait_for_upstream(config: dict, wait_seconds: float,
                       controller_status: Path) -> None:
    upstream = ROOT / config["execution"]["upstream_wait_status"]
    if not upstream.exists():
        raise RuntimeError(f"upstream status is absent: {upstream}")
    while _state(upstream) in {"RUNNING", "WAITING_FOR_UPSTREAM"}:
        controller_status.write_text(
            f"WAITING_FOR_UPSTREAM path={upstream} checked_at={time.time()}\n"
        )
        time.sleep(wait_seconds)
    if _state(upstream) not in {"COMPLETE", "SUCCESS"}:
        raise RuntimeError(f"upstream did not finish cleanly: {upstream.read_text()}")


def _run_stage(stage: str, jobs: list[dict], *, config: dict, config_path: Path,
               commit: str, unit_prefix: str, wait_seconds: float,
               controller_status: Path, memory_path: Path) -> None:
    completed = [job for job in jobs if _complete(job, commit=commit)]
    pending = [job for job in jobs if job not in completed]
    controller_status.write_text(
        f"RUNNING stage={stage} commit={commit} total={len(jobs)} "
        f"completed={len(completed)}\n"
    )
    if pending and not memory_path.exists():
        sentinel = pending.pop(0)
        _launch(sentinel, config_path=config_path, commit=commit,
                unit_prefix=f"{unit_prefix}-sentinel")
        while not _complete(sentinel, commit=commit):
            if _state(sentinel["status"]) == "FAILED":
                raise RuntimeError("formal memory sentinel failed")
            time.sleep(wait_seconds)
        completed.append(sentinel)
        peak = _peak_rss_kib(sentinel["log"])
        maximum, available = _max_workers(config, peak)
        memory_path.write_text(json.dumps({
            "status": "FORMAL_MEMORY_SENTINEL_COMPLETE",
            "sentinel": {
                "candidate_id": sentinel["candidate"],
                "seed": sentinel["seed"],
                "peak_rss_kib": peak,
                "log": str(sentinel["log"].relative_to(ROOT)),
                "log_sha256": _sha256(sentinel["log"]),
            },
            "mem_available_kib_after_sentinel": available,
            "selected_max_workers": maximum,
            "free_disk_gib_after_sentinel": _free_disk_gib(),
            "commit": commit,
        }, indent=2))
    maximum = int(json.loads(memory_path.read_text())["selected_max_workers"])

    active, failed = [], []
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
            raise RuntimeError(f"{len(failed)} formal workers failed: "
                               f"{[job['candidate'] for job in failed]}")
        free = _free_disk_gib()
        if free < MINIMUM_FREE_GIB and pending:
            controller_status.write_text(
                f"PAUSED_LOW_DISK stage={stage} free_gib={free:.1f} "
                f"needed={MINIMUM_FREE_GIB}\n"
            )
            time.sleep(wait_seconds)
            continue
        while pending and len(active) < maximum:
            job = pending.pop(0)
            _launch(job, config_path=config_path, commit=commit,
                    unit_prefix=unit_prefix)
            active.append(job)
            print(json.dumps({
                "stage": stage, "launched": job["candidate"], "seed": job["seed"],
                "active": len(active), "pending": len(pending),
                "free_disk_gib": round(free, 1),
            }), flush=True)
        controller_status.write_text(
            f"RUNNING stage={stage} commit={commit} completed={len(completed)} "
            f"total={len(jobs)} active={len(active)} pending={len(pending)}\n"
        )
        if active:
            time.sleep(wait_seconds)


def _aggregate(stage: str, config_path: Path, commit: str) -> None:
    subprocess.run(
        [str(PYTHON), str(AGGREGATOR), "--config", str(config_path),
         "--stage", stage, "--expected-commit", commit],
        cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--unit-prefix", default="topic4-formal")
    parser.add_argument("--wait-seconds", type=float)
    parser.add_argument("--skip-upstream-wait", action="store_true")
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
        raise RuntimeError(f"formal launcher commit {commit} is not HEAD {head}")
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    controller_status = output_root / "controller.status"
    wait_seconds = float(args.wait_seconds or config["execution"]["wait_seconds"])
    if not args.skip_upstream_wait:
        _wait_for_upstream(config, wait_seconds, controller_status)

    manifest_path = output_root / "candidate_manifest.json"
    if not manifest_path.exists():
        subprocess.run(
            [str(PYTHON), str(FREEZER), "--config", str(config_path),
             "--expected-commit", commit],
            cwd=ROOT, check=True, env={**os.environ, **NUMERIC_ENV},
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("config", {}).get("sha256") != _sha256(config_path):
        raise RuntimeError("formal candidate manifest uses another config")
    memory_path = output_root / "screen_memory_audit.json"
    search = config["search"]

    if not (output_root / "stage_a_selection.json").exists():
        _run_stage(
            "a",
            _jobs(output_root,
                  [row["candidate_id"] for row in manifest["candidate_set"]["candidates"]],
                  [int(seed) for seed in search["fit_network_seeds"]]),
            config=config, config_path=config_path, commit=commit,
            unit_prefix=args.unit_prefix, wait_seconds=wait_seconds,
            controller_status=controller_status, memory_path=memory_path,
        )
        _aggregate("a", config_path, commit)
    stage_a = json.loads((output_root / "stage_a_selection.json").read_text())

    if not (output_root / "stage_b_selection.json").exists():
        _run_stage(
            "b",
            _jobs(output_root, stage_a["stage_b_candidates"],
                  [int(seed) for seed in search["selection_network_seeds"]]),
            config=config, config_path=config_path, commit=commit,
            unit_prefix=args.unit_prefix, wait_seconds=wait_seconds,
            controller_status=controller_status, memory_path=memory_path,
        )
        _aggregate("b", config_path, commit)
    stage_b = json.loads((output_root / "stage_b_selection.json").read_text())

    if not (output_root / "cohort_result.json").exists():
        _run_stage(
            "c",
            _jobs(output_root, stage_b["stage_c_candidates"],
                  [int(seed) for seed in search["confirmation_network_seeds"]]),
            config=config, config_path=config_path, commit=commit,
            unit_prefix=args.unit_prefix, wait_seconds=wait_seconds,
            controller_status=controller_status, memory_path=memory_path,
        )
        _aggregate("c", config_path, commit)
    result = json.loads((output_root / "cohort_result.json").read_text())

    if FIGURE.exists():
        subprocess.run(
            [str(PYTHON), str(FIGURE), "--config", str(config_path),
             "--expected-commit", commit],
            cwd=ROOT, check=False, env={**os.environ, **NUMERIC_ENV},
        )
    controller_status.write_text(
        f"COMPLETE status={result['status']} commit={commit}\n"
    )
    subprocess.run([
        "notify-send", "Topic 4 formal data-driven SNN cohort",
        f"{result['status']}; pass={result['cohort']['pass_fraction']:.2f}",
    ], check=False)
    print(json.dumps({
        "status": result["status"],
        "pass_fraction": result["cohort"]["pass_fraction"],
        "representative": result["representative_subject"]["subject_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
