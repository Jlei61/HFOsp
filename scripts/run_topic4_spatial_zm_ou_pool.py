#!/usr/bin/env python3
"""Resumable worker pool for the spatial Z/M + persistent-OU round.

Jobs are declared in a JSONL manifest, one object per line with ``job_id``,
``out`` and ``argv``.  A job is claimed by atomically creating ``<out>.claim``;
a job whose ``<out>.json`` already exists is skipped, so an interrupted round
resumes without recomputing or overwriting finished results.  Every job keeps
its own log file whether it succeeded or failed.

Worker count is sized from measured per-worker RSS and the live machine state
rather than from a fixed number, per the round's execution contract.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
SINGLE_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def available_memory_gib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return float(line.split()[1]) / (1024.0 * 1024.0)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


def size_pool(*, per_worker_rss_gib, reserve_gib, free_core_margin, cap):
    memory_workers = int(
        max(0.0, available_memory_gib() - reserve_gib) / per_worker_rss_gib)
    load = os.getloadavg()[0]
    core_workers = int(max(1.0, os.cpu_count() - load - free_core_margin))
    return max(1, min(cap, memory_workers, core_workers))


def _claim(path: Path) -> bool:
    try:
        handle = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except OSError as error:
        if error.errno == errno.EEXIST:
            return False
        raise
    os.write(handle, f"{os.getpid()} {time.time()}\n".encode())
    os.close(handle)
    return True


def run_job(job, log_dir: Path, dry_run: bool):
    out = Path(job["out"])
    if not out.is_absolute():
        out = ROOT / out
    done = out.with_suffix(".json")
    if done.exists():
        return {"job_id": job["job_id"], "state": "already_done"}
    claim = out.with_suffix(".claim")
    claim.parent.mkdir(parents=True, exist_ok=True)
    if not _claim(claim):
        return {"job_id": job["job_id"], "state": "claimed_elsewhere"}
    log_path = log_dir / f"{job['job_id']}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        claim.unlink(missing_ok=True)
        return {"job_id": job["job_id"], "state": "dry_run",
                "argv": [PYTHON, *job["argv"]]}
    env = dict(os.environ)
    env.update(SINGLE_THREAD_ENV)
    started = time.time()
    with log_path.open("w") as log:
        log.write(" ".join([PYTHON, *job["argv"]]) + "\n")
        log.flush()
        code = subprocess.call([PYTHON, *job["argv"]], cwd=str(ROOT), env=env,
                               stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    if code == 0 and done.exists():
        claim.unlink(missing_ok=True)
        return {"job_id": job["job_id"], "state": "done",
                "wall_seconds": round(elapsed, 1)}
    # Leave the claim in place only for a genuine crash, so a resume can see it.
    return {"job_id": job["job_id"], "state": "failed", "returncode": code,
            "wall_seconds": round(elapsed, 1), "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--per-worker-rss-gib", type=float, default=8.0)
    parser.add_argument("--reserve-gib", type=float, default=40.0)
    parser.add_argument("--free-core-margin", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--summary", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Clear stale claim files whose result is missing.")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    jobs = [json.loads(line) for line in manifest.read_text().splitlines()
            if line.strip()]
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = ROOT / log_dir

    if args.retry_failed:
        for job in jobs:
            out = Path(job["out"])
            if not out.is_absolute():
                out = ROOT / out
            if not out.with_suffix(".json").exists():
                out.with_suffix(".claim").unlink(missing_ok=True)

    workers = size_pool(per_worker_rss_gib=args.per_worker_rss_gib,
                        reserve_gib=args.reserve_gib,
                        free_core_margin=args.free_core_margin,
                        cap=args.max_workers)
    print(json.dumps({
        "n_jobs": len(jobs), "workers": workers,
        "available_gib": round(available_memory_gib(), 1),
        "load1": round(os.getloadavg()[0], 2),
    }), flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda job: run_job(job, log_dir, args.dry_run),
                               jobs):
            results.append(result)
            print(json.dumps(result), flush=True)

    summary = {
        "n_jobs": len(jobs),
        "n_done": sum(r["state"] in {"done", "already_done"} for r in results),
        "n_failed": sum(r["state"] == "failed" for r in results),
        "workers": workers,
        "results": results,
    }
    if args.summary:
        path = Path(args.summary)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}),
          flush=True)
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
