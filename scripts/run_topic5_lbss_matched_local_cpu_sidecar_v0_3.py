#!/usr/bin/env python3
"""Fill one restart-safe attenuation target on CPU in parallel.

This is a scheduling-only sidecar. It imports the immutable attenuation
snapshot and writes the same per-unit cache contract as the formal GPU runner.
The formal runner later verifies and reuses each cache before aggregation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--tail-jobs", type=int, default=None,
        help=(
            "Process only the last N currently missing units. This lets a CPU "
            "sidecar work from the opposite end of the deterministic GPU queue "
            "without duplicating in-flight units."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("L1_ADDED", "L2_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL"),
        default="L3_MATCHED_LOCAL",
    )
    parser.add_argument(
        "--fit-ids", nargs="*", default=None,
        help="Optional explicit fit ids; useful for bounded tail scheduling.",
    )
    args = parser.parse_args()
    out = args.out_root.resolve()
    snapshot = args.snapshot.resolve()
    sys.path.insert(0, str(snapshot))
    sys.path.insert(0, str(snapshot / "scripts"))
    from run_topic5_lbss_attenuation_v0_2 import (  # noqa: PLC0415
        attenuation_unit_cache_path,
        load_attenuation_unit_cache,
        unit_target_worker,
    )

    target_to_arm = {
        "L1_ADDED": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_ADDED": "L2_LOCAL_PLUS_RANDOM_LR",
        "L3_ADDED": "L3_LOCAL_PLUS_LEARNED_LR",
        "L3_MATCHED_LOCAL": "L3_LOCAL_PLUS_LEARNED_LR",
    }
    target = str(args.target)
    metrics = sorted(out.glob(
        f"per_fit/*/{target_to_arm[target]}/seed*/metrics.json"
    ))
    if len(metrics) != 31 * 3:
        raise RuntimeError(f"{target} denominator changed: {len(metrics)}")
    if args.fit_ids:
        requested = set(args.fit_ids)
        metrics = [path for path in metrics if path.parents[2].name in requested]
        observed = {path.parents[2].name for path in metrics}
        if observed != requested:
            raise RuntimeError(f"requested fit ids not found: {sorted(requested - observed)}")
    jobs = []
    reused = 0
    for path in metrics:
        destination = attenuation_unit_cache_path(out, path, target)
        if load_attenuation_unit_cache(destination, path, target) is not None:
            reused += 1
        else:
            jobs.append((str(out), str(path), target, "cpu"))
    if args.tail_jobs is not None:
        if args.tail_jobs < 1:
            raise ValueError("tail-jobs must be positive")
        jobs = jobs[-int(args.tail_jobs):]
    scheduled_jobs = len(jobs)
    marker_slug = target.lower() + (
        f"_tail{int(args.tail_jobs)}" if args.tail_jobs is not None else ""
    )
    if args.fit_ids:
        marker_slug += "_fits_" + str(abs(hash(tuple(sorted(args.fit_ids)))))
    status = out / f"ATTENUATION_CPU_SIDECAR_{marker_slug}_STATUS.json"
    atomic(status, {
        "status": "RUNNING", "scheduled": scheduled_jobs, "reused": reused,
        "remaining": len(jobs), "complete": 0, "workers": int(args.workers),
        "target": target, "target_values_read": False,
        "updated_at": now(), "pid": os.getpid(),
        "snapshot": str(snapshot),
    })
    complete = 0
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        futures = [executor.submit(unit_target_worker, job) for job in jobs]
        for future in as_completed(futures):
            future.result()
            complete += 1
            atomic(status, {
                "status": "RUNNING", "scheduled": scheduled_jobs, "reused": reused,
                "remaining": len(jobs) - complete, "complete": complete,
                "workers": int(args.workers), "target": target,
                "target_values_read": False,
                "updated_at": now(), "pid": os.getpid(), "snapshot": str(snapshot),
            })
    atomic(out / f"ATTENUATION_CPU_SIDECAR_{marker_slug}_COMPLETE.json", {
        "status": "COMPLETE", "scheduled": scheduled_jobs, "reused": reused,
        "complete": complete, "workers": int(args.workers),
        "target": target, "target_values_read": False,
        "updated_at": now(), "snapshot": str(snapshot),
    })
    atomic(status, {
        "status": "COMPLETE", "scheduled": scheduled_jobs, "reused": reused,
        "remaining": 0, "complete": complete, "workers": int(args.workers),
        "target": target, "target_values_read": False,
        "updated_at": now(), "pid": os.getpid(),
        "snapshot": str(snapshot),
    })


if __name__ == "__main__":
    main()
