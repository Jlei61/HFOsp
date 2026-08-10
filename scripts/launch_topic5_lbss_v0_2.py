#!/usr/bin/env python3
"""Concurrent, resumable launcher for LBSS v0.2 training units."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2_LOCAL_PLUS_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def job_directory(out: Path, fit: str, arm: str, seed: int, stage: str) -> Path:
    root = "diagnostic_smoke_units" if stage == "smoke" else "per_fit"
    return out / root / fit / arm / f"seed{seed}"


def unit_complete(out: Path, fit: str, arm: str, seed: int, stage: str) -> bool:
    directory = job_directory(out, fit, arm, seed, stage)
    done, metrics = directory / "DONE.json", directory / "metrics.json"
    if not done.exists() or not metrics.exists():
        return False
    try:
        marker = json.loads(done.read_text())
        result = json.loads(metrics.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        marker.get("ok")
        and (stage == "smoke" or marker.get("converged"))
        and result.get("target_values_read") is False
        and result.get("best_checkpoint_eligible")
    )


def run_job(job: dict, trainer: Path, out: Path, device: str, log_root: Path, stage: str) -> dict:
    fit, arm, seed = job["fit_id"], job["arm"], int(job["seed"])
    log = log_root / fit / arm / f"seed{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(trainer), "--fit-id", fit, "--arm", arm,
        "--seed", str(seed), "--out-root", str(out), "--device", device,
    ]
    if job.get("epochs_freeze") is not None:
        command += ["--epochs-freeze", str(job["epochs_freeze"])]
    if stage == "smoke":
        command += ["--unit-root-name", "diagnostic_smoke_units"]
    started = time.time()
    with log.open("a") as stream:
        stream.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        stream.flush()
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="replace")[-12000:]
    oom = "out of memory" in text.lower() or "cuda error: out of memory" in text.lower()
    return {
        **job,
        "returncode": int(completed.returncode),
        "seconds": round(time.time() - started, 2),
        "log": str(log),
        "oom": bool(oom),
        "complete": unit_complete(out, fit, arm, seed, stage),
    }


def jobs_for(stage: str, fits: list[str]) -> list[dict]:
    if stage == "smoke":
        selected = [fit for fit in fits if fit in {
            "epilepsiae_1084__shared", "epilepsiae_1146__shared",
            "yuquan_chengshuai__shared",
        }]
        return [
            {"fit_id": fit, "arm": arm, "seed": 0, "epochs_freeze": 30}
            for fit in selected for arm in ARMS
        ]
    if stage == "formal":
        return [
            {"fit_id": fit, "arm": arm, "seed": seed}
            for fit in fits for arm in ARMS for seed in (0, 1, 2)
        ]
    raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal", "status"), required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--trainer", type=Path)
    args = parser.parse_args()
    out = args.out_root.resolve()
    manifest = json.loads((out / "INPUT_CACHE_MANIFEST.json").read_text())
    fits = sorted({item["fit_id"] for item in manifest["files"]})
    trainer = (args.trainer or (out / "run_snapshot/scripts/train_topic5_lbss_unit_v0_2.py")).resolve()
    if args.stage != "status" and not trainer.exists():
        raise FileNotFoundError(f"immutable trainer missing: {trainer}")
    stage = "formal" if args.stage == "status" else args.stage
    jobs = jobs_for(stage, fits)
    completed_before = [job for job in jobs if unit_complete(
        out, job["fit_id"], job["arm"], job["seed"], stage
    )]
    pending = [job for job in jobs if job not in completed_before]
    status_path = out / f"{stage.upper()}_TRAINING_STATUS.json"
    if args.stage == "status":
        write_json(status_path, {
            "stage": stage,
            "scheduled": len(jobs),
            "complete": len(completed_before),
            "pending": len(pending),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        print(status_path.read_text())
        return

    log_root = out / "run_logs" / stage
    results: list[dict] = []
    started = time.time()
    write_json(status_path, {
        "stage": stage, "scheduled": len(jobs), "complete_before": len(completed_before),
        "launched": len(pending), "workers": int(args.workers), "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.workers)) as executor:
        future_map = {
            executor.submit(run_job, job, trainer, out, args.device, log_root, stage): job
            for job in pending
        }
        for future in concurrent.futures.as_completed(future_map):
            result = future.result()
            results.append(result)
            now_complete = len(completed_before) + sum(item["complete"] for item in results)
            write_json(status_path, {
                "stage": stage,
                "scheduled": len(jobs),
                "complete": now_complete,
                "processed_this_run": len(results),
                "failed_this_run": sum(not item["complete"] for item in results),
                "oom_this_run": sum(item["oom"] for item in results),
                "workers": int(args.workers),
                "pid": os.getpid(),
                "elapsed_seconds": round(time.time() - started, 1),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "last_result": result,
            })

    failures = [result for result in results if not result["complete"]]
    # OOM may be caused by excessive concurrency; retry only those units one at
    # a time without changing batch size or the model contract.
    recovered: list[dict] = []
    for failure in [item for item in failures if item["oom"]]:
        retry = run_job(failure, trainer, out, args.device, log_root / "oom_serial_retry", stage)
        recovered.append(retry)
    all_complete = [job for job in jobs if unit_complete(
        out, job["fit_id"], job["arm"], job["seed"], stage
    )]
    unresolved = [job for job in jobs if job not in all_complete]
    final = {
        "stage": stage,
        "scheduled": len(jobs),
        "complete": len(all_complete),
        "unresolved": len(unresolved),
        "unresolved_units": unresolved,
        "oom_observed": sum(item["oom"] for item in results),
        "oom_recovered": sum(item["complete"] for item in recovered),
        "workers": int(args.workers),
        "elapsed_seconds": round(time.time() - started, 1),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
    }
    write_json(status_path, final)
    marker = out / f"{stage.upper()}_TRAINING_COMPLETE.json"
    failed_marker = out / f"{stage.upper()}_TRAINING_FAILED.json"
    if unresolved:
        failed_marker.write_text(json.dumps(final, indent=2) + "\n")
        marker.unlink(missing_ok=True)
        raise SystemExit(2)
    marker.write_text(json.dumps(final, indent=2) + "\n")
    failed_marker.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
