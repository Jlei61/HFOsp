#!/usr/bin/env python3
"""Monitor v0.2 units and persist five-minute completion/OOM/NaN snapshots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
OOM_PATTERN = re.compile(r"out of memory|cuda error.*memory|cublas_status_alloc", re.I)
NAN_PATTERN = re.compile(r"\b(?:nan|inf)\b|non-finite", re.I)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tail_text(path: Path, *, bytes_to_read: int = 1 << 18) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - int(bytes_to_read)))
        return handle.read().decode(errors="replace")


def expected_tasks(config: Mapping[str, Any], *, smoke: bool) -> list[tuple[str, str, int]]:
    if smoke:
        return [
            (subject, model, int(config["smoke"]["seed"]))
            for subject in config["smoke"]["subjects"]
            for model in config["smoke"]["models"]
        ]
    dataset_root = (
        Path(config["dataset_artifact_root"]).resolve() / config["dataset_root"]
    )
    manifest = json.loads((dataset_root / "dataset_manifest.json").read_text())
    return [
        (subject, model, int(seed))
        for subject in manifest["cohort_subjects"]
        for model in config["models"]["names"]
        for seed in config["training"]["seeds"]
    ]


def snapshot(
    config: Mapping[str, Any],
    *,
    output_root: Path,
    smoke: bool,
) -> dict[str, Any]:
    tasks = expected_tasks(config, smoke=smoke)
    unit_root = output_root / ("smoke" if smoke else "") / "per_subject"
    log_root = output_root / "logs" / ("smoke" if smoke else "formal")
    rows = []
    status_counts = {name: 0 for name in ("COMPLETE", "FAILED", "PARTIAL", "PENDING")}
    n_oom = 0
    n_nan = 0
    max_peak_gpu_mb = 0.0
    max_peak_rss_gb = 0.0
    for subject, model, seed in tasks:
        run_dir = unit_root / subject / model / f"seed_{seed}"
        done_path = run_dir / "DONE.json"
        failed_path = run_dir / "FAILED.json"
        progress_path = run_dir / "resume_state.pt"
        log_path = log_root / f"{subject}__{model}__seed{seed}.log"
        done: dict[str, Any] = {}
        failed: dict[str, Any] = {}
        if done_path.exists():
            try:
                done = json.loads(done_path.read_text())
            except json.JSONDecodeError:
                done = {}
        if failed_path.exists():
            try:
                failed = json.loads(failed_path.read_text())
            except json.JSONDecodeError:
                failed = {}
        if done.get("status") == "COMPLETE":
            status = "COMPLETE"
        elif failed.get("status") == "FAILED":
            status = "FAILED"
        elif progress_path.exists() or log_path.exists():
            status = "PARTIAL"
        else:
            status = "PENDING"
        status_counts[status] += 1
        text = tail_text(log_path)
        oom = bool(OOM_PATTERN.search(text))
        nonfinite = bool(NAN_PATTERN.search(text))
        n_oom += int(oom)
        n_nan += int(nonfinite)
        peak_gpu = float(done.get("peak_gpu_memory_mb", 0.0) or 0.0)
        peak_rss = float(done.get("peak_rss_gb", 0.0) or 0.0)
        max_peak_gpu_mb = max(max_peak_gpu_mb, peak_gpu)
        max_peak_rss_gb = max(max_peak_rss_gb, peak_rss)
        if status != "PENDING" or oom or nonfinite:
            rows.append(
                {
                    "subject": subject,
                    "model": model,
                    "seed": seed,
                    "status": status,
                    "oom_in_log": oom,
                    "nonfinite_in_log": nonfinite,
                    "best_cycle": done.get("best_cycle"),
                    "peak_gpu_memory_mb": peak_gpu,
                    "peak_rss_gb": peak_rss,
                    "error": failed.get("error"),
                }
            )
    total = len(tasks)
    if status_counts["COMPLETE"] == total:
        overall_status = "COMPLETE"
    elif status_counts["COMPLETE"] + status_counts["FAILED"] == total:
        overall_status = "FAILED"
    elif status_counts["PARTIAL"] or status_counts["COMPLETE"] or status_counts["FAILED"]:
        overall_status = "RUNNING"
    else:
        overall_status = "PENDING"
    return {
        "status": overall_status,
        "smoke": smoke,
        "updated_unix": time.time(),
        "n_total": total,
        "n_complete": status_counts["COMPLETE"],
        "n_failed": status_counts["FAILED"],
        "n_partial": status_counts["PARTIAL"],
        "n_pending": status_counts["PENDING"],
        "n_oom_logs": n_oom,
        "n_nonfinite_logs": n_nan,
        "max_peak_gpu_memory_mb": max_peak_gpu_mb,
        "max_peak_rss_gb": max_peak_rss_gb,
        "units_with_state": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=None)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.resolve().read_text())
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    monitor_root = output_root / "monitor"
    monitor_root.mkdir(parents=True, exist_ok=True)
    status_path = monitor_root / ("smoke_status.json" if args.smoke else "status.json")
    log_path = monitor_root / ("smoke_status.log" if args.smoke else "status.log")
    interval = float(
        args.interval
        if args.interval is not None
        else config["resources"]["monitor_interval_seconds"]
    )
    if interval <= 0:
        raise ValueError("interval must be positive")
    while True:
        current = snapshot(config, output_root=output_root, smoke=bool(args.smoke))
        atomic_json(status_path, current)
        line = (
            f"unix={current['updated_unix']:.3f} status={current['status']} "
            f"complete={current['n_complete']}/{current['n_total']} "
            f"failed={current['n_failed']} partial={current['n_partial']} "
            f"pending={current['n_pending']} oom={current['n_oom_logs']} "
            f"nonfinite={current['n_nonfinite_logs']} "
            f"peak_gpu_mb={current['max_peak_gpu_memory_mb']:.1f}\n"
        )
        with log_path.open("a") as handle:
            handle.write(line)
        print(line, end="", flush=True)
        if not args.watch or current["status"] in {"COMPLETE", "FAILED"}:
            raise SystemExit(1 if current["status"] == "FAILED" else 0)
        time.sleep(interval)


if __name__ == "__main__":
    main()
