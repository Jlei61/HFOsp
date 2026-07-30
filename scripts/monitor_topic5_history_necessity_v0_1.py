#!/usr/bin/env python3
"""Continuously record formal history-training progress and resources."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / (1024.0**2)
    return float("nan")


def _gpu() -> tuple[float, float]:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        memory, utilization = raw.split(",")[:2]
        return float(memory), float(utilization)
    except (OSError, subprocess.SubprocessError, ValueError):
        return float("nan"), float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-folds", type=int, default=102)
    parser.add_argument("--expected-models", type=int, default=306)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    progress_csv = args.run_root / "watcher_progress.csv"
    fields = [
        "timestamp_utc",
        "complete_folds",
        "complete_models",
        "expected_folds",
        "expected_models",
        "mem_available_gb",
        "gpu_memory_used_mb",
        "gpu_utilization_percent",
        "oom_log_hits",
    ]
    with progress_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0:
            writer.writeheader()
        while _alive(args.launcher_pid):
            complete_folds = len(
                [
                    path
                    for path in args.run_root.glob("seed_*/*/DONE.json")
                    if path.parent.parent.name != "logs"
                    and json.loads(path.read_text()).get("status") == "complete"
                ]
            )
            complete_models = len(
                [
                    path
                    for path in args.run_root.glob(
                        "seed_*/*/history_*_gru/DONE.json"
                    )
                    if json.loads(path.read_text()).get("status") == "complete"
                ]
            )
            oom_hits = 0
            for log in args.run_root.glob("seed_*/logs/*.log"):
                text = log.read_text(errors="ignore").lower()
                oom_hits += text.count("out of memory")
            gpu_memory, gpu_utilization = _gpu()
            row = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "complete_folds": complete_folds,
                "complete_models": complete_models,
                "expected_folds": int(args.expected_folds),
                "expected_models": int(args.expected_models),
                "mem_available_gb": _mem_available_gb(),
                "gpu_memory_used_mb": gpu_memory,
                "gpu_utilization_percent": gpu_utilization,
                "oom_log_hits": oom_hits,
            }
            writer.writerow(row)
            handle.flush()
            (args.run_root / "watcher_status.json").write_text(
                json.dumps(row, indent=2)
            )
            time.sleep(max(1.0, float(args.interval_seconds)))


if __name__ == "__main__":
    main()
