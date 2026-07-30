#!/usr/bin/env python3
"""Append CPU, RAM, swap and GPU usage while a launcher PID is alive."""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _meminfo() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw = line.split(":", 1)
        values[key] = float(raw.strip().split()[0]) / (1024.0 * 1024.0)
    return values


def _gpu() -> list[str]:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10).strip()
        return [item.strip() for item in output.split(",")]
    except (OSError, subprocess.SubprocessError):
        return ["nan", "nan", "nan", "nan"]


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp_utc",
        "load_1m",
        "mem_available_gb",
        "swap_total_gb",
        "swap_free_gb",
        "gpu_memory_used_mb",
        "gpu_memory_free_mb",
        "gpu_utilization_percent",
        "gpu_temperature_c",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while _alive(args.pid):
            memory = _meminfo()
            gpu_used, gpu_free, gpu_util, gpu_temp = _gpu()
            writer.writerow(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "load_1m": os.getloadavg()[0],
                    "mem_available_gb": memory.get("MemAvailable", float("nan")),
                    "swap_total_gb": memory.get("SwapTotal", float("nan")),
                    "swap_free_gb": memory.get("SwapFree", float("nan")),
                    "gpu_memory_used_mb": gpu_used,
                    "gpu_memory_free_mb": gpu_free,
                    "gpu_utilization_percent": gpu_util,
                    "gpu_temperature_c": gpu_temp,
                }
            )
            handle.flush()
            time.sleep(max(float(args.interval_seconds), 1.0))


if __name__ == "__main__":
    main()
