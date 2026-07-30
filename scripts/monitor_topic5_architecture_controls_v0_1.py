#!/usr/bin/env python3
"""Monitor architecture-control workers and write resumable status artifacts."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def meminfo() -> dict[str, float]:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        out[key] = float(value.strip().split()[0]) / (1024.0 * 1024.0)
    return out


def gpu() -> dict[str, float]:
    try:
        value = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
        memory, utilization, temperature = map(
            float, (item.strip() for item in value.split(","))
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        memory = utilization = temperature = float("nan")
    return {
        "gpu_memory_used_mb": memory,
        "gpu_utilization_percent": utilization,
        "gpu_temperature_c": temperature,
    }


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def error_hits(root: Path) -> list[dict[str, str]]:
    patterns = ("traceback", "out of memory", "cuda error", "killed")
    hits = []
    for path in root.rglob("*.log"):
        try:
            lines = path.read_text(errors="replace").splitlines()[-100:]
        except OSError:
            continue
        for line in lines:
            if any(pattern in line.lower() for pattern in patterns):
                hits.append({"path": str(path), "line": line[-1000:]})
    return hits[-20:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    while True:
        done_paths = list(args.root.rglob("DONE.json"))
        failures = []
        complete = 0
        for path in done_paths:
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if value.get("status") == "COMPLETE" and value.get(
                "engineering_pass"
            ):
                complete += 1
            else:
                failures.append({"path": str(path), "value": value})
        memory = meminfo()
        errors = error_hits(args.root)
        launcher_alive = alive(args.launcher_pid)
        status = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "COMPLETE"
                if complete == args.expected and not failures
                else "RUNNING"
                if launcher_alive
                else "LAUNCHER_EXITED"
            ),
            "launcher_pid": args.launcher_pid,
            "launcher_alive": launcher_alive,
            "expected_cells": args.expected,
            "completed_cells": complete,
            "failed_cells": failures,
            "recent_error_hits": errors,
            "mem_available_gb": memory.get("MemAvailable"),
            "swap_used_gb": memory.get("SwapTotal", 0.0)
            - memory.get("SwapFree", 0.0),
            **gpu(),
        }
        atomic_json(args.root / "MONITOR_STATUS.json", status)
        if complete == args.expected and not failures:
            atomic_json(
                args.root / "MONITOR_DONE.json",
                {
                    "status": "ALL_CELLS_COMPLETE",
                    "completed_cells": complete,
                    "timestamp_utc": status["timestamp_utc"],
                },
            )
            break
        if not launcher_alive:
            atomic_json(
                args.root / "MONITOR_ALERT.json",
                {
                    "status": "LAUNCHER_EXITED_BEFORE_COMPLETE",
                    "completed_cells": complete,
                    "expected_cells": args.expected,
                    "failed_cells": failures,
                    "recent_error_hits": errors,
                    "timestamp_utc": status["timestamp_utc"],
                },
            )
            break
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
