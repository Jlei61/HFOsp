#!/usr/bin/env python3
"""Monitor the 34x3 target-blind hidden-state extraction."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_rnn_internal_state_reduction"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def resource_snapshot() -> dict[str, str | float]:
    memory = {}
    for line in subprocess.check_output(["free", "-b"], text=True).splitlines():
        if line.startswith("Mem:"):
            fields = line.split()
            memory = {
                "ram_used_gb": int(fields[2]) / 1024**3,
                "ram_available_gb": int(fields[6]) / 1024**3,
            }
            break
    load = os.getloadavg()
    return {
        **memory,
        "load_1m": float(load[0]),
        "load_5m": float(load[1]),
        "load_15m": float(load[2]),
    }


def snapshot() -> dict:
    statuses = list(
        (OUT / "interictal/cells").glob("seed_*/**/CELL_STATUS.json")
    )
    counts = {"COMPLETE": 0, "RUNNING": 0, "FAILED": 0}
    runtime = []
    for path in statuses:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        status = str(payload.get("status", "FAILED"))
        counts[status] = counts.get(status, 0) + 1
        if status == "COMPLETE":
            runtime.append(float(payload.get("runtime_seconds", 0.0)))
    return {
        "timestamp": time.time(),
        "expected_cells": 102,
        "status_files": len(statuses),
        "complete": counts.get("COMPLETE", 0),
        "running": counts.get("RUNNING", 0),
        "failed": counts.get("FAILED", 0),
        "pending": 102 - len(statuses),
        "median_cell_runtime_seconds": (
            sorted(runtime)[len(runtime) // 2] if runtime else None
        ),
        "extraction_done_marker": (OUT / "EXTRACTION_DONE.json").exists(),
        "target_values_read": False,
        **resource_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=60.0)
    args = parser.parse_args()
    resource_path = OUT / "logs/resource_monitor.csv"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        current = snapshot()
        atomic_json(OUT / "MONITOR_STATUS.json", current)
        exists = resource_path.exists()
        with resource_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(current))
            if not exists:
                writer.writeheader()
            writer.writerow(current)
        print(json.dumps(current), flush=True)
        if current["complete"] == 102 and current["extraction_done_marker"]:
            atomic_json(
                OUT / "MONITOR_DONE.json", {**current, "status": "COMPLETE"}
            )
            return
        time.sleep(float(args.interval))


if __name__ == "__main__":
    main()
