#!/usr/bin/env python3
"""Write a target-blind heartbeat for the Topic 5 history-RNN pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def _command(args: list[str]) -> str:
    try:
        return subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _gpu() -> list[dict]:
    output = _command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 6:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "name": values[1],
                "memory_used_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "utilization_percent": int(values[4]),
                "temperature_c": int(values[5]),
            }
        )
    return rows


def _processes() -> list[dict]:
    output = _command(["ps", "-eo", "pid=,etimes=,%cpu=,rss=,args="])
    rows = []
    for line in output.splitlines():
        if "topic5_history_rnn" not in line or "monitor_topic5_history_rnn" in line:
            continue
        values = line.strip().split(maxsplit=4)
        if len(values) != 5:
            continue
        rows.append(
            {
                "pid": int(values[0]),
                "elapsed_seconds": int(values[1]),
                "cpu_percent": float(values[2]),
                "rss_kib": int(values[3]),
                "command": values[4],
            }
        )
    return rows


def _phase(path: Path) -> dict:
    done = list(path.glob("*/DONE.json")) if path.exists() else []
    return {
        "exists": path.exists(),
        "done_folds": len(done),
        "finalized_folds": (
            len(done)
            if path.name.startswith("g2_")
            else sum((item.parent / "ORDER_CONTROLS.json").exists() for item in done)
        ),
        "failed_folds": (
            len(list(path.glob("*.FAILED.json")))
            + len(list(path.glob("*/FAILED.json")))
        )
        if path.exists()
        else 0,
        "summary_files": sorted(
            str(item.relative_to(path)) for item in path.glob("*SUMMARY.json")
        )
        if path.exists()
        else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/topic5_history_rnn_early_ictal_field"),
    )
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    heartbeat = root / "PIPELINE_HEARTBEAT.json"
    history = root / "pipeline_heartbeat.jsonl"
    while True:
        formal = root / "g1_sequential_formal_v0_1"
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "monitor_pid": os.getpid(),
            "monitor_reads_target_values": False,
            "phases": {
                "g1_seed_20260725": _phase(formal / "seed_20260725"),
                "g1_seed_20260726": _phase(formal / "seed_20260726"),
                "g1_seed_20260727": _phase(formal / "seed_20260727"),
                "g2": _phase(root / "g2_early_ictal_loso_v0_1"),
            },
            "gpu": _gpu(),
            "processes": _processes(),
        }
        temporary = heartbeat.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(heartbeat)
        with history.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        if args.once:
            break
        time.sleep(max(float(args.interval_seconds), 5.0))


if __name__ == "__main__":
    main()
