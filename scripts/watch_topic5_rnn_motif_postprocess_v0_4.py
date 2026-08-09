#!/usr/bin/env python3
"""Read-only live monitor for the long v0.4 post-processing chain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


def process_rows(pattern: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["pgrep", "-af", pattern], capture_output=True, text=True, check=False
    )
    rows = []
    for line in completed.stdout.splitlines():
        pid, _, command = line.partition(" ")
        if not pid.isdigit() or "pgrep -af" in command:
            continue
        rows.append({"pid": int(pid), "command": command})
    return rows


def gpu_status() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            continue
        rows.append(
            {
                "index": int(values[0]),
                "utilization_percent": int(values[1]),
                "memory_used_mib": int(values[2]),
                "memory_total_mib": int(values[3]),
                "temperature_c": int(values[4]),
            }
        )
    return rows


def current_stage(out_root: Path) -> str:
    order = (
        "G_lesion_shards",
        "G_lesion_aggregate",
        "G_theory",
        "G_figure",
        "F_early_ictal",
        "F_lesion_early",
        "F_figure",
        "H_common",
        "I_figure",
        "I_tests",
    )
    status = out_root / "postprocess_status"
    for stage in order:
        if not (status / f"{stage}.DONE.json").exists():
            return stage
    if (out_root / "PIPELINE_COMPLETE.json").exists():
        return "COMPLETE"
    if (out_root / "POSTPROCESS_READY_FOR_VISUAL_QA.json").exists():
        return "READY_FOR_VISUAL_QA"
    return "POSTPROCESS_COMPLETE_PENDING_QA"


def snapshot(out_root: Path) -> dict[str, Any]:
    lesion_done = sum(
        1 for _ in (out_root / "matched_lesions").glob("**/LESION_DONE.json")
    )
    failed = [
        str(path)
        for path in out_root.glob("**/*FAILED*.json")
        if "pre_plot_fix" not in path.name and "diagnostic_archives" not in path.parts
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "current_stage": current_stage(out_root),
        "matched_lesion": {
            "complete": lesion_done,
            "total": 217,
            "fraction": lesion_done / 217.0,
        },
        "target_values_read": (out_root / "target_access_audit.json").exists(),
        "ready_for_visual_qa": (out_root / "POSTPROCESS_READY_FOR_VISUAL_QA.json").exists(),
        "pipeline_complete": (out_root / "PIPELINE_COMPLETE.json").exists(),
        "failure_markers": failed,
        "postprocess_processes": process_rows("run_topic5_rnn_motif_postprocess_v0_4.py"),
        "lesion_workers": process_rows("run_topic5_rnn_motif_matched_lesions_v0_4.py"),
        "gpu": gpu_status(),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=48.0)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    deadline = time.time() + args.max_hours * 3600.0
    log_path = out_root / "POSTPROCESS_LIVE_WATCHER.jsonl"
    while time.time() < deadline:
        payload = snapshot(out_root)
        atomic_json(out_root / "POSTPROCESS_LIVE_STATUS.json", payload)
        with log_path.open("a") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        if payload["pipeline_complete"]:
            return 0
        time.sleep(max(30, int(args.interval_seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
