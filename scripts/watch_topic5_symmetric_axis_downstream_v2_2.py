#!/usr/bin/env python3
"""Persistent progress watcher for gated Claim-3 and Claim-4 execution."""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = (
    ROOT / "results/topic5_symmetric_axis_propagation_state_v2_2/formal"
)
PIPELINE = FORMAL / "PIPELINE_SUPERVISOR_STATE.json"
STATE = FORMAL / "DOWNSTREAM_WATCHER_STATE.json"
LOG = FORMAL / "downstream_watch.jsonl"
POLL_SECONDS = 300


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def monitor(script: str) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "cuda_env",
            "python",
            script,
        ],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def write(payload: dict[str, Any]) -> None:
    record = {
        "unix_time": time.time(),
        "pid": os.getpid(),
        "target_values_read": False,
        **payload,
    }
    atomic_json(STATE, record)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def main() -> None:
    while True:
        pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
        upstream_status = str(pipeline.get("status"))
        stage = str(pipeline.get("stage"))
        if upstream_status == "FAILED":
            write(
                {
                    "status": "FAILED",
                    "stage": stage,
                    "error": pipeline.get("error"),
                }
            )
            raise SystemExit("pipeline supervisor failed")
        if upstream_status == "COMPLETE":
            write(
                {
                    "status": "COMPLETE",
                    "stage": "downstream_pipeline_complete",
                    "upstream_stage": stage,
                }
            )
            return
        if stage in {
            "prepare_claim3_random_axes",
            "benchmark_claim3",
            "run_claim3",
        }:
            progress = monitor(
                "scripts/monitor_topic5_symmetric_axis_formal_claim3_v2_2.py"
            )
            payload = {
                "status": "RUNNING",
                "stage": "monitor_claim3",
                "upstream_stage": stage,
                **progress,
            }
        elif stage == "run_claim4":
            progress = monitor(
                "scripts/monitor_topic5_symmetric_axis_formal_claim4_v2_2.py"
            )
            payload = {
                "status": "RUNNING",
                "stage": "monitor_claim4",
                "upstream_stage": stage,
                **progress,
            }
        else:
            payload = {
                "status": "RUNNING",
                "stage": "wait_downstream_gate",
                "upstream_stage": stage,
            }
        write(payload)
        if payload.get("failures"):
            raise SystemExit("downstream grid contains failed tasks")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
