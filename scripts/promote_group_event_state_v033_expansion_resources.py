#!/usr/bin/env python3
"""Add GPU 0 to the expansion controller after O1b replications finish."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp.{time.time_ns()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication-log", type=Path, required=True)
    parser.add_argument("--expansion-status", type=Path, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=20.0)
    args = parser.parse_args()
    while True:
        replication_done = args.replication_log.is_file() and \
            "REPLICATION_COMPLETE" in args.replication_log.read_text(errors="replace")
        expansion_started = args.expansion_status.is_file()
        if replication_done and expansion_started:
            break
        time.sleep(args.poll)
    lease = json.loads(args.lease.read_text())
    lease["gpu_ids"] = [0, 1]
    lease["max_workers"] = 4
    lease["max_jobs_per_gpu_before_sentinel_review"] = 2
    lease["forbidden"] = [
        value for value in lease.get("forbidden", [])
        if "more than two expansion jobs on GPU 1" not in str(value)
    ] + ["more than two expansion jobs per GPU"]
    lease["note"] = (
        "O1b replication completed; expansion now uses two measured workers on each GPU. "
        "STATE_TRAIN plus chronological STATE_SELECTION only."
    )
    atomic_json(args.lease, lease)
    atomic_json(args.receipt, {
        "format": "group_event_state_v0_3_3_expansion_resource_promotion",
        "promoted_epoch": time.time(), "gpu_ids": [0, 1], "max_workers": 4,
        "reason": "O1b replication completed and expansion controller is live",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
