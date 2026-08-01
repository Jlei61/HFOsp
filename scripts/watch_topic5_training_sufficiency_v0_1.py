#!/usr/bin/env python3
"""Watch a Topic 5 sufficiency run root for COMPLETE / FAILED / OOM / NaN.

Emits one line per check so it can drive a terminal, a tmux pane or an agent
monitor.  It never writes into the run tree, so it is safe to start and stop at
any time.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FAILURE_PATTERNS = re.compile(
    r"Traceback|CUDA out of memory|RuntimeError|AssertionError|Killed|"
    r"nan|NaN|inf detected",
    re.IGNORECASE,
)
OOM_PATTERN = re.compile(r"CUDA out of memory|Killed", re.IGNORECASE)
NAN_PATTERN = re.compile(r"\bnan\b", re.IGNORECASE)


def _scan(root: Path) -> dict:
    done = list(root.rglob("DONE.json"))
    running = [
        path.parent
        for path in root.rglob("run_state.json")
        if json.loads(path.read_text()).get("status") == "RUNNING"
    ]
    oom, nan, failed = [], [], []
    for log in sorted((root / "logs").glob("*.log")) if (root / "logs").is_dir() else []:
        text = log.read_text(errors="replace")
        if OOM_PATTERN.search(text):
            oom.append(log.name)
        elif NAN_PATTERN.search(text):
            nan.append(log.name)
        elif FAILURE_PATTERNS.search(text):
            failed.append(log.name)
    state_path = root / "LAUNCHER_STATE.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    return {
        "status": state.get("status", "UNKNOWN"),
        "n_cells": state.get("n_cells"),
        "n_complete": len(done),
        "n_running": len(running),
        "n_oom": len(oom),
        "n_nan": len(nan),
        "n_failed_logs": len(failed),
        "oom": oom[:3],
        "nan": nan[:3],
        "failed": failed[:3],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    while True:
        report = _scan(root)
        print(
            json.dumps({"time": time.strftime("%H:%M:%S"), **report}),
            flush=True,
        )
        if args.once or report["status"] in {"COMPLETE", "INCOMPLETE", "BLOCKED_PARTIAL_CELLS"}:
            break
        time.sleep(int(args.interval_seconds))
    if report["status"] == "INCOMPLETE" or report["n_oom"] or report["n_nan"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
