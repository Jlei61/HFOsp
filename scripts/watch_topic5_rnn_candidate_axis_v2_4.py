#!/usr/bin/env python3
"""Monitor the detached v2.4 candidate-axis search."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4/formal"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING"}
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    state_path = BASE / "AXIS_SEARCH_LAUNCHER_STATE.json"
    watcher_path = BASE / "AXIS_SEARCH_WATCHER_STATE.json"
    while True:
        state = read_json(state_path)
        status = state.get("status")
        payload = {
            "status": (
                "COMPLETE"
                if status == "COMPLETE"
                else "FAILED"
                if status == "FAILED"
                else "MONITORING"
            ),
            "launcher_status": status,
            "tasks_finished": state.get("n_tasks_finished", 0),
            "tasks_total": state.get("n_tasks_total", 27),
            "tasks_failed": state.get("n_tasks_failed", 0),
            "updated_unix": time.time(),
            "target_values_read": False,
        }
        atomic_json(watcher_path, payload)
        print(
            f"axis-search={status} {payload['tasks_finished']}/"
            f"{payload['tasks_total']} failed={payload['tasks_failed']}",
            flush=True,
        )
        if status in {"COMPLETE", "FAILED"}:
            raise SystemExit(0 if status == "COMPLETE" else 1)
        time.sleep(30)


if __name__ == "__main__":
    main()
