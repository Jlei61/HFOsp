#!/usr/bin/env python3
"""Monitor formal v2.3 artifacts and advance to analysis when complete."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FORMAL = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3/formal"
)
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
ANALYZER = ROOT / "scripts/analyze_topic5_competitive_propagation_formal_v2_3.py"


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
    watcher = FORMAL / "WATCHER_STATE.json"
    while True:
        launcher = read_json(FORMAL / "LAUNCHER_STATE.json")
        markov = read_json(FORMAL / "MARKOV_BENCHMARK_STATE.json")
        formal_status = launcher.get("status")
        markov_status = markov.get("status")
        if formal_status == "FAILED" or markov_status == "FAILED":
            atomic_json(
                watcher,
                {
                    "status": "BLOCKED_UPSTREAM_FAILURE",
                    "formal_status": formal_status,
                    "markov_status": markov_status,
                    "updated_unix": time.time(),
                    "target_values_read": False,
                },
            )
            raise SystemExit("formal or Markov upstream failed")
        if formal_status == "COMPLETE" and markov_status == "COMPLETE":
            atomic_json(
                watcher,
                {
                    "status": "ANALYZING",
                    "formal_status": formal_status,
                    "markov_status": markov_status,
                    "updated_unix": time.time(),
                    "target_values_read": False,
                },
            )
            result = subprocess.run(
                [PYTHON, str(ANALYZER)],
                cwd=ROOT,
                check=False,
            )
            status = "COMPLETE" if result.returncode == 0 else "ANALYSIS_FAILED"
            atomic_json(
                watcher,
                {
                    "status": status,
                    "analysis_returncode": result.returncode,
                    "formal_status": formal_status,
                    "markov_status": markov_status,
                    "finished_unix": time.time(),
                    "target_values_read": False,
                },
            )
            raise SystemExit(result.returncode)
        atomic_json(
            watcher,
            {
                "status": "MONITORING",
                "formal_status": formal_status,
                "formal_tasks_finished": launcher.get("n_tasks_finished", 0),
                "formal_tasks_total": launcher.get("n_tasks_total", 66),
                "formal_tasks_failed": launcher.get("n_tasks_failed", 0),
                "markov_status": markov_status,
                "updated_unix": time.time(),
                "target_values_read": False,
            },
        )
        print(
            f"formal={formal_status} "
            f"{launcher.get('n_tasks_finished', 0)}/"
            f"{launcher.get('n_tasks_total', 66)} "
            f"markov={markov_status}",
            flush=True,
        )
        time.sleep(30)


if __name__ == "__main__":
    main()
