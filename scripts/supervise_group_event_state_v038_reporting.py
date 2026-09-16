#!/usr/bin/env python3
"""Continuously refresh v0.3.8 reports and render final decision figures.

The incremental JSON is explicitly non-final.  Paper decision figures are only
rendered after the master closure queue reports COMPLETE.  A FAILED master
queue is propagated and never converted into a scientific closeout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
BASE = Path("/data/hfosp_group_event_state_v0_3_8_core_expansion")
MASTER = BASE / "supervisor/queue_status.json"
STATUS = BASE / "reporting/queue_status.json"


def _read(path: Path) -> dict:
    if not path.exists():
        return {"status": "PENDING"}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _count_cards() -> dict[str, int]:
    roots = {
        "h1": (BASE / "h1_long_8h", BASE / "h1_medium_2h"),
        "h2a": (BASE / "h2a_long_8h", BASE / "h2a_medium_2h"),
        "h2b": (BASE / "h2b_long_8h/outcomes", BASE / "h2b_medium_2h/outcomes"),
        "trained_credit": (BASE / "trained_credit_long",),
        "random_background": (
            BASE / "dual_random_background_long_v2", BASE / "dual_random_background_medium_v2",
        ),
    }
    return {
        name: sum(1 for root in directories for _path in root.glob("**/card.json"))
        if name in {"h1", "h2a", "h2b"} else
        sum(1 for root in directories for path in root.glob("**/*.json")
            if path.name != "queue_status.json")
        for name, directories in roots.items()
    }


def _run(command: list[str], log_name: str) -> None:
    log = BASE / "reporting" / log_name
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
            check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )


def main() -> None:
    previous_counts: dict[str, int] | None = None
    while True:
        master = _read(MASTER)
        counts = _count_cards()
        if master.get("status") == "FAILED":
            _atomic(STATUS, {
                "status": "FAILED", "reason": "master queue failed",
                "master": master, "card_counts": counts,
            })
            raise SystemExit(1)

        if counts != previous_counts:
            _run([
                str(PYTHON), "scripts/finalize_group_event_state_v038.py",
                "--data-root", str(BASE), "--allow-incomplete",
            ], "incremental.log")
            previous_counts = counts

        _atomic(STATUS, {
            "status": "RUNNING" if master.get("status") != "COMPLETE" else "FINALIZING",
            "master_status": master.get("status", "PENDING"),
            "card_counts": counts,
            "incremental_summary": str(BASE / "final_reports/summary_incremental.json"),
        })

        if master.get("status") == "COMPLETE":
            _run([
                str(PYTHON), "scripts/finalize_group_event_state_v038.py",
                "--data-root", str(BASE),
            ], "finalize.log")
            summary = BASE / "final_reports/summary_main.json"
            _run([
                str(PYTHON), "scripts/paper_figures/plot_group_event_state_v038_core_closure.py",
                "--summary", str(summary),
                "--out-dir", str(BASE / "final_reports/figures"),
            ], "figures.log")
            _atomic(STATUS, {
                "status": "COMPLETE", "master_status": "COMPLETE",
                "card_counts": _count_cards(), "summary": str(summary),
                "figures": str(BASE / "final_reports/figures"),
            })
            return
        time.sleep(60.0)


if __name__ == "__main__":
    main()
