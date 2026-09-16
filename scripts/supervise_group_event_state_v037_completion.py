#!/usr/bin/env python3
"""Finish expanded optimizer search, freeze recipes, then launch formal H1."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
BASE = Path("/data/hfosp_group_event_state_v0_3_7")


def _run(script: str, *args: str) -> None:
    command = [str(PYTHON), str(ROOT / "scripts" / script), *args]
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    first = BASE / "optimizer_search/supervisor/queue_status.json"
    while True:
        if first.exists() and json.loads(first.read_text()).get("status") in {"COMPLETE", "FAILED"}:
            break
        time.sleep(20.0)
    if json.loads(first.read_text()).get("status") != "COMPLETE":
        raise SystemExit("initial optimizer search failed")
    # The running supervisor was started before the pilot was expanded to four
    # subjects and the dual model. A second idempotent pass fills those cells.
    _run("supervise_group_event_state_v037_optimizer_search.py", "--workers-per-gpu", "2", "--gpus", "0,1")
    _run("finalize_group_event_state_v037_optimizer_search.py")
    _run("supervise_group_event_state_v037_optimized_h1.py", "--workers-per-gpu", "2", "--gpus", "0,1")
    h2a = subprocess.Popen([
        str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v037_optimized_h2a.py"),
        "--workers-per-gpu", "2", "--gpus", "0,1",
    ], cwd=ROOT)
    h2b = subprocess.Popen([
        str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v037_optimized_h2b.py"),
        "--workers", "2",
    ], cwd=ROOT)
    if h2a.wait() != 0 or h2b.wait() != 0:
        raise SystemExit("optimized H2a or H2b failed")
    _run(
        "supervise_group_event_state_v037_h2a_joint.py",
        "--workers", "2", "--gpus", "0,1",
        "--h1-root", str(BASE / "h1_optimized/event"),
        "--primary-h2a-root", str(BASE / "h2a_optimized/event"),
        "--out-root", str(BASE / "h2a_joint_sensitivity_optimized"),
    )


if __name__ == "__main__": main()
