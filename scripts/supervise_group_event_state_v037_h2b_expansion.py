#!/usr/bin/env python3
"""Freeze and evaluate E916 only after both registered H1 families finish."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
EXPANSION = Path("/data/hfosp_group_event_state_v0_3_7/h1_h2b_expansion/queue_status.json")
OUT = Path("/data/hfosp_group_event_state_v0_3_7/h2b_frozen_transfer")
SUBJECT = "epilepsiae_916"
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _status() -> str:
    try: return json.loads(EXPANSION.read_text(encoding="utf-8"))["status"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError): return "WAITING"


def main() -> None:
    supervisor = OUT / "supervisor_expansion"; supervisor.mkdir(parents=True, exist_ok=True)
    while _status() not in {"COMPLETE", "FAILED"}: time.sleep(20)
    if _status() != "COMPLETE": raise RuntimeError("H1 H2b expansion failed")
    for seed in SEEDS:
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/freeze_group_event_state_v037_h2b.py"),
            "--subject", SUBJECT, "--seed", str(seed), "--out-root", str(OUT),
        ], cwd=ROOT, check=True)
    marker = supervisor / "ALL_E916_FEATURES_FROZEN"
    marker.write_text("All five E916 interictal feature files frozen before outcomes.\n", encoding="utf-8")
    for seed in SEEDS:
        subprocess.run([
            str(PYTHON), str(ROOT / "scripts/run_group_event_state_v037_h2b.py"),
            "--subject", SUBJECT, "--seed", str(seed), "--out-root", str(OUT),
        ], cwd=ROOT, check=True)
    payload = {
        "format": "group_event_state_v0_3_7_h2b_expansion_queue_v1", "status": "COMPLETE",
        "subject": SUBJECT, "features_frozen": 5, "outcomes_complete": 5,
        "selection_reason": "outcome-count estimability only; no H1-effect selection",
        "development_targets_read": False, "sealed_partition_opened": False,
    }
    temporary = supervisor / "queue_status.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, supervisor / "queue_status.json")


if __name__ == "__main__": main()

