#!/usr/bin/env python3
"""Detached wait -> aggregate -> single-candidate extension pipeline for LC5v2.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode"
BLOCK_DONE = OUT / "lc5v2p1_boundary_patch_block_DONE.json"
BLOCK_FAILED = OUT / "lc5v2p1_boundary_patch_block_FAILED.json"
AGGREGATE = ROOT / "scripts/aggregate_topic4_fcxr_lc5v2p1_phase_map.py"
EXTEND = ROOT / "scripts/run_topic4_fcxr_lc5v2p1_candidate_extension.py"
PHASE_MAP = OUT / "lc5v2p1_joint_phase_map/phase_map.json"
PYTHON = "/home/honglab/leijiaxin/anaconda3/bin/python"


def wait_for_block(poll_s):
    while True:
        if BLOCK_DONE.is_file():
            return "DONE"
        if BLOCK_FAILED.is_file():
            return "FAILED"
        time.sleep(poll_s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()
    if args.poll_s <= 0:
        raise ValueError("poll interval must be positive")
    status = wait_for_block(args.poll_s)
    if status != "DONE":
        raise SystemExit("boundary patch failed; aggregation withheld")
    subprocess.run([PYTHON, str(AGGREGATE)], cwd=ROOT, check=True)
    payload = json.loads(PHASE_MAP.read_text())
    candidate = payload.get("primary_extension_candidate")
    if candidate is None:
        print(json.dumps({"status": "DONE_NO_EXTENSION_CANDIDATE"}, indent=2), flush=True)
        return
    subprocess.run(
        [PYTHON, str(EXTEND), "--source-summary", candidate["source_summary"], "--confirm-run"],
        cwd=ROOT, check=True,
    )
    print(json.dumps({
        "status": "DONE_WITH_EXTENSION", "candidate": candidate,
        "finished_epoch_s": time.time(),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
