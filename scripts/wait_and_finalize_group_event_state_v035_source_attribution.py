#!/usr/bin/env python3
"""Wait for source attribution and render its report exactly once."""

from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DONE = Path("/data/hfosp_group_event_state_v0_3_5/source_attribution/supervisor/queue_done.json")
REPORT = Path("/data/hfosp_group_event_state_v0_3_5/source_attribution/reports/summary.json")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")

while not DONE.is_file():
    time.sleep(30)
if not REPORT.is_file():
    subprocess.run([str(PYTHON), str(ROOT / "scripts/finalize_group_event_state_v035_source_attribution.py")], cwd=ROOT, check=True)
