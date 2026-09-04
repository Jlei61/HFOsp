#!/usr/bin/env python3
"""Wait for the corrected long-horizon H1/H2 queue and render its report."""

from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
LONG = Path("/data/hfosp_group_event_state_v0_3_5_long_observed_support")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
while not (LONG / "supervisor" / "queue_done.json").is_file():
    time.sleep(30)
subprocess.run(
    [str(PYTHON), str(ROOT / "scripts/finalize_group_event_state_v035_long_horizon.py")],
    cwd=ROOT, check=True,
)
