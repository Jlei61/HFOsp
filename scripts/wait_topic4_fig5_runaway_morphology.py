#!/usr/bin/env python3
"""Low-frequency completion monitor for the two Figure 5 morphology canaries."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = (ROOT / "results" / "topic4_sef_hfo"
       / "data_driven_zm_ictal_transition" / "runaway_morphology")
EXPECTED = (
    OUT / "qigk_e1146_reference.json",
    OUT / "joint_seed_1801_post2s.json",
)
UNITS = (
    "topic4-fig5-qigk-morphology.service",
    "topic4-fig5-joint-morphology.service",
)


def _active(unit):
    result = subprocess.run(
        ["systemctl", "--user", "is-active", "--quiet", unit], check=False)
    return result.returncode == 0


def main():
    status_path = OUT / "monitor_status.json"
    OUT.mkdir(parents=True, exist_ok=True)
    while True:
        complete = [path.exists() for path in EXPECTED]
        active = [_active(unit) for unit in UNITS]
        status = {
            "checked_unix_s": time.time(),
            "poll_interval_s": 600,
            "artifacts_complete": dict(zip(map(str, EXPECTED), complete)),
            "units_active": dict(zip(UNITS, active)),
        }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        if all(complete):
            result = subprocess.run([
                str(Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")),
                str(ROOT / "scripts" / "compare_topic4_fig5_runaway_morphology.py"),
            ], cwd=ROOT, check=False, capture_output=True, text=True)
            status["comparison_returncode"] = result.returncode
            status["comparison_stdout"] = result.stdout
            status["comparison_stderr"] = result.stderr
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            message = ("Fig5 morphology comparison complete" if result.returncode == 0
                       else "Fig5 morphology comparison failed")
            subprocess.run(["notify-send", "Topic 4", message], check=False)
            raise SystemExit(result.returncode)
        failed = [
            unit for unit, is_active, is_complete in zip(UNITS, active, complete)
            if not is_active and not is_complete
        ]
        if failed:
            status["failed_units"] = failed
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            subprocess.run([
                "notify-send", "Topic 4", "Fig5 morphology worker failed"
            ], check=False)
            raise SystemExit(1)
        time.sleep(600)


if __name__ == "__main__":
    main()
