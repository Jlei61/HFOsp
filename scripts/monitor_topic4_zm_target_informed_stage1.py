#!/usr/bin/env python3
"""Low-frequency monitor and Stage-1 rescore trigger for Topic 4 rev5."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/target_informed_bridge_v1"
FIT_IDS = (
    "si070_tz2500", "si070_tz5000", "si070_tz10000",
    "si080_tz2500", "si080_tz10000",
    "si090_tz2500", "si090_tz5000", "si090_tz10000",
)


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _snapshot():
    fit_done = {candidate: (OUT / "fit" / f"{candidate}.json").exists()
                for candidate in FIT_IDS}
    usage = shutil.disk_usage(ROOT)
    mem_available_kib = None
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_available_kib = int(line.split()[1])
            break
    return {
        "timestamp_epoch": time.time(),
        "fit_done": fit_done,
        "n_fit_done": int(sum(fit_done.values())),
        "clinical_target_done": (OUT / "clinical_target.json").exists(),
        "paired_baseline_done": (OUT / "paired_baseline/seed1801_zmoff.npz").exists(),
        "memory_available_gib": (None if mem_available_kib is None
                                  else mem_available_kib / 1024**2),
        "disk_free_gib": usage.free / 1024**3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=float, default=600.0)
    args = parser.parse_args()
    status_path = OUT / "stage1_monitor.json"
    while True:
        status = _snapshot()
        ready = (status["n_fit_done"] == len(FIT_IDS)
                 and status["clinical_target_done"]
                 and status["paired_baseline_done"])
        status["status"] = "STAGE1_INPUTS_COMPLETE" if ready else "STAGE1_RUNNING"
        _atomic_json(status_path, status)
        print(json.dumps(status), flush=True)
        if ready:
            break
        if status["memory_available_gib"] is not None and status["memory_available_gib"] < 32.0:
            status["status"] = "MEMORY_RESERVE_VIOLATED"
            _atomic_json(status_path, status)
            raise SystemExit("memory reserve below 32 GiB")
        if status["disk_free_gib"] < 20.0:
            status["status"] = "DISK_RESERVE_VIOLATED"
            _atomic_json(status_path, status)
            raise SystemExit("disk reserve below 20 GiB")
        time.sleep(float(args.interval_seconds))
    command = [
        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
        "scripts/rescore_topic4_fig5_target_informed_candidates.py",
        "--config", "config/topic4_data_driven_zm_target_informed_bridge_v1.json",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    status.update({
        "status": ("STAGE1_RESCORE_COMPLETE" if completed.returncode == 0
                   else "STAGE1_RESCORE_FAILED"),
        "rescore_returncode": int(completed.returncode),
        "rescore_stdout": completed.stdout[-4000:],
        "rescore_stderr": completed.stderr[-4000:],
    })
    _atomic_json(status_path, status)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Topic 4 rev5", "Stage 1 rescore complete"],
                       check=False)


if __name__ == "__main__":
    main()
