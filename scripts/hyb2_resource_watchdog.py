"""Live resource watchdog for the HYB2 sprint (plan section 9).

The plan's swap / memory thresholds were only evaluated at stage ENTRY, which cannot catch a
breach that develops mid-run.  This polls continuously and enforces them.

It only ever signals processes matching `run_topic4_fcxr_hyb2.py` owned by this user, and it kills
the NEWEST one first -- never a sibling, never a user process, never SIGKILL.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

OUT = sys.argv[1]
SWAP_BASELINE_MB = float(sys.argv[2])
MEM_FLOOR_GB = 96.0
SOFT_SWAP_MB, HARD_SWAP_MB = 256.0, 512.0
POLL_S = 15.0
PATTERN = "run_topic4_fcxr_hyb2.py"


def _mem():
    mi = dict(re.findall(r"(\w+):\s+(\d+)", open("/proc/meminfo").read()))
    avail = int(mi["MemAvailable"]) / 1048576.0
    swap_mb = (int(mi["SwapTotal"]) - int(mi["SwapFree"])) / 1024.0
    return avail, swap_mb


def _mine():
    """(pid, start_seconds, rss_gb) for my HYB2 runs, newest first."""
    out = subprocess.run(["ps", "-eo", "pid,lstart,rss,cmd", "--no-headers"],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        if PATTERN in ln and "watchdog" not in ln and "/bin/bash" not in ln:
            f = ln.split()
            try:
                rows.append((int(f[0]), int(f[5]) if f[5].isdigit() else 0, int(f[6]) / 1048576.0))
            except (ValueError, IndexError):
                continue
    return sorted(rows, key=lambda r: -r[0])          # highest pid ~ newest


def _log(rec):
    with open(os.path.join(OUT, "watchdog.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def main():
    growing = 0
    prev_swap = None
    while True:
        avail, swap = _mem()
        delta = swap - SWAP_BASELINE_MB
        mine = _mine()
        rec = dict(t=datetime.now(timezone.utc).isoformat(), mem_available_gb=round(avail, 1),
                   swap_used_mb=round(swap, 1), swap_delta_mb=round(delta, 1),
                   n_runs=len(mine), rss_gb=[round(r[2], 1) for r in mine])
        breach = None
        if delta > HARD_SWAP_MB and prev_swap is not None and swap > prev_swap:
            growing += 1
            if growing >= 2:
                breach = f"swap delta {delta:.0f} MB > {HARD_SWAP_MB} and still growing"
        else:
            growing = 0
        if avail < MEM_FLOOR_GB:
            breach = f"MemAvailable {avail:.0f} GB < {MEM_FLOOR_GB} GB floor"
        if delta > SOFT_SWAP_MB and breach is None:
            rec["soft"] = "stop submitting new work"
            open(os.path.join(OUT, "STOP_SUBMITTING.flag"), "w").write(rec["t"])
        if breach and mine:
            pid = mine[0][0]
            rec["action"] = f"SIGTERM newest own run pid {pid}"
            rec["breach"] = breach
            os.kill(pid, signal.SIGTERM)              # SIGTERM only; never SIGKILL, never siblings
            with open(os.path.join(OUT, "RESOURCE_PAUSED.json"), "w") as f:
                json.dump(rec, f, indent=2)
            _log(rec)
            return 1
        _log(rec)
        prev_swap = swap
        time.sleep(POLL_S)


if __name__ == "__main__":
    raise SystemExit(main())
