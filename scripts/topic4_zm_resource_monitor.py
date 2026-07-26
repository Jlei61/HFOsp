#!/usr/bin/env python
"""Durable resource log for the Z/M branch-decision line (spec rev3.1 §14.10-§14.12).

Appends one JSON line every `--interval` seconds with CPU%, per-worker RSS, MemAvailable, swap, and
the PID/cmdline of every live branch-decision worker, so a stopped/killed run still leaves evidence
of what the machine was doing. Read-only: it never kills anything -- the launcher owns that decision.

  python scripts/topic4_zm_resource_monitor.py --interval 120 &
"""
from __future__ import annotations

import argparse
import json
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision", "resource_log.jsonl")
MATCH = ("topic4_zm", "zm_branch_decision", "run_zm_snn_native_exit", "audit_topic4_zm")


def meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                out[k] = int(v.split()[0]) // 1024  # MB
    out["SwapUsed"] = out.get("SwapTotal", 0) - out.get("SwapFree", 0)
    return out


def workers():
    rows = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\x00", b" ").decode(errors="replace").strip()
            if not any(m in cmd for m in MATCH):
                continue
            rss = 0
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) // 1024
                        break
            rows.append(dict(pid=int(pid), rss_mb=rss, cmd=cmd[:220]))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--max-hours", type=float, default=14.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    t_end = time.time() + a.max_hours * 3600.0
    while time.time() < t_end:
        rec = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), load=os.getloadavg(),
                   mem=meminfo(), workers=workers())
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
