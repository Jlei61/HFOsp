#!/usr/bin/env python
"""One-shot crash/disconnect-safe resume guard for an intentionally paused Z/M worker.

It never launches a new simulation.  It resumes exactly one pre-existing PID
only after fewer than ``max_active`` watched full-SNN workers remain and the
locked memory/swap guards pass.  Every decision is appended to JSONL.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


def pid_cmd(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return ""


def pid_state(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return "missing"


def memory_status_mb():
    vals = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemAvailable", "SwapTotal", "SwapFree"}:
            vals[key] = int(value.strip().split()[0]) // 1024
    vals["SwapUsed"] = vals["SwapTotal"] - vals["SwapFree"]
    return vals


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-pid", type=int, required=True)
    ap.add_argument("--wait-pids", type=int, nargs="+", required=True)
    ap.add_argument("--max-active", type=int, default=2)
    ap.add_argument("--min-mem-gb", type=float, default=96.0)
    ap.add_argument("--max-swap-mb", type=int, required=True)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--expect", required=True,
                    help="substring that must remain present in resume PID cmdline")
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    log = Path(a.log)

    while True:
        cmd = pid_cmd(a.resume_pid)
        states = {p: pid_state(p) for p in a.wait_pids}
        alive_active = [p for p, st in states.items() if st != "missing"]
        mem = memory_status_mb()
        row = dict(ts=time.strftime("%Y-%m-%dT%H:%M:%S"), event="poll",
                   resume_pid=a.resume_pid, resume_state=pid_state(a.resume_pid),
                   active_wait_pids=alive_active, wait_states=states, mem=mem)
        if not cmd or a.expect not in cmd:
            row.update(event="abort", reason="resume_pid_missing_or_cmd_mismatch", cmd=cmd)
            append_jsonl(log, row)
            return 2
        if len(alive_active) < a.max_active:
            if mem["MemAvailable"] < int(a.min_mem_gb * 1024):
                row.update(event="hold", reason="memory_guard")
            elif mem["SwapUsed"] > a.max_swap_mb:
                row.update(event="hold", reason="swap_growth_guard")
            else:
                os.kill(a.resume_pid, signal.SIGCONT)
                row.update(event="resumed", reason="worker_slot_available", cmd=cmd)
                append_jsonl(log, row)
                return 0
        append_jsonl(log, row)
        time.sleep(max(5.0, min(a.poll, 60.0)))


if __name__ == "__main__":
    raise SystemExit(main())
