#!/usr/bin/env python3
"""Exact-PID resource/deadline watchdog for detached LC2 jobs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import tempfile
import time
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _meminfo():
    with open("/proc/meminfo") as f:
        x = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=x["MemAvailable"] / 1024.0 / 1024.0,
                swap_used_mib=(x["SwapTotal"] - x["SwapFree"]) / 1024.0)


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".watchdog_", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--sid", type=int, required=True)
    ap.add_argument("--hours", type=float, default=9.5)
    ap.add_argument("--min-mem-gib", type=float, default=96.0)
    ap.add_argument("--max-swap-delta-mib", type=float, default=512.0)
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--status", required=True)
    args = ap.parse_args()
    if os.getsid(args.pid) != args.sid or args.pid != args.sid:
        raise SystemExit("refusing watchdog: target PID must be the verified session leader")
    start = time.monotonic()
    baseline = _meminfo()
    _write(args.status, dict(status="WATCHING", pid=args.pid, sid=args.sid, started=_now(),
                             baseline=baseline, deadline_hours=args.hours))
    reason = None
    last = baseline
    while alive(args.pid):
        last = _meminfo()
        elapsed_h = (time.monotonic() - start) / 3600.0
        if elapsed_h >= args.hours:
            reason = "deadline"
        elif last["mem_available_gib"] < args.min_mem_gib:
            reason = "memory_floor"
        elif last["swap_used_mib"] - baseline["swap_used_mib"] >= args.max_swap_delta_mib:
            reason = "swap_delta"
        if reason:
            os.killpg(args.sid, signal.SIGTERM)
            break
        time.sleep(args.poll_s)
    _write(args.status, dict(status="TERMINATED" if reason else "TARGET_EXITED", reason=reason,
                             pid=args.pid, sid=args.sid, started_baseline=baseline, last=last,
                             elapsed_hours=(time.monotonic() - start) / 3600.0, finished=_now()))


if __name__ == "__main__":
    main()
