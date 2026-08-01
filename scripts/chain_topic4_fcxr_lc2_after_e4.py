#!/usr/bin/env python3
"""Run the one-candidate E5 pilot only after canonical E4 geometry unlocks it."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core", "closed_loop_exploration")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(payload):
    path = os.path.join(OUT, "E4_TO_E5_CHAIN.json")
    fd, tmp = tempfile.mkstemp(prefix=".e5chain_", dir=OUT)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _run(*args):
    print("[E5-chain]", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4-chain-pid", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=30.0)
    args = ap.parse_args()
    _write(dict(status="WAITING_E4", pid=os.getpid(), e4_chain_pid=args.e4_chain_pid, started=_now()))
    while _alive(args.e4_chain_pid):
        time.sleep(args.poll_s)
    if not os.path.isfile(os.path.join(OUT, "E4_DONE.json")):
        _write(dict(status="STOPPED_WITHOUT_CANONICAL_E4", pid=os.getpid(), finished=_now()))
        return
    try:
        _run("scripts/run_topic4_fcxr_lc2_dynamic.py", "manifest")
        m = json.load(open(os.path.join(OUT, "dynamic_pilot_manifest.json")))
        if m["status"] != "LOCKED":
            _write(dict(status="COMPLETE_NOT_UNLOCKED", pid=os.getpid(), finished=_now()))
            return
        _write(dict(status="RUNNING_E5", pid=os.getpid(), n_rows=len(m["rows"]), started_e5=_now()))
        _run("scripts/run_topic4_fcxr_lc2_dynamic.py", "run", "--confirm-run")
        _run("scripts/plot_topic4_fcxr_lc2_exploration.py")
        _write(dict(status="COMPLETE", pid=os.getpid(), finished=_now()))
    except Exception as exc:
        _write(dict(status="FAILED", pid=os.getpid(), error=repr(exc), finished=_now()))
        raise


if __name__ == "__main__":
    main()
