#!/usr/bin/env python3
"""Resume LC2 E4 automatically after the detached E3 grid reaches a canonical terminal."""
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


def _write(name, payload):
    path = os.path.join(OUT, name)
    fd, tmp = tempfile.mkstemp(prefix=".chain_", dir=OUT)
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


def _run(*args):
    print("[chain]", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e3-pid", type=int, required=True)
    ap.add_argument("--poll-s", type=float, default=30.0)
    args = ap.parse_args()
    _write("E3_TO_E4_CHAIN.json", dict(status="WAITING_E3", chain_pid=os.getpid(),
                                        e3_pid=args.e3_pid, started=_now()))
    while alive(args.e3_pid):
        time.sleep(args.poll_s)
    if not os.path.isfile(os.path.join(OUT, "E3_DONE.json")):
        _write("E3_TO_E4_CHAIN.json", dict(status="STOPPED_WITHOUT_CANONICAL_E3",
                                            chain_pid=os.getpid(), e3_pid=args.e3_pid,
                                            e3_failed=os.path.isfile(os.path.join(OUT, "E3_FAILED.json")),
                                            watchdog=json.load(open(os.path.join(OUT, "E3_WATCHDOG.json")))
                                            if os.path.isfile(os.path.join(OUT, "E3_WATCHDOG.json")) else None,
                                            finished=_now()))
        return
    try:
        _run("scripts/plot_topic4_fcxr_lc2_exploration.py")
        _run("scripts/run_topic4_fcxr_lc2_forks.py", "manifest")
        manifest = json.load(open(os.path.join(OUT, "frozen_fork_manifest.json")))
        if manifest["n_finalists"] == 0:
            _write("E3_TO_E4_CHAIN.json", dict(status="COMPLETE_NO_SCREEN_SURVIVOR",
                                                chain_pid=os.getpid(), finished=_now()))
            return
        _write("E3_TO_E4_CHAIN.json", dict(status="RUNNING_E4", chain_pid=os.getpid(),
                                            n_finalists=manifest["n_finalists"],
                                            n_rows=manifest["n_rows"], started_e4=_now()))
        _run("scripts/run_topic4_fcxr_lc2_forks.py", "all", "--workers", "2", "--confirm-run")
        _run("scripts/plot_topic4_fcxr_lc2_exploration.py")
        _write("E3_TO_E4_CHAIN.json", dict(status="COMPLETE", chain_pid=os.getpid(), finished=_now()))
    except Exception as exc:
        _write("E3_TO_E4_CHAIN.json", dict(status="FAILED", chain_pid=os.getpid(),
                                            error=repr(exc), finished=_now()))
        raise


if __name__ == "__main__":
    main()
