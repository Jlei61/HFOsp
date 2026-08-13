#!/usr/bin/env python3
"""Resume the LC5v2.1 block finalizer after a manually refilled cell finishes.

The original block dispatcher can be paused while spare worker slots are refilled manually.
This monitor waits for the last manual cell to publish an atomic DONE/FAILED sentinel, validates
that the paused PID is still the intended block dispatcher, resumes it, and then waits for the
block-level terminal sentinel.  It performs no simulation and changes no scientific result.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time


EXPECTED_DISPATCHER = "run_topic4_fcxr_lc5v2p1_phase_map_block.py"


def _cmdline(pid: int) -> str:
    path = Path("/proc") / str(pid) / "cmdline"
    if not path.is_file():
        raise RuntimeError(f"dispatcher PID {pid} no longer exists")
    return path.read_bytes().replace(b"\0", b" ").decode(errors="replace")


def validate_dispatcher(pid: int) -> str:
    cmdline = _cmdline(pid)
    if EXPECTED_DISPATCHER not in cmdline:
        raise RuntimeError(
            f"refusing to signal PID {pid}: expected {EXPECTED_DISPATCHER!r}, got {cmdline!r}"
        )
    return cmdline


def _wait_for_either(done: Path, failed: Path, poll_s: float) -> tuple[str, Path]:
    while True:
        if done.is_file():
            return "DONE", done
        if failed.is_file():
            return "FAILED", failed
        time.sleep(poll_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher-pid", type=int, required=True)
    parser.add_argument("--cell-done", type=Path, required=True)
    parser.add_argument("--cell-failed", type=Path, required=True)
    parser.add_argument("--block-done", type=Path, required=True)
    parser.add_argument("--block-failed", type=Path, required=True)
    parser.add_argument("--poll-s", type=float, default=30.0)
    args = parser.parse_args()

    if args.poll_s <= 0:
        raise ValueError("poll interval must be positive")
    initial_cmdline = validate_dispatcher(args.dispatcher_pid)
    cell_status, cell_sentinel = _wait_for_either(
        args.cell_done.resolve(), args.cell_failed.resolve(), args.poll_s
    )
    # Resume even after a failed manual cell so that the canonical dispatcher can publish its own
    # block-level failure instead of leaving a stale RUNNING sentinel.
    validate_dispatcher(args.dispatcher_pid)
    os.kill(args.dispatcher_pid, signal.SIGCONT)
    block_status, block_sentinel = _wait_for_either(
        args.block_done.resolve(), args.block_failed.resolve(), args.poll_s
    )
    print(json.dumps({
        "status": block_status,
        "cell_status": cell_status,
        "cell_sentinel": str(cell_sentinel),
        "block_sentinel": str(block_sentinel),
        "dispatcher_pid": args.dispatcher_pid,
        "dispatcher_cmdline": initial_cmdline,
        "finished_epoch_s": time.time(),
    }, indent=2, sort_keys=True), flush=True)
    if cell_status != "DONE" or block_status != "DONE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
