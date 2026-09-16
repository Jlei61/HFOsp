#!/usr/bin/env python3
"""Recoverably finish v0.3.7 after the first completion supervisor exits.

The first detached supervisor was launched while the downstream queue was
still being extended.  This watcher deliberately waits for that process to
exit before it starts anything, so two supervisors can never own the same H2
queue.  Every stage is idempotent and requires the preceding machine status.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
BASE = Path("/data/hfosp_group_event_state_v0_3_7")


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _status(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status", "UNKNOWN"))


def _run(script: str, *args: str) -> None:
    subprocess.run([str(PYTHON), str(ROOT / "scripts" / script), *args], cwd=ROOT, check=True)


def _write(value: dict) -> None:
    path = BASE / "supervisor/post_completion_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-pid", type=int, required=True)
    args = parser.parse_args()
    _write({"status": "WAITING_FOR_FIRST_SUPERVISOR", "upstream_pid": args.upstream_pid})
    while _alive(args.upstream_pid):
        time.sleep(30.0)

    h1_status = BASE / "h1_optimized/supervisor/queue_status.json"
    if _status(h1_status) != "COMPLETE":
        # If the older supervisor stopped after recipe selection, finish H1.
        optimizer = BASE / "optimizer_search/supervisor/queue_status.json"
        if _status(optimizer) != "COMPLETE":
            raise RuntimeError("optimizer search did not complete")
        if not (BASE / "optimizer_search/summary.json").exists():
            _run("finalize_group_event_state_v037_optimizer_search.py")
        _run("supervise_group_event_state_v037_optimized_h1.py", "--workers-per-gpu", "2", "--gpus", "0,1")
    if _status(h1_status) != "COMPLETE":
        raise RuntimeError("optimized H1 is not complete")

    h2a_status = BASE / "h2a_optimized/supervisor/queue_status.json"
    h2b_status = BASE / "h2b_optimized/supervisor/queue_status.json"
    children: list[subprocess.Popen] = []
    if _status(h2a_status) != "COMPLETE":
        children.append(subprocess.Popen([
            str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v037_optimized_h2a.py"),
            "--workers-per-gpu", "2", "--gpus", "0,1",
        ], cwd=ROOT))
    if _status(h2b_status) != "COMPLETE":
        children.append(subprocess.Popen([
            str(PYTHON), str(ROOT / "scripts/supervise_group_event_state_v037_optimized_h2b.py"),
            "--workers", "2",
        ], cwd=ROOT))
    if any(child.wait() != 0 for child in children):
        raise RuntimeError("optimized H2a/H2b queue failed")
    if _status(h2a_status) != "COMPLETE" or _status(h2b_status) != "COMPLETE":
        raise RuntimeError("optimized H2a/H2b did not close")

    joint_status = BASE / "h2a_joint_sensitivity_optimized/supervisor/queue_status.json"
    if _status(joint_status) != "COMPLETE":
        _run(
            "supervise_group_event_state_v037_h2a_joint.py",
            "--workers", "2", "--gpus", "0,1",
            "--h1-root", str(BASE / "h1_optimized/event"),
            "--primary-h2a-root", str(BASE / "h2a_optimized/event"),
            "--out-root", str(BASE / "h2a_joint_sensitivity_optimized"),
        )

    _write({"status": "RUNNING_FINAL_TESTS_AND_REPORTS", "upstream_pid": args.upstream_pid})
    subprocess.run([
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_group_event_state_v037_ctssm.py",
        "tests/test_group_event_state_v037_decoder_audit.py",
        "tests/test_group_event_state_v037_h1_dual.py",
        "tests/test_group_event_state_v037_h1_train.py",
        "tests/test_group_event_state_v037_h2a.py",
        "tests/test_group_event_state_v037_h2b.py",
        "tests/test_group_event_state_v037_h3_generative.py",
        "tests/test_group_event_state_v037_strict_decoder_cache.py",
    ], cwd=ROOT, check=True)
    _run("finalize_group_event_state_v037.py")
    _write({
        "status": "COMPLETE", "upstream_pid": args.upstream_pid,
        "summary": str(BASE / "final_reports/integrated_summary_v2.json"),
        "manifest": str(BASE / "final_reports/manifest_v3.json"),
    })


if __name__ == "__main__":
    main()
