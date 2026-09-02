#!/usr/bin/env python3
"""Atomic resumable worker for the v0.3 two-GPU pilot queue."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _with_manifest(path: Path, callback):
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        payload = json.loads(path.read_text())
        result = callback(payload)
        _atomic(path, payload)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return result


def _output_exists(task: dict[str, Any]) -> bool:
    return Path(task["expected_output"]).exists()


def _claim(manifest: Path, worker: str):
    now = time.time()

    def callback(payload):
        by_id = {task["id"]: task for task in payload["tasks"]}
        for task in payload["tasks"]:
            if (
                task["status"] == "running"
                and not _output_exists(task)
                and now - float(task.get("started_epoch", now)) > 5 * 3600
            ):
                task["status"] = "pending"
                task["reclaimed_after_seconds"] = now - float(task["started_epoch"])
        for task in payload["tasks"]:
            if task["status"] == "complete" or _output_exists(task):
                task["status"] = "complete"
                continue
            if task["status"] != "pending":
                continue
            dependencies = [by_id[d]["status"] for d in task.get("depends_on", [])]
            if any(v == "failed" for v in dependencies):
                task["status"] = "failed"
                task["failure_reason"] = "blocked_by_failed_dependency"
                continue
            if dependencies and not all(v == "complete" for v in dependencies):
                continue
            task.update({
                "status": "running",
                "worker": worker,
                "started_epoch": now,
                "attempts": int(task.get("attempts", 0)) + 1,
            })
            payload["updated_epoch"] = now
            return dict(task)
        return None

    return _with_manifest(manifest, callback)


def _finish(manifest: Path, task_id: str, returncode: int, worker: str) -> None:
    now = time.time()

    def callback(payload):
        for task in payload["tasks"]:
            if task["id"] != task_id:
                continue
            success = returncode == 0 and _output_exists(task)
            task.update({
                "status": "complete" if success else "failed",
                "returncode": int(returncode),
                "finished_epoch": now,
                "worker": worker,
            })
        payload["updated_epoch"] = now

    _with_manifest(manifest, callback)


def _queue_state(manifest: Path) -> tuple[int, int, int]:
    def callback(payload):
        counts = {key: 0 for key in ("pending", "running", "complete", "failed")}
        for task in payload["tasks"]:
            if _output_exists(task):
                task["status"] = "complete"
            counts[task["status"]] += 1
        return counts["pending"], counts["running"], counts["failed"]

    return _with_manifest(manifest, callback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()
    worker = f"{socket.gethostname()}:{os.getpid()}:gpu{args.gpu}"
    while True:
        task = _claim(args.manifest, worker)
        if task is None:
            pending, running, failed = _queue_state(args.manifest)
            if pending == 0 and running == 0:
                print(f"queue closed: failed={failed}", flush=True)
                return
            time.sleep(args.poll_seconds)
            continue
        log = Path(task["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "GROUP_EVENT_STATE_SOURCE_COMMIT": str(task["source_commit"]),
        })
        print(f"claim {task['id']} -> {log}", flush=True)
        with log.open("a") as handle:
            handle.write(f"\nworker={worker} started={time.time()}\n")
            handle.flush()
            completed = subprocess.run(
                task["command"],
                cwd=task["workdir"],
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        _finish(args.manifest, task["id"], completed.returncode, worker)


if __name__ == "__main__":
    main()
