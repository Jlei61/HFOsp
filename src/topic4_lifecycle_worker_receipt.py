"""Durable lifecycle-worker status with heartbeat and terminal cause.

The scientific result remains in its immutable summary/NPZ.  This mutable
receipt exists only to make a missing result auditable: a stale ``running``
receipt is distinguishable from a Python exception or a clean success.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import tempfile
import threading
import time
import traceback


TERMINAL_STATUSES = {
    "success", "scientific_early_stop", "python_exception", "signal_exit",
    "oom_suspected", "timeout", "resource_abort",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


class WorkerReceipt:
    def __init__(self, path, *, config_hash, git_sha, command, heartbeat_s=60.0):
        self.path = Path(path)
        self.heartbeat_s = float(heartbeat_s)
        self.payload = {
            "schema": "topic4_lifecycle_worker_receipt_v1_2026-08-02",
            "status": "running",
            "pid": os.getpid(),
            "start_time_utc": _utc_now(),
            "heartbeat_time_utc": _utc_now(),
            "config_hash": str(config_hash),
            "git_sha": str(git_sha),
            "command": str(command),
            "checkpoint_hash": None,
            "artifact_path": None,
            "peak_rss_gb": _rss_gb(),
        }
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._old_handlers = {}

    def _write(self, **updates) -> None:
        with self._lock:
            self.payload.update(updates)
            self.payload["heartbeat_time_utc"] = _utc_now()
            self.payload["peak_rss_gb"] = max(
                float(self.payload.get("peak_rss_gb", 0.0)), _rss_gb()
            )
            _atomic_json(self.path, self.payload)

    def update_context(self, *, checkpoint_hash=None, **fields) -> None:
        updates = dict(fields)
        if checkpoint_hash is not None:
            updates["checkpoint_hash"] = str(checkpoint_hash)
        self._write(**updates)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_s):
            self._write()

    def _handle_signal(self, signum, _frame):
        self._write(
            status="signal_exit", terminal_time_utc=_utc_now(), signal=int(signum)
        )
        raise SystemExit(128 + int(signum))

    def __enter__(self):
        self._write()
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        return self

    def finish(self, status="success", *, artifact_path=None, **fields) -> None:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unknown terminal status {status!r}")
        self._write(
            status=status,
            terminal_time_utc=_utc_now(),
            artifact_path=None if artifact_path is None else str(artifact_path),
            **fields,
        )

    def __exit__(self, exc_type, exc, _tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(1.0, self.heartbeat_s))
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)
        if exc_type is None:
            if self.payload.get("status") == "running":
                self.finish("success")
            return False
        if self.payload.get("status") == "running":
            self.finish(
                "python_exception",
                exception_type=exc_type.__name__,
                exception_message=str(exc),
                traceback="".join(traceback.format_exception(exc_type, exc, _tb))[-12000:],
            )
        return False


def classify_stale_receipt(payload: dict, *, stale_after_s: float, now_epoch=None) -> str:
    """Supervisor-side classification; never rewrites a receipt itself."""
    if payload.get("status") != "running":
        return str(payload.get("status"))
    stamp = datetime.fromisoformat(str(payload["heartbeat_time_utc"]))
    now = time.time() if now_epoch is None else float(now_epoch)
    age = now - stamp.timestamp()
    return "resource_abort" if age > float(stale_after_s) else "running"
