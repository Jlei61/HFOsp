"""Queue ownership, resource leases and idempotent per-subject outputs (EI 4-5).

Three agents share two GPUs and one filesystem, and an old v0.1 queue is still
running.  The rules encoded here:

* a lease is created with ``O_EXCL`` and carries PID, PGID and a heartbeat, so a
  stale lease can be reclaimed only after both the PID is gone *and* the
  heartbeat is old -- never on a hunch;
* processes are managed by recorded PID.  ``pkill -f`` is banned in this
  repository because the pattern once matched the shell that issued it, killing
  the caller mid-script while every later line silently did not run;
* a result exists only after a complete payload has been renamed into place, and
  a rerun with the same configuration hash skips it.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import time
from typing import Any, Mapping

from .registry import atomic_write_json, payload_hash

STALE_HEARTBEAT_SECONDS = 900.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class ResourceLease:
    path: Path
    agent: str

    def payload(self, **extra: Any) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "pid": os.getpid(),
            "pgid": os.getpgid(0),
            "started_epoch": getattr(self, "_started", time.time()),
            "heartbeat_epoch": time.time(),
            **extra,
        }

    def acquire(self, **extra: Any) -> None:
        self._started = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing = json.loads(self.path.read_text())
            pid = int(existing.get("pid", -1))
            age = time.time() - float(existing.get("heartbeat_epoch", 0.0))
            if _pid_alive(pid) or age < STALE_HEARTBEAT_SECONDS:
                raise RuntimeError(
                    f"lease held by pid {pid} (heartbeat {age:.0f}s ago): "
                    f"{self.path}"
                )
        atomic_write_json(self.path, self.payload(**extra))

    def beat(self, **extra: Any) -> None:
        atomic_write_json(self.path, self.payload(**extra))

    def release(self) -> None:
        self.path.unlink(missing_ok=True)


def write_status(path: Path, **fields: Any) -> None:
    atomic_write_json(path, {"updated_epoch": time.time(), **fields})


def already_done(path: Path, config_hash: str) -> bool:
    """A result counts as done only if it parses *and* matches this config."""

    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return str(payload.get("config_hash", "")) == str(config_hash)


def save_result(path: Path, payload: Mapping[str, Any], config_hash: str) -> None:
    body = dict(payload)
    body["config_hash"] = str(config_hash)
    atomic_write_json(Path(path), body)


def config_fingerprint(*parts: Any) -> str:
    return payload_hash(list(parts))
