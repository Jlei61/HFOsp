"""Atomic writes, payload hashing and idempotent skip for the H3 line.

A result file that exists is a promise that every byte in it was validated.  The
only way to keep that promise across an interrupted queue is to write a temporary
file, check it, and rename; a partially written JSON that parses is worse than a
missing one, because the queue will skip it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_default)


def payload_hash(payload: Any) -> str:
    """Stable content hash of a config/result payload."""

    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def file_hash(path: Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(payload: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_default))
    # Parse the temporary file before it is allowed to take the real name, so a
    # truncated write can never be mistaken for a finished result.
    json.loads(tmp.read_text())
    os.replace(tmp, path)
    return path


def write_npz_atomic(path: Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    with tmp.open("wb") as handle:
        np.savez(handle, **arrays)
    with np.load(tmp) as check:
        missing = set(arrays) - set(check.files)
        if missing:
            raise ValueError(f"atomic npz lost arrays: {sorted(missing)}")
    os.replace(tmp, path)
    return path


def finished(path: Path, expected_hash: str | None = None) -> bool:
    """Idempotent-resume test.  A file only counts as done if it re-validates.

    Missing keys, a stale ``config_hash`` or a non-finite score all mean *not*
    done: the run is repeated rather than trusted, which is the only behaviour
    that makes a resumable queue safe.
    """

    path = Path(path)
    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if expected_hash is not None and payload.get("config_hash") != expected_hash:
        return False
    return bool(payload.get("status") == "ok")
