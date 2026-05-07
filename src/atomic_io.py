"""Atomic JSON write helpers for long-running cohort drivers.

A cohort run that writes per-subject JSON over hours can be interrupted
(Ctrl-C, kernel OOM, machine reboot). A non-atomic ``open(path, "w")``
truncates the destination first; an interruption mid-``json.dump`` then
leaves a half-written JSON which downstream consumers will silently
read as malformed or partial data.

The fix is to write to a sibling tempfile and atomically rename it
into place (``os.replace`` is POSIX-atomic when source and destination
are on the same filesystem). On failure the tempfile is removed and the
original destination is left untouched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Optional


def write_json_atomic(
    out_path: Path,
    payload: Any,
    *,
    indent: int = 2,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Atomically serialize ``payload`` as JSON to ``out_path``.

    Writes to ``out_path.with_suffix(out_path.suffix + ".tmp")`` first,
    then ``os.replace``s into the final location. On serialization
    failure the tempfile is unlinked and the existing destination
    (if any) is preserved verbatim.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=indent, default=default)
        os.replace(tmp, out_path)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def purge_stale_tmp(out_dir: Path, *, suffix: str = ".json.tmp") -> int:
    """Remove leftover ``*<suffix>`` files from a previous interrupted run.

    Returns count of files removed. Missing directory returns 0.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return 0
    n = 0
    for stale in out_dir.glob(f"*{suffix}"):
        try:
            stale.unlink()
            n += 1
        except OSError:
            pass
    return n
