"""Traceability guard for the gitignored SNN engine (spec 2026-06-08 §7 hard
contract). The engine lives under results/.../engine and is NOT git-tracked, so a
silent edit there would make Step-3 results unreproducible. record_versions()
snapshots sha256 of the engine files a runner imports; the runner calls
assert_versions() at startup so any later drift fails loudly."""
from __future__ import annotations

import hashlib
from pathlib import Path


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record_versions(paths) -> dict:
    """Map each engine file path -> its sha256 (snapshot for engine_versions.json)."""
    return {str(p): _sha256(str(p)) for p in paths}


def assert_versions(recorded: dict) -> None:
    """Raise if any recorded engine file's current sha256 differs (drift = loud fail)."""
    for p, h in recorded.items():
        cur = _sha256(p)
        if cur != h:
            raise RuntimeError(
                f"SNN engine drift: {p} sha256 {cur[:12]} != recorded {h[:12]}. "
                "Re-snapshot engine_versions.json only after reviewing the change.")
