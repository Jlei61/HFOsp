"""Traceability guard for the SNN engine (spec 2026-06-08 §7 hard contract).
The engine now lives git-tracked under src/snn_engine/ (moved out of the
gitignored results/ tree 2026-06-15); git history is the primary integrity
record. This guard remains as an explicit reproducibility assertion: a runner
records sha256 of the engine files it imports (record_versions) and calls
assert_versions() at startup, so an unreviewed in-place edit fails loudly
before a run even if it isn't yet committed."""
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
