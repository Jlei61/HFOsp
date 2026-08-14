"""Portable access to the accepted FCXR-LC1 seed-1 classifier constants.

The original full baseline JSON lived under an ignored, now-removed sibling worktree.  Later
protocols froze the small subset of its fields that the lifecycle reducer actually consumes.  This
module loads that tracked subset, validates its provenance and shape, and makes the loss of the
original full artifact explicit.  It does not recreate a baseline from a tested trajectory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "config/topic4_fcxr_lc1_seed1_classifier_snapshot.json"
EXPECTED_SCHEMA = "fcxr_lc1_seed1_classifier_snapshot_v1"
REQUIRED_TOP = ("frozen_event_bar", "af_bin_ms", "floor_af", "band")
REQUIRED_BAND = (
    "win_ms", "event_lookback_ms", "roll_hi", "recruit_p90",
    "event_rate_lo", "event_rate_hi",
)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_classifier_snapshot(payload: dict) -> dict:
    if payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("wrong FCXR-LC1 classifier snapshot schema")
    missing = [key for key in REQUIRED_TOP if key not in payload]
    if missing:
        raise ValueError(f"classifier snapshot missing fields: {missing}")
    band = payload["band"]
    if not isinstance(band, dict):
        raise TypeError("classifier snapshot band must be a mapping")
    missing = [key for key in REQUIRED_BAND if key not in band]
    if missing:
        raise ValueError(f"classifier snapshot band missing fields: {missing}")
    numeric = {
        "frozen_event_bar": payload["frozen_event_bar"],
        "af_bin_ms": payload["af_bin_ms"],
        "floor_af": payload["floor_af"],
        **{f"band.{key}": band[key] for key in REQUIRED_BAND},
    }
    for key, value in numeric.items():
        if not isinstance(value, (int, float)) or not (float(value) >= 0.0):
            raise ValueError(f"classifier snapshot {key} must be a finite non-negative number")
    if not float(band["event_rate_lo"]) < float(band["event_rate_hi"]):
        raise ValueError("classifier event-rate band is empty")
    if not 0.0 < float(band["recruit_p90"]) < 1.0:
        raise ValueError("classifier recruit_p90 must be a fraction")
    original = payload.get("original_full_contract", {})
    digest = str(original.get("sha256", ""))
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("classifier snapshot lacks the frozen original-contract sha256")
    provenance = payload.get("provenance", {})
    if not provenance:
        raise ValueError("classifier snapshot lacks provenance")
    for relative in provenance.values():
        path = ROOT / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"classifier snapshot provenance is missing: {path}")
    return payload


def load_classifier_snapshot(path: Path | str = DEFAULT_SNAPSHOT) -> dict:
    path = Path(path)
    payload = validate_classifier_snapshot(json.loads(path.read_text()))
    # Return a fresh object so callers cannot mutate a module-global contract in place.
    return json.loads(json.dumps(payload))

