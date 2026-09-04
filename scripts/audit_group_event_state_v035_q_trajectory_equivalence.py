#!/usr/bin/env python3
"""Prove that provisional and locked q(t) trajectories are array-identical.

W3 was launched from the provisional dynamic-rate root while W1's registered
scores live under ``dynamic_rate_final``.  The causal multiscale q(t) is
deterministic and is not a learned residual, but that fact must be checked on
the materialised arrays rather than assumed from code structure.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.contracts import OUTPUT_ROOT, atomic_json  # noqa: E402


KEYS = ("anchor_time", "segment", "phase", "q_standardized")


def _digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def main() -> None:
    rows = []
    for final_path in sorted((OUTPUT_ROOT / "dynamic_rate_final").glob("*/seed*/trajectory_and_scores.npz")):
        relative = final_path.relative_to(OUTPUT_ROOT / "dynamic_rate_final")
        provisional_path = OUTPUT_ROOT / "dynamic_rate" / relative
        if not provisional_path.exists():
            raise FileNotFoundError(f"missing provisional q trajectory: {provisional_path}")
        current = {"unit": str(relative), "arrays": {}}
        with np.load(provisional_path, allow_pickle=False) as old, np.load(final_path, allow_pickle=False) as new:
            for key in KEYS:
                equal = key in old.files and key in new.files and np.array_equal(old[key], new[key])
                current["arrays"][key] = {
                    "equal": bool(equal),
                    "provisional_sha256": _digest(old[key]) if key in old.files else None,
                    "final_sha256": _digest(new[key]) if key in new.files else None,
                }
                if not equal:
                    raise ValueError(f"q trajectory mismatch for {relative}: {key}")
        rows.append(current)
    if not rows:
        raise RuntimeError("no locked dynamic-rate trajectories found")
    payload = {
        "format": "group_event_state_v0_3_5_q_trajectory_equivalence_v1",
        "n_units": len(rows),
        "keys": list(KEYS),
        "all_array_identical": True,
        "units": rows,
        "scientific_consequence": (
            "W3 consumes exactly the registered deterministic causal q(t); "
            "the learned W1 residual scores are not used as event-content input"
        ),
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    out = OUTPUT_ROOT / "audit" / "q_trajectory_equivalence.json"
    atomic_json(out, payload)
    print(json.dumps({"output": str(out), "n_units": len(rows), "all_array_identical": True}, indent=2))


if __name__ == "__main__":
    main()
