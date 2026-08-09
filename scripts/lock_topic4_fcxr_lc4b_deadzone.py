#!/usr/bin/env python3
"""Create the zero-compute LC4b candidate lock from the frozen per-cell artifact."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc4b_deadzone import build_locked_candidate, sha256_file  # noqa: E402


SOURCE = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
          / "percell_separation/per_cell.npz")
OUT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
       / "lc4b_deadzone_lifecycle/candidate_lock.json")
EXPECTED_SHA = "81d2c97a16eaf69753951dcd91f55153c7102d5d7ca4c0f75e12f9b24208a33d"


def main() -> None:
    live = sha256_file(SOURCE)
    if live != EXPECTED_SHA:
        raise SystemExit(f"per-cell calibration hash drift: {live} != {EXPECTED_SHA}")
    z = np.load(SOURCE)
    candidate = build_locked_candidate(z["interictal_peak_tau1000"],
                                       z["ictal_settled_tau1000"])
    record = {
        "status": "D0_PASS",
        "verdict": "DEADZONE_IDENTIFIABLE",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": live,
        "selection_rule": {
            "deadzone": "arithmetic midpoint of interictal maximum and ictal minimum",
            "excess_scale": "ictal median minus deadzone",
            "n": "locked n=4 termination-authority lineage",
            "g_m_max": "force match to 44.8619393917937 using mean ictal activation",
        },
        "candidate": candidate,
        "claim_boundary": (
            "D0 proves only that an exact dead zone is identifiable on the frozen load artifact; "
            "D1 must test whether it remains exactly inert in a 12 s live baseline."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    os.replace(tmp, OUT)
    print(OUT)


if __name__ == "__main__":
    main()
