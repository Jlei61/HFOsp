#!/usr/bin/env python3
"""Lock the single FCXR-LC4c entry/offset-aligned candidate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc4b_deadzone import sha256_file
from src.topic4_fcxr_lc4c import derive_aligned_candidate


BASE = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
ENTRY = BASE / "stage1_entry_window/cell_tauz5000_theta110.json"
LC4B = BASE / "lc4b_deadzone_lifecycle"
BASE_LOCK = LC4B / "candidate_lock.json"
NOMINAL = LC4B / "nominal_lifecycle.json"
TRACE = LC4B / "nominal_lifecycle_traces.npz"
LOAD = BASE / "percell_separation/per_cell.npz"
OUT = BASE / "lc4c_entry_offset_alignment/candidate_lock.json"
EXPECTED = {
    ENTRY: "5586a33fbbd624f9ac9d266987da02c9c6254e0c748aa6cbd1d963d035eef3c2",
    BASE_LOCK: "0381e0cef908450fb7485c5810af3aa9b7c2091da9da7d8e017d9d5d55ae05f0",
    NOMINAL: "27ce55c78e7abb6827209dc95af330fd09ea48a233deb11a29fc237e4fa3f479",
    TRACE: "93cc29d9b99a9c489959f756cc9ad409562d8901bebfa9280afc156ffb373ad9",
    LOAD: "81d2c97a16eaf69753951dcd91f55153c7102d5d7ca4c0f75e12f9b24208a33d",
}


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(f"artifact drift: {path}: {got} != {expected}")
    entry = json.loads(ENTRY.read_text())
    base_lock = json.loads(BASE_LOCK.read_text())
    nominal = json.loads(NOMINAL.read_text())
    if nominal["nominal_gate"]["offset_ms"] is not None:
        raise SystemExit("LC4c dose repair requires the locked LC4b no-offset result")
    zt = np.load(TRACE)
    zl = np.load(LOAD)
    candidate = derive_aligned_candidate(
        base_lock["candidate"], entry, zt["a_mean"], zl["interictal_peak_tau1000"])
    payload = {
        "status": "C0_PASS",
        "verdict": "ENTRY_OFFSET_REPAIR_IDENTIFIABLE",
        "candidate": candidate,
        "sources": {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
        "rules": {
            "entry": "existing theta_scale=1.1 row; no interpolation or scan",
            "offset": "g_m_max = locked target / observed LC4b closed-high a_mean_max",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
