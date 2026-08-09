#!/usr/bin/env python3
"""Lock the single FCXR-LC4d offset-latency candidate."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fcxr_lc4b_deadzone import sha256_file  # noqa: E402
from src.topic4_fcxr_lc4d import derive_latency_candidate  # noqa: E402


BASE = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
LC4C = BASE / "lc4c_entry_offset_alignment"
BASE_LOCK = LC4C / "candidate_lock.json"
ENTRY = LC4C / "entry_gate.json"
NOMINAL = LC4C / "nominal_lifecycle.json"
TRACE = LC4C / "nominal_lifecycle_traces.npz"
LOAD = BASE / "percell_separation/per_cell.npz"
OUT = BASE / "lc4d_offset_latency_alignment/candidate_lock.json"
EXPECTED = {
    BASE_LOCK: "9e560ff415bf9c4d951e5cd8001a681bc9ee872f6d4eaf7110639e48d3e6d6e2",
    ENTRY: "186eca44cb72d8a95046828bc6ec737ebeca37b3bb883e222b0e057bd55e2da6",
    NOMINAL: "e31b2d19430ce40e277b8ca7bc7a8972a8dc7b7f605587ed74d8fc407566908d",
    TRACE: "22534ed100743b356873e5e0b3f727a95a94bc24d50c8016292933f8b587a3db",
    LOAD: "81d2c97a16eaf69753951dcd91f55153c7102d5d7ca4c0f75e12f9b24208a33d",
}


def main() -> None:
    for path, expected in EXPECTED.items():
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(f"artifact drift: {path}: {got} != {expected}")
    base = json.loads(BASE_LOCK.read_text())["candidate"]
    entry = json.loads(ENTRY.read_text())
    nominal = json.loads(NOMINAL.read_text())
    zt = np.load(TRACE)
    zl = np.load(LOAD)
    candidate = derive_latency_candidate(
        base, entry, nominal, zt["a_mean"], float(zt["trace_dt_ms"][0]),
        zl["interictal_peak_tau1000"], zt["adap_current"],
        float(zt["trace_dt_ms"][0]))
    payload = {
        "status": "L0_PASS",
        "verdict": "OFFSET_LATENCY_REPAIR_IDENTIFIABLE",
        "candidate": candidate,
        "sources": {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
        "rule": "g_m_max = I_target / a_mean at exactly onset+4000 ms; no sweep or safety factor",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
