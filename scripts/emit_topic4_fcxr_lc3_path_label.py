#!/usr/bin/env python
"""Emit the registered four-way temporal-geometry label from frozen evidence.

The slow-vector stage completed without emitting its registered label.  Nothing
is re-simulated here: the frozen cells, the drift vectors and the observed
ignition times are all on disk, and the label is a reading of them.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_fcxr_lc3_pathlabel import temporal_geometry_label  # noqa: E402

BASE = os.path.join(ROOT, "results", "topic4_sef_hfo",
                    "fcxr_lc3_dx_spatial_instability")
CELLS = os.path.join(BASE, "geometry_cells")
SLOWFLOW = os.path.join(BASE, "slow_vector_field", "slow_vector_field.json")
RECON = os.path.join(BASE, "dynamic_reconnaissance")
OUT = os.path.join(BASE, "slow_vector_field", "temporal_geometry_label.json")
POINT_PREFIX = "H1_"          # the nominal point; H6 is the sparse sentinel


def _cells():
    low, high = [], []
    for path in sorted(glob.glob(os.path.join(CELLS, "*.json"))):
        if path.endswith(".DONE.json") or ".RUNNING" in path:
            continue
        c = json.load(open(path))
        if c.get("status") != "COMPLETE" or not c["row_id"].startswith(POINT_PREFIX):
            continue
        (low if c["state_kind"] == "low" else high).append(c)
    return low, high


def _ignitions():
    times, seeds = [], []
    for path in sorted(glob.glob(os.path.join(RECON, "recon_noise*.json"))):
        if ".DONE." in path or ".RUNNING" in path:
            continue
        r = json.load(open(path))
        bout = r.get("lifecycle", {}).get("bout")
        if bout is not None:
            times.append(float(bout[0]) * 1000.0)
            seeds.append(r["noise_seed"])
    return times, seeds


def main():
    low, high = _cells()
    vectors = json.load(open(SLOWFLOW))["rows"]
    times, seeds = _ignitions()
    if not times:
        raise SystemExit("no observed ignition time; the label needs one to judge "
                         "whether a quiet window was long enough")
    out = temporal_geometry_label(low_cells=low, high_cells=high, vectors=vectors,
                                  ignition_times_ms=times)
    out["provenance"] = dict(
        point_prefix=POINT_PREFIX, n_low_cells=len(low), n_high_cells=len(high),
        n_vectors=len(vectors), ignition_seeds=seeds, ignition_times_ms=times,
        emitted_post_hoc=("the stage completed without emitting it; nothing was "
                          "re-simulated and no threshold was tuned to the answer"),
        vector_state_kinds=sorted({v["state_kind"] for v in vectors}),
    )
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps({k: out[k] for k in
                      ("label", "reason", "entry", "return_bracket", "drift")}, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
