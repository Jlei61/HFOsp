"""Assemble the complete Stage-D branch map from ALL saved cells (grid + cells runs), re-classifying each from
its rate trace with the workpoint classifier + the empirical interictal band, resolving T1/T2 per cell, and
aggregating per-D. No re-simulation. Writes fast_slow_dynamics/branch_map.json + prints the verdict.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    workpoint_metrics, classify_run_workpoint, resolve_high_ic_wp, classify_branch_D_wp, WP_THRESHOLDS,
)

BASE = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay", "fast_slow_dynamics")
D_GRID = [0.0, 0.05, 0.075, 0.085, 0.09, 0.1, 0.125, 0.15]


def _label_from_trace(jf, band):
    row = json.load(open(jf))
    npz = np.load(jf.replace(".json", "_trace.npz"))
    wm = workpoint_metrics(np.asarray(npz["rate_E"], float), float(npz["rate_dt_ms"][0]), band,
                           float(row["analysis_start_ms"]))
    lab = classify_run_workpoint(dict(numerical_unsafe=bool(row["numerical_unsafe"]),
                                      af_tail=float(row.get("af_tail", 0.0)), **wm))
    return dict(D=row["D"], slot=row["slot"], window=row["window"], label=lab, T_ms=float(row["T_ms"]),
                end_rate_hz=row["end_rate_hz"], roll_occ=wm["roll_occ"], roll_end_occ=wm["roll_end_occ"],
                roll_high_ms=wm["roll_high_ms"])


def main():
    bref = json.load(open(os.path.join(BASE, "baseline_ref.json")))
    band = float(bref["rate_roll_hi"])
    # collect every cell across grid+cells runs; latest run wins per (D, slot, window)
    cells = {}
    for jf in sorted(glob.glob(os.path.join(BASE, "runs", "*grid*", "per_cell", "*.json"))
                     + glob.glob(os.path.join(BASE, "runs", "*cells*", "per_cell", "*.json"))):
        if jf.endswith("_trace.json") or not os.path.exists(jf.replace(".json", "_trace.npz")):
            continue
        c = _label_from_trace(jf, band)
        if c["T_ms"] < WP_THRESHOLDS["HIGH_MS"]:       # exclude smoke/validation runs (T<HIGH_MS)
            continue
        cells[(c["D"], c["slot"], c["window"])] = c    # sorted glob -> newest real run overwrites

    per_D, verdict_cells = [], []
    for D in D_GRID:
        def resolved(slot):
            t1 = cells.get((D, slot, "T1")); t2 = cells.get((D, slot, "T2"))
            if t1 is None:
                return None, None
            lab = resolve_high_ic_wp(t1["label"], t2["label"]) if t2 is not None else t1["label"]
            return lab, (t1["end_rate_hz"])
        low, _ = resolved("low")
        h1, p1 = resolved("high1"); h2, p2 = resolved("high2")
        if low is None or h1 is None or h2 is None:
            per_D.append(dict(D=D, D_label="INCOMPLETE", low=low, high=[h1, h2])); continue
        d = classify_branch_D_wp(low, [h1, h2], [p1, p2])
        per_D.append(dict(D=D, **d))
        if d["D_label"] not in ("INTERICTAL_WORKPOINT",):
            verdict_cells.append((D, d["D_label"]))

    finite = [D for D, l in verdict_cells if l in ("FINITE_HIGH", "BISTABLE")]
    meta = [D for D, l in verdict_cells if l == "METASTABLE_TRANSIENT"]
    elev = [D for D, l in verdict_cells if l == "ELEVATED_EVENT_TRAIN"]
    if finite:
        verdict = f"FINITE-HIGH / BISTABLE at D={finite} -> proceed to sech^2/eigenmode + seed3 + spatial confirm"
    elif meta:
        verdict = (f"NO persistent high branch. Near-transition (D={meta}) shows bounded METASTABLE transients "
                   "that decay by the longer T2; RC1 saturation caps the amplitude but gives no high attractor.")
    elif elev:
        verdict = (f"NO high branch. Near-transition (D={elev}) is an ELEVATED_EVENT_TRAIN (more active than the "
                   "interictal band but not a distinct sustained high branch); rest is the interictal workpoint.")
    else:
        verdict = "Interictal workpoint across all D in [0,0.15] (seed1); no elevation, no high branch."

    print(f"empirical interictal band = {band:.2f} Hz  (workpoint classifier, seed1, dt=0.05)\n")
    for p in per_D:
        print(f"  D={p['D']:g}: {p['D_label']:22s} (low={p.get('low_label', p.get('low'))}, "
              f"high={p.get('high_labels', p.get('high'))})")
    print(f"\nVERDICT: {verdict}")
    n_t2 = sum(1 for k in cells if k[2] == "T2")
    json.dump(dict(seed=1, dt=0.05, band_hz=band, D_grid=D_GRID, thresholds=WP_THRESHOLDS,
                   per_D=per_D, cells=list(cells.values()), n_T2=n_t2, verdict=verdict,
                   classifier="workpoint_relative"),
              open(os.path.join(BASE, "branch_map.json"), "w"), indent=2)
    print(f"wrote {BASE}/branch_map.json  ({len(cells)} cells, {n_t2} T2)")


if __name__ == "__main__":
    main()
