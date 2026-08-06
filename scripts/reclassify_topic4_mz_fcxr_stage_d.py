"""Offline re-classify saved Stage-D T1 cells with the workpoint-relative classifier (reviewer step 4/6).

Reads each cell's saved rate trace + the empirical interictal band (baseline_ref.json rate_roll_hi) and applies
classify_run_workpoint -- NO re-simulation. Prints per-cell (new label vs the old flawed envelope label + roll
metrics) and a per-D summary, and writes reclassified_labels.json. Use it to decide which cells (truly above the
interictal band) warrant a T2 run.

Usage: python scripts/reclassify_topic4_mz_fcxr_stage_d.py [run_dir]   (default: newest grid run)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.topic4_mz_fcxr_dynamics import workpoint_metrics, classify_run_workpoint  # noqa: E402

BASE = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay", "fast_slow_dynamics")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(BASE, "runs", "*grid*")))[-1]
    bref = json.load(open(os.path.join(BASE, "baseline_ref.json")))
    band = float(bref["rate_roll_hi"])
    print(f"empirical interictal band upper edge (rate_roll_hi, {bref.get('roll_ms',300):.0f}ms rolling "
          f"q{bref.get('baseline_q',99):.0f}) = {band:.2f} Hz\nrun: {os.path.basename(run_dir)}\n")

    cells = sorted(glob.glob(os.path.join(run_dir, "per_cell", "*_T1.json")))
    out = []
    for jf in cells:
        row = json.load(open(jf))
        npz = np.load(jf.replace(".json", "_trace.npz"))
        rate = np.asarray(npz["rate_E"], float); dt_ms = float(npz["rate_dt_ms"][0])
        wm = workpoint_metrics(rate, dt_ms, band, float(row["analysis_start_ms"]))
        lab = classify_run_workpoint(dict(numerical_unsafe=bool(row["numerical_unsafe"]),
                                          af_tail=float(row.get("af_tail", 0.0)), **wm))
        out.append(dict(label=row["label"], D=row["D"], slot=row["slot"], workpoint_label=lab,
                        old_envelope_label=row.get("provisional_label"), end_rate_hz=row["end_rate_hz"], **wm))
        print(f"  {row['label']:16s} -> {lab:20s} roll_occ={wm['roll_occ']:.3f} roll_end={wm['roll_end_occ']:.3f} "
              f"roll_hi={wm['roll_high_ms']:6.0f}ms end={row['end_rate_hz']:5.1f}Hz  (old envelope: {row.get('provisional_label')})")

    print("\nper-D (T1, workpoint classifier):")
    for D in sorted(set(r["D"] for r in out)):
        by = {r["slot"]: r["workpoint_label"] for r in out if r["D"] == D}
        print(f"  D={D:g}: low={by.get('low','-'):22s} high1={by.get('high1','-'):22s} high2={by.get('high2','-')}")

    above = [r["label"] for r in out if r["workpoint_label"] not in ("INTERICTAL_WORKPOINT",)]
    print(f"\ncells NOT plain interictal (T2 candidates): {above or 'NONE'}")
    json.dump(dict(run=os.path.basename(run_dir), band_hz=band, cells=out, t2_candidates=above),
              open(os.path.join(run_dir, "reclassified_labels.json"), "w"), indent=2)
    print(f"wrote {run_dir}/reclassified_labels.json")


if __name__ == "__main__":
    main()
