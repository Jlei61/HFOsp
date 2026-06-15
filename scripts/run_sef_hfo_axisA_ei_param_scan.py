#!/usr/bin/env python3
"""Axis-A E/I parameter scan (user 2026-06-15: "scan some E/I parameter ranges").

A3 + relay screen showed local E/I (single knob, or + larger core / + seed=17.5) does NOT
open a "bare-sheet-quiet AND clean directional template" window: weak = trapped-local,
strong = breaks quiet. The relay's E/I+seed=17.5 broke quiet because the combination was
too hot. This scan sweeps the GENTLER E/I space (lower-strength single knobs, COMBINED
both-knobs, and inhib + a MILD V_th seed) to test whether any milder combination finds the
window. Gate-only feasibility (1 seed, T=2000); survivors -> 3-seed confirm.

GO gate per cell: clean directional >=6 AND bare-sheet quiet (true_floor<0.001) AND
clean-event n_part stably covers the axis (median >= 7.5). Reuses engine (src/snn_engine,
post-migration) + the frozen read-out. NOT a parameter sea -- a bounded gentler-window probe.

Usage: python scripts/run_sef_hfo_axisA_ei_param_scan.py --workers 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SP = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
OUT = SP / "ei_param_scan"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
T = 2000.0
THREADS = 8
QUIET = 0.001
NPART_COVER = 7.5

# each cell: (name, lesion, ei_scale, ee_gain, ei_vth_seed)  -- None = omit that knob
CELLS = [
    # --- inhibition-collapse strength (single knob, gentler -> A3 point) ---
    ("inhib_ei0.7",        "oneend_inhib", 0.7, None, None),
    ("inhib_ei0.6",        "oneend_inhib", 0.6, None, None),
    ("inhib_ei0.5",        "oneend_inhib", 0.5, None, None),
    # --- recurrent-excitation strength (single knob, gentle) ---
    ("recur_ee1.3",        "oneend_recur", None, 1.3, None),
    ("recur_ee1.5",        "oneend_recur", None, 1.5, None),
    # --- COMBINED both knobs (milder each) ---
    ("comb_ei0.7_ee1.3",   "oneend_combined", 0.7, 1.3, None),
    ("comb_ei0.6_ee1.4",   "oneend_combined", 0.6, 1.4, None),
    # --- inhib + MILD V_th seed (gentler than the relay's 0.5+17.5 that broke quiet) ---
    ("inhib_ei0.7_seed17.8", "oneend_inhib", 0.7, None, 17.8),
    ("inhib_ei0.6_seed17.6", "oneend_inhib", 0.6, None, 17.6),
    ("inhib_ei0.7_seed17.5", "oneend_inhib", 0.7, None, 17.5),
]


def _args(lesion, ei, ee, seed):
    a = ["--lesion", lesion]
    if ei is not None:
        a += ["--ei-scale", str(ei)]
    if ee is not None:
        a += ["--ee-gain", str(ee)]
    if seed is not None:
        a += ["--ei-vth-seed", str(seed)]
    return a


def _run(name, lesion, ei, ee, seed):
    tag = f"eips_{name}_s1"
    jp = OUT / f"readout_{tag}.json"
    if jp.exists():
        return {"name": name, "skipped": True}
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--seed", "1", "--T", str(T),
           "--tag", tag, "--out", str(OUT)] + _args(lesion, ei, ee, seed)
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"name": name, "skipped": False, "rc": rc}


def gate(name):
    jp = OUT / f"readout_eips_{name}_s1.json"
    if not jp.exists():
        return dict(status="MISSING")
    d = json.loads(jp.read_text())
    nclean = int(d["n_clean_forward"]) + int(d["n_clean_reverse"])
    clean = [e for e in d["events"] if e["returned"] and e["axis_err"] is not None
             and e["axis_err"] < 25 and e["n_part"] >= 7]
    cnp = [e["n_part"] for e in clean]
    tf = d["detector"].get("true_inter_event_floor")
    npm = (statistics.median(cnp) if cnp else 0)
    quiet = tf is not None and tf < QUIET
    return dict(status="ok", n_events=int(d["n_events"]), n_clean_directional=nclean,
                n_part_clean_median=npm, true_floor=tf, bare_sheet_quiet=quiet,
                GO=bool(nclean >= 6 and quiet and npm >= NPART_COVER))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if not a.gate_only:
        print(f"E/I param scan: {len(CELLS)} cells (T={T}, seed 1), workers={a.workers}")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_run, *c) for c in CELLS]
            for f in as_completed(futs):
                r = f.result()
                print(f"  {r['name']}: " + ("resumed" if r.get("skipped") else f"rc={r['rc']}"))
    results = {name: gate(name) for name, *_ in CELLS}
    survivors = [n for n, g in results.items() if g.get("GO")]
    out = dict(stage="E/I parameter scan (gentler-window probe; gate-only, 1 seed)",
               T=T, gate=dict(clean_dir_min=6, quiet_floor=QUIET, npart_cover_min=NPART_COVER),
               cells=results, survivors=survivors,
               next_step=("3-seed confirm on survivors" if survivors
                          else "NO survivor across the gentler E/I window -> the E/I read-out "
                               "NULL holds on a broader basis; STOP (no tau/sigma sea)"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ei_param_scan.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'ei_param_scan.json'}")
    for name, g in results.items():
        if g["status"] != "ok":
            print(f"  {name}: {g['status']}"); continue
        print(f"  {name}: clean_dir={g['n_clean_directional']} n_part_med={g['n_part_clean_median']} "
              f"true_floor={g['true_floor']} quiet={g['bare_sheet_quiet']} -> GO={g['GO']}")
    print(f"SURVIVORS: {survivors or 'none'} -> {out['next_step']}")


if __name__ == "__main__":
    main()
