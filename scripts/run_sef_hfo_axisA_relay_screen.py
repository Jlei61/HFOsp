#!/usr/bin/env python3
"""Axis-A nucleation-vs-propagation-relay 2x2 screen (user 2026-06-15).

The A3 result reframed: local IGNITION ability != outward PROPAGATION-RELAY ability.
V_th-down makes the core form an outgoing pulse that recruits normal-threshold surround;
local E/I ignites but the activity is trapped in-core. This minimal 2x2 asks "what
condition turns local nucleation into a propagable directional template" -- NOT a big
parameter sweep.

  axis 1 (core nucleation mechanism): V_th-down  vs  E/I
  axis 2 (outward-relay condition):   original   vs  larger core / mild V_th seed

4 conditions, 1 seed, T=2000, GATE only (no formal stats):
  1. V_th-down r=1.5            -> positive reference (existing run)
  2. E/I r=1.5 (ei=0.5)         -> NULL reference     (existing run)
  3. E/I larger core r=2.5/3.0  -> is the core too small to relay out?
  4. E/I + mild V_th seed 17.5  -> does E/I need a threshold seed to output a pulse?

GO gate per condition: clean directional >=6 AND bare-sheet quiet (true_floor<0.001)
AND clean-event n_part stably covers the axis (median >= 7.5, not just-barely-7).
Survivors (if any) -> 3-seed confirm (separate step). No survivor -> STOP (do NOT scan
tau_m/tau_ref/sigma/tau_I).

Usage: python scripts/run_sef_hfo_axisA_relay_screen.py --workers 3
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
OUT = SP / "relay_screen"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
T = 2000.0
THREADS = 8
QUIET = 0.001          # bare-sheet-quiet threshold (V_th baseline ~0.00016)
NPART_COVER = 7.5      # clean-event n_part median must exceed this (not just-barely-7)

# name -> (readout_json_path, run_args_or_None). None = reference (already on disk).
COND = {
    "1_vth_down_r1.5":   (SP / "a1_0a_feasibility/readout_a10a_neg_m17.0_std1.5_s1.json", None),
    "2_ei_r1.5":         (SP / "a3_seed_confirm/readout_a3sc_oneend_inhib_ei_scale0.5_s1.json", None),
    "3a_ei_r2.5":        (OUT / "readout_relay_ei_r2.5_s1.json",
                          ["--lesion", "oneend_inhib", "--ei-scale", "0.5", "--core-r", "2.5"]),
    "3b_ei_r3.0":        (OUT / "readout_relay_ei_r3.0_s1.json",
                          ["--lesion", "oneend_inhib", "--ei-scale", "0.5", "--core-r", "3.0"]),
    "4_ei_vthseed17.5":  (OUT / "readout_relay_ei_vthseed17.5_s1.json",
                          ["--lesion", "oneend_inhib", "--ei-scale", "0.5", "--ei-vth-seed", "17.5"]),
}


def _run(name, jp, args):
    tag = jp.stem.replace("readout_", "")
    if jp.exists():
        return {"name": name, "skipped": True}
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--seed", "1", "--T", str(T),
           "--tag", tag, "--out", str(OUT)] + args
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"name": name, "skipped": False, "rc": rc}


def gate(jp):
    if not jp.exists():
        return dict(status="MISSING")
    d = json.loads(jp.read_text())
    nclean = int(d["n_clean_forward"]) + int(d["n_clean_reverse"])
    clean = [e for e in d["events"] if e["returned"] and e["axis_err"] is not None
             and e["axis_err"] < 25 and e["n_part"] >= 7]
    cnp = [e["n_part"] for e in clean]
    cerr = [e["axis_err"] for e in clean]
    tf = d["detector"].get("true_inter_event_floor")
    npm = (statistics.median(cnp) if cnp else 0)
    quiet = tf is not None and tf < QUIET
    go = bool(nclean >= 6 and quiet and npm >= NPART_COVER)
    return dict(status="ok", n_events=int(d["n_events"]), n_clean_directional=nclean,
                n_part_clean_median=npm, n_part_clean=sorted(cnp),
                axis_err_clean_median=(round(statistics.median(cerr), 2) if cerr else None),
                true_floor=tf, bare_sheet_quiet=quiet, GO=go)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--gate-only", action="store_true")
    a = ap.parse_args()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if not a.gate_only:
        to_run = [(n, jp, args) for n, (jp, args) in COND.items() if args is not None]
        print(f"relay screen: running {len(to_run)} new cells (T={T}, seed 1), workers={a.workers}")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(_run, n, jp, args) for n, jp, args in to_run]
            for f in as_completed(futs):
                r = f.result()
                print(f"  {r['name']}: " + ("resumed" if r.get("skipped") else f"rc={r['rc']}"))
    results = {name: gate(jp) for name, (jp, _) in COND.items()}
    survivors = [n for n, g in results.items() if g.get("GO") and not n.startswith(("1_", "2_"))]
    out = dict(stage="nucleation-vs-propagation-relay 2x2 screen (gate-only, 1 seed)",
               T=T, gate=dict(clean_dir_min=6, quiet_floor=QUIET, npart_cover_min=NPART_COVER),
               conditions=results, survivors=survivors,
               next_step=("3-seed confirm on survivors" if survivors
                          else "NO survivor -> STOP; do NOT scan tau_m/tau_ref/sigma/tau_I"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "relay_screen.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'relay_screen.json'}")
    for name, g in results.items():
        if g["status"] != "ok":
            print(f"  {name}: {g['status']}"); continue
        print(f"  {name}: clean_dir={g['n_clean_directional']} n_part_clean_med={g['n_part_clean_median']} "
              f"axis_err_med={g['axis_err_clean_median']} quiet={g['bare_sheet_quiet']} -> GO={g['GO']}")
    print(f"SURVIVORS: {survivors or 'none'} -> {out['next_step']}")


if __name__ == "__main__":
    main()
