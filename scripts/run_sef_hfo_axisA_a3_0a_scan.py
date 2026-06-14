#!/usr/bin/env python3
"""Axis-A Stage A3-0a — E/I lesion MAGNITUDE feasibility scan (NOT a science comparison).

A3 pilot (ei=0.5 / ee=1.5) showed the local E/I lesion nucleates self-terminating but
SPATIALLY-LOCAL events (n_part mostly < PART_MIN=7, axis unreadable) -> ~0 clean directional
templates. Question here: does a STRONGER lesion recruit enough of the network to produce
readable directional templates, and at what rate? REPORTS feasibility only (event count,
clean-directional count, readability, n_part spread) — the formal rate-matched comparison
is A3 (run_sef_hfo_axisA_a3_ei.py), gated on this finding.

Usage: python scripts/run_sef_hfo_axisA_a3_0a_scan.py --workers 6
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
OUT = BASE / "a3_0a_scan"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
T = 2000.0
THREADS = 8
PART_MIN = 7

# (lesion, knob, value) — stronger = lower ei_scale / higher ee_gain
CELLS = [("oneend_inhib", "ei_scale", v) for v in (0.5, 0.35, 0.2)] + \
        [("oneend_recur", "ee_gain", v) for v in (1.5, 2.0, 2.5)]


def _tag(les, knob, v):
    return f"a30a_{les}_{knob}{v}_s1"


def _run(les, knob, v):
    tag = _tag(les, knob, v)
    if (OUT / f"readout_{tag}.json").exists():
        return {"tag": tag, "rc": 0, "skipped": True}
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--lesion", les, f"--{knob.replace('_', '-')}", str(v),
           "--seed", "1", "--T", str(T), "--tag", tag, "--out", str(OUT)]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"tag": tag, "rc": rc, "skipped": False}


def diagnose():
    rows = []
    for les, knob, v in CELLS:
        tag = _tag(les, knob, v)
        jp = OUT / f"readout_{tag}.json"
        if not jp.exists():
            rows.append(dict(lesion=les, knob=knob, value=v, tag=tag, status="MISSING")); continue
        d = json.loads(jp.read_text())
        evs = d["events"]
        nparts = [e["n_part"] for e in evs]
        n_readable = sum(1 for e in evs if e["axis_err"] is not None)
        n_clean = int(d["n_clean_forward"]) + int(d["n_clean_reverse"])
        rows.append(dict(
            lesion=les, knob=knob, value=v, tag=tag, status="ok",
            n_events=int(d["n_events"]), n_clean_directional=n_clean,
            n_readable_axis=n_readable, n_part_max=(max(nparts) if nparts else 0),
            n_part_ge_PARTMIN=sum(1 for n in nparts if n >= PART_MIN),
            true_floor=d["detector"].get("true_inter_event_floor"),
            clean_fwd=int(d["n_clean_forward"]), clean_rev=int(d["n_clean_reverse"])))
    out = dict(stage="A3-0a E/I magnitude feasibility (feasibility only; not a comparison)",
               T=T, part_min=PART_MIN, cells=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_0a_scan.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'a3_0a_scan.json'}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['tag']}: {r['status']}"); continue
        print(f"  {r['lesion']} {r['knob']}={r['value']}: events={r['n_events']} "
              f"clean_dir={r['n_clean_directional']} readable={r['n_readable_axis']} "
              f"n_part_max={r['n_part_max']} n_part>=7={r['n_part_ge_PARTMIN']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--diagnose-only", action="store_true")
    a = ap.parse_args()
    if not a.diagnose_only:
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        print(f"A3-0a scan: {len(CELLS)} cells T={T} workers={a.workers}")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run, *c): c for c in CELLS}
            for f in as_completed(futs):
                r = f.result(); print(f"  done {r['tag']} rc={r['rc']}" + (" (resumed)" if r.get("skipped") else ""))
    diagnose()


if __name__ == "__main__":
    main()
