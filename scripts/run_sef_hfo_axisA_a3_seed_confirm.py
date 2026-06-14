#!/usr/bin/env python3
"""Axis-A A3 multi-seed confirmation of the local-event NULL (firms the A3-0a screen).

A3-0a (single seed) found local E/I lesions nucleate spatially-LOCAL, non-directional
events. Main caveat = single seed. This runs the two matched-NUCLEATION-rate magnitudes
(oneend_inhib ei=0.5 / oneend_recur ee=1.5 -- the points whose nucleation count ~ the V_th
baseline) across seeds {1,2,3} and asks: is "events stay local (n_part<7) + ~0 clean
directional + bare-sheet quiet" ROBUST across seeds? If yes, the A3 NULL is a firm screen.

REPORTS the per-seed read-out; not a mechanism claim. Usage: --workers 6
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
OUT = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/a3_seed_confirm"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"
T = 2000.0
THREADS = 8
PART_MIN = 7
QUIET_FLOOR = 0.001                # bare-sheet-quiet threshold (V_th baseline ~0.00016)
CELLS = [("oneend_inhib", "ei_scale", 0.5, s) for s in (1, 2, 3)] + \
        [("oneend_recur", "ee_gain", 1.5, s) for s in (1, 2, 3)]


def _tag(les, knob, v, s):
    return f"a3sc_{les}_{knob}{v}_s{s}"


def _run(les, knob, v, s):
    tag = _tag(les, knob, v, s)
    if (OUT / f"readout_{tag}.json").exists():
        return {"tag": tag, "rc": 0, "skipped": True}
    log = OUT / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--lesion", les, f"--{knob.replace('_', '-')}", str(v),
           "--seed", str(s), "--T", str(T), "--tag", tag, "--out", str(OUT)]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"tag": tag, "rc": rc, "skipped": False}


def diagnose():
    rows = []
    for les, knob, v, s in CELLS:
        tag = _tag(les, knob, v, s)
        jp = OUT / f"readout_{tag}.json"
        if not jp.exists():
            rows.append(dict(lesion=les, value=v, seed=s, tag=tag, status="MISSING")); continue
        d = json.loads(jp.read_text())
        nparts = [e["n_part"] for e in d["events"]]
        n_clean = int(d["n_clean_forward"]) + int(d["n_clean_reverse"])
        tf = d["detector"].get("true_inter_event_floor")
        rows.append(dict(lesion=les, value=v, seed=s, tag=tag, status="ok",
                         n_events=int(d["n_events"]), n_clean_directional=n_clean,
                         n_part_max=(max(nparts) if nparts else 0),
                         n_part_ge_PARTMIN=sum(1 for n in nparts if n >= PART_MIN),
                         true_floor=tf, bare_sheet_quiet=bool(tf is not None and tf < QUIET_FLOOR)))
    # robustness verdict per lesion: across all seeds, stays local + ~0 clean + quiet
    verdict = {}
    for les in ("oneend_inhib", "oneend_recur"):
        ok = [r for r in rows if r["lesion"] == les and r["status"] == "ok"]
        verdict[les] = dict(
            n_seeds=len(ok),
            clean_dir_per_seed=[r["n_clean_directional"] for r in ok],
            n_part_max_per_seed=[r["n_part_max"] for r in ok],
            bare_sheet_quiet_all=all(r["bare_sheet_quiet"] for r in ok) if ok else None,
            local_robust_all_seeds=all(r["n_clean_directional"] <= 1 for r in ok) if ok else None)
    out = dict(stage="A3 multi-seed confirmation of the local-event NULL (matched-nucleation magnitudes)",
               T=T, part_min=PART_MIN, cells=rows, verdict=verdict)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a3_seed_confirm.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {OUT/'a3_seed_confirm.json'}")
    for les, vd in verdict.items():
        print(f"  {les}: clean_dir/seed={vd['clean_dir_per_seed']} n_part_max/seed={vd['n_part_max_per_seed']} "
              f"quiet_all={vd['bare_sheet_quiet_all']} local_robust={vd['local_robust_all_seeds']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--diagnose-only", action="store_true")
    a = ap.parse_args()
    if not a.diagnose_only:
        (OUT / "logs").mkdir(parents=True, exist_ok=True)
        print(f"A3 seed-confirm: {len(CELLS)} cells T={T} workers={a.workers}")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_run, *c): c for c in CELLS}
            for f in as_completed(futs):
                r = f.result(); print(f"  done {r['tag']} rc={r['rc']}" + (" (resumed)" if r.get("skipped") else ""))
    diagnose()


if __name__ == "__main__":
    main()
