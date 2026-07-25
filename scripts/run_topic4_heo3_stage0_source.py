"""FCXR-HEO3 Stage H3.0 part B — source-space audit (review P1-b).

Sensor-level 15/15 may be one loud core seen by every contact. This re-runs the precursor arm
(fast-τ/10%) and the 16 Hz reference with the SAME configs as Phase 1, then reduces the E-cell spike
field to per-region activity (core-source / core-sink / axis-corridor / off-axis), whole-field
participation ratio and the activity centroid + its axis coordinate — small summaries, no big arrays
saved. Answers: is the tissue actually recruited, and does the centroid move along the axis?
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-heo3")

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_fcxr_heo1 as HEO1R  # noqa: E402
import run_topic4_heo2_phase1 as P1  # noqa: E402
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402
PP = HEO1R.PP  # same placement module HEO1 uses (CORE_R)

DT = HEO1R.DT
OUT = os.path.join(HEO1R.FCXR.OUT_ROOT, "heo3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("pass --confirm-run")
    HEO1R.FCXR._assert_engine_blessed()
    os.makedirs(OUT, exist_ok=True)
    with HEO1R.FCXR._launcher_lock():
        HEO1R.FCXR._write_json(os.path.join(OUT, "SOURCE_RUNNING.json"),
                               dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
        try:
            pa = json.load(open(os.path.join(HEO1R.FCXR.OUT_ROOT, "broadband_diagnostic", "phase1_arms.json")))
            eta = float(pa["etas"]["250.0/0.1"])
            ref, u_c, roll_hi, scl, c0 = HEO1R._load_baseline(1)
            S, p_i = HEO1R._build_and_align(1)
            contacts, names, scl2 = HEO1R._montage(S)
            anchor = HEO1R._heo_cfg(P1.ANCHOR_A, u_c[P1.ANCHOR_GQ], P1.ANCHOR_D, p_i)
            regions = H3.build_regions(S["posE"], S["src_xy"], S["snk_xy"], PP.CORE_R)
            sizes = {k: int(v.sum()) for k, v in regions.items() if k != "axis_coord"}
            print(f"[h3.0b] region sizes (E cells): {sizes}", flush=True)

            arms = {"m_off": dict(anchor),
                    "dyn_tau250_frac0.1": {**anchor, "use_m": True, "eta_m": eta,
                                           "tau_adp": 250.0, "m_enable_ms": P1.M_ENABLE_MS}}
            out = dict(region_sizes=sizes, arms={})
            for lab, cfg in arms.items():
                t0 = time.time()
                res, slow = HEO1R._heo_run(S, cfg, 5000.0, kick_boost=0.0, t_kick=1e9, seed=1,
                                           lfp_sites=contacts, early_stop=True)
                E = np.asarray(res["E_spk_bool"])
                rows = H3.source_space_audit(E, S["posE"], regions, DT, win_ms=1000.0, hop_ms=100.0)
                out["arms"][lab] = dict(wall_s=round(time.time() - t0, 1), n_windows=len(rows), rows=rows)
                pr = [r["participation_ratio"] for r in rows]
                off = [r["rate_off_axis"] for r in rows]; cor = [r["rate_axis_corridor"] for r in rows]
                cs = [r["rate_core_source"] for r in rows]; ck = [r["rate_core_sink"] for r in rows]
                ac = [r["centroid_axis_coord"] for r in rows]
                print(f"[h3.0b] {lab}: PR med {np.median(pr):.3f} (min {np.min(pr):.3f})  "
                      f"rate core_src {np.median(cs):.1f} core_snk {np.median(ck):.1f} "
                      f"corridor {np.median(cor):.1f} off_axis {np.median(off):.1f} Hz  "
                      f"centroid_axis med {np.nanmedian(ac):.3f} range {np.nanmin(ac):.2f}-{np.nanmax(ac):.2f}",
                      flush=True)
                del E
            HEO1R.FCXR._write_json(os.path.join(OUT, "stage0_source_space.json"), out)
            HEO1R.FCXR._write_json(os.path.join(OUT, "SOURCE_DONE.json"),
                                   dict(finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "SOURCE_RUNNING.json")):
                os.remove(os.path.join(OUT, "SOURCE_RUNNING.json"))
            print("[h3.0b] DONE -> heo3/stage0_source_space.json", flush=True)
        except Exception as exc:
            HEO1R.FCXR._write_json(os.path.join(OUT, "SOURCE_FAILED.json"),
                                   dict(error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
