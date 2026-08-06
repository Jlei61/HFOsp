"""FCXR-HEO3 Stage H3.1b — geometry correction + the load-matched mean-field control.

H3.1 P0: with phase_shift=0 and stripe width D/3 both core CENTRES sat exactly on stripe boundaries
(source at axis coord s=0, sink at s=D=3w), so each core ended up ~50/50 fast/slow (measured 51.1% /
48.4% slow). The two regions therefore never received different recovery times, and H3.1's "no region
alternation" does NOT refute the region-level hypothesis. Centring the stripes (phase_shift=w/2) puts
the source core entirely in a FAST stripe and the sink core entirely in a SLOW one, boundaries between
them — that is the actual test.

Arms (anchor, seed 1, delayed at m_enable_ms=1000, T=5000ms):
  patch_dyn_centered          : source FAST / sink SLOW (the corrected hypothesis test)
  patch_dyn_centered_shuffled : same tau values permuted -> marginal mix kept, placement destroyed
  patch_dyn_centered_swapped  : source SLOW / sink FAST -> rules out a direction accident
  meanfield_dyn_loadmatched   : mean-field m with eta rescaled so the REALIZED mean gM matches uniform
                                dynamic (0.1909); H3.1's mean-field arm ran at gM 0.4381, so its
                                "broadening abolished" was confounded with ~2.3x more potassium.
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
import multiprocessing as mp
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_fcxr_heo1 as HEO1R  # noqa: E402
import run_topic4_heo2_phase1 as P1  # noqa: E402
import run_topic4_heo3_stage1_2x2 as S1  # noqa: E402  (reuse _arm_cell / _CTX / OUT / TAU0)
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402
PP = HEO1R.PP

OUT = S1.OUT
TAU0 = S1.TAU0
GM_UNIFORM_DYN = 0.1909            # realized mean gM of uniform dynamic (the load-match target)
GM_MEANFIELD_H31 = 0.4381          # realized mean gM of the unmatched H3.1 mean-field arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--confirm-run", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("pass --confirm-run to launch H3.1b")
    HEO1R.FCXR._assert_engine_blessed()
    os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
    with HEO1R.FCXR._launcher_lock():
        HEO1R.FCXR._write_json(os.path.join(OUT, "H31B_RUNNING.json"),
                               dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
        try:
            pa = json.load(open(os.path.join(HEO1R.FCXR.OUT_ROOT, "broadband_diagnostic", "phase1_arms.json")))
            eta0 = float(pa["etas"]["250.0/0.1"])
            ref, u_c, roll_hi, scl, c0 = HEO1R._load_baseline(1)
            S, p_i = HEO1R._build_and_align(1)
            contacts, names, scl2 = HEO1R._montage(S)
            anchor = HEO1R._heo_cfg(P1.ANCHOR_A, u_c[P1.ANCHOR_GQ], P1.ANCHOR_D, p_i)
            regions = H3.build_regions(S["posE"], S["src_xy"], S["snk_xy"], PP.CORE_R)
            S1._CTX.update(S=S, contacts=contacts, regions=regions)

            D = float(np.linalg.norm(np.asarray(S["snk_xy"], float) - np.asarray(S["src_xy"], float)))
            w = D / 3.0
            def mk(fast, slow, sh=None):
                return H3.build_patch_field(S["posE"], S["src_xy"], S["snk_xy"], w, fast, slow,
                                            TAU0, eta0, shuffle_seed=sh, phase_shift=w / 2.0)
            tau_c, eta_c, pid_c = mk(TAU0 / 2, TAU0 * 2)                 # source fast / sink slow
            tau_sh, eta_sh, _ = mk(TAU0 / 2, TAU0 * 2, sh=23)
            tau_sw, eta_sw, _ = mk(TAU0 * 2, TAU0 / 2)                   # swapped direction
            fs_src = float((tau_c[regions["core_source"]] == TAU0 * 2).mean())
            fs_snk = float((tau_c[regions["core_sink"]] == TAU0 * 2).mean())
            print(f"[h3.1b] stripe_w={w:.3f} phase_shift={w/2:.3f} | slow-fraction: core_source "
                  f"{100*fs_src:.1f}%  core_sink {100*fs_snk:.1f}%  (H3.1 bug was 51.1% / 48.4%)", flush=True)
            assert fs_src < 0.02 and fs_snk > 0.98, "centred stripes must separate the two cores"

            eta_mf = eta0 * (GM_UNIFORM_DYN / GM_MEANFIELD_H31)          # first-order load rescale
            print(f"[h3.1b] mean-field eta rescale {eta0:.5f} -> {eta_mf:.5f} "
                  f"(target realized gM {GM_UNIFORM_DYN}); realized gM is reported per arm", flush=True)

            def dyn(tau_E, eta_E):
                return {**anchor, "use_m": True, "eta_m": eta0, "tau_adp": TAU0,
                        "tau_adp_E": tau_E, "eta_m_E": eta_E, "m_enable_ms": P1.M_ENABLE_MS}

            arms = [
                ("patch_dyn_centered", dyn(tau_c, eta_c)),
                ("patch_dyn_centered_shuffled", dyn(tau_sh, eta_sh)),
                ("patch_dyn_centered_swapped", dyn(tau_sw, eta_sw)),
                ("meanfield_dyn_loadmatched", {**anchor, "use_m": True, "eta_m": eta_mf,
                                               "tau_adp": TAU0, "m_enable_ms": P1.M_ENABLE_MS,
                                               "m_mean_field": True}),
            ]
            with mp.get_context("fork").Pool(min(a.workers, 2)) as pool:
                rows = pool.map(S1._arm_cell, arms)
            for r in rows:
                print(f"[h3.1b] {r['label']:30s} rate {r['mean_rate_hz']:6.1f}Hz  gM {r['mean_gM']:.4f}  "
                      f"PR {r['participation_ratio_med']:.3f}  alternation {r['region_alternation']:+.3f}  "
                      f"unsafe={r['numerical_unsafe']} runaway={r['runaway_early_stop_ms']}", flush=True)
            HEO1R.FCXR._write_json(os.path.join(OUT, "stage1b_arms.json"),
                                   dict(eta0=eta0, eta_meanfield=eta_mf, stripe_w=w, phase_shift=w / 2.0,
                                        slow_frac_core_source=fs_src, slow_frac_core_sink=fs_snk,
                                        gm_target=GM_UNIFORM_DYN, arms=rows))
            HEO1R.FCXR._write_json(os.path.join(OUT, "H31B_DONE.json"),
                                   dict(n=len(rows), finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "H31B_RUNNING.json")):
                os.remove(os.path.join(OUT, "H31B_RUNNING.json"))
            print("[h3.1b] DONE", flush=True)
        except Exception as exc:
            HEO1R.FCXR._write_json(os.path.join(OUT, "H31B_FAILED.json"),
                                   dict(error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
