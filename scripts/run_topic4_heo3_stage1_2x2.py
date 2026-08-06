"""FCXR-HEO3 Stage H3.1 — causal 2x2: {uniform, patch} x {static load, dynamic adaptation}.

Question (review lock): can SPATIALLY ORGANIZED RECOVERY TIME let regions take turns carrying the
activity, so the tissue stays recruited while phase relationships decorrelate and the broadened state
PERSISTS? H3.0 floor to beat: 0/8 arms reached a single joint target window.

Design invariant — patchy tau with per-cell eta_i = eta0*tau0/tau_i holds every cell's steady-state K
LOAD fixed and varies only WHEN it recovers, so a positive result cannot be "patches just added load".
Patch stripes run perpendicular to the source->sink axis at corridor scale (width D/3 ~ 4.35, vs core
radius 1.5). Contrast: weak = 2x tau ratio, strong = 4x, both geometric-mean-preserving around 250 ms.

Arms (all delayed at m_enable_ms=1000 on the same 16Hz anchor, seed 1, T=5000ms):
  uniform_static  : mean-matched frozen m (uniform)                     [2x2 cell]
  patch_static    : frozen m in stripes, SAME global mean (4x contrast) [2x2 cell] - spatial, no dynamics
  patch_dyn_weak  : patchy tau 177/354 + compensated eta                [2x2 cell]
  patch_dyn_strong: patchy tau 125/500 + compensated eta                [2x2 cell]
  patch_dyn_strong_shuffled : same tau values permuted across cells     [control] - histogram/mean/var
                    preserved, spatial organization destroyed
  meanfield_dyn   : population-mean m applied to every cell             [control] - pure temporal
                    modulation, no inter-cell differences
`uniform_dynamic` (= HEO2 dyn_tau250_frac0.1) and `m_off` are reused from the existing runs.
Each arm saves rate/LFP for the joint-window gate AND its source-space rows (computed in-process).
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
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_fcxr_heo1 as HEO1R  # noqa: E402
import run_topic4_heo2_phase1 as P1  # noqa: E402
import src.topic4_mz_fcxr_heo3 as H3  # noqa: E402
PP = HEO1R.PP

DT = HEO1R.DT
OUT = os.path.join(HEO1R.FCXR.OUT_ROOT, "heo3")
TAU0 = 250.0
M_ENABLE = P1.M_ENABLE_MS
T_MS = 5000.0
_CTX = {}


def _arm_cell(task):
    label, cfg = task
    S, contacts, regions = _CTX["S"], _CTX["contacts"], _CTX["regions"]
    t0 = time.time()
    res, slow = HEO1R._heo_run(S, cfg, T_MS, kick_boost=0.0, t_kick=1e9, seed=1,
                               lfp_sites=contacts, early_stop=True)
    num = HEO1R._numerical(S, res, slow)
    rows = H3.source_space_audit(np.asarray(res["E_spk_bool"]), S["posE"], regions, DT,
                                 win_ms=1000.0, hop_ms=100.0)
    enable = cfg.get("m_enable_ms") or cfg.get("m_frozen_enable_ms") or 0.0
    HEO1R.FCXR._write_npz(os.path.join(OUT, "arms", f"{label}_trace.npz"),
                          rate_E=np.asarray(res["rate_E"], np.float32),
                          lfp_trace=np.asarray(res["lfp_trace"], np.float32),
                          m_mean=np.asarray(slow.trace_m_mean, np.float32),
                          gM_mean=np.asarray(slow.trace_gM_mean, np.float32),
                          m_enable_ms=float(enable), dt=DT)
    row = dict(label=label, wall_s=round(time.time() - t0, 1),
               mean_rate_hz=float(np.asarray(res["rate_E"]).mean()),
               mean_gM=float(np.mean(slow.trace_gM_mean)),           # the K-load match check
               participation_ratio_med=float(np.median([r["participation_ratio"] for r in rows])),
               region_alternation=H3.region_alternation(rows), source_rows=rows, **num)
    HEO1R.FCXR._write_json(os.path.join(OUT, "arms", f"{label}.json"), row)
    return {k: v for k, v in row.items() if k != "source_rows"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--confirm-run", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("pass --confirm-run to launch H3.1")
    HEO1R.FCXR._assert_engine_blessed()
    os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
    with HEO1R.FCXR._launcher_lock():
        HEO1R.FCXR._write_json(os.path.join(OUT, "H31_RUNNING.json"),
                               dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
        try:
            pa = json.load(open(os.path.join(HEO1R.FCXR.OUT_ROOT, "broadband_diagnostic", "phase1_arms.json")))
            eta0 = float(pa["etas"]["250.0/0.1"])
            ctrl = json.load(open(os.path.join(HEO1R.FCXR.OUT_ROOT, "broadband_diagnostic", "phase1_controls.json")))
            m_mean = float(ctrl["mean_m_250_10"])                     # matched static load from HEO2.1

            ref, u_c, roll_hi, scl, c0 = HEO1R._load_baseline(1)
            S, p_i = HEO1R._build_and_align(1)
            contacts, names, scl2 = HEO1R._montage(S)
            anchor = HEO1R._heo_cfg(P1.ANCHOR_A, u_c[P1.ANCHOR_GQ], P1.ANCHOR_D, p_i)
            NE = int(S["NE"])
            regions = H3.build_regions(S["posE"], S["src_xy"], S["snk_xy"], PP.CORE_R)
            _CTX.update(S=S, contacts=contacts, regions=regions)

            D = float(np.linalg.norm(np.asarray(S["snk_xy"], float) - np.asarray(S["src_xy"], float)))
            pw = D / 3.0                                              # corridor-scale stripe width
            r2, r4 = np.sqrt(2.0), 2.0
            mk = lambda ratio, sh=None: H3.build_patch_field(                       # noqa: E731
                S["posE"], S["src_xy"], S["snk_xy"], pw, TAU0 / ratio, TAU0 * ratio, TAU0, eta0, sh)
            tau_w, eta_w, _ = mk(r2)
            tau_s, eta_s, pid_s = mk(r4)
            tau_sh, eta_sh, _ = mk(r4, sh=11)
            # patchy STATIC load: same stripes, frozen m, SAME global mean as uniform_static.
            # Solve for `lo` instead of assuming a 50/50 stripe split — the stripes do not divide the
            # cells evenly, and an unmatched mean would confound "spatial pattern" with "more K".
            f0 = float((pid_s == 0).mean())
            lo = m_mean / (f0 + 4.0 * (1.0 - f0))
            m_patch = np.where(pid_s == 0, lo, 4.0 * lo)
            assert abs(m_patch.mean() - m_mean) < 1e-9, "patch_static global K load must match uniform_static"
            print(f"[h3.1] eta0={eta0:.5f} m_mean={m_mean:.3f} axis D={D:.2f} stripe_w={pw:.2f} "
                  f"| tau weak {tau_w.min():.0f}/{tau_w.max():.0f} strong {tau_s.min():.0f}/{tau_s.max():.0f} "
                  f"| load inv max|eta*tau-eta0*tau0| = {np.abs(eta_s*tau_s - eta0*TAU0).max():.2e} "
                  f"| m_patch mean {m_patch.mean():.3f} (target {m_mean:.3f})", flush=True)

            def dyn(tau_E, eta_E, **kw):
                return {**anchor, "use_m": True, "eta_m": eta0, "tau_adp": TAU0,
                        "tau_adp_E": tau_E, "eta_m_E": eta_E, "m_enable_ms": M_ENABLE, **kw}

            arms = [
                ("uniform_static", {**anchor, "use_m": False, "eta_m": eta0,
                                    "m_frozen_E": np.full(NE, m_mean), "m_frozen_enable_ms": M_ENABLE}),
                ("patch_static", {**anchor, "use_m": False, "eta_m": eta0,
                                  "m_frozen_E": m_patch, "m_frozen_enable_ms": M_ENABLE}),
                ("patch_dyn_weak", dyn(tau_w, eta_w)),
                ("patch_dyn_strong", dyn(tau_s, eta_s)),
                ("patch_dyn_strong_shuffled", dyn(tau_sh, eta_sh)),
                ("meanfield_dyn", {**anchor, "use_m": True, "eta_m": eta0, "tau_adp": TAU0,
                                   "m_enable_ms": M_ENABLE, "m_mean_field": True}),
            ]
            workers = min(a.workers, 2)
            with mp.get_context("fork").Pool(workers) as pool:
                rows = pool.map(_arm_cell, arms)
            for r in rows:
                print(f"[h3.1] {r['label']:26s} rate {r['mean_rate_hz']:6.1f}Hz  gM {r['mean_gM']:.4f}  "
                      f"PR {r['participation_ratio_med']:.3f}  alternation {r['region_alternation']}  "
                      f"unsafe={r['numerical_unsafe']} runaway={r['runaway_early_stop_ms']}", flush=True)
            HEO1R.FCXR._write_json(os.path.join(OUT, "stage1_arms.json"),
                                   dict(eta0=eta0, tau0=TAU0, m_mean=m_mean, stripe_w=pw, axis_D=D,
                                        contrast_weak=float(r2 ** 2), contrast_strong=float(r4 ** 2),
                                        arms=rows))
            HEO1R.FCXR._write_json(os.path.join(OUT, "H31_DONE.json"),
                                   dict(n=len(rows), finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "H31_RUNNING.json")):
                os.remove(os.path.join(OUT, "H31_RUNNING.json"))
            print("[h3.1] DONE", flush=True)
        except Exception as exc:
            HEO1R.FCXR._write_json(os.path.join(OUT, "H31_FAILED.json"),
                                   dict(error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
