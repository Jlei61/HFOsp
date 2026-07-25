"""FCXR-HEO2.1 P1-c controls (2 sims, parallel):
(A) `mean_static_K` — delayed static-K matched to the fast-τ/10% dynamic arm's MEAN load (constant K =
    eta_m·mean(m_post), injected after the 1 s establishment). Isolates dynamics vs mean-load: if the
    partial broadening/desync of the dynamic arm is NOT reproduced by the matched constant K, it comes
    from the time-course, not the mean. (The old peak-matched static_K over-applied K and silenced.)
(B) `dyn_tau750_frac0.1_ext` — the slow-τ/10% termination arm extended to 9 s, to distinguish true
    termination from no-recovery-within-the-5 s-window.
Reuses the Phase-1 anchor + force-matched etas (no re-force-match). setsid nohup / flock / sentinels / ≤2 workers.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-heo2")

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

DT = HEO1R.DT
OUT = P1.OUT
_CTX = {}


def _cfg_mean_static_K(anchor, eta_m, mean_m, NE):
    cfg = dict(anchor)
    cfg.update(use_m=False, eta_m=float(eta_m), m_frozen_E=np.full(NE, float(mean_m), float),
               m_frozen_enable_ms=P1.M_ENABLE_MS)                 # delayed constant-K = matched mean load
    return cfg


def _cfg_dyn(anchor, eta_m, tau):
    cfg = dict(anchor)
    cfg.update(use_m=True, eta_m=float(eta_m), tau_adp=float(tau), m_enable_ms=P1.M_ENABLE_MS)
    return cfg


def _arm_cell(task):
    label, cfg, T = task
    S, contacts = _CTX["S"], _CTX["contacts"]
    t0 = time.time()
    res, slow = HEO1R._heo_run(S, cfg, T, kick_boost=0.0, t_kick=1e9, seed=1, lfp_sites=contacts, early_stop=True)
    num = HEO1R._numerical(S, res, slow)
    enable = cfg.get("m_enable_ms") or cfg.get("m_frozen_enable_ms") or 0.0
    HEO1R.FCXR._write_npz(os.path.join(OUT, "arms", f"{label}_trace.npz"),
                          rate_E=np.asarray(res["rate_E"], np.float32),
                          lfp_trace=np.asarray(res["lfp_trace"], np.float32),
                          m_mean=np.asarray(slow.trace_m_mean, np.float32),
                          gM_mean=np.asarray(slow.trace_gM_mean, np.float32),
                          coop_engaged_frac=np.asarray(slow.trace_coop_engaged_frac, np.float32),
                          m_enable_ms=float(enable), dt=DT)
    row = dict(label=label, T_ms=T, wall_s=round(time.time() - t0, 1),
               mean_rate_hz=float(np.asarray(res["rate_E"]).mean()), eta_m=cfg.get("eta_m"),
               tau_adp=cfg.get("tau_adp"), m_enable_ms=cfg.get("m_enable_ms"),
               m_frozen_enable_ms=cfg.get("m_frozen_enable_ms"),
               static_K=bool(cfg.get("m_frozen_E") is not None), **num)
    HEO1R.FCXR._write_json(os.path.join(OUT, "arms", f"{label}.json"), row)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--confirm-run", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("pass --confirm-run to launch the HEO2.1 controls")
    HEO1R.FCXR._assert_engine_blessed()
    os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
    with HEO1R.FCXR._launcher_lock():
        HEO1R.FCXR._write_json(os.path.join(OUT, "CONTROLS_RUNNING.json"),
                               dict(pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
        try:
            pa = json.load(open(os.path.join(OUT, "phase1_arms.json")))
            eta_250_10 = float(pa["etas"]["250.0/0.1"]); eta_750_10 = float(pa["etas"]["750.0/0.1"])
            t = np.load(os.path.join(OUT, "arms", "dyn_tau250_frac0.1_trace.npz"), allow_pickle=True)
            mm = np.asarray(t["m_mean"], float); k = int(float(t["m_enable_ms"]) / DT)
            mean_m_250_10 = float(mm[k:].mean())                 # post-enable mean load of the fast/10% arm

            ref, u_c, roll_hi, scl, contacts0 = HEO1R._load_baseline(1)
            S, p_i = HEO1R._build_and_align(1)
            contacts, names, scl2 = HEO1R._montage(S)
            uc = u_c[P1.ANCHOR_GQ]
            anchor = HEO1R._heo_cfg(P1.ANCHOR_A, uc, P1.ANCHOR_D, p_i)
            NE = int(S["NE"])
            _CTX.update(S=S, contacts=contacts)
            print(f"[controls] mean_m(250/10% post)={mean_m_250_10:.3f} eta250_10={eta_250_10:.5f} "
                  f"eta750_10={eta_750_10:.5f} -> matched static-K gM load", flush=True)

            arms = [("mean_static_K", _cfg_mean_static_K(anchor, eta_250_10, mean_m_250_10, NE), 5000.0),
                    ("dyn_tau750_frac0.1_ext", _cfg_dyn(anchor, eta_750_10, 750.0), 9000.0)]
            rows = P1._run_tasks(_arm_cell, arms, min(a.workers, 2))
            HEO1R.FCXR._write_json(os.path.join(OUT, "phase1_controls.json"),
                                   dict(mean_m_250_10=mean_m_250_10, eta_250_10=eta_250_10,
                                        eta_750_10=eta_750_10, arms=rows))
            HEO1R.FCXR._write_json(os.path.join(OUT, "CONTROLS_DONE.json"),
                                   dict(n=len(rows), finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "CONTROLS_RUNNING.json")):
                os.remove(os.path.join(OUT, "CONTROLS_RUNNING.json"))
            print(f"[controls] DONE {[r['label'] for r in rows]}", flush=True)
        except Exception as exc:
            HEO1R.FCXR._write_json(os.path.join(OUT, "CONTROLS_FAILED.json"),
                                   dict(error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
