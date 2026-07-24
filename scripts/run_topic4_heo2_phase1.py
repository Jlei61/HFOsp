"""FCXR-HEO2 Phase 1 — delayed, force-matched adaptation wedge on the sustained-16 Hz anchor.

Question: can delayed spike-frequency adaptation break the anchor's sustained ~16 Hz high state into a
spiky ~3-8 Hz broadband rhythm WHILE keeping coverage (not just stall/collapse it)? 1 s m-off to
establish the 16 Hz state, THEN enable adaptation. eta_m force-matched per tau (peak adaptation current =
frac of recurrent drive, self-consistent from the anchor's own high state) so timescale is not confounded
with strength. Static m_frozen_E arm = matched mean-K load without dynamics. Nothing runs on import.
Spec: docs/superpowers/specs/2026-07-24-topic4-heo2-broadband-diagnostic-design.md
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
import multiprocessing as mp
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_fcxr_heo1 as HEO1R  # noqa: E402  (_build_and_align/_montage/_heo_cfg/_heo_run/_numerical/_load_baseline/FCXR/DT)
from src.topic4_mz_slowvars import replay_adaptation_peak, eta_m_from_frac  # noqa: E402
from src.topic4_mz_fcxr_dynamics import workpoint_metrics, classify_run_workpoint  # noqa: E402

DT = HEO1R.DT
DENOM_E = 58.0 - 18.0                     # E_E - v_match: gErec conductance -> recurrent current-equivalent
ANCHOR_GQ, ANCHOR_A, ANCHOR_D = 0.999, 8.0, 0.15   # strongest sustained-16Hz state (Phase-0 pick)
TAUS = [250.0, 750.0]
FRACS = [0.05, 0.10]
M_ENABLE_MS = 1000.0                      # establish the 16 Hz high state for 1 s, then enable adaptation
OUT = os.path.join(HEO1R.FCXR.OUT_ROOT, "broadband_diagnostic")
_CTX = {}


def _forcematch(S, anchor_cfg, contacts, seed):
    """Run the anchor m-off high state; return (I_EE_scale, {tau: peak_m_repr}) self-consistently."""
    res, slow = HEO1R._heo_run(S, anchor_cfg, 2000.0, kick_boost=0.0, t_kick=1e9, seed=seed,
                               lfp_sites=contacts, early_stop=False)
    ge = np.asarray(slow.trace_gErec_mean, float)
    n0 = int(500.0 / DT)                                  # skip the first 500 ms establishment
    I_EE = float(np.mean(ge[n0:]) * DENOM_E) if ge.size > n0 else float(np.mean(ge) * DENOM_E)
    E = np.asarray(res["E_spk_bool"])[n0:]
    peaks = {tau: float(np.percentile(replay_adaptation_peak(E, DT, tau), 90)) for tau in TAUS}
    return I_EE, peaks, res, slow


def _adapt_cfg(base_cfg, *, eta_m, tau_adp, m_enable_ms=None, m_frozen_E=None):
    cfg = dict(base_cfg)
    cfg.update(eta_m=float(eta_m), tau_adp=float(tau_adp))
    if m_frozen_E is not None:
        cfg.update(use_m=False, m_frozen_E=m_frozen_E, m_enable_ms=None)
    else:
        cfg.update(use_m=True, m_enable_ms=m_enable_ms)
    return cfg


def _workpoint_cell(task):
    """D=0 workpoint gate: adaptation from t=0 at baseline must NOT erase interictal IED."""
    tau, frac, eta_m = task
    S, p_i, contacts, u_c, roll_hi, anchor_base0 = (_CTX[k] for k in ("S", "p_i", "contacts", "u_c", "roll_hi", "base0"))
    cfg = _adapt_cfg(anchor_base0, eta_m=eta_m, tau_adp=tau, m_enable_ms=None)   # D=0 base, adaptation on
    res, slow = HEO1R._heo_run(S, cfg, 4000.0, kick_boost=0.0, t_kick=1e9, seed=1, lfp_sites=contacts, early_stop=True)
    num = HEO1R._numerical(S, res, slow)
    rate = np.asarray(res["rate_E"], float)
    wp = classify_run_workpoint(dict(numerical_unsafe=num["numerical_unsafe"], **workpoint_metrics(rate, DT, roll_hi)))
    preserved = bool((not num["numerical_unsafe"]) and wp == "INTERICTAL_WORKPOINT" and num["runaway_early_stop_ms"] is None)
    return dict(tau=tau, frac=frac, eta_m=eta_m, workpoint_label=wp, preserved=preserved, mean_rate_hz=float(rate.mean()))


def _arm_cell(task):
    label, cfg, T = task
    S, contacts = _CTX["S"], _CTX["contacts"]
    t0 = time.time()
    res, slow = HEO1R._heo_run(S, cfg, T, kick_boost=0.0, t_kick=1e9, seed=1, lfp_sites=contacts, early_stop=True)
    num = HEO1R._numerical(S, res, slow)
    HEO1R.FCXR._write_npz(os.path.join(OUT, "arms", f"{label}_trace.npz"),
                          rate_E=np.asarray(res["rate_E"], np.float32),
                          lfp_trace=np.asarray(res["lfp_trace"], np.float32),
                          m_mean=np.asarray(slow.trace_m_mean, np.float32),
                          gM_mean=np.asarray(slow.trace_gM_mean, np.float32),
                          coop_engaged_frac=np.asarray(slow.trace_coop_engaged_frac, np.float32),
                          m_enable_ms=cfg.get("m_enable_ms") or 0.0, dt=DT)
    row = dict(label=label, T_ms=T, wall_s=round(time.time() - t0, 1), mean_rate_hz=float(np.asarray(res["rate_E"]).mean()),
               eta_m=cfg.get("eta_m"), tau_adp=cfg.get("tau_adp"), m_enable_ms=cfg.get("m_enable_ms"),
               static_K=bool(cfg.get("m_frozen_E") is not None), **num)
    HEO1R.FCXR._write_json(os.path.join(OUT, "arms", f"{label}.json"), row)
    return row


def _run_tasks(fn, tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, tasks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--confirm-run", action="store_true")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("pass --confirm-run to launch the Phase-1 wedge")
    HEO1R.FCXR._assert_engine_blessed()
    os.makedirs(os.path.join(OUT, "arms"), exist_ok=True)
    with HEO1R.FCXR._launcher_lock():
        HEO1R.FCXR._resource_log(OUT, "phase1_start", HEO1R.FCXR._plan_workers(6000.0, a.workers))
        HEO1R.FCXR._write_json(os.path.join(OUT, "RUNNING.json"),
                               dict(mode="heo2_phase1", pid=os.getpid(), started=datetime.now(timezone.utc).isoformat()))
        try:
            ref, u_c, roll_hi, scl, contacts0 = HEO1R._load_baseline(1)
            S, p_i = HEO1R._build_and_align(1)
            contacts, names, scl2 = HEO1R._montage(S)
            uc = u_c[ANCHOR_GQ]
            anchor = HEO1R._heo_cfg(ANCHOR_A, uc, ANCHOR_D, p_i)            # D=0.15 sustained-16Hz anchor
            base0 = HEO1R._heo_cfg(ANCHOR_A, uc, None, p_i)                 # D=0 baseline (workpoint gate)
            NE = int(S["NE"])
            print("[heo2] force-matching from the anchor m-off high state ...", flush=True)
            I_EE, peaks, res0, slow0 = _forcematch(S, anchor, contacts, 1)
            etas = {(tau, frac): eta_m_from_frac(frac, I_EE, peaks[tau]) for tau in TAUS for frac in FRACS}
            print(f"[heo2] I_EE_scale={I_EE:.2f} peak_m={ {k: round(v,3) for k,v in peaks.items()} } "
                  f"etas={ {f'{t}/{f}': round(e,5) for (t,f),e in etas.items()} }", flush=True)

            _CTX.update(S=S, p_i=p_i, contacts=contacts, u_c=u_c, roll_hi=roll_hi, base0=base0)
            workers = min(HEO1R.FCXR._plan_workers(6000.0, a.workers)["workers"], 2)
            # workpoint gate per (tau,frac)
            wp_rows = _run_tasks(_workpoint_cell, [(t, f, etas[(t, f)]) for t in TAUS for f in FRACS], workers)
            survivors = [(r["tau"], r["frac"]) for r in wp_rows if r["preserved"]]
            HEO1R.FCXR._write_json(os.path.join(OUT, "phase1_workpoint_gate.json"), dict(rows=wp_rows, survivors=survivors))
            print(f"[heo2] workpoint survivors {len(survivors)}/{len(wp_rows)}: {survivors}", flush=True)

            # 6 arms: m_off + surviving dyn(tau,frac) + static_K
            arms = [("m_off", dict(anchor), 5000.0)]
            for tau, frac in survivors:
                T = max(5000.0, 1000.0 + 5.0 * tau)
                arms.append((f"dyn_tau{tau:g}_frac{frac:g}",
                             _adapt_cfg(anchor, eta_m=etas[(tau, frac)], tau_adp=tau, m_enable_ms=M_ENABLE_MS), T))
            arms.append(("static_K",
                         _adapt_cfg(anchor, eta_m=etas[(750.0, 0.10)], tau_adp=750.0,
                                    m_frozen_E=np.full(NE, peaks[750.0], float)), 5000.0))
            arm_rows = _run_tasks(_arm_cell, arms, workers)
            HEO1R.FCXR._write_json(os.path.join(OUT, "phase1_arms.json"),
                                   dict(I_EE_scale=I_EE, peak_m=peaks, etas={f"{t}/{f}": etas[(t, f)] for t in TAUS for f in FRACS},
                                        survivors=survivors, arms=arm_rows))
            HEO1R.FCXR._write_json(os.path.join(OUT, "DONE.json"),
                                   dict(mode="heo2_phase1", n_arms=len(arm_rows), finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "RUNNING.json")):
                os.remove(os.path.join(OUT, "RUNNING.json"))
            print(f"[heo2] DONE arms={[r['label'] for r in arm_rows]}", flush=True)
        except Exception as exc:
            HEO1R.FCXR._write_json(os.path.join(OUT, "FAILED.json"),
                                   dict(mode="heo2_phase1", error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
