#!/usr/bin/env python
"""Z/M-native containment-to-exit lifecycle -- CORRECT-substrate rebuild (2026-07-22).

The prior q_I+S_G+p/H sandbox (run_m4_snn_native_exit.py) ran the WRONG substrate. This runs the
LOCKED Z/M model -- per-neuron z (inhibitory efficacy) + m (adaptation), lockpoint
zA_q75_tz5000__mA0p001_tau500 -- on the SAME E1146 twoend_equal substrate, optionally adds the S_G
divisive containment pool + H slow memory, and tests the lifecycle:

  spontaneous interictal core firing --(z depletes on the top-q75 inhibited cells)--> onset -->
  bounded ictal --(S_G/H containment)--> termination --> does z recover(->1) + m decay(->0) return
  the substrate to interictal IED generation?

Novel vs the q_I sandbox: q_I recovery FAILED because the q_I substrate had no stable interictal
attractor (IEDs were a q_I-depletion entry transient). The Z/M substrate DIFFERS -- z heals toward 1
and m decays after an event, potentially RE-CREATING the pre-ictal excitable state -> recovery may
work here. This runner CHARACTERIZES that on the correct substrate.

z threshold I_th_EI = q75 of the slow-OFF interictal E-cell inhibitory current (calibrated in-run).
Reuses PP.build_substrate + M4._e_disk_mask + slow_field (z+m ported, byte-parity vs mz_slow_vars) +
classify_termination. Spontaneous protocol (KICK_BOOST=0, cores self-ignite). OMP=1. --confirm-run gated.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_m4_phaseplane as PP          # noqa: E402  (build_substrate + constants)
import run_m4_dynamic_qi as M4          # noqa: E402  (forces OMP=1; _e_disk_mask; DT)
from kick_probe import simulate_kick    # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_m4_termination import classify_termination  # noqa: E402

DT = 0.1
# locked Z/M working point (zA_q75_tz5000__mA0p001_tau500)
TAU_Z, TAU_ADP, ETA_M = 5000.0, 500.0, 0.001
OUT = os.path.join(PP.ROOT, "results", "topic4_sef_hfo", "zm_snn_native_exit")


class _IIObserver:
    """Byte-parity slow object (apply_currents == I_E - I_I) that SAMPLES E-cell I_I at a step stride
    for the I_th_EI percentile calibration (slow-off interictal baseline)."""
    def __init__(self, nE, stride=50):
        self.nE = int(nE); self.stride = int(stride); self._i = 0; self.samples = []
    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        if self._i % self.stride == 0:
            self.samples.append(np.asarray(I_I, float)[:self.nE].copy())
        return np.asarray(I_E, float) - np.asarray(I_I, float)
    def threshold(self, V_th_base):
        return V_th_base
    def step(self, spk, labels, dt):
        self._i += 1


def _calibrate_I_th_EI(S, T_ms=1500.0, q=75.0, settle_frac=1.0 / 3.0):
    """Slow-OFF interictal baseline -> q-th percentile of the SETTLED E-cell inhibitory current I_I."""
    p = dataclasses.replace(S["p"], T=float(T_ms))
    obs = _IIObserver(S["NE"], stride=50)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    simulate_kick(p, S["net"], 0.0, slow=obs, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                  t_kick=1e9, V_th_per_neuron=S["vth"], verbose=False)
    n = len(obs.samples)
    settled = np.concatenate(obs.samples[int(n * settle_frac):], axis=0)
    return float(np.percentile(settled, q))


def _zm_cfg(I_th_EI, *, use_SG=False, alpha_G=0.0, use_H=False, alpha_H=0.0, tau_H=6000.0,
            H_sensor="active", use_persist=False, tau_p=5000.0, tau_p_down=None):
    """Locked Z/M base + optional S_G divisive containment + H slow memory. q_I/g_K OFF (use_qI=False
    -> q_I==1 -> z*q_I*I_I == z*I_I). H rides S_G's recurrent-divisive term (needs use_SG) and its p
    sensor (needs use_persist, eta_r=0 -> sensor only, no subtractive current). H_sensor='active' =
    active-focus intensity (the Z/M bursting focus is localized -> the global spatial mean starves H)."""
    return SpatialSlowFieldConfig(
        use_qI=False, use_gK=False,
        use_z=True, use_m=True, tau_z=TAU_Z, I_th_EI=float(I_th_EI), tau_adp=TAU_ADP, eta_m=ETA_M,
        use_SG=use_SG, alpha_G=alpha_G, r0_psi=0.0, r50_psi=M4.R50_PSI, n_psi=M4.N_PSI,
        p_pool=M4.P_POOL, tau_mu=M4.TAU_MU, tau_S=M4.TAU_S, S_max=M4.S_MAX,
        use_persist=use_persist, tau_p=tau_p, tau_p_down=tau_p_down, sigma_p=1.5, eta_r=0.0,
        use_H=use_H, alpha_H=alpha_H, tau_H=tau_H, H_sensor=H_sensor)


def _core_mask_E(S):
    """E-only bool mask of the two low-V_th cores (E cells within PP.CORE_R of source/sink centroid)."""
    return M4._e_disk_mask(S, [S["src_xy"], S["snk_xy"]], PP.CORE_R)[:S["NE"]]


def _rate_and_af(spk_E, mask, bin_ms=25.0):
    """Per-bin firing rate (Hz) over the masked E cells + active fraction (frac of masked E cells that
    fired >=1 spike in the bin). mask None -> all E cells."""
    m = np.ones(spk_E.shape[1], bool) if mask is None else mask
    nsel = max(1, int(m.sum()))
    bs = int(round(bin_ms / DT))
    rate, af = [], []
    for b0 in range(0, spk_E.shape[0], bs):
        seg = spk_E[b0:b0 + bs][:, m]
        rate.append(float(seg.sum()) / nsel / (seg.shape[0] * DT) * 1e3)
        af.append(float((seg.sum(axis=0) > 0).mean()))
    return np.asarray(rate, np.float32), np.asarray(af, np.float32)


def _run_arm(S, label, cfg, T_ms, early_stop=True, es_thresh_hz=120.0):
    t0 = time.time()
    p = dataclasses.replace(S["p"], T=float(T_ms))
    core = _core_mask_E(S)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], core_mask_E=core, cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], verbose=False,
                        early_stop_runaway=early_stop, es_thresh_hz=es_thresh_hz, es_dur_ms=100.0)
    spk = res["E_spk_bool"]
    core_rate, _ = _rate_and_af(spk, core, bin_ms=25.0)
    surr_rate, _ = _rate_and_af(spk, ~core, bin_ms=25.0)
    all_rate, af = _rate_and_af(spk, None, bin_ms=25.0)
    baseline = float(np.median(af[:max(1, len(af) // 20)]))    # first ~5% = interictal baseline af
    runaway_ms = res.get("runaway_early_stop_ms")              # kick_probe key: sustained >=es_thresh_hz -> truncated
    cls, info = classify_termination(af, 25.0, baseline=baseline, runaway_ms=runaway_ms)
    row = dict(
        label=label, seed=int(S["seed"]), T_ms=float(T_ms), n_steps=int(spk.shape[0]),
        termination_class=cls, offset_ms=info.get("offset_ms"), runaway_ms=runaway_ms,
        peak_all_hz=float(all_rate.max()), tail_all_hz=float(all_rate[-max(1, len(all_rate) // 20):].mean()),
        z_min_final=float(slow.trace_z_min[-1]), z_mean_final=float(slow.trace_z_mean[-1]),
        z_core_final=float(slow.trace_z_core_mean[-1]) if slow.trace_z_core_mean else None,
        m_mean_final=float(slow.trace_m_mean[-1]), S_G_max=float(max(slow.trace_SG) if slow.trace_SG else 0.0),
        H_max=float(max(slow.trace_H) if slow.trace_H else 0.0), wall_s=round(time.time() - t0, 1),
        cfg=dataclasses.asdict(cfg))
    arrays = dict(
        core_rate=core_rate, surr_rate=surr_rate, all_rate=all_rate, af=af,
        z_mean=np.asarray(slow.trace_z_mean, np.float32), z_min=np.asarray(slow.trace_z_min, np.float32),
        z_core=np.asarray(slow.trace_z_core_mean, np.float32),
        z_surround=np.asarray(slow.trace_z_surround_mean, np.float32),
        m_mean=np.asarray(slow.trace_m_mean, np.float32), m_max=np.asarray(slow.trace_m_max, np.float32),
        SG=np.asarray(slow.trace_SG, np.float32), H=np.asarray(slow.trace_H, np.float32),
        p_mean=np.asarray(slow.trace_p_mean, np.float32), p_max=np.asarray(slow.trace_p_max, np.float32))
    row["p_max_final"] = float(slow.trace_p_max[-1]) if slow.trace_p_max else 0.0
    return row, arrays


ARMS = {
    "bare":  dict(),                                                        # Z/M only (no containment)
    "sg":    dict(use_SG=True, alpha_G=16.0),                               # + divisive containment pool
    "sgh":   dict(use_SG=True, alpha_G=16.0, use_H=True, alpha_H=16.0,      # + slow memory H (active-focus sensor)
                  tau_H=6000.0, H_sensor="active", use_persist=True,        # asymmetric p: fast charge (2000) / slow
                  tau_p=2000.0, tau_p_down=10000.0),                        # decay (10000) -> p accumulates the bursty
                                                                           # IED train into a sustained-seizure memory
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true", help="required: guards the multi-minute sim")
    ap.add_argument("--arms", default="bare", help="comma list of " + ",".join(ARMS))
    ap.add_argument("--T", type=float, default=12000.0, help="sim length ms (onset~2k + bounded + recovery)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--alpha-G", dest="alpha_G", type=float, default=None, help="override S_G strength")
    ap.add_argument("--alpha-H", dest="alpha_H", type=float, default=None, help="override H strength")
    ap.add_argument("--no-early-stop", action="store_true", help="run full T even on runaway")
    ap.add_argument("--es-thresh", dest="es_thresh", type=float, default=120.0,
                    help="runaway early-stop Hz threshold; raise (~250) for containment arms so a bounded-but-"
                         "high plateau runs full T instead of being truncated as runaway")
    a = ap.parse_args()
    if not a.confirm_run:
        raise SystemExit("refusing to run without --confirm-run (each arm is multi-minute at N=40000)")
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    assert all(x in ARMS for x in arms), f"unknown arm; choices={list(ARMS)}"

    os.makedirs(OUT, exist_ok=True)
    S = PP.build_substrate(seed=a.seed)
    t_cal = time.time()
    I_th_EI = _calibrate_I_th_EI(S)
    print(f"[calib] I_th_EI = q75(slow-off interictal I_I) = {I_th_EI:.4f}  ({time.time()-t_cal:.0f}s)")

    rows = []
    for name in arms:
        kw = dict(ARMS[name])
        if a.alpha_G is not None and "alpha_G" in kw:
            kw["alpha_G"] = a.alpha_G
        if a.alpha_H is not None and "alpha_H" in kw:
            kw["alpha_H"] = a.alpha_H
        cfg = _zm_cfg(I_th_EI, **kw)
        print(f"[arm {name}] running T={a.T:.0f}ms (es_thresh={a.es_thresh:.0f}Hz) ...", flush=True)
        row, arrays = _run_arm(S, name, cfg, a.T, early_stop=not a.no_early_stop, es_thresh_hz=a.es_thresh)
        rows.append(row)
        np.savez_compressed(os.path.join(OUT, f"{name}_seed{a.seed}.npz"), **arrays)
        print(f"[arm {name}] cls={row['termination_class']} peak={row['peak_all_hz']:.1f}Hz "
              f"tail={row['tail_all_hz']:.1f}Hz z_min={row['z_min_final']:.3f} "
              f"S_G_max={row['S_G_max']:.3f} H_max={row['H_max']:.3f} wall={row['wall_s']}s", flush=True)

    jpath = os.path.join(OUT, f"lifecycle_seed{a.seed}.json")
    by_label = {}                                              # accumulate across runs: this run's arms update by label
    if os.path.exists(jpath):
        try:
            for rr in json.load(open(jpath)).get("rows", []):
                by_label[rr["label"]] = rr
        except Exception:
            pass
    for rr in rows:
        by_label[rr["label"]] = dict(rr, T_ms=float(a.T))
    order = ["bare", "sg", "sgh"]
    merged = [by_label[k] for k in order if k in by_label] + [v for k, v in by_label.items() if k not in order]
    manifest = dict(subject="epilepsiae_1146", placement="twoend_equal", substrate=dict(
        L=float(S["L"]), N=int(S["N"]), NE=int(S["NE"])), I_th_EI=I_th_EI, seed=a.seed,
        lockpoint="zA_q75_tz5000__mA0p001_tau500", tau_z=TAU_Z, tau_adp=TAU_ADP, eta_m=ETA_M, rows=merged)
    with open(jpath, "w") as f:
        json.dump(manifest, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    print(f"[done] wrote {jpath} ({len(merged)} arms: {[r['label'] for r in merged]})")


if __name__ == "__main__":
    main()
