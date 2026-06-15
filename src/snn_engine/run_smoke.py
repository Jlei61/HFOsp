"""
Smoke run for the spatially-structured E-I LIF wave engine.

Produces (in ./outputs/):
  smoke_overview.png  -- raster, E/I population rate, LFP traces, PSD
  smoke_lfp_frames.png -- LFP snapshots across the grid (look for a wave)
and prints verification numbers:
  * single E->E PSP peak amplitude (sanity vs Table-1 J~0.1 mV)
  * dominant LFP PSD frequency (expect a beta-band-ish peak in the SI regime)

Usage:
    python run_smoke.py                 # smoke config (fast)
    python run_smoke.py --g 4.1 --ratio 0.9
"""

from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal

from params import Params, compute_nu_theta
from model import build_network, simulate
from lfp import LFPRecorder

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def single_psp_peak(p: Params):
    """Integrate one isolated E neuron receiving a single E->E spike at t=0,
    using the SAME exp-Euler scheme, and return the peak V deflection (mV).
    Verifies the synaptic gain (Table 1 realized EPSP ~0.1 mV)."""
    dt = p.dt
    w_EE = p.w_EE
    decay_sE = np.exp(-dt / p.tau_r_AMPA)
    decay_IE = np.exp(-dt / p.tau_d_AMPA)
    decay_V = np.exp(-dt / p.tau_m_E)
    s = (p.tau_m_E / p.tau_r_AMPA) * w_EE     # single-spike jump on s
    I = 0.0; V = 0.0; peak = 0.0
    for _ in range(int(60 / dt)):             # 60 ms
        s *= decay_sE
        I = s + (I - s) * decay_IE
        V = I + (V - I) * decay_V
        peak = max(peak, V)
    return peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None, help="nu_ext/nu_theta")
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--density", type=float, default=None)
    args = ap.parse_args()

    p = Params()
    if args.g is not None: p.g = args.g
    if args.ratio is not None: p.nu_ext_ratio = args.ratio
    if args.T is not None: p.T = args.T
    if args.L is not None: p.L = args.L
    if args.density is not None: p.density = args.density

    nu_theta, nE, nI = compute_nu_theta(p)
    print(f"nu_theta = {nu_theta*1e3:.1f} Hz (E {nE*1e3:.0f}, I {nI*1e3:.0f}); "
          f"nu_signal = {p.nu_ext_ratio*nu_theta*1e3:.1f} Hz; g={p.g}")

    psp = single_psp_peak(p)
    print(f"[verify] single E->E PSP peak = {psp:.4f} mV  "
          f"(Table-1 realized EPSP ~0.10-0.11 mV)")

    net = build_network(p)
    rec = LFPRecorder(p, net["pos"], net["labels"])
    print(f"LFP grid: {len(rec.sites)} sites")

    res = simulate(p, net, record_lfp=rec, lfp_every=2)

    # ---- oscillation diagnostic: E population-rate PSD (cleanest) ----
    dt = p.dt
    kbin = int(round(1.0 / dt))
    nn = (len(res["rate_E"]) // kbin) * kbin
    rE_1ms = res["rate_E"][:nn].reshape(-1, kbin).mean(1)
    fr, Pr = signal.welch(rE_1ms - rE_1ms.mean(), fs=1000.0,
                          nperseg=min(len(rE_1ms), 1024))
    bandr = (fr >= 10) & (fr <= 45)
    fpeak = fr[bandr][np.argmax(Pr[bandr])] if bandr.any() else float("nan")
    prom = (Pr[bandr].max() / np.median(Pr[(fr >= 5) & (fr <= 80)])) if bandr.any() else 0
    print(f"[verify] E population-rate PSD peak (10-45 Hz) = {fpeak:.1f} Hz  "
          f"prominence = {prom:.1f}")

    # ---- spatial-mean LFP PSD (search 12-45 Hz to skip OU low-freq) ----
    lfp = res["lfp"]                       # (frames, sites)
    lfp_dt = (res["lfp_t"][1] - res["lfp_t"][0]) / 1e3  # s
    fs = 1.0 / lfp_dt
    mean_lfp = lfp.mean(axis=1)
    mean_lfp = mean_lfp - mean_lfp.mean()
    nper = min(len(mean_lfp), 512)
    f, P = signal.welch(mean_lfp, fs=fs, nperseg=nper)
    band = (f >= 12) & (f <= 45)
    flfp = f[band][np.argmax(P[band])] if band.any() else float("nan")
    print(f"[verify] LFP PSD peak (12-45 Hz) = {flfp:.1f} Hz")

    print("VERIFY_SUMMARY", dict(nu_theta_Hz=float(nu_theta*1e3),
          psp_mV=float(psp), rateE_Hz=float(res['rate_E'].mean()),
          rateI_Hz=float(res['rate_I'].mean()), Erate_PSD_peak_Hz=float(fpeak),
          Erate_PSD_prom=float(prom), LFP_PSD_peak_Hz=float(flfp)))


if __name__ == "__main__":
    main()
