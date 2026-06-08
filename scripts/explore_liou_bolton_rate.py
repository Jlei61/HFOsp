"""EXPLORATORY (capability/feasibility, NOT a data-locked conclusion):

Bolt the three Liou-Abbott (eLife 2020, Liou…Abbott) seizure-model mechanisms onto OUR
Siegert LIF rate field and ask: does the rate model then produce adaptation-driven
repetitive bursting (the ictal tonic→clonic relaxation oscillation)? Comparison is the
SAME model with the mechanisms ON vs OFF (advisor 2026-06-08), in an EXCITABLE regime
(w_ee_mult≈2 — the locked SEF-HFO operating point is monostable-quiet and CANNOT burst;
this exploration is deliberately separated from those data-locked nulls).

Three mechanisms (paper eqs, docs/paper/abbott_model.md):
  1. adaptive threshold   τ_φ φ̇ = (φ0−φ) + Δ_φ f      (eq 2) — firing raises the threshold
  2. sAHP recovery        τ_a ȧ  = −a + r ; μ ← μ − b_a a (eq 4 analogue; our existing recovery)
  3. Mexican-hat surround  inhibition wider than excitation (l_inh > ell)  (eq 5)
(chloride/eq-3 is the slow wavefront-expansion mechanism, τ=5 s — NOT bursting — skipped.)

1-D neural field (Liou's primary reduction, their Fig 2). Headline = per-contact firing
rate(t): single self-limited/tonic response (baseline) vs repetitive bursting (ON).

Run: PYTHONPATH="$PWD" python scripts/explore_liou_bolton_rate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import (  # noqa: E402
    mean_field, lif_rate, V_TH, TAU_ME, TAU_MI, TREF_E, TREF_I,
    C_EE, W_EE, C_EI, W_EI, C_IE, W_IE, C_II, W_II, JX_E, JX_I,
    TAU_AMPA, TAU_GABA,
)

OUT = "results/topic4_sef_hfo/observation_layer/figures"
N = 200            # 1-D populations
L = 12.0           # mm
DX = L / N
ELL = 0.54         # E->E kernel width (mm)


def _gauss_kernel(width):
    half = N // 2
    x = (np.arange(N) - half) * DX
    g = np.exp(-0.5 * (x / width) ** 2)
    return g / g.sum()


def _conv(field, kern):
    # periodic 1-D convolution (kernel centered)
    return np.real(np.fft.ifft(np.fft.fft(field) * np.fft.fft(np.fft.ifftshift(kern))))


def _build_rate_lut(sigma):
    mus = np.linspace(-20.0, 220.0, 700)
    vths = np.linspace(V_TH, V_TH + 160.0, 320)
    tab = np.empty((mus.size, vths.size))
    for i, m in enumerate(mus):
        for j, v in enumerate(vths):
            try:
                tab[i, j] = lif_rate(float(m), sigma, TAU_ME, TREF_E, v_th=float(v))
            except Exception:
                tab[i, j] = 0.0
    tab = np.nan_to_num(tab, nan=0.0, posinf=0.5, neginf=0.0)
    return RegularGridInterpolator((mus, vths), tab, bounds_error=False, fill_value=None), (mus, vths)


def integrate_1d(*, mechanisms, w_ee_mult=2.0, dphi=1.0, b_a=3.0, tphi=150.0, ta=800.0,
                 ell_inh_on=1.6, ell_inh_off=0.5, dt=0.5, t_max=8000.0,
                 kick_amp=10.0, kick_t=(200.0, 600.0), kick_frac=0.12):
    """1-D bolt-on rate field. mechanisms=False → baseline (no adaptive threshold, no sAHP,
    narrow inhibition); True → all three ON. sigma fixed at the canonical op's sE (as in
    integrate_lif_field). Returns (t, frames[nsteps,N] in Hz)."""
    op = mean_field(1.0)
    sE, sI, nuext = op["sE"], op["sI"], op["nuext"]
    rate_E, _grids = _build_rate_lut(sE)
    musI = np.linspace(-20.0, 220.0, 1500)
    rI_lut = np.array([lif_rate(m, sI, TAU_MI, TREF_I) for m in musI])

    wee = w_ee_mult * W_EE
    KE = _gauss_kernel(ELL)
    KI = _gauss_kernel(ell_inh_on if mechanisms else ell_inh_off)
    dphi_e = dphi if mechanisms else 0.0
    ba_e = b_a if mechanisms else 0.0

    muxE = TAU_ME * JX_E * nuext
    muxI = TAU_MI * JX_I * nuext
    rE = np.full(N, 1e-4)
    rI = np.full(N, 1e-4)
    vth = np.full(N, V_TH)
    a = np.full(N, 1e-4)
    sEE = _conv(rE, KE).copy(); sEI = _conv(rI, KI).copy()
    sIE = _conv(rE, KI).copy(); sII = _conv(rI, KI).copy()

    nsteps = int(t_max / dt)
    frames = np.empty((nsteps, N))
    kick_mask = np.arange(N) < int(kick_frac * N)
    for t in range(nsteps):
        stim = kick_amp * kick_mask if (kick_t[0] <= t * dt < kick_t[1]) else 0.0
        sEE += dt / TAU_AMPA * (_conv(rE, KE) - sEE)
        sEI += dt / TAU_GABA * (_conv(rI, KI) - sEI)
        sIE += dt / TAU_AMPA * (_conv(rE, KI) - sIE)
        sII += dt / TAU_GABA * (_conv(rI, KI) - sII)
        muE = TAU_ME * (C_EE * wee * sEE - C_EI * W_EI * sEI) + muxE + stim - ba_e * a
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * W_II * sII) + muxI
        fE = rate_E(np.column_stack([np.clip(muE, -20, 220), np.clip(vth, V_TH, V_TH + 160)]))
        fI = np.interp(muI, musI, rI_lut)
        rE = rE + dt / TAU_ME * (-rE + fE)
        rI = rI + dt / TAU_MI * (-rI + fI)
        vth = vth + dt / tphi * ((V_TH - vth) + dphi_e * rE * 1000.0)
        a = a + dt / ta * (-a + rE)
        frames[t] = rE * 1000.0      # Hz
    return np.arange(nsteps) * dt, frames


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[bolt-on] baseline (mechanisms OFF) ...", flush=True)
    t, fb = integrate_1d(mechanisms=False)
    print("[bolt-on] mechanisms ON (adaptive threshold + sAHP + surround) ...", flush=True)
    _t, fo = integrate_1d(mechanisms=True)

    ts = t / 1000.0
    xmm = np.arange(N) * DX
    contacts = [int(0.10 * N), int(0.25 * N), int(0.45 * N), int(0.65 * N)]   # along the line
    fig, ax = plt.subplots(2, 2, figsize=(13, 8),
                           gridspec_kw={"height_ratios": [1.2, 1.0]})
    for col, (frames, ttl) in enumerate([(fb, "baseline — mechanisms OFF"),
                                         (fo, "ON — adaptive threshold + sAHP + surround")]):
        im = ax[0, col].imshow(frames.T, aspect="auto", origin="lower",
                               extent=(ts[0], ts[-1], 0, L), cmap="magma",
                               vmin=0, vmax=max(50, np.percentile(fo, 99.5)))
        ax[0, col].set_title(ttl, fontsize=11)
        ax[0, col].set_ylabel("space (mm)")
        for c in contacts:
            ax[0, col].axhline(c * DX, color="cyan", lw=0.5, alpha=0.5)
        fig.colorbar(im, ax=ax[0, col], label="rate (Hz)", fraction=0.046)
        for k, c in enumerate(contacts):
            ax[1, col].plot(ts, frames[:, c] + k * 0, lw=0.8,
                            label=f"x={c*DX:.1f} mm")
        ax[1, col].set_title("per-contact firing rate(t)", fontsize=10)
        ax[1, col].set_xlabel("time (s)"); ax[1, col].set_ylabel("rate (Hz)")
        ax[1, col].legend(fontsize=7, ncol=2)
    fig.suptitle("Liou-Abbott mechanisms bolted onto the LIF rate field (1-D, excitable regime "
                 "w_ee×2) — adaptation-driven bursting vs baseline\n"
                 "EXPLORATORY capability test in an excitable regime — NOT the data-locked "
                 "SEF-HFO operating point", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "liou_bolton_bursting_vs_baseline.png")
    fig.savefig(path, dpi=95)
    print(f"wrote {path}")
    # quick numeric summary
    def burst_count(frames, c, after=700):
        x = frames[int(after / 0.5):, c]
        if x.max() < 5: return 0
        thr = 0.3 * x.max(); pk = 0
        for i in range(1, len(x) - 1):
            if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > thr: pk += 1
        return pk
    c = contacts[1]
    print(f"  contact x={c*DX:.1f}mm: baseline peaks={burst_count(fb,c)} "
          f"max={fb[:,c].max():.0f}Hz | ON peaks={burst_count(fo,c)} max={fo[:,c].max():.0f}Hz")


if __name__ == "__main__":
    main()
