"""EXPLORATORY (capability/feasibility in an EXCITABLE regime — NOT a data-locked
SEF-HFO conclusion; deliberately separated from the locked-op heterogeneity nulls).

Bolt the three Liou-Abbott (eLife 2020, Liou…Abbott "A model for focal seizure onset,
propagation, evolution, and progression") mechanisms onto OUR Siegert LIF rate field and
study, with mechanism-ablation controls and a parameter sweep, whether the rate model
produces adaptation-driven repetitive bursting (the ictal tonic→clonic relaxation
oscillation + inward traveling waves of paper Fig 2).

Three mechanisms (paper eqs, docs/paper/abbott_model.md):
  1. adaptive threshold   τ_φ φ̇ = (φ0−φ) + Δ_φ f          (eq 2) — firing raises threshold
  2. sAHP recovery        τ_a ȧ  = −a + r ; μ ← μ − b_a a   (eq 4 analogue; our recovery)
  3. Mexican-hat surround inhibition wider than excitation (l_inh > ell)  (eq 5)
(chloride/eq-3, the slow wavefront-expansion mechanism τ=5 s, is NOT a bursting driver — skipped.)

1-D neural field (Liou's primary reduction, their Fig 2). Modes (argv[1]):
  explore  — grid of candidate space-time rasters (pick a paper-like regime)
  main     — paper Fig-2-style full-model figure (space-time + zoom + per-contact readout)
  ablation — mechanism A/B controls (full / −threshold / −sAHP / −surround / none)
  sweep    — regime phase diagram (Δ_φ × excitability) + burst-period dependence

Run: PYTHONPATH="$PWD" python scripts/explore_liou_bolton_rate.py <mode>
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
N = 200
L = 12.0
DX = L / N
ELL = 0.54
DT = 0.5

# default bursting recipe (excitable regime); fast threshold = clonic rhythm,
# slow sAHP = tonic→clonic→termination envelope.
DEF = dict(w_ee_mult=2.0, dphi=1.0, b_a=3.0, l_inh=1.6, tphi=100.0, ta=2500.0)
NARROW_INH = 0.5

_LUT = {}          # cache: sigma -> (interpolator, sI, rI_lut, musI, nuext, muxE, muxI)


def _gauss_kernel(width):
    x = (np.arange(N) - N // 2) * DX
    g = np.exp(-0.5 * (x / width) ** 2)
    return g / g.sum()


def _conv(field, kern):
    return np.real(np.fft.ifft(np.fft.fft(field) * np.fft.fft(np.fft.ifftshift(kern))))


def _setup():
    if "rateE" not in _LUT:
        op = mean_field(1.0)
        sE, sI, nuext = op["sE"], op["sI"], op["nuext"]
        mus = np.linspace(-20.0, 220.0, 700)
        vths = np.linspace(V_TH, V_TH + 160.0, 320)
        tab = np.array([[lif_rate(float(m), sE, TAU_ME, TREF_E, v_th=float(v)) for v in vths]
                        for m in mus])
        tab = np.nan_to_num(tab, nan=0.0, posinf=1.0 / TREF_E, neginf=0.0)
        musI = np.linspace(-20.0, 220.0, 1500)
        _LUT.update(rateE=RegularGridInterpolator((mus, vths), tab, bounds_error=False,
                                                  fill_value=None),
                    musI=musI, rI_lut=np.array([lif_rate(m, sI, TAU_MI, TREF_I) for m in musI]),
                    nuext=nuext, muxE=TAU_ME * JX_E * nuext, muxI=TAU_MI * JX_I * nuext)
    return _LUT


def integrate(*, w_ee_mult, dphi, b_a, l_inh, tphi, ta, t_max=10000.0,
              kick_amp=10.0, kick_t=(200.0, 600.0), kick_frac=0.12):
    """1-D bolt-on rate field. dphi=0 → no adaptive threshold; b_a=0 → no sAHP; l_inh small
    → no surround. Returns (t_ms, frames[nsteps,N] Hz). sigma fixed at canonical sE."""
    s = _setup()
    wee = w_ee_mult * W_EE
    KE, KI = _gauss_kernel(ELL), _gauss_kernel(l_inh)
    rE = np.full(N, 1e-4); rI = np.full(N, 1e-4); vth = np.full(N, V_TH); a = np.full(N, 1e-4)
    sEE = _conv(rE, KE).copy(); sEI = _conv(rI, KI).copy()
    sIE = _conv(rE, KI).copy(); sII = _conv(rI, KI).copy()
    nsteps = int(t_max / DT)
    frames = np.empty((nsteps, N))
    kick_mask = np.arange(N) < int(kick_frac * N)
    for t in range(nsteps):
        stim = kick_amp * kick_mask if (kick_t[0] <= t * DT < kick_t[1]) else 0.0
        sEE += DT / TAU_AMPA * (_conv(rE, KE) - sEE)
        sEI += DT / TAU_GABA * (_conv(rI, KI) - sEI)
        sIE += DT / TAU_AMPA * (_conv(rE, KI) - sIE)
        sII += DT / TAU_GABA * (_conv(rI, KI) - sII)
        muE = TAU_ME * (C_EE * wee * sEE - C_EI * W_EI * sEI) + s["muxE"] + stim - b_a * a
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * W_II * sII) + s["muxI"]
        fE = s["rateE"](np.column_stack([np.clip(muE, -20, 220), np.clip(vth, V_TH, V_TH + 160)]))
        fI = np.interp(muI, s["musI"], s["rI_lut"])
        rE = rE + DT / TAU_ME * (-rE + fE)
        rI = rI + DT / TAU_MI * (-rI + fI)
        vth = vth + DT / tphi * ((V_TH - vth) + dphi * rE * 1000.0)
        a = a + DT / ta * (-a + rE)
        frames[t] = rE * 1000.0
    return np.arange(nsteps) * DT, frames


def classify(frames, kick_end=600.0):
    """Outcome at the seed contact: regime, burst count/period, spatial extent, termination."""
    c = int(0.12 * N) // 2
    s = int((kick_end + 200) / DT)
    x = frames[s:, c]
    peak = float(frames.max())
    if peak < 5.0:
        return dict(regime="silent", n_bursts=0, period_s=np.nan, peak_hz=peak,
                    extent_mm=0.0, terminated=False)
    thr = 0.35 * x.max() if x.max() > 0 else 1e9
    pk = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > thr]
    nb = len(pk)
    per = float(np.mean(np.diff(pk)) * DT / 1000.0) if nb >= 2 else np.nan
    final = frames[-int(500 / DT):].max()
    terminated = final < 0.1 * peak
    active = (frames > 0.1 * peak).any(axis=0)
    extent = float(active.sum() * DX)
    regime = ("bursting" if nb >= 3 and not (np.isnan(per))
              else "tonic" if x[-len(x) // 5:].mean() > 0.3 * x.max() else "single")
    return dict(regime=regime, n_bursts=nb, period_s=per, peak_hz=peak,
                extent_mm=extent, terminated=bool(terminated))


# ---------------------------------------------------------------------------
def mode_explore():
    cands = [
        ("w_ee2.0 dphi1.0 ba3 ta2500", dict(DEF)),
        ("slower sAHP ta=4500 (termination?)", {**DEF, "ta": 4500.0}),
        ("ta=1200 (sustained)", {**DEF, "ta": 1200.0}),
        ("dphi0.6 ba5", {**DEF, "dphi": 0.6, "b_a": 5.0}),
        ("w_ee1.8", {**DEF, "w_ee_mult": 1.8}),
        ("w_ee2.4 ta4000", {**DEF, "w_ee_mult": 2.4, "ta": 4000.0}),
    ]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    for k, (lbl, p) in enumerate(cands):
        t, fr = integrate(t_max=10000.0, **p)
        info = classify(fr)
        a = ax.flat[k]
        a.imshow(fr.T, aspect="auto", origin="lower", extent=(0, t[-1] / 1000, 0, L),
                 cmap="magma", vmin=0, vmax=np.percentile(fr, 99.5))
        a.set_title(f"{lbl}\n{info['regime']} nb={info['n_bursts']} "
                    f"T={info['period_s']:.2f}s term={info['terminated']}", fontsize=8)
        a.set_xlabel("time (s)"); a.set_ylabel("space (mm)")
    fig.tight_layout(); p = os.path.join(OUT, "_explore_liou_grid.png")
    fig.savefig(p, dpi=85); print(f"wrote {p}")
    for lbl, pp in cands:
        print(f"  {lbl}: {classify(integrate(t_max=10000.0, **pp)[1])}")


def _raster(ax, t, fr, vmax, title, ylab=True):
    im = ax.imshow(fr.T, aspect="auto", origin="lower", extent=(0, t[-1] / 1000, 0, L),
                   cmap="magma", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (s)")
    if ylab:
        ax.set_ylabel("cortical position (mm)")
    ax.margins(0)
    return im


def mode_main():
    """Paper Fig-2-style full-model figure: spatiotemporal evolution + traveling-wave zoom +
    single-location readout (the three mechanisms ON, excitable regime)."""
    t, fr = integrate(t_max=9000.0, **DEF)
    vmax = float(np.percentile(fr, 99.5))
    z0, z1 = 3000.0, 4800.0                                  # clonic-window zoom
    iz = slice(int(z0 / DT), int(z1 / DT))
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.85], width_ratios=[1.5, 1.0])
    axA = fig.add_subplot(gs[0, :])
    imA = _raster(axA, t, fr, vmax, "Spatiotemporal evolution — recurrent excitation + "
                  "adaptive threshold + sAHP + surround → sustained repetitive bursting")
    axA.axvspan(0.2, 0.6, color="cyan", alpha=0.25)
    axA.text(0.4, L * 0.96, "seed", color="cyan", fontsize=8, ha="center", va="top")
    axA.axvspan(z0 / 1000, z1 / 1000, ec="white", fc="none", lw=1.2)
    fig.colorbar(imA, ax=axA, label="firing rate (Hz)", fraction=0.04, pad=0.01)
    axB = fig.add_subplot(gs[1, 0])
    imB = axB.imshow(fr[iz].T, aspect="auto", origin="lower",
                     extent=(z0 / 1000, z1 / 1000, 0, L), cmap="magma", vmin=0, vmax=vmax)
    axB.set_title("Zoom: each burst is a travelling wave", fontsize=10)
    axB.set_xlabel("time (s)"); axB.set_ylabel("cortical position (mm)"); axB.margins(0)
    fig.colorbar(imB, ax=axB, label="firing rate (Hz)", fraction=0.046, pad=0.02)
    axC = fig.add_subplot(gs[1, 1])
    for c, lab in [(int(0.12 * N), "near seed"), (int(0.40 * N), "mid"), (int(0.70 * N), "far")]:
        axC.plot(t / 1000, fr[:, c], lw=0.8, label=lab)
    axC.set_title("Single-location firing rate over time", fontsize=10)
    axC.set_xlabel("time (s)"); axC.set_ylabel("firing rate (Hz)"); axC.margins(x=0)
    axC.legend(fontsize=8, title="cortical site", loc="upper right")
    fig.suptitle("Adaptation-driven seizure-like bursting reproduced on the LIF rate field "
                 "(1-D)  ·  excitable-regime capability test, not the data-locked operating point",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(OUT, "liou_bolton_main.png"); fig.savefig(p, dpi=110)
    print(f"wrote {p}  ({classify(fr)})")


def mode_ablation():
    """Mechanism A/B controls: remove one mechanism at a time, same excitable regime."""
    conds = [
        ("all three on", DEF),
        ("no adaptive threshold", {**DEF, "dphi": 0.0}),
        ("no sAHP recovery", {**DEF, "b_a": 0.0}),
        ("no surround (narrow inhibition)", {**DEF, "l_inh": NARROW_INH}),
        ("none (no adaptation, no surround)", {**DEF, "dphi": 0.0, "b_a": 0.0, "l_inh": NARROW_INH}),
    ]
    res = [(lab, integrate(t_max=8000.0, **p)) for lab, p in conds]
    vmax = max(float(np.percentile(fr, 99.5)) for _, (_t, fr) in res)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    im = None
    for k, (lab, (t, fr)) in enumerate(res):
        info = classify(fr)
        tag = (f"bursting · period {info['period_s']:.2f}s" if info["regime"] == "bursting"
               else info["regime"])
        im = _raster(ax.flat[k], t, fr, vmax, f"{lab}\n→ {tag}", ylab=(k % 3 == 0))
    ax.flat[5].axis("off")               # 6th cell holds the shared colourbar only
    fig.colorbar(im, ax=ax[1, 2], label="firing rate (Hz)", fraction=0.05)
    fig.suptitle("Which mechanism does what — remove one at a time (same excitable regime).  "
                 "Adaptive threshold / sAHP drive the bursting RHYTHM; surround shapes spatial spread;\n"
                 "with no adaptation the field locks into uncontrolled tonic saturation.  "
                 "Excitable-regime capability test, not the data-locked operating point.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "liou_bolton_ablation.png"); fig.savefig(p, dpi=105)
    print(f"wrote {p}")
    for lab, (_t, fr) in res:
        print(f"  {lab:38s} {classify(fr)}")


def mode_sweep():
    """Regime phase diagram (threshold-adaptation strength × excitability) + burst-rhythm
    dependence on the threshold time constant."""
    dphis = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    wees = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
    code = {"silent": 0, "single": 1, "tonic": 2, "bursting": 3}
    grid = np.zeros((len(wees), len(dphis)))
    for i, wee in enumerate(wees):
        for j, dp in enumerate(dphis):
            info = classify(integrate(t_max=6000.0, **{**DEF, "w_ee_mult": wee, "dphi": dp})[1])
            grid[i, j] = code[info["regime"]]
    tphis = [50.0, 100.0, 150.0, 200.0, 300.0]
    pers = [classify(integrate(t_max=8000.0, **{**DEF, "tphi": tp})[1])["period_s"] for tp in tphis]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#2b2b2b", "#4575b4", "#f4a300", "#d73027"])
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                      norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N),
                      extent=(dphis[0] - .15, dphis[-1] + .15, wees[0] - .1, wees[-1] + .1))
    ax[0].set_xlabel("threshold-adaptation strength  Δφ (mV per kHz)")
    ax[0].set_ylabel("recurrent excitability  (× baseline E→E)")
    ax[0].set_title("Where bursting lives in parameter space")
    cb = fig.colorbar(im, ax=ax[0], ticks=[0, 1, 2, 3], fraction=0.046)
    cb.ax.set_yticklabels(["silent", "single", "tonic", "bursting"])
    ax[1].plot(tphis, pers, "o-", color="#d73027")
    ax[1].set_xlabel("threshold recovery time constant  τφ (ms)")
    ax[1].set_ylabel("burst period (s)")
    ax[1].set_title("Threshold recovery sets the burst rhythm"); ax[1].margins(0.05)
    fig.suptitle("Parameter exploration of the adaptation-driven bursting  ·  "
                 "excitable-regime capability test, not the data-locked operating point",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, "liou_bolton_sweep.png"); fig.savefig(p, dpi=110)
    print(f"wrote {p}")
    print(f"  period vs tphi: {list(zip(tphis, [round(x,3) if x==x else None for x in pers]))}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    modes = {"explore": mode_explore, "main": mode_main, "ablation": mode_ablation,
             "sweep": mode_sweep}
    if mode == "all":
        _setup()
        for m in ("main", "ablation", "sweep"):
            print(f"[{m}] ...", flush=True); modes[m]()
    else:
        modes.get(mode, mode_explore)()
