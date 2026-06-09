"""
Liou-Abbott (eLife 2020) seizure dynamics on the Siegert LIF rate field — v2.

Minimal correct bone structure:
  - Local inhibition:  narrow Gaussian kernel (sigma_L), Mexican-hat surround
  - Global inhibition: uniform γ·mean(rE), drives termination as recruited area grows
  - Spatial z(x,t):   inhibition efficacy field (eq 8), exhausts locally under heavy I→E use
  - Adaptive threshold φ(x,t): fast negative feedback, sets clonic rhythm (eq 2)
  - sAHP a(x,t):       slow adaptation, tonic→clonic envelope (eq 4)

Three deliverable figures:
  lifecycle  — four spatial fields (rE, φ, z, a) + per-contact traces; mechanism causality
  ablation   — remove one mechanism at a time; wavefront + termination metrics annotated
  threshold  — kick-amplitude spectrum: silent → self-limited → sustained seizure

Excitable-regime capability test (w_ee_mult≈2.0). NOT the data-locked SEF-HFO op.

Run:
  PYTHONPATH="$PWD" python scripts/explore_liou_bolton_rate.py [lifecycle|ablation|threshold|all]
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

OUT  = "results/topic4_sef_hfo/observation_layer/figures"
N    = 200
L    = 12.0      # mm
DX   = L / N     # 0.06 mm / node
ELL  = 0.54      # E spatial scale (mm)
DT   = 0.5       # ms
KICK_FRAC = 0.12  # fraction of domain = SOZ seed

# Default recipe: excitable regime, all mechanisms on
DEF = dict(
    w_ee_mult = 2.0,      # recurrent E→E multiplier
    dphi      = 1.0,      # adaptive threshold strength (mV·ms / kHz)
    b_a       = 3.0,      # sAHP strength
    sigma_L   = 1.5,      # local I kernel width (mm); > ELL → Mexican hat
    gamma     = 0.0,      # global inhibition coeff (set >0 to enable termination drive)
    tau_G     = 4000.0,   # τ of slow global inhibition pool (ms)
    tphi      = 100.0,    # threshold recovery τ (ms)
    ta        = 2500.0,   # sAHP recovery τ (ms)
    tau_z     = 100000.0, # z τ (ms); slow → visible spatial gradient without killing bursting
)

_LUT: dict = {}


def _gauss_kernel(width: float) -> np.ndarray:
    x = (np.arange(N) - N // 2) * DX
    g = np.exp(-0.5 * (x / width) ** 2)
    return g / g.sum()


def _conv(field: np.ndarray, kern: np.ndarray) -> np.ndarray:
    return np.real(np.fft.ifft(np.fft.fft(field) * np.fft.fft(np.fft.ifftshift(kern))))


def _setup() -> dict:
    if "rateE" not in _LUT:
        op = mean_field(1.0)
        sE, sI, nuext = op["sE"], op["sI"], op["nuext"]
        mus  = np.linspace(-20.0, 220.0, 700)
        vths = np.linspace(V_TH, V_TH + 160.0, 320)
        tab  = np.array([[lif_rate(float(m), sE, TAU_ME, TREF_E, v_th=float(v))
                          for v in vths] for m in mus])
        tab  = np.nan_to_num(tab, nan=0.0, posinf=1.0 / TREF_E, neginf=0.0)
        musI = np.linspace(-20.0, 220.0, 1500)
        _LUT.update(
            rateE=RegularGridInterpolator(
                (mus, vths), tab, bounds_error=False, fill_value=None),
            musI=musI,
            rI_lut=np.array([lif_rate(m, sI, TAU_MI, TREF_I) for m in musI]),
            nuext=nuext,
            muxE=TAU_ME * JX_E * nuext,
            muxI=TAU_MI * JX_I * nuext,
        )
    return _LUT


def integrate(
    *, w_ee_mult, dphi, b_a, sigma_L, gamma, tau_G, tphi, ta,
    t_max=10000.0, kick_amp=10.0, kick_t=(200.0, 600.0),
    cl=True, tau_z=100000.0, cl_th=0.05, cl_w=0.012,
    soz_delta_mu=0.0, return_full=False,
):
    """
    Spatial LIF rate field with local / global inhibition decomposed.

    muE(x,t) = τ_E [C_EE·wee·sEE(x) − z(x)·C_EI·W_EI·sEI(x)]
               − γ·r̄_E(t)  +  μ_ext,E + μ_offset(x) + stim(x,t) − b_a·a(x,t)

    Local inhibition : sEI = conv(rI, K_L),  K_L ~ Gauss(sigma_L > ELL).
    Global inhibition: γ·G(t), where G is a slow scalar (τ_G≈4s) accumulating mean(rE).
                       Slow accumulation lets burst rhythm develop; termination happens
                       when G saturates and γ·G overcomes local excitatory drive.
    Efficacy z(x,t)  : per-node; τ_z decay toward z_inf(sEI) when cl=True.

    Returns
    -------
    (t_ms, rE_Hz)                          when return_full=False
    (t_ms, rE_Hz, vth, z, a_scaled)       when return_full=True  [shape (nsteps, N)]
    """
    s    = _setup()
    wee  = w_ee_mult * W_EE
    KE   = _gauss_kernel(ELL)
    KI   = _gauss_kernel(sigma_L)

    n_kick = int(KICK_FRAC * N)
    mu_off = np.zeros(N)
    mu_off[:n_kick] = soz_delta_mu

    rE  = np.full(N, 1e-4)
    rI  = np.full(N, 1e-4)
    vth = np.full(N, V_TH)
    a   = np.zeros(N)
    z   = np.ones(N)
    G   = 0.0   # slow global inhibition pool (scalar); τ_G accumulation of mean(rE)

    sEE = _conv(rE, KE); sEI = _conv(rI, KI)
    sIE = _conv(rE, KI); sII = _conv(rI, KI)

    nsteps = int(t_max / DT)
    rE_f  = np.empty((nsteps, N), dtype=np.float32)
    vth_f = np.empty((nsteps, N), dtype=np.float32) if return_full else None
    z_f   = np.empty((nsteps, N), dtype=np.float32) if return_full else None
    a_f   = np.empty((nsteps, N), dtype=np.float32) if return_full else None

    kick_mask = np.arange(N) < n_kick

    for ti in range(nsteps):
        stim = kick_amp * kick_mask if kick_t[0] <= ti * DT < kick_t[1] else 0.0

        sEE += DT / TAU_AMPA * (_conv(rE, KE) - sEE)
        sEI += DT / TAU_GABA * (_conv(rI, KI) - sEI)
        sIE += DT / TAU_AMPA * (_conv(rE, KI) - sIE)
        sII += DT / TAU_GABA * (_conv(rI, KI) - sII)

        if cl:
            z_inf = 1.0 / (1.0 + np.exp((sEI - cl_th) / cl_w))
            z    += DT / tau_z * (z_inf - z)

        # global inhibition: slow pool G accumulates with mean activity; drives termination
        G += DT / tau_G * (-G + rE.mean())

        muE = (TAU_ME * (C_EE * wee * sEE - z * C_EI * W_EI * sEI)
               - gamma * G + s["muxE"] + mu_off + stim - b_a * a)
        muI = TAU_MI * (C_IE * W_IE * sIE - C_II * W_II * sII) + s["muxI"]

        fE = s["rateE"](np.column_stack([
            np.clip(muE, -20.0, 220.0),
            np.clip(vth, V_TH, V_TH + 160.0),
        ]))
        fI = np.interp(muI, s["musI"], s["rI_lut"])

        rE  += DT / TAU_ME * (-rE + fE)
        rI  += DT / TAU_MI * (-rI + fI)
        vth += DT / tphi * ((V_TH - vth) + dphi * rE * 1000.0)
        a   += DT / ta   * (-a + rE)

        rE_f[ti] = (rE * 1000.0).astype(np.float32)
        if return_full:
            vth_f[ti] = vth.astype(np.float32)
            z_f[ti]   = z.astype(np.float32)
            a_f[ti]   = (a * 1000.0).astype(np.float32)

    t_ms = np.arange(nsteps, dtype=np.float32) * DT
    if return_full:
        return t_ms, rE_f, vth_f, z_f, a_f
    return t_ms, rE_f


def classify(frames: np.ndarray, kick_end: float = 600.0) -> dict:
    """
    Classify spatiotemporal outcome from rE_frames (Hz, shape nsteps×N).

    Returns: regime, n_bursts, period_s, peak_hz, wavefront_speed_mms,
             terminated, recruitment_mm_max.
    """
    n_kick = int(KICK_FRAC * N)
    c_seed = n_kick // 2
    s0     = int((kick_end + 200) / DT)
    peak   = float(frames.max())

    if peak < 5.0:
        return dict(regime="silent", n_bursts=0, period_s=float("nan"),
                    peak_hz=peak, wavefront_speed_mms=0.0,
                    terminated=False, recruitment_mm_max=0.0)

    thr  = 0.2 * peak
    post = frames[s0:]

    # Rightmost active node at each post-kick time step → wavefront position
    edge = np.array([
        r.nonzero()[0][-1] * DX if (r > thr).any() else 0.0
        for r in (post > thr)
    ])
    t_edge = np.arange(len(edge)) * DT / 1000.0
    grow   = np.where(np.diff(edge) > 0.0)[0]
    if len(grow) >= 10:
        half = grow[: max(1, len(grow) // 2)]
        wf_speed = float(np.polyfit(t_edge[half], edge[half], 1)[0])
    else:
        wf_speed = 0.0

    x      = post[:, c_seed]
    thr_pk = 0.35 * x.max() if x.max() > 0 else 1e9
    pk = [i for i in range(1, len(x) - 1)
          if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > thr_pk]
    nb  = len(pk)
    per = float(np.mean(np.diff(pk)) * DT / 1000.0) if nb >= 2 else float("nan")

    terminated      = float(frames[-int(500 / DT):].max()) < 0.1 * peak
    recruitment_max = float((frames > thr).any(axis=0).sum() * DX)

    # regime: ≥3 periodic peaks → bursting; terminated → transient; else → tonic
    if nb >= 3 and per == per:
        regime = "bursting"
    elif terminated:
        regime = "transient"    # brief after-discharge, self-limited (interictal-like)
    else:
        regime = "tonic"        # sustained but non-oscillatory
    return dict(regime=regime, n_bursts=nb, period_s=per, peak_hz=peak,
                wavefront_speed_mms=wf_speed, terminated=bool(terminated),
                recruitment_mm_max=recruitment_max)


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 1: lifecycle — four spatial fields + contact traces
# ─────────────────────────────────────────────────────────────────────────────

def mode_lifecycle():
    """Seizure lifecycle: four spatial fields + per-contact traces.

    Layout (rows, share time axis):
      0 – firing rate rE(x,t)
      1 – spike threshold φ(x,t)
      2 – inhibition efficacy z(x,t)  [1=intact, 0=exhausted]
      3 – sAHP adaptation current a(x,t)
      4 – single-contact traces at seed / border / outside
    """
    print("[lifecycle] integrating …", flush=True)
    t, rE, vth, z, a = integrate(t_max=14000.0, return_full=True, **DEF)
    t_s  = t / 1000.0
    info = classify(rE)
    print(f"  outcome: {info}")

    vmax_r = float(np.percentile(rE, 99.5))

    fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True,
                              gridspec_kw={"hspace": 0.42})

    def hmap(ax, data, cmap, vmin=None, vmax=None, cbar_label=""):
        im = ax.imshow(data.T, aspect="auto", origin="lower",
                       extent=(t_s[0], t_s[-1], 0, L),
                       cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.012, pad=0.01, label=cbar_label)
        ax.set_ylabel("position (mm)")
        ax.margins(0)

    hmap(axes[0], rE,  "magma",   0,  vmax_r, "Hz")
    axes[0].set_title("Firing rate", fontsize=11, fontweight="bold")

    hmap(axes[1], vth, "plasma",  cbar_label="mV")
    axes[1].set_title(
        "Spike threshold — rises inside active zone, drives tonic→clonic rhythm", fontsize=10)

    hmap(axes[2], z,   "RdYlGn", 0,  1,      "efficacy")
    axes[2].set_title(
        "Inhibition efficacy — exhausts at wavefront, recovers when quiet", fontsize=10)

    hmap(axes[3], a,   "Blues",   cbar_label="a.u.")
    axes[3].set_title(
        "sAHP current — accumulated activity, contributes to termination", fontsize=10)

    idxs = [int(0.05 * N), int(0.40 * N), int(0.78 * N)]
    cols = ["#d73027", "#f4a300", "#4575b4"]
    labs = [f"inside SOZ  (x={idxs[0]*DX:.1f} mm)",
            f"border      (x={idxs[1]*DX:.1f} mm)",
            f"outside     (x={idxs[2]*DX:.1f} mm)"]
    for idx, lab, col in zip(idxs, labs, cols):
        axes[4].plot(t_s, rE[:, idx], lw=0.7, label=lab, color=col)
    axes[4].set_ylabel("firing rate (Hz)")
    axes[4].set_xlabel("time (s)")
    axes[4].set_title(
        "Single-location traces — phase delay reveals propagating wavefront", fontsize=10)
    axes[4].legend(fontsize=9, loc="upper right")
    axes[4].margins(x=0)

    fig.tight_layout()
    p = os.path.join(OUT, "liou_bolton_lifecycle.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 2: mechanism ablation
# ─────────────────────────────────────────────────────────────────────────────

def mode_ablation():
    """Mechanism ablation — remove one mechanism per panel, annotate outcome metrics.

    Conditions:
      Full model              — adaptive threshold + sAHP + Mexican-hat surround + z
      No adaptive threshold   — dphi=0; what drives the clonic rhythm?
      No sAHP                 — b_a=0; what shapes the tonic→clonic envelope?
      No surround             — sigma_L=ELL; what does wider inhibition do spatially?
    """
    conds = [
        ("Full model\n(adaptive thresh + sAHP + surround)",
         {**DEF}),
        ("No adaptive threshold\n(dphi = 0)",
         {**DEF, "dphi": 0.0}),
        ("No sAHP recovery\n(b_a = 0)",
         {**DEF, "b_a": 0.0}),
        ("No Mexican-hat surround\n(σ_I = σ_E, narrow inhibition)",
         {**DEF, "sigma_L": ELL}),
    ]
    T_MAX = 10000.0
    results = []
    for lab, p in conds:
        tag = lab.split("\n")[0]
        print(f"  [{tag}] …", flush=True)
        t, fr = integrate(t_max=T_MAX, return_full=False, **p)
        results.append((lab, t, fr, classify(fr)))

    vmax = max(float(np.percentile(fr, 99.5)) for _, _, fr, _ in results)
    t_s  = results[0][1] / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (lab, _t, fr, info) in zip(axes.flat, results):
        im = ax.imshow(fr.T, aspect="auto", origin="lower",
                       extent=(t_s[0], t_s[-1], 0, L),
                       cmap="magma", vmin=0, vmax=vmax)
        r   = info["regime"]
        ann = [r]
        if r == "bursting" and info["period_s"] == info["period_s"]:
            ann.append(f"period={info['period_s']:.2f}s")
        if info["wavefront_speed_mms"] > 0.5:
            ann.append(f"wf={info['wavefront_speed_mms']:.1f}mm/s")
        ann.append("terminates" if info["terminated"] else "sustained")
        ax.set_title(f"{lab}\n→ {', '.join(ann)}", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (mm)")
        ax.margins(0)
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="firing rate (Hz)")

    fig.suptitle(
        "Mechanism ablation: which variable controls onset / wavefront / clonic rhythm / termination?\n"
        "Excitable-regime capability test — not the data-locked operating point",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = os.path.join(OUT, "liou_bolton_ablation.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")
    for lab, _, _, info in results:
        print(f"  {lab.split(chr(10))[0]:42s}  {info}")


# ─────────────────────────────────────────────────────────────────────────────
#  Figure 3: stimulus-amplitude threshold scan
# ─────────────────────────────────────────────────────────────────────────────

def mode_threshold():
    """Stimulus-amplitude spectrum: silent → self-limited → sustained seizure.

    Varies kick amplitude to reveal the sub-/supra-threshold boundary.
    Uses a near-threshold excitability (w_ee_mult=1.7) so that small kicks produce
    brief self-limited after-discharges while larger kicks trigger sustained bursting.
    Two example rasters (one on each side) shown alongside the spectrum.
    """
    # Near-threshold excitability: spontaneous quiescent, kick-triggered seizure
    THRESH_PARAMS = {**DEF, "w_ee_mult": 1.7}
    kick_amps = [0, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    T_MAX     = 8000.0
    print("[threshold] scanning at w_ee_mult=1.7 …", flush=True)
    results = []
    for amp in kick_amps:
        _, fr = integrate(t_max=T_MAX, kick_amp=float(amp), **THRESH_PARAMS)
        info  = classify(fr)
        results.append((amp, info, fr))
        print(f"  amp={amp:4g}  {info['regime']:12s}  nb={info['n_bursts']}  "
              f"recruited={info['recruitment_mm_max']:.1f}mm  term={info['terminated']}")

    # colour by regime code
    code = {"silent": 0, "transient": 1, "tonic": 2, "bursting": 3}
    cpal = {0: "#333333", 1: "#4575b4", 2: "#f46d43", 3: "#d73027"}
    llbl = {0: "silent",
            1: "self-limited (interictal-like)",
            2: "sustained tonic",
            3: "sustained bursting (seizure-like)"}

    sub_i   = next((i for i, (_, inf, _) in enumerate(results)
                    if inf["regime"] == "transient"), 0)
    supra_i = next((i for i, (_, inf, _) in enumerate(results)
                    if inf["regime"] == "bursting"), -1)

    fig = plt.figure(figsize=(13, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], hspace=0.55, wspace=0.38)

    # Spectrum: recruited area vs amplitude
    ax_s = fig.add_subplot(gs[0, 0])
    amps_all = [a for a, _, _ in results]
    recr_all = [inf["recruitment_mm_max"] for _, inf, _ in results]
    ax_s.plot(amps_all, recr_all, "-", color="#aaaaaa", lw=0.8, zorder=1)
    for amp, info, _ in results:
        c = code.get(info["regime"], 0)
        ax_s.scatter(amp, info["recruitment_mm_max"], color=cpal[c], s=70, zorder=4)
    from matplotlib.patches import Patch
    ax_s.legend(handles=[Patch(facecolor=cpal[v], label=llbl[v])
                          for v in sorted(cpal)],
                fontsize=8, loc="upper left")
    ax_s.set_xlabel("stimulus strength (a.u.)")
    ax_s.set_ylabel("max recruited area (mm)")
    ax_s.set_title("Recruited area vs stimulus")
    ax_s.margins(0.06)

    # Event count vs amplitude — distinguishes interictal (few) from ictal (many)
    ax_p = fig.add_subplot(gs[1, 0])
    for amp, info, _ in results:
        c = code.get(info["regime"], 0)
        ax_p.scatter(amp, info["n_bursts"], color=cpal[c], s=60, zorder=4)
    ax_p.plot(amps_all, [inf["n_bursts"] for _, inf, _ in results],
              "-", color="#aaaaaa", lw=0.8, zorder=1)
    ax_p.set_xlabel("stimulus strength (a.u.)")
    ax_p.set_ylabel("number of discharge events")
    ax_p.set_title("Event count — interictal (1–2) vs ictal (many)")
    ax_p.margins(0.06)

    # Two example rasters
    for row, idx, ttl_prefix in [
        (0, sub_i,   "Self-limited after-discharge"),
        (1, supra_i, "Sustained seizure-like activity"),
    ]:
        amp_ex, info_ex, fr_ex = results[idx]
        t_ex   = np.arange(fr_ex.shape[0]) * DT / 1000.0
        vmax_ex = float(np.percentile(fr_ex, 99.5))
        ax = fig.add_subplot(gs[row, 1])
        im = ax.imshow(fr_ex.T, aspect="auto", origin="lower",
                       extent=(t_ex[0], t_ex[-1], 0, L),
                       cmap="magma", vmin=0, vmax=vmax_ex)
        ax.set_title(f"{ttl_prefix}  (stimulus = {amp_ex})", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("position (mm)")
        ax.margins(0)
        fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01, label="firing rate (Hz)")

    fig.suptitle(
        "Stimulus threshold scan — interictal-like after-discharge vs sustained seizure\n"
        "Excitable-regime capability test — not the data-locked operating point",
        fontsize=10,
    )
    p = os.path.join(OUT, "liou_bolton_threshold.png")
    fig.savefig(p, dpi=110)
    print(f"wrote {p}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode  = sys.argv[1] if len(sys.argv) > 1 else "all"
    MODES = {
        "lifecycle": mode_lifecycle,
        "ablation":  mode_ablation,
        "threshold": mode_threshold,
    }
    if mode == "all":
        _setup()
        for m in ("lifecycle", "ablation", "threshold"):
            print(f"\n[{m}] starting …", flush=True)
            MODES[m]()
    elif mode in MODES:
        MODES[mode]()
    else:
        print(f"unknown mode '{mode}'. choose: lifecycle ablation threshold all")
        sys.exit(1)
