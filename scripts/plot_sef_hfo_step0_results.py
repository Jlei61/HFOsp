"""Paper-grade SEF-HFO Step-0 result figures (LIF rate field), field-based redesign.

step0b + step0d are drawn from the REAL spatial field (re-run canonical
integrate_lif_field), so the figures SHOW the event and the directionality rather
than plotting summary scalars. step0a stays the corrected operating-point figure.

  step0b_lif_self_limited.png   -> field snapshots (ignite→advance→terminate) +
                                   front/active-fraction time course + amplitude response
  step0d_anisotropy.png         -> peak-field snapshots at θ_EE = 0/45/90° + isotropic
  step0a_lif_operating_point.png -> canonical mean_field (fsolve), NOT step0a_lif.json
                                   (that JSON is the damped-iteration artifact)

Run: python scripts/plot_sef_hfo_step0_results.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sef_hfo_lif import (  # noqa: E402
    mean_field, integrate_lif_field, lif_rate,
    _STIM_X0, _STIM_R, _STIM_T, ELL_PAR, ELL_PERP, TAU_ME, TREF_E, V_TH,
)
from src.sef_hfo_field import _grid  # noqa: E402

ROOT = "results/topic4_sef_hfo"
N, L = 96, 12.0
C_E, C_I = "#c0392b", "#2c6fbb"


def _disk(x0, y0):
    X, Y = _grid(N, L)
    return ((X - x0) ** 2 + (Y - y0) ** 2) <= _STIM_R ** 2


def _heat(ax, field, vmax, title):
    im = ax.imshow(field.T * 1000.0, origin="lower", extent=[-6, 5.875, -6, 5.875],
                   cmap="magma", vmin=0, vmax=vmax, aspect="equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9.5)
    return im


# ===========================================================================
# step0b — the actual self-limited event (field snapshots + time course)
# ===========================================================================
def fig_step0b():
    op = mean_field(1.0)
    disk = _disk(_STIM_X0, 0.0)
    A = 8.0
    stim = lambda t: (A * disk if t < _STIM_T else 0.0)

    ext, front, fld_end = integrate_lif_field(op, stim, t_max=90, return_field=True)
    tms = np.arange(len(ext)) * 0.25
    snaps = []
    for tm in (8.0, 24.0, 44.0):
        _e, _f, fld = integrate_lif_field(op, stim, t_max=tm, return_field=True)
        snaps.append((tm, fld))
    snaps.append((90.0, fld_end))
    vmax = max(s[1].max() for s in snaps) * 1000.0

    fig = plt.figure(figsize=(12.6, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 0.9], hspace=0.32, wspace=0.12,
                          left=0.06, right=0.93, top=0.9, bottom=0.09)

    # top: field snapshots
    labels = ["t = 8 ms  ·  ignite", "t = 24 ms  ·  advance", "t = 44 ms  ·  propagated",
              "t = 90 ms  ·  quiet (self-terminated)"]
    for i, ((tm, fld), lab) in enumerate(zip(snaps, labels)):
        ax = fig.add_subplot(gs[0, i])
        im = _heat(ax, fld, vmax, lab)
        if i == 0:
            ax.add_patch(plt.Circle((_STIM_X0, 0), _STIM_R, fill=False, ec="w", lw=1.2, ls=":"))
            ax.annotate("", (3.5, 0), (-1.5, 0), arrowprops=dict(arrowstyle="-|>", color="w", lw=1.6))
            ax.text(1.0, 0.9, "θ$_{EE}$ axis", color="w", fontsize=8)
    cax = fig.add_axes([0.94, 0.52, 0.012, 0.36])
    fig.colorbar(im, cax=cax).set_label("firing rate (Hz)", fontsize=8)

    # bottom-left: time course (self-limited in time)
    axt = fig.add_subplot(gs[1, :2])
    axt.plot(tms, front, color=C_E, lw=2.2, label="front position (mm)")
    axt.set_xlabel("time (ms)"); axt.set_ylabel("front position (mm)", color=C_E)
    axt.tick_params(axis="y", labelcolor=C_E); axt.set_xlim(0, 90)
    axt2 = axt.twinx()
    axt2.plot(tms, ext, color="#444", lw=1.8, ls="--", label="active fraction")
    axt2.set_ylabel("active fraction", color="#444"); axt2.tick_params(axis="y", labelcolor="#444")
    axt2.set_ylim(0, 0.3)
    axt.axvspan(0, _STIM_T, color="#f0c14b", alpha=0.15)
    axt.text(2, 5.4, "stim on", fontsize=7.5, color="#9a7d1e")
    axt.text(46, 1.2, "front advances, then\nactivity returns to rest\n= self-terminates", fontsize=8.3)
    axt.set_title("the event in time: propagate, then self-terminate", fontsize=10)

    # bottom-right: amplitude response (threshold → all-or-none → bounded)
    d = json.load(open(f"{ROOT}/finite_pulse/step0b_lif.json"))
    g = next(p for p in d["runs"]["recovery_off"]["points"] if p["ratio"] == 1.0)["grid"]
    As = [x["A"] for x in g]; adv = [x["adv_mm"] for x in g]; mx = [x["max_ext"] for x in g]
    axr = fig.add_subplot(gs[1, 2:])
    axr.plot(As, adv, "-o", color=C_E, lw=2, ms=5, label="front advance (mm)")
    axr.set_xlabel("pulse amplitude  A"); axr.set_ylabel("front advance (mm)", color=C_E)
    axr.tick_params(axis="y", labelcolor=C_E); axr.set_ylim(-0.4, 8.5)
    axr2 = axr.twinx()
    axr2.plot(As, mx, "-s", color="#444", lw=1.6, ms=4, label="peak active fraction")
    axr2.axhline(0.5, color="#999", ls="--", lw=1.2)
    axr2.text(13, 0.52, "runaway threshold", fontsize=7.5, color="#777")
    axr2.set_ylabel("peak active fraction", color="#444"); axr2.tick_params(axis="y", labelcolor="#444")
    axr2.set_ylim(0, 0.62)
    axr.axvspan(0.5, 3.5, color="#d9d9d9", alpha=0.5)
    axr.text(2, 7.6, "below\nthreshold:\nextinction", fontsize=7.6, ha="center", color="#555")
    axr.text(15, 4.0, "above threshold: full event,\nadvance independent of A\n(all-or-none), never runaway",
             fontsize=7.8, color="#333")
    axr.set_title("amplitude response: thresholded, all-or-none, bounded", fontsize=10)

    fig.suptitle("Step 0b — a finite pulse ignites, propagates directionally, then self-terminates "
                 "(LIF rate field; the sigmoid field could not)", fontsize=12, fontweight="bold")
    out = f"{ROOT}/finite_pulse/figures/step0b_lif_self_limited.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ===========================================================================
# step0d — anisotropy: the field rotates with connectivity
# ===========================================================================
def fig_step0d():
    op = mean_field(1.0)
    disk = _disk(0.0, 0.0)  # centered, so elongation is symmetric about the axis
    A = 8.0
    stim = lambda t: (A * disk if t < _STIM_T else 0.0)
    ell_iso = float(np.sqrt(ELL_PAR * ELL_PERP))
    cases = [("θ$_{EE}$ = 0°", 0.0, False), ("θ$_{EE}$ = 45°", np.pi / 4, False),
             ("θ$_{EE}$ = 90°", np.pi / 2, False), ("isotropic (control)", 0.0, True)]
    fields = []
    for lab, th, iso in cases:
        kw = dict(theta_EE=th, t_max=55, return_peak_field=True)
        if iso:
            kw.update(ell_par=ell_iso, ell_perp=ell_iso)
        _e, _f, pf = integrate_lif_field(op, stim, **kw)
        fields.append((lab, th, iso, pf))
    vmax = max(f[3].max() for f in fields) * 1000.0

    fig, ax = plt.subplots(1, 4, figsize=(13.0, 3.8))
    for a, (lab, th, iso, pf) in zip(ax, fields):
        im = _heat(a, pf, vmax, lab)
        if not iso:
            dx, dy = 4.0 * np.cos(th), 4.0 * np.sin(th)
            a.annotate("", (dx, dy), (-dx, -dy),
                       arrowprops=dict(arrowstyle="-", color="w", lw=1.4, ls=(0, (4, 3))))
            a.text(0.04, 0.06, "ratio ≈ 4.2", transform=a.transAxes, color="w", fontsize=8.5)
        else:
            a.text(0.04, 0.06, "ratio = 1.0\n(no axis)", transform=a.transAxes, color="w", fontsize=8.5)
    cax = fig.add_axes([0.94, 0.2, 0.011, 0.55])
    fig.colorbar(im, cax=cax).set_label("firing rate (Hz)", fontsize=8)
    fig.suptitle("Step 0d — the active region elongates along the E→E connectivity axis and rotates with it "
                 "(dashed = imposed θ$_{EE}$); isotropic control has no axis", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.93, 0.92])
    out = f"{ROOT}/finite_pulse/figures/step0d_anisotropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


# ===========================================================================
# step0a — corrected operating point (unchanged)
# ===========================================================================
def fig_step0a():
    ratios = np.array([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25])
    ops = [mean_field(float(r)) for r in ratios]
    nuE = np.array([o["nuE"] for o in ops]) * 1000.0
    nuI = np.array([o["nuI"] for o in ops]) * 1000.0
    op0 = ops[3]

    fig, ax = plt.subplots(1, 2, figsize=(10.2, 4.3))
    ax[0].semilogy(ratios, nuE, "-o", color=C_E, ms=5, label="E rate")
    ax[0].semilogy(ratios, nuI, "-s", color=C_I, ms=5, label="I rate")
    ax[0].set_xlabel("external drive ratio"); ax[0].set_ylabel("operating-point firing rate  (Hz)")
    ax[0].set_title("self-consistent rest: E rate robustly low", fontsize=10)
    ax[0].legend(fontsize=9, loc="center left")
    ax[0].text(0.86, 0.30, "E ≈ 0.2 Hz, flat across drive\n→ robustly stable, sub-critical",
               fontsize=8.5, color=C_E)
    ax[0].grid(True, which="both", alpha=0.2)

    mu_lo = min(6.5, op0["muE"] - 1.0)
    mu = np.linspace(mu_lo, 22, 140)
    phi = np.array([lif_rate(m, op0["sE"], TAU_ME, TREF_E) for m in mu]) * 1000.0
    ax[1].plot(mu, phi, color=C_E, lw=2.6, label="Φ$_{LIF}$ (σ at rest)")
    ax[1].axvline(V_TH, color="#bbb", ls=":", lw=1.0)
    ax[1].text(V_TH + 0.2, phi.max() * 0.52, "spike\nthreshold", fontsize=7.5, color="#888")
    mu_ig = 17.0
    ax[1].annotate("", (mu_ig, lif_rate(mu_ig, op0["sE"], TAU_ME, TREF_E) * 1000.0),
                   (op0["muE"], op0["nuE"] * 1000.0),
                   arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.8,
                                   connectionstyle="arc3,rad=-0.3"))
    ax[1].text(12.6, phi.max() * 0.13, "finite pulse\npushes μ$_E$ up\n→ ignites",
               color="#444", fontsize=8.3, ha="center")
    ax[1].plot([op0["muE"]], [op0["nuE"] * 1000.0], "o", color="#222", ms=9, zorder=5)
    ax[1].annotate("stable rest:\nsub-threshold, low rate",
                   (op0["muE"], op0["nuE"] * 1000.0), (op0["muE"] + 0.4, phi.max() * 0.40),
                   fontsize=8.3, color="#222", arrowprops=dict(arrowstyle="->", lw=1.0))
    ax[1].set_xlabel("mean input drive  μ$_E$  (mV)"); ax[1].set_ylabel("firing rate  (Hz)")
    ax[1].set_xlim(mu_lo, 22); ax[1].set_ylim(0, phi.max() * 1.05); ax[1].margins(0)
    ax[1].set_title("stable but excitable, not near-critical", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left", frameon=False)

    fig.suptitle("Step 0a — self-consistent LIF operating point: robustly stable, sub-critical, "
                 "excitable  (fsolve; near-critical Hopf in committed JSON was a damped-iteration artifact)",
                 fontsize=10.6, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{ROOT}/linear_stability/figures/step0a_lif_operating_point.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_step0b()
    fig_step0d()
    fig_step0a()
