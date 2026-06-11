"""SEF-HFO schematic, v3 — (a) mechanism setup only; transfer moved to a supplementary.

Per user feedback: drop the old (c)/(d) panels (the real result figures in
finite_pulse/ + linear_stability/ now carry those); the transfer panel becomes a
separate supplementary figure; the setup panel is redrawn properly with neuron
shapes (▲ excitatory, ● inhibitory) and size encoding threshold heterogeneity
(uniform size = low-heterogeneity patch; varied size = heterogeneous surround).

Outputs:
  sef_hfo_mechanism_schematic.png        — the setup / mechanism cartoon
  sef_hfo_transfer_supplementary.png     — Φ_LIF vs sigmoid (why LIF), supplementary

References: Brunel et al. 2026 (LIF E-I traveling waves, anisotropic E→E);
Liou/Abbott et al. 2020 (focal propagation, Mexican-hat + adaptation).

Run: python scripts/plot_sef_hfo_schematic.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Circle, RegularPolygon, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E  # noqa: E402

OUTDIR = "results/topic4_sef_hfo/schematic/figures"
C_E, C_I = "#c0392b", "#2c6fbb"
C_SHAFT = "#555555"
C_SIG = "#9aa3ab"
TH = np.deg2rad(25.0)
AX = np.array([np.cos(TH), np.sin(TH)])
PERP = np.array([-np.sin(TH), np.cos(TH)])
CEN = np.array([5.0, 4.1])


def _tri(ax, xy, r, color):
    ax.add_patch(RegularPolygon(xy, 3, radius=r, orientation=0.0, fc=color, ec="white", lw=0.6, zorder=5))


def _cir(ax, xy, r, color):
    ax.add_patch(Circle(xy, r, fc=color, ec="white", lw=0.6, zorder=5))


def fig_mechanism():
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8.2); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # tissue + low-heterogeneity patch (elongated along the connectivity axis)
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 7.6, boxstyle="round,pad=0.02,rounding_size=0.3",
                                fc="#f4f7fa", ec="#cdd6df", lw=1.0))
    ax.add_patch(Ellipse(CEN, 6.6, 3.4, angle=25, fc="#fbe3dd", ec="#e3a99c", lw=1.2))
    ax.text(6.7, 7.8, "heterogeneous surround (varied thresholds)", ha="center", fontsize=8.5, color="#8a7d70")
    ax.text(5.0, 0.95, "low-heterogeneity patch\n(uniform thresholds, anisotropic E→E)",
            ha="center", fontsize=8.6, color="#b5503f")

    # patch neurons — UNIFORM size (low heterogeneity); ▲ = E (larger), ● = I (smaller)
    rE, rI = 0.20, 0.13
    e_slots = [(-1.7, 0.45), (-0.9, -0.35), (-0.1, 0.5), (-0.2, -0.45),
               (0.8, 0.3), (1.0, -0.5), (1.7, 0.15), (0.0, 0.0)]
    i_slots = [(-1.2, 0.0), (-0.4, 0.05), (0.5, -0.1), (1.3, -0.25), (0.4, 0.55)]
    e_pos = [CEN + s * AX + p * PERP for s, p in e_slots]
    for xy in e_pos:
        _tri(ax, xy, rE, C_E)
    for s, p in i_slots:
        _cir(ax, CEN + s * AX + p * PERP, rI, C_I)

    # anisotropic E→E connectivity motif (from one hub E neuron): reach is LONG along
    # θ_EE, SHORT across it — the dashed ellipse is the connection kernel shape.
    hub = CEN
    ax.add_patch(Ellipse(hub, 4.4, 2.0, angle=25, fill=False, ec=C_E, lw=1.5,
                         ls=(0, (5, 3)), alpha=0.85, zorder=3))
    for s, p in [(2.1, 0.0), (-2.1, 0.0), (0.0, 0.92), (0.0, -0.92)]:
        ax.add_patch(FancyArrowPatch(hub, hub + s * AX + p * PERP, arrowstyle="-|>",
                                     mutation_scale=10, color=C_E, lw=1.5, alpha=0.85,
                                     shrinkA=2, shrinkB=2, zorder=4))
    ax.text(*(hub + 2.7 * AX + 0.15 * PERP), "θ$_{EE}$", color=C_E, fontsize=13,
            fontweight="bold", ha="left", va="center")

    # surround neurons — VARIED size (heterogeneous thresholds); corners, leaving lower-left for legend
    for x, y, r in [(1.5, 6.6, 0.30), (8.7, 2.0, 0.36), (8.9, 6.3, 0.25), (1.15, 4.3, 0.22)]:
        _tri(ax, (x, y), r, "#d98b7d")
    for x, y, r in [(2.4, 6.95, 0.20), (8.95, 3.3, 0.11), (8.95, 5.0, 0.15)]:
        _cir(ax, (x, y), r, "#9bb6d6")

    # electrode shaft — different angle from θ_EE; contacts are SQUARES (not neurons)
    sh = np.deg2rad(72)
    base = np.array([4.2, 4.1])
    p0, p1 = base - 3.0 * np.array([np.cos(sh), np.sin(sh)]), base + 3.0 * np.array([np.cos(sh), np.sin(sh)])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=C_SHAFT, lw=1.7, zorder=6)
    cc = np.array([base + t * np.array([np.cos(sh), np.sin(sh)]) for t in np.linspace(-2.6, 2.6, 6)])
    ax.plot(cc[:, 0], cc[:, 1], "s", color=C_SHAFT, mec="white", mew=0.9, ms=9, zorder=7)
    ax.text(2.8, 7.5, "electrode shaft\n(samples the template)", ha="center", fontsize=8.5, color=C_SHAFT)

    # legend (proxy), lower-left corner
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="^", color="w", mfc=C_E, mec="w", ms=11, label="excitatory (E)"),
           Line2D([0], [0], marker="o", color="w", mfc=C_I, mec="w", ms=9, label="inhibitory (I)")]
    ax.legend(handles=leg, loc="lower left", fontsize=8.5, frameon=True, framealpha=0.92,
              handletextpad=0.3, borderpad=0.5)

    ax.set_title("SEF-HFO setup — a low-heterogeneity, anisotropically-connected, "
                 "stable-but-excitable E–I patch",
                 fontsize=12, fontweight="bold", pad=10)
    fig.text(0.5, 0.015, "Setup schematic. Neuron size encodes threshold heterogeneity "
             "(uniform in patch, varied in surround); θ$_{EE}$ = E→E connectivity axis ≠ electrode shaft. "
             "After Brunel 2026 & Liou/Abbott 2020.", ha="center", fontsize=7.4, color="#888")
    os.makedirs(OUTDIR, exist_ok=True)
    out = f"{OUTDIR}/sef_hfo_mechanism_schematic.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def fig_transfer_supplementary():
    sigma = 4.0
    mu = np.linspace(8, 22, 120)
    lif = np.array([lif_rate(m, sigma, TAU_ME, TREF_E) for m in mu]) * 1000.0
    rmax = lif.max()
    mu0, s = 18.5, 0.85
    sig = rmax / (1 + np.exp(-(mu - mu0) / s))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.axvspan(10.8, 14.5, color="#f0c14b", alpha=0.16, zorder=0)
    ax.text(12.65, rmax * 0.015, "interictal operating band", ha="center", va="bottom",
            fontsize=7.8, color="#9a7d1e")
    ax.plot(mu, lif, color=C_E, lw=2.8, label="LIF  Φ(μ,σ)  — current")
    ax.plot(mu, sig, color=C_SIG, lw=2.4, ls="--", label="sigmoid F$_{eff}$ — demoted")
    ax.axvline(V_TH, color="#bbb", ls=":", lw=1.0)
    ax.text(V_TH + 0.2, rmax * 0.55, "spike\nthreshold", fontsize=7.5, color="#888")
    ax.text(12.6, rmax * 0.60, "LIF stays\nsensitive\nat low rate", color=C_E, fontsize=8.5,
            ha="center", fontweight="bold")
    ax.text(15.3, rmax * 0.12, "sigmoid flat\n→ can't ignite", color="#7f8c8d", fontsize=8, ha="left")
    ax.set_xlabel("mean input drive  μ  (mV)"); ax.set_ylabel("population firing rate  (Hz)")
    ax.set_xlim(8, 22); ax.set_ylim(0, rmax * 1.05); ax.margins(0)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.set_title("Supplementary — why LIF transfer, not sigmoid", fontsize=11, fontweight="bold")
    fig.text(0.5, 0.005, "Real Φ_LIF Siegert curve; sigmoid is an illustrative contrast (slope ratio "
             "indicative, not a measured loop gain).", ha="center", fontsize=7.0, color="#999")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = f"{OUTDIR}/sef_hfo_transfer_supplementary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_mechanism()
    fig_transfer_supplementary()
