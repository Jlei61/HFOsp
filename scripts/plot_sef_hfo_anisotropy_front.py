"""Supplementary figure — onset-front direction tracks the excitatory connectivity axis.

The load-bearing Step-0 discriminator in the spiking ground truth: when a localized
kick triggers a self-limited event, does the EARLY recruitment front elongate ALONG
the excitatory-to-excitatory connectivity long axis, and rotate with it? An isotropic
control must show no preferred direction.

Two layers (CLAUDE.md §7 — one question per panel):
  (a-d) spatial: representative single realization (seed 1), the first 8 ms of the
        front colored by recruitment time (early = dark). The cloud elongates along
        the imposed axis (dashed) for 0/45/90 deg and stays round for isotropic.
  (e)   elongation ratio vs condition — is the front elongated at all? (3-seed mean+-sd)
  (f)   direction error vs imposed angle — does the elongation point along the imposed
        axis? (3-seed mean+-sd, aniso only; isotropic has no axis to track)

Spiking ground truth = Bachschmid-Romano/Hatsopoulos/Brunel (2026) spatial E-I LIF.
Spatial data: results/topic4_sef_hfo/lif_snn/data/front_shapes.npz (seed 1).
Verdict data: results/topic4_sef_hfo/lif_snn/data/anisotropy_front_numbers.json (3 seeds).
Run from repo root: python scripts/plot_sef_hfo_anisotropy_front.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (  # noqa: E402
    style_panel, COL_CLUSTER_T0, COL_CLUSTER_T1, COL_SIG, COL_NONSIG,
    FS_LABEL, FS_TICK,
)

DATA = "results/topic4_sef_hfo/lif_snn/data"
OUT_BASE = "results/topic4_sef_hfo/lif_snn/figures/anisotropy_front_test"

C_E = COL_CLUSTER_T1   # excitatory substrate — faint red
C_I = COL_CLUSTER_T0   # inhibitory substrate — faint blue
KICK_R = 0.15          # mm — radius of the kicked disk (R_KICK in the engine)
FRONT_LO = 168.0       # ms — kick offset = start of the onset-front window
W_FIG = 8.0            # ms — the clean discriminator window (NOT the 12 ms wide one)
RATIO_MIN = 1.3        # elongation threshold (spec)
AXIS_ERR_MAX = 25.0    # direction-error threshold (deg, spec)

SPATIAL = [("c0", 0.0, "connectivity axis  0°"),
           ("c45", 45.0, "45°"),
           ("c90", 90.0, "90°"),
           ("ciso", None, "isotropic control")]


def panel_spatial(ax, npz, key, theta, title, center, L, cmap, norm):
    """Faint E/I sheet + first-8 ms front colored by recruitment time + kick disk + axis."""
    # faint resting substrate for 2-D context (E red, I blue)
    ax.scatter(npz["allI_xy"][:, 0], npz["allI_xy"][:, 1], s=1.0, c=C_I,
               alpha=0.10, linewidths=0)
    ax.scatter(npz["allE_xy"][:, 0], npz["allE_xy"][:, 1], s=0.9, c=C_E,
               alpha=0.07, linewidths=0)
    xy = npz[f"{key}_xy"]
    onset = npz[f"{key}_onset"]
    m = onset < (FRONT_LO + W_FIG)                      # first 8 ms only
    xy, onset = xy[m], onset[m]
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=onset - FRONT_LO, cmap=cmap, norm=norm,
                    s=5.0, linewidths=0, zorder=4)
    ax.add_patch(Circle(center, KICK_R, fill=False, ec="black", lw=1.0, zorder=6))
    if theta is not None:
        th = np.radians(theta)
        dx, dy = 0.80 * np.cos(th), 0.80 * np.sin(th)
        ax.plot([center[0] - dx, center[0] + dx], [center[1] - dy, center[1] + dy],
                color="black", lw=1.6, ls="--", zorder=7)
    else:
        ax.text(0.5, 0.06, "no preferred axis", transform=ax.transAxes,
                fontsize=11, va="bottom", ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(1.2); s.set_visible(True)
    ax.set_title(title, fontsize=13, pad=6)
    return sc


def panel_ratio(ax, agg):
    """Elongation ratio vs condition (3-seed mean+-sd): aniso elongated, iso round."""
    xs = [0, 1, 2, 3]
    keys = ["aniso_theta0", "aniso_theta45", "aniso_theta90", "iso_theta0"]
    cols = [COL_SIG, COL_SIG, COL_SIG, COL_NONSIG]
    means = [agg[k]["excess"]["ratio_mean"] for k in keys]
    sds = [agg[k]["excess"]["ratio_sd"] for k in keys]
    ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#555555", capsize=4, lw=1.4, zorder=2)
    ax.scatter(xs, means, c=cols, s=110, zorder=3, edgecolors="white", linewidths=1.0)
    ax.axhline(RATIO_MIN, ls="--", lw=1.4, color="#888888")
    ax.text(3.35, RATIO_MIN + 0.05, "elongation\nthreshold", fontsize=10,
            color="#666666", va="bottom", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels(["0°", "45°", "90°", "isotropic"], fontsize=FS_TICK)
    ax.set_ylabel("front elongation  (long/short)", fontsize=FS_LABEL)
    ax.set_xlim(-0.4, 3.6); ax.set_ylim(0.8, None)
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "rotated connectivity axis",
            fontsize=11, color=COL_SIG, ha="center", va="top")


def panel_error(ax, agg):
    """Direction error vs imposed angle (3-seed mean+-sd, aniso only)."""
    angs = [0, 45, 90]
    keys = ["aniso_theta0", "aniso_theta45", "aniso_theta90"]
    means = [agg[k]["excess"]["axis_error_mean"] for k in keys]
    sds = [agg[k]["excess"]["axis_error_sd"] for k in keys]
    ax.errorbar(angs, means, yerr=sds, fmt="o", color=COL_SIG, ms=9, capsize=4,
                lw=1.4, ecolor="#555555", zorder=3, mec="white", mew=1.0)
    ax.axhline(AXIS_ERR_MAX, ls="--", lw=1.4, color="#888888")
    ax.text(90, AXIS_ERR_MAX - 1.0, "tolerance", fontsize=10, color="#666666",
            va="top", ha="right")
    ax.set_xticks(angs); ax.set_xticklabels(["0°", "45°", "90°"], fontsize=FS_TICK)
    ax.set_xlabel("imposed connectivity axis", fontsize=FS_LABEL)
    ax.set_ylabel("front direction error (°)", fontsize=FS_LABEL)
    ax.set_xlim(-8, 98); ax.set_ylim(0, AXIS_ERR_MAX + 4)
    ax.text(45, 3.0, "front points within ~1° of the imposed axis",
            fontsize=10.5, color="#444444", ha="center", va="bottom")


def main():
    npz = np.load(os.path.join(DATA, "front_shapes.npz"), allow_pickle=True)
    doc = json.load(open(os.path.join(DATA, "anisotropy_front_numbers.json")))
    agg = doc["aggregate"]
    L = float(npz["L"]); center = np.asarray(npz["kick_center"], float)

    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=W_FIG)

    fig = plt.figure(figsize=(14.0, 8.4))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.32, 1.0], hspace=0.42, wspace=0.30,
                          left=0.055, right=0.90, top=0.93, bottom=0.085)
    axsp = [fig.add_subplot(gs[0, j]) for j in range(4)]
    axe = fig.add_subplot(gs[1, 0:2])
    axf = fig.add_subplot(gs[1, 2:4])

    sc = None
    for ax, (key, theta, title) in zip(axsp, SPATIAL):
        sc = panel_spatial(ax, npz, key, theta, title, center, L, cmap, norm)

    cbar = fig.colorbar(sc, ax=axsp, fraction=0.013, pad=0.015)
    cbar.set_label("recruitment time after kick (ms)\n(early = dark)", fontsize=11)
    cbar.ax.tick_params(labelsize=FS_TICK)

    panel_ratio(axe, agg)
    panel_error(axf, agg)
    style_panel(axe, "e", label_x=-0.13, label_y=1.02)
    style_panel(axf, "f", label_x=-0.13, label_y=1.02)
    for ax, letter in zip(axsp, "abcd"):
        ax.text(-0.06, 1.06, letter, transform=ax.transAxes, fontsize=20,
                fontweight="bold", va="bottom", ha="right")

    fig.savefig(OUT_BASE + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {OUT_BASE}.png + .pdf")


if __name__ == "__main__":
    main()
