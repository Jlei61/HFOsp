#!/usr/bin/env python3
"""Topic 4 MZ early-onset dynamics — natural z+m trajectory figure (temporal phase-diagram
Panels A + C). Consumes the per-cell continuous trajectories written by
``run_topic4_mz_onset_dynamics.py focused-m`` (``per_seed/traj_*.npz``). No simulation here.

Panel A: single-cell push-pull time course (rate, disinhibition D=1-z̄, adaptation a=η_m·m̄/I_EE)
         for one representative z+m cell — each interictal event pushes D and a up, then a relaxes
         faster (τ_adp) than D (τ_z).
Panel C: D–a state plane across adaptation strengths × seeds — z alone travels along the D axis to
         run-off (runaway ✦); adaptation redirects the path up the a axis, away from the run-off corner.

Frozen-system Panels B/D (α₁ / ε_c) are added once the frozen (D,a) grid is computed.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_onset_dynamics")
TRAJ_DIR = os.path.join(OUT, "per_seed")
FIG_DIR = os.path.join(OUT, "figures")

REGIME = "zA_q75_tz5000"
NONZERO_FRACS = [0.01, 0.025, 0.05, 0.10, 0.20]   # adaptation-strength axis (fraction of I_EE_scale)


def _load():
    """Return {(z_regime, a_frac, seed): npz-dict}. a_frac read from the npz field, not the name."""
    cells = {}
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "traj_*.npz"))):
        z = np.load(f)
        key = (str(z["z_regime"]), round(float(z["A_frac"]), 3), int(z["seed"]))
        cells[key] = {k: z[k] for k in z.files}
    return cells


def _frac_color(frac):
    """Sequential viridis over the non-zero adaptation strengths (light→dark = weak→strong)."""
    i = NONZERO_FRACS.index(frac)
    return cm.viridis(0.12 + 0.80 * i / (len(NONZERO_FRACS) - 1))


def panel_a(fig, gs, cells, rep_frac, rep_seed):
    """Three stacked stripes sharing the time axis for one representative z+m cell."""
    c = cells[(REGIME, rep_frac, rep_seed)]
    t = c["t_ms"] / 1000.0
    on, off = c["event_on_ms"] / 1000.0, c["event_off_ms"] / 1000.0
    sub = gs.subgridspec(3, 1, hspace=0.12)
    axes = [fig.add_subplot(sub[i]) for i in range(3)]
    series = [("rate_E_hz", "E rate (Hz)", "#3a3a3a"),
              (None, "disinhibition\nD = 1 − z̄", "#A35E48"),
              (None, "adaptation\na = η$_m$ m̄ / I$_{EE}$", "#3d6b9c")]
    data = [c["rate_E_hz"], c["D_allE"], c["a_allE"]]
    for ax, (_, ylab, col), y in zip(axes, series, data):
        for eo, ef in zip(on, off):
            ax.axvspan(eo, ef, color="#d9c27a", alpha=0.35, lw=0)
        ax.plot(t, y, color=col, lw=1.1)
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.margins(x=0)
        ax.grid(True, alpha=0.18, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_title(f"A · single-cell push–pull  (adaptation = {rep_frac:g}, seed {rep_seed})",
                      fontsize=10, loc="left", weight="bold")
    axes[0].tick_params(labelbottom=False)
    axes[1].tick_params(labelbottom=False)
    axes[2].set_xlabel("time (s)", fontsize=9)
    axes[0].annotate("interictal events (shaded): each pushes D↑ and a↑;\n"
                     "between events a relaxes faster (τ$_{adp}$=2 s) than D (τ$_z$=5 s)",
                     xy=(0.985, 0.9), xycoords="axes fraction", ha="right", va="top", fontsize=7.3,
                     color="#555")


def panel_c(fig, gs, cells):
    """D–a state plane: all adaptation strengths × 3 seeds; z-only run-off vs adaptation redirection."""
    ax = fig.add_subplot(gs)
    seeds = [1, 3, 4]
    # z-only (a_frac=0): travels along D axis to run-off
    for seed in seeds:
        c = cells.get((REGIME, 0.0, seed))
        if c is None:
            continue
        ax.plot(c["D_allE"], c["a_allE"], color="#c0392b", lw=1.0, alpha=0.5, zorder=3)
        ra = float(c["runaway_ms"])
        if np.isfinite(ra):
            k = int(np.argmax(c["D_allE"]))
            ax.scatter([c["D_allE"][k]], [c["a_allE"][k]], marker="*", s=150, color="#c0392b",
                       edgecolor="white", lw=0.8, zorder=6)
    # z+m cells: redirected up the a axis
    for frac in NONZERO_FRACS:
        col = _frac_color(frac)
        for seed in seeds:
            c = cells.get((REGIME, frac, seed))
            if c is None:
                continue
            ax.plot(c["D_allE"], c["a_allE"], color=col, lw=0.9, alpha=0.55, zorder=4)
    # slow-off reference at the origin
    ax.scatter([0], [0], marker="o", s=45, facecolor="#dddddd", edgecolor="#888", lw=0.8, zorder=5)
    ax.set_xlabel("disinhibition   D = 1 − z̄", fontsize=9.5)
    ax.set_ylabel("adaptation   a = η$_m$ m̄ / I$_{EE}$", fontsize=9.5)
    ax.set_title("C · D–a state plane  (primary regime, seeds 1/3/4)", fontsize=10, loc="left", weight="bold")
    ax.grid(True, alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.margins(x=0.02, y=0.04)
    # legend: adaptation ramp + z-only + slow-off + runaway marker
    handles = [Line2D([0], [0], color="#c0392b", lw=2, label="z-only (a=0) → run-off")]
    handles += [Line2D([0], [0], color=_frac_color(f), lw=2, label=f"a={f:g}") for f in NONZERO_FRACS]
    handles += [Line2D([0], [0], marker="*", color="#c0392b", lw=0, markersize=11,
                       markeredgecolor="white", label="run-off onset"),
                Line2D([0], [0], marker="o", color="#888", markerfacecolor="#ddd", lw=0,
                       markersize=8, label="slow-off (origin)")]
    ax.legend(handles=handles, fontsize=7.6, loc="upper right", frameon=True, framealpha=0.9,
              edgecolor="#ccc", borderpad=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep-frac", type=float, default=0.05, help="representative z+m adaptation strength for Panel A")
    ap.add_argument("--rep-seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    cells = _load()

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.22,
                          left=0.06, right=0.985, top=0.9, bottom=0.12)
    panel_a(fig, gs[0], cells, round(args.rep_frac, 3), args.rep_seed)
    panel_c(fig, gs[1], cells)
    fig.suptitle("MZ early-onset dynamics · natural z+m trajectories  (E1146, primary regime zA_q75_tz5000)",
                 fontsize=11.5, weight="bold", x=0.06, ha="left")
    base = os.path.join(FIG_DIR, "mz_onset_natural_trajectories")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    print("wrote", base + ".png / .pdf", "| n_cells=", len(cells))


if __name__ == "__main__":
    main()
