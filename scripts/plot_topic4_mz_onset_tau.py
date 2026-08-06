#!/usr/bin/env python3
"""MZ onset dynamics — tau_adp near-critical sensitivity figure (review §5). At the near-critical
adaptation strength (target frac 0.001), does faster adaptation *recovery* (shorter tau_adp) turn the
bounded sub-onset plateau into a containment-recovery cycle? Consumes the tau-tagged trajectories.

Panel A: disinhibition D(t) overlaid by tau_adp (3 seeds each) + the z-only run-off level.
Panel B: D_max vs tau_adp (per seed), run-off marked — the plateau->run-off transition as tau shortens.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_onset_dynamics")
TRAJ = os.path.join(OUT, "per_seed")
D_ONSET = 0.0869                                  # z-only run-off corridor: D at first-crossing (0.087±0.002)
SEEDS = [1, 3, 4]
TAUS = [2000, 1000, 500]                          # ms; 2000 = the gap-grid a=0.001 (no tau tag)
COL = {2000: "#2a6f97", 1000: "#e8a13a", 500: "#c0392b"}   # CVD-safe blue / amber / red


def _path(tau, s):
    tag = "" if tau == 2000 else f"_tau{tau}"
    return os.path.join(TRAJ, f"traj_zA_q75_tz5000_A0.001{tag}_seed{s}.npz")


def main():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.5, 1.0]))

    # Panel A: D(t) by tau
    for tau in TAUS:
        for s in SEEDS:
            p = _path(tau, s)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            axA.plot(z["t_ms"] / 1000.0, z["D_allE"], color=COL[tau], lw=1.0, alpha=0.7)
    axA.axhline(D_ONSET, color="#555", ls="--", lw=1.0)
    axA.annotate("z-only run-off onset (corridor ~0.087)", xy=(0.4, D_ONSET), xytext=(0.4, D_ONSET + 0.006),
                 fontsize=7.8, color="#555")
    axA.annotate("shorter τ → D shoots off-scale to 0.6–0.8 (run-off)", xy=(0.5, 0.118), fontsize=7.8,
                 color="#c0392b", style="italic")
    axA.set_xlabel("time (s)", fontsize=9.5)
    axA.set_ylabel("disinhibition   D = 1 − z̄", fontsize=9.5)
    axA.set_title("A · faster adaptation recovery (shorter τ$_{adp}$) at fixed η$_m$=0.00745  (frac 0.001 @ τ=2s)",
                  fontsize=9.5, loc="left", weight="bold")
    axA.grid(True, alpha=0.18, lw=0.5)
    for sp in ("top", "right"):
        axA.spines[sp].set_visible(False)
    axA.margins(x=0)
    axA.set_ylim(-0.004, 0.13)                    # clip the run-off shoot-up (D→0.6-0.8) so the 0.05-0.10 band is legible
    handles = [Line2D([0], [0], color=COL[t], lw=2, label=f"τ$_{{adp}}$ = {t/1000:g} s") for t in TAUS]
    handles.append(Line2D([0], [0], color="#555", ls="--", lw=1, label="run-off level"))
    axA.legend(handles=handles, fontsize=8.2, loc="center right", frameon=True, framealpha=0.9, edgecolor="#ccc")

    # Panel B: the D each cell reaches vs tau — bounded plateau D_max (o), or run-off ONSET D at crossing
    # (X, on the corridor; NOT the post-run-off peak ~0.8, which would be a different-instant quantity).
    for tau in TAUS:
        for s in SEEDS:
            p = _path(tau, s)
            if not os.path.exists(p):
                continue
            z = np.load(p)
            t, D = z["t_ms"], z["D_allE"]
            ra = float(z["runaway_ms"])
            if np.isfinite(ra):
                k = min(int(np.searchsorted(t, ra)), len(D) - 1)
                y, mk = float(D[k]), "X"
            else:
                y, mk = float(D.max()), "o"
            axB.scatter([tau / 1000.0], [y], s=90, color=COL[tau], marker=mk, edgecolor="white", lw=0.8, zorder=4)
    axB.axhline(D_ONSET, color="#555", ls="--", lw=1.0)
    axB.set_xticks([0.5, 1.0, 2.0])
    axB.set_xlim(0.35, 2.15)
    axB.set_xlabel("adaptation recovery τ$_{adp}$ (s)", fontsize=9.5)
    axB.set_ylabel("D reached (plateau, or onset D at run-off)", fontsize=9.5)
    axB.set_title("B · plateau rises → run-off as τ$_{adp}$ shortens", fontsize=10, loc="left", weight="bold")
    axB.grid(True, alpha=0.18, lw=0.5)
    for sp in ("top", "right"):
        axB.spines[sp].set_visible(False)
    axB.legend(handles=[Line2D([0], [0], marker="o", color="#888", lw=0, markersize=9, label="bounded (plateau)"),
                        Line2D([0], [0], marker="X", color="#888", lw=0, markersize=9, label="run-off")],
               fontsize=8.2, loc="upper right", frameon=True, framealpha=0.9, edgecolor="#ccc")

    fig.suptitle("MZ onset dynamics · τ$_{adp}$ sensitivity — faster recovery weakens the brake, does not create a recovery cycle",
                 fontsize=11, weight="bold", x=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    base = os.path.join(OUT, "figures", "mz_onset_tau_sensitivity")
    fig.savefig(base + ".png", dpi=150)
    fig.savefig(base + ".pdf")
    print("wrote", base + ".png / .pdf")


if __name__ == "__main__":
    main()
