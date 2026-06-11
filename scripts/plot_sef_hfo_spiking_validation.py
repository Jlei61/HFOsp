"""Supplementary figure — Step-0 SEF-HFO mechanism in the spiking ground truth (pictorial).

Three questions, pictorial panels (CLAUDE.md §7; house style src/plot_style.py):
  (a) Drive-axis phase structure — a quiet excitable operating point (low drive ~0.6) below
      an oscillatory regime (~1.0).
  (b) Regime time series — E-population rate is flat/quiet at drive 0.6, rhythmic at 1.0.
  (c) Kick from quiet rest — the 2-D E (red) / I (blue) sheet at rest, then a local kick
      seeds an event that spreads as a front and self-terminates.

The onset-front direction discriminator (front elongates along / rotates with the imposed
excitatory connectivity axis) is its own figure: scripts/plot_sef_hfo_anisotropy_front.py.

Spiking ground truth = Bachschmid-Romano/Hatsopoulos/Brunel (2026) spatial E-I LIF engine.
Data in results/topic4_sef_hfo/lif_snn/data/.  Run: python scripts/plot_sef_hfo_spiking_validation.py
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
    style_panel, COL_CLUSTER_T0, COL_CLUSTER_T1, COL_OSCILLATOR, COL_NONSIG,
    FS_LABEL, FS_TICK, FS_PANEL_LETTER,
)

DATA = "results/topic4_sef_hfo/lif_snn/data"
OUT_BASE = "results/topic4_sef_hfo/lif_snn/figures/sef_hfo_spiking_validation"

C_E = COL_CLUSTER_T1     # excitatory neurons — red
C_I = COL_CLUSTER_T0     # inhibitory neurons — blue
C_OSC = COL_OSCILLATOR   # oscillatory drive / oscillation strength — warm brown
C_QUIET = COL_NONSIG     # quiet drive — neutral gray
LEG_FS = 10


def _load_json(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# (a) drive-axis phase structure
# ---------------------------------------------------------------------------
def panel_a(ax, d):
    r = np.asarray(d["ratios"], float)
    ax.errorbar(r, d["rateE_mean"], yerr=d["rateE_sd"], color=C_E, marker="o",
                ms=5, lw=1.8, capsize=2, label="excitatory rate")
    ax.errorbar(r, d["rateI_mean"], yerr=d["rateI_sd"], color=C_I, marker="s",
                ms=5, lw=1.8, capsize=2, label="inhibitory rate")
    ax.set_xlabel(r"external drive  $\nu_{ext}/\nu_\theta$", fontsize=FS_LABEL)
    ax.set_ylabel("firing rate (Hz)", fontsize=FS_LABEL)
    ax.set_xlim(r.min(), r.max())
    ax.set_ylim(0, None)
    ax2 = ax.twinx()
    ax2.plot(r, d["prominence_mean"], color=C_OSC, marker="^", ms=5, lw=1.8, ls="--",
             label="oscillation strength")
    ax2.set_ylabel("oscillation strength", fontsize=FS_LABEL, color=C_OSC)
    ax2.tick_params(axis="y", labelcolor=C_OSC, labelsize=FS_TICK)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylim(0, None)
    ax.axvspan(0.55, 0.68, color="#cfe0cf", alpha=0.45, lw=0, zorder=0)
    ax.axvspan(0.70, r.max(), color="#e8d0c8", alpha=0.45, lw=0, zorder=0)
    ytop = ax.get_ylim()[1]
    ax.text(0.605, ytop * 0.97, "interictal-like\n(quiet, excitable)", ha="center",
            va="top", fontsize=8.5, color="#3a6b3a")
    ax.text(1.0, ytop * 0.97, "oscillatory\nregime", ha="center", va="top",
            fontsize=8.5, color="#8a4a38")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=LEG_FS, loc="center left",
              bbox_to_anchor=(0.0, 0.42), frameon=False)


# ---------------------------------------------------------------------------
# (b) regime time series — quiet vs oscillating (NOT red/blue: those are E/I)
# ---------------------------------------------------------------------------
def panel_b(ax, d):
    t = np.asarray(d["t_ms"], float)
    m = (t >= 200.0) & (t <= 600.0)
    ax.plot(t[m], np.asarray(d["rateE_osc"], float)[m], color=C_OSC, lw=1.0,
            label=f"drive {d['drive_osc']:.1f} (oscillatory)")
    ax.plot(t[m], np.asarray(d["rateE_quiet"], float)[m], color=C_QUIET, lw=1.2,
            label=f"drive {d['drive_quiet']:.1f} (quiet)")
    ax.set_xlabel("time (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("E population rate (Hz)", fontsize=FS_LABEL)
    ax.set_xlim(200, 600)
    ax.set_ylim(0, None)
    ax.legend(fontsize=LEG_FS, loc="upper right", frameon=False)


# ---------------------------------------------------------------------------
# spatial-panel helpers — E (red) / I (blue) substrate + bright recruited E
# ---------------------------------------------------------------------------
def _substrate(ax, npz, L, cx, cy, R):
    ax.scatter(npz["allI_xy"][:, 0], npz["allI_xy"][:, 1], s=1.4, c=C_I, alpha=0.22, linewidths=0)
    ax.scatter(npz["allE_xy"][:, 0], npz["allE_xy"][:, 1], s=1.2, c=C_E, alpha=0.16, linewidths=0)
    ax.add_patch(Circle((cx, cy), R, fill=False, ec="black", lw=0.9))
    ax.set_xlim(0, L); ax.set_ylim(0, L)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# ---------------------------------------------------------------------------
# (c) kick from quiet rest — substrate then seed -> spread -> self-terminate
# ---------------------------------------------------------------------------
def panel_c(axes, npz):
    L = float(npz["L"]); cx, cy = npz["kick_center"]; R = float(npz["kick_R"])
    wins = npz["windows"]
    specs = [("rest\n(before kick)", None), ("kick", 0), ("spread (front)", 2),
             ("self-terminated", 3)]
    for ax, (title, wi) in zip(axes, specs):
        _substrate(ax, npz, L, cx, cy, R)
        if wi is not None:
            xy = npz[f"w{wi}_xy"]
            ax.scatter(xy[:, 0], xy[:, 1], s=3.5, c=C_E, alpha=0.85, linewidths=0)
            lo, hi = wins[wi]
            ax.set_title(f"{title}\n{int(lo)}–{int(hi)} ms", fontsize=8.5, pad=4)
        else:
            ax.set_title(title, fontsize=8.5, pad=4)


def _group_letter(ax, letter):
    ax.text(-0.22, 1.05, letter, transform=ax.transAxes, fontsize=FS_PANEL_LETTER,
            fontweight="bold", va="bottom", ha="right")


def main():
    drive = _load_json("drive_sweep.json")
    regime = _load_json("regime_timeseries.json")
    snaps = np.load(os.path.join(DATA, "kick_snapshots.npz"))

    fig = plt.figure(figsize=(14.5, 8.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.5, wspace=0.55,
                          left=0.07, right=0.965, top=0.93, bottom=0.10)
    axa = fig.add_subplot(gs[0, 0:2])
    axb = fig.add_subplot(gs[0, 2:4])
    axc = [fig.add_subplot(gs[1, j]) for j in range(4)]

    panel_a(axa, drive)
    panel_b(axb, regime)
    panel_c(axc, snaps)

    style_panel(axa, "a", label_x=-0.14, label_y=1.04)
    style_panel(axb, "b", label_x=-0.16, label_y=1.04)
    _group_letter(axc[0], "c")

    # shared E/I legend for the spatial row
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=C_E, ms=7, label="excitatory"),
               Line2D([0], [0], marker="o", ls="", color=C_I, ms=7, label="inhibitory")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=LEG_FS,
               frameon=False, bbox_to_anchor=(0.5, 0.01))

    os.makedirs(os.path.dirname(OUT_BASE), exist_ok=True)
    fig.savefig(OUT_BASE + ".png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_BASE + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {OUT_BASE}.png + .pdf")


if __name__ == "__main__":
    main()
