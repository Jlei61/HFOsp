"""SNN spontaneous-event scan figure — PRELIMINARY (verdict downgraded, user review
2026-06-06; see archive §7). Shows the DATA only; the rate-band comparison is NOT a verdict.

What the figure honestly shows:
  (a) (drive x noise) regime map — every cell annotated with its spontaneous event rate.
      sigma=0 is silent; with slow spatial noise the homogeneous sheet robustly produces
      discrete self-terminating events. The "nominal band" [0.01,1]/s is the DETECTED
      group-event rate (Topic-2); the model counts ALL coherent nucleations, so cells
      "above the band" are NOT established as too-fast — the comparison is denominator-
      mismatched (footnote).
  (b) event rate vs noise correlation time — flat across a 6.7x range (the within-model
      observation that the high-rate cells' rate is not a noise-timescale artifact; does
      NOT settle the data comparison).

Data: results/topic4_sef_hfo/lif_snn/data/{snn_noise_grid,snn_noise_tausweep}.json
Run from repo root: python scripts/plot_sef_hfo_snn_noise_scan.py
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.plot_style import (  # noqa: E402
    style_panel, savefig_pub, COL_NONSIG, COL_SIG, COL_SURROGATE, COL_CLUSTER_T0,
    COL_CLUSTER_T1, FS_LABEL, FS_TICK,
)

DATA = "results/topic4_sef_hfo/lif_snn/data"
OUT_BASE = "results/topic4_sef_hfo/lif_snn/figures/snn_noise_spontaneous_scan"
RATE_LO, RATE_HI = 0.01, 1.0      # data-compatible event-rate band (events/s)

C_EXT = "#E6E6E6"     # extinction (no events)
C_ACC = COL_SURROGATE # accepted: data-compatible discrete (sage)
C_FAST = COL_SIG      # discrete but too frequent (rust)
C_WEAK = "#D9C2A6"    # discrete in <60% seeds (pale)


def _load(name):
    with open(os.path.join(DATA, name)) as f:
        return json.load(f)


def panel_map(ax, grid):
    drives = sorted({c["drive"] for c in grid["cells"]})
    fracs = sorted({c["sigma_frac"] for c in grid["cells"]})
    by = {(c["drive"], c["sigma_frac"]): c for c in grid["cells"]}
    for i, dr in enumerate(drives):
        for j, fr in enumerate(fracs):
            c = by[(dr, fr)]
            nd, ns = c["n_discrete"], c["n_seeds"]
            if nd == 0:
                col, txt = C_EXT, "ext"
            elif c["accepted"]:
                col, txt = C_ACC, f"{c['mean_rate']:.1f}/s"
            else:
                # discrete but not data-compatible (rate >1/s and/or <60% seeds robust)
                col = C_FAST
                txt = f"{c['mean_rate']:.1f}" + ("" if nd / ns >= 0.60 else "*")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=col,
                                   edgecolor="white", lw=1.5))
            if c["accepted"]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                       edgecolor="#2c6e49", lw=2.6, zorder=5))
            ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                    color="#222222", zorder=6)
    ax.set_xticks(range(len(fracs)))
    ax.set_xticklabels([f"{f:.2f}" for f in fracs], fontsize=FS_TICK)
    ax.set_yticks(range(len(drives)))
    ax.set_yticklabels([f"{d:.2f}" for d in drives], fontsize=FS_TICK)
    ax.set_xlabel("noise strength  (s.d. / external drive)", fontsize=FS_LABEL)
    ax.set_ylabel(r"external drive  $\nu_{ext}/\nu_\theta$", fontsize=FS_LABEL)
    ax.set_xlim(-0.5, len(fracs) - 0.5); ax.set_ylim(-0.5, len(drives) - 0.5)
    ax.set_aspect("equal")
    # legend
    from matplotlib.patches import Patch
    handles = [Patch(fc=C_EXT, ec="white", label="silent (no events)"),
               Patch(fc=C_FAST, ec="white", label="discrete, rate >1/s (* = <60% seeds)"),
               Patch(fc=C_ACC, ec="#2c6e49", lw=2, label="discrete, rate in nominal band*")]
    ax.legend(handles=handles, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False)


def panel_tau(ax, tw):
    taus = sorted({r["tau"] for r in tw["rows"]})
    fracs = sorted({r["frac"] for r in tw["rows"]})
    by = {(r["tau"], r["frac"]): r for r in tw["rows"]}
    cols = {fracs[0]: COL_CLUSTER_T0, fracs[1]: COL_CLUSTER_T1}
    mk = {fracs[0]: "o", fracs[1]: "s"}
    ax.axhspan(RATE_LO, RATE_HI, color=C_ACC, alpha=0.35, lw=0)
    ax.text(taus[-1], RATE_HI, " nominal band (≤1/s)*", fontsize=9.5,
            color="#2c6e49", va="center", ha="right")
    for fr in fracs:
        ys = [by[(t, fr)]["mean_rate"] for t in taus]
        ax.plot(taus, ys, marker=mk[fr], ms=8, lw=1.8, color=cols[fr],
                label=f"noise {fr:.2f}", mec="white", mew=1.0)
    ax.set_xlabel("noise correlation time  $\\tau$  (ms)", fontsize=FS_LABEL)
    ax.set_ylabel("spontaneous event rate (events/s)", fontsize=FS_LABEL)
    ax.set_xticks(taus); ax.set_xticklabels([f"{int(t)}" for t in taus], fontsize=FS_TICK)
    ax.set_ylim(0, None)
    ax.legend(fontsize=10, loc="upper right", frameon=False, title="drive 0.6")


def main():
    grid = _load("snn_noise_grid.json")
    tw = _load("snn_noise_tausweep.json")
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    fig.patch.set_facecolor("white")
    panel_map(axa, grid)
    panel_tau(axb, tw)
    style_panel(axa, "a", label_x=-0.16, label_y=1.04)
    style_panel(axb, "b", label_x=-0.14, label_y=1.04)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.235, wspace=0.32)
    fig.text(0.5, 0.018,
             "* Preliminary, NOT a verdict. The nominal rate band is the DETECTED group-event "
             "rate (Topic-2); the model counts ALL coherent nucleations (many never propagate "
             "out) — the comparison is denominator-mismatched. See archive §7.",
             ha="center", va="bottom", fontsize=8.0, color="#555555", wrap=True)
    savefig_pub(fig, OUT_BASE + ".png")
    print(f"  saved {OUT_BASE}.png + .pdf")


if __name__ == "__main__":
    main()
