"""FCXR-HEO3 H3.0 part B figure — source-space audit (review P1-b), three independent questions:
(1) is the TISSUE recruited and does it stay recruited (participation ratio over time)?
(2) does the activity centre MOVE along the source->sink axis, or is it frozen?
(3) WHERE does the activity sit (per-region rate profile)?
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay", "heo3")
COL = {"m_off": "#4c72b0", "dyn_tau250_frac0.1": "#c44e52"}
NAME = {"m_off": "16 Hz reference", "dyn_tau250_frac0.1": "fast-τ/10% precursor"}
REGIONS = ["core_source", "core_sink", "axis_corridor", "off_axis"]


def main():
    d = json.load(open(os.path.join(OUT, "stage0_source_space.json")))
    sizes = d["region_sizes"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))

    for lab, a in d["arms"].items():
        rows = a["rows"]
        t = np.array([r["t_ms"] for r in rows]) / 1000.0
        ax[0].plot(t, [r["participation_ratio"] for r in rows], lw=1.3, color=COL[lab], label=NAME[lab])
        ax[1].plot(t, [r["centroid_axis_coord"] for r in rows], lw=1.3, color=COL[lab], label=NAME[lab])
    ax[0].axvline(1.0, ls=":", c="0.5", lw=1)
    ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("participation ratio")
    ax[0].set_title("(1) is the TISSUE recruited?\nPR≈0.6 → ~60% of E-cells carry it, not one loud core", fontsize=9)
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=7, loc="lower left")

    ax[1].axvline(1.0, ls=":", c="0.5", lw=1)
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("centroid coord along source→sink axis")
    ax[1].set_title("(2) does the activity centre MOVE?\nboth ~frozen at 0.43; the one excursion is the COLLAPSE, not alternation", fontsize=8.5)
    ax[1].annotate("collapse\n(PR→0.08)", xy=(3.3, 0.10), xytext=(3.55, 0.26), fontsize=6.5, color="0.35",
                   arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8))
    ax[1].set_ylim(0, 1); ax[1].legend(fontsize=7, loc="lower left")

    x = np.arange(len(REGIONS)); w = 0.36
    for i, (lab, a) in enumerate(d["arms"].items()):
        med = [float(np.median([r[f"rate_{g}"] for r in a["rows"]])) for g in REGIONS]
        ax[2].bar(x + (i - 0.5) * w, med, w, color=COL[lab], label=NAME[lab])
    ax[2].set_xticks(x)
    ax[2].set_xticklabels([f"{g.replace('_', chr(10))}\nn={sizes[g]}" for g in REGIONS], fontsize=7)
    ax[2].set_ylabel("median per-neuron rate (Hz)")
    ax[2].set_title("(3) WHERE is the activity?\ngraded core→corridor→off-axis (~4×), not confined", fontsize=9)
    ax[2].legend(fontsize=7)

    fig.text(0.5, 0.005, "FCXR-HEO3 H3.0b — source-space audit: real tissue recruitment (PR≈0.6), graded not confined, and NO sustained region alternation yet",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(OUT, "figures", "stage0_source_space.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    for lab, a in d["arms"].items():
        pr = [r["participation_ratio"] for r in a["rows"]]
        ac = [r["centroid_axis_coord"] for r in a["rows"]]
        print(f"  {lab:22s} PR med {np.median(pr):.3f} min {np.min(pr):.3f} | centroid axis "
              f"med {np.nanmedian(ac):.3f} span {np.nanmax(ac)-np.nanmin(ac):.3f}")
    print("[h3.0b] wrote figures/stage0_source_space.png")


if __name__ == "__main__":
    main()
