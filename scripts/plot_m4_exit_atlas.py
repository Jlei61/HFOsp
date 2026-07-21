#!/usr/bin/env python
"""Phase-2 frozen exit-atlas phase diagram (task brief §5).

Reads an `arms_<tag>_seed<seed>.json` produced by run_m4_snn_native_exit.py --mode frozen_atlas and draws,
one panel per q_core, the S_G x J_exit grid coloured by REGIME:
  low-only  = both cold and warm ICs settle low  -> a monostable interictal basin
  bistable  = cold->low but warm(kick)->high/runaway -> the ictal branch is reachable
  high-only = both ICs -> high/runaway -> no low basin here
Each cell is annotated with the warm-IC outcome (settled rate Hz, or "RA" for runaway) so the magnitude
gradient is visible; a small dot marks cells where the COLD IC is also high. Self-contained: real-unit axes,
one shared legend, no internal codes. Honest diagnostic (frozen landscape, NOT the dynamic trajectory).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REG_COLOR = {"low-only": "#92C5DE", "bistable": "#FDB863", "high-only": "#B2182B"}
REG_ORDER = ["low-only", "bistable", "high-only"]
_HIGH = ("runaway", "bounded_high", "bounded_oscillatory")


def _regime(cold_cls, warm_cls):
    ch, wh = cold_cls in _HIGH, warm_cls in _HIGH
    return "low-only" if not (ch or wh) else "high-only" if (ch and wh) else "bistable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = [r for r in json.load(open(a.json))["rows"] if "error" not in r]
    d = {(r["q_core"], r["S_G"], r["J_exit"], r["warm"]): r for r in rows}
    qs = sorted({r["q_core"] for r in rows})
    sgs = sorted({r["S_G"] for r in rows})
    js = sorted({r["J_exit"] for r in rows})
    q_label = {qs[0]: "depleted", qs[len(qs) // 2]: "middle", qs[-1]: "recovered"}

    fig, axes = plt.subplots(1, len(qs), figsize=(3.5 * len(qs) + 0.5, 3.9), squeeze=False)
    for qi, q in enumerate(qs):
        ax = axes[0][qi]
        grid = np.zeros((len(sgs), len(js), 3))
        for i, sg in enumerate(sgs):
            for k, j in enumerate(js):
                cc = d[(q, sg, j, False)]["atlas_class"]
                wc = d[(q, sg, j, True)]["atlas_class"]
                reg = _regime(cc, wc)
                grid[i, k] = matplotlib.colors.to_rgb(REG_COLOR[reg])
                wr = d[(q, sg, j, True)]
                lab = "RA" if wr["atlas_class"] == "runaway" else f"{wr['settled_rate_hz']:.0f}"
                txt_c = "white" if reg == "high-only" else "#222222"
                ax.text(k, i, lab, ha="center", va="center", fontsize=9, color=txt_c, fontweight="bold")
                if cc in _HIGH:                                   # cold IC also high -> corner dot
                    ax.plot(k + 0.34, i - 0.34, "o", ms=4, color="#222222")
        ax.imshow(grid, origin="lower", aspect="auto", extent=(-0.5, len(js) - 0.5, -0.5, len(sgs) - 0.5))
        ax.set_xticks(range(len(js))); ax.set_xticklabels([f"{j:g}" for j in js])
        ax.set_yticks(range(len(sgs))); ax.set_yticklabels([f"{sg:g}" for sg in sgs])
        ax.set_xlabel("J_exit — recovery current (mV)")
        if qi == 0:
            ax.set_ylabel("S_G — divisive containment")
        ax.set_title(f"q_core = {q:g}  ({q_label.get(q, '')})", fontsize=11)
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [Patch(facecolor=REG_COLOR[r], edgecolor="#555", label=r) for r in REG_ORDER]
    handles.append(plt.Line2D([0], [0], marker="o", ls="", color="#222", ms=5, label="cold IC also high"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Frozen exit atlas — does a low/interictal basin exist?  (E1146 M4, seed 1; "
                 "cell text = warm-kick settled rate Hz / RA=runaway)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = a.out or os.path.join(os.path.dirname(a.json), "figures",
                                os.path.basename(a.json).replace("arms_", "exit_atlas_").replace(".json", ".png"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
