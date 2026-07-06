"""M4 dynamic (k_q x alpha_G) phase diagram (Topic 4). The dynamic analog of the frozen-q_I phase-plane:
does the shared divisive pool S_G bound the q_I-depletion runaway, and how strong must it be? Rows = q_I
depletion rate k_q; cols = pool strength alpha_G (0 = pool off). 3 panels (§7):
  A verdict     : per-cell runaway outcome (no_runaway = the pool prevented it; one_shot/train = it ran away).
  B peak rate   : sustained per-neuron rate (Hz) -- the braking GRADIENT (does more alpha_G lower it, and
                  does it ever fall below the runaway level?).
  C S_G_max     : how hard the pool pushed (0 in the no_pool column by construction).

Reads results/topic4_m4_dynamic_sweep/dynamic_qi_summary.json. Output:
results/paper-ready-figure/fig_m4_dynamic_qi/figures/fig_m4_phase_diagram.png. Plotting-only.
UNITS: summary max_rate_hz is ALREADY per-neuron Hz (kick_probe returns Hz); used directly.
SCREEN CAVEAT: this sweep is T=1000 ms, so a "no_runaway" cell only means "no runaway WITHIN 1 s" -- a
survivor, NOT a confirmed bounded state (high alpha_G may merely delay runaway past 1 s). Survivor cells
must be re-run at T=5000+ to earn a "bounded candidate" label.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results/topic4_m4_dynamic_sweep/dynamic_qi_summary.json"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"

RUNAWAY_HZ = 120.0                                   # runner detection level (summary rate is already Hz)
# verdict -> integer code + colour. GREEN = survived the 1 s window (NOT confirmed bounded); reds = ran away.
VERDICT_CODE = {"no_runaway": 0, "train_then_runaway": 1, "few_events_then_runaway": 2, "one_shot_burst": 3}
VERDICT_COL = ["#2e7d32", "#f6c344", "#ef8a3a", "#c1272d"]
VERDICT_LAB = ["no runaway ≤1s\n(survivor, unconfirmed)", "train→runaway", "few→runaway", "one_shot_burst"]


def _grid(rows, kq_grid, ag_grid, key, fn=lambda r: r):
    """rows -> (len(kq) x len(ag)) grid of fn(row[key]); ag index 0 == no_pool (alpha_G 0)."""
    ag_axis = [0.0] + list(ag_grid)
    G = np.full((len(kq_grid), len(ag_axis)), np.nan)
    for r in rows:
        i = kq_grid.index(r["k_q"]) if r["k_q"] in kq_grid else None
        aval = 0.0 if not r["use_SG"] else r["alpha_G"]
        j = ag_axis.index(aval) if aval in ag_axis else None
        if i is not None and j is not None:
            G[i, j] = fn(r[key]) if key else fn(r)
    return G, ag_axis


def main():
    d = json.load(open(SUMMARY))
    meta = d["meta"]; rows = d["rows"]
    kq_grid = meta["kq_grid"]; ag_grid = meta["alpha_grid"]
    vg, ag_axis = _grid(rows, kq_grid, ag_grid, "verdict", lambda v: VERDICT_CODE.get(v, 3))
    rg, _ = _grid(rows, kq_grid, ag_grid, "max_rate_hz", float)      # summary max_rate_hz is already Hz
    areag, _ = _grid(rows, kq_grid, ag_grid, "active_area_tail", float)
    sg, _ = _grid(rows, kq_grid, ag_grid, "S_G_max", float)
    OUT.mkdir(parents=True, exist_ok=True)

    xt = [("off" if a == 0 else f"{a:g}") for a in ag_axis]
    yt = [f"{k:g}" for k in kq_grid]
    fig, ax = plt.subplots(1, 4, figsize=(18.5, 4.6))
    fig.suptitle("M4 dynamic phase diagram (T=1000 ms SURVIVOR screen): does $S_G$ delay/prevent the "
                 "$q_I$-depletion runaway?  (E1146, L=20)   green = no runaway ≤1s (unconfirmed, needs "
                 "T=5000)", fontsize=11.5, y=1.02)

    def _fmt(a, grid, title, cbar_lab, cmap=None, norm=None, annot="{:.0f}"):
        im = a.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
        a.set_xticks(range(len(ag_axis))); a.set_xticklabels(xt)
        a.set_yticks(range(len(kq_grid))); a.set_yticklabels(yt)
        a.set_xlabel("pool strength $\\alpha_G$"); a.set_ylabel("depletion rate $k_q$")
        a.set_title(title, fontsize=11, loc="left")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    a.text(j, i, annot.format(grid[i, j]), ha="center", va="center", fontsize=7.5,
                           color="white" if cmap is None else "black")
        return im

    # A verdict (categorical)
    cmap = ListedColormap(VERDICT_COL); norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    a0 = ax[0]; a0.imshow(vg, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    a0.set_xticks(range(len(ag_axis))); a0.set_xticklabels(xt)
    a0.set_yticks(range(len(kq_grid))); a0.set_yticklabels(yt)
    a0.set_xlabel("pool strength $\\alpha_G$"); a0.set_ylabel("depletion rate $k_q$")
    a0.set_title("A  runaway verdict", fontsize=11, loc="left")
    a0.legend(handles=[Patch(fc=VERDICT_COL[i], label=VERDICT_LAB[i].replace("\n", " ")) for i in range(4)],
              fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False)

    # B peak rate (Hz) -- the braking gradient
    im1 = _fmt(ax[1], rg, f"B  peak rate (Hz/neuron; runaway$\\geq${RUNAWAY_HZ:.0f})", "Hz", cmap="magma_r")
    cb1 = fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04); cb1.set_label("Hz/neuron")

    # C spatial extent (active-area tail) -- the CONTAINMENT gradient (1.0 = whole-field, ->0 = core-local)
    im2 = _fmt(ax[2], areag, "C  spatial extent (tail active-area)", "area", cmap="cividis", annot="{:.2f}")
    cb2 = fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04); cb2.set_label("tail active-area frac")

    # D S_G_max
    im3 = _fmt(ax[3], sg, "D  pool output $S_G^{max}$", "S_G", cmap="viridis", annot="{:.2f}")
    cb3 = fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04); cb3.set_label("$S_G$ max")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT / "fig_m4_phase_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "fig_m4_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    # recap: any bounded cell? min rate per k_q row
    from collections import Counter
    print("verdicts:", dict(Counter(r["verdict"] for r in rows)))
    print("peak-rate range (Hz):", f"{np.nanmin(rg):.0f}..{np.nanmax(rg):.0f}")
    surv = [r["label"] for r in rows if r["verdict"] == "no_runaway"]
    print("SURVIVOR cells (no runaway <=1s; UNCONFIRMED, need T=5000):",
          surv if surv else "NONE -- every cell ran away within 1 s in this grid")


if __name__ == "__main__":
    main()
