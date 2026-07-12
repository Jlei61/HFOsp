"""M4 dynamic (k_q x alpha_G) phase diagram — the CORRECTED, EXTENDED 2D heatmap. Supersedes the old
T=1000-only sweep heatmap: for every cell it takes the LONGEST-T verdict available across all runs
(sweep T=1000 / confirm+delay+extend T=5000-8000 / longconfirm+multiseed T=15000), because the T=1000
sweep mislabels delayed-runaways as survivors (the aG20 trap). alpha_G extended to 24.

Two panels (§7):
  A verdict (categorical): bounded (green) / runaway (red) / uncertain-short-T (orange) / no-data (gray).
     Cell text = runaway onset ms, or 'bnd'. The (0.10,16) cell is annotated with the 4-seed split.
  B sustained peak rate (Hz) — the intensity, with the 120 runaway level as the colormap pivot.

The ONLY bounded region is the narrow (k_q=0.10, alpha_G=16) cell (3/4 seeds); everything else runs away.
This is a non-runaway bounded attractor candidate window, NOT a full seizure cycle. res rate_E is already Hz.
Reads results/topic4_m4_dynamic_{sweep,confirm,delay,longconfirm,extend,multiseed/seed*}/.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"
KQS = [0.10, 0.18, 0.25, 0.35, 0.50]
ALPHAS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]
RUNAWAY_HZ, Q_FLOOR = 120.0, 0.05
DIRS = ["topic4_m4_dynamic_sweep", "topic4_m4_dynamic_confirm", "topic4_m4_dynamic_delay",
        "topic4_m4_dynamic_longconfirm", "topic4_m4_dynamic_extend", "topic4_m4_dynamic_confirm2",
        "topic4_m4_dynamic_multiseed/seed2", "topic4_m4_dynamic_multiseed/seed3",
        "topic4_m4_dynamic_multiseed/seed4"]
# verdict codes / colours: 0 bounded, 1 uncertain, 2 runaway, 3 no-data
COL = ListedColormap(["#2e7d32", "#e8873a", "#c1272d", "#d9d9d9"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], COL.N)


def _load(d):
    p = RES / d / "dynamic_qi_summary.json"
    return json.load(open(p))["rows"] if p.exists() else []


def _best_rows():
    """allrows[(k_q, alpha_G)] = every row across dirs; best = the longest-T row (most trustworthy verdict)."""
    allrows = {}
    for d in DIRS:
        for r in _load(d):
            ag = 0.0 if not r["use_SG"] else r["alpha_G"]
            allrows.setdefault((round(r["k_q"], 2), round(ag, 1)), []).append(r)
    best = {k: max(v, key=lambda r: r["T"]) for k, v in allrows.items()}
    return best, allrows


def _verdict(r):
    if r is None:
        return 3
    if r["runaway_ms"] is not None:
        return 2                                   # ran away
    if r["T"] >= 15000 and r["max_rate_hz"] < RUNAWAY_HZ and r["q_mean_final"] > Q_FLOOR + 0.01:
        return 0                                   # long-confirmed bounded
    return 1                                       # no-runaway but T<15000 -> uncertain (could flip)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best, allrows = _best_rows()
    nk, na = len(KQS), len(ALPHAS)
    vg = np.full((nk, na), 3)
    rate = np.full((nk, na), np.nan)
    txt = [["" for _ in range(na)] for _ in range(nk)]
    for i, kq in enumerate(KQS):
        for j, ag in enumerate(ALPHAS):
            r = best.get((round(kq, 2), round(ag, 1)))
            vg[i, j] = _verdict(r)
            if r is not None:
                rate[i, j] = r["max_rate_hz"]
                txt[i][j] = "bnd" if vg[i, j] == 0 else ("?" if vg[i, j] == 1 else
                                                         (f"{r['runaway_ms']:.0f}" if r["runaway_ms"] else "·"))
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.4))
    fig.suptitle("M4 dynamic phase diagram ($k_q\\times\\alpha_G$, longest-T verdict per cell): a narrow MARGINAL "
                 "bound strip at $\\alpha_G$=16 — confirmed bounded (0.10,16)+(0.25,16), marginal (0.18/0.35,16); "
                 "a bounded attractor candidate, not a seizure cycle",
                 fontsize=10.3, y=1.03)
    xt = [("off" if a == 0 else f"{a:g}") for a in ALPHAS]
    yt = [f"{k:g}" for k in KQS]

    # A verdict
    ax[0].imshow(vg, origin="lower", aspect="auto", cmap=COL, norm=NORM)
    for i in range(nk):
        for j in range(na):
            ax[0].text(j, i, txt[i][j], ha="center", va="center", fontsize=7,
                       color="white" if vg[i, j] in (0, 2) else "black")
    ax[0].set_xticks(range(na)); ax[0].set_xticklabels(xt); ax[0].set_yticks(range(nk)); ax[0].set_yticklabels(yt)
    ax[0].set_xlabel("pool strength $\\alpha_G$"); ax[0].set_ylabel("depletion rate $k_q$")
    ax[0].set_title("A  verdict (text = runaway onset ms / bnd)", fontsize=10, loc="left")
    # annotate the bounded cell's seed split
    if (0.10, 16.0) in allrows:
        seeds = [r for r in allrows[(0.10, 16.0)] if r["T"] >= 15000]
        nb = sum(1 for r in seeds if _verdict(r) == 0)
        ax[0].text(ALPHAS.index(16.0), KQS.index(0.10) + 0.34, f"{nb}/{len(seeds)} seed", ha="center",
                   fontsize=6.5, color="white", fontweight="bold")
    ax[0].legend(handles=[Patch(fc="#2e7d32", label="confirmed bounded (T=15000)"), Patch(fc="#c1272d", label="runaway"),
                          Patch(fc="#e8873a", label="marginal / not confirmed bounded"), Patch(fc="#d9d9d9", label="no data")],
                 fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False)

    # B peak rate
    im = ax[1].imshow(rate, origin="lower", aspect="auto", cmap="magma_r", vmin=60, vmax=480)
    for i in range(nk):
        for j in range(na):
            if np.isfinite(rate[i, j]):
                ax[1].text(j, i, f"{rate[i, j]:.0f}", ha="center", va="center", fontsize=7,
                           color="white" if rate[i, j] > 300 else "black")
    ax[1].set_xticks(range(na)); ax[1].set_xticklabels(xt); ax[1].set_yticks(range(nk)); ax[1].set_yticklabels(yt)
    ax[1].set_xlabel("pool strength $\\alpha_G$"); ax[1].set_ylabel("depletion rate $k_q$")
    ax[1].set_title("B  peak rate (Hz/neuron)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04).set_label("Hz/neuron")

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out = OUT / "fig_m4_phase_diagram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(OUT / "fig_m4_phase_diagram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    nb = (vg == 0).sum(); nr = (vg == 2).sum(); nu = (vg == 1).sum(); nd = (vg == 3).sum()
    print(f"cells: bounded={nb} runaway={nr} uncertain={nu} no-data={nd}")
    for i, kq in enumerate(KQS):
        print(f"  k_q={kq}: " + " ".join(f"{a:g}:{['bnd','?','run','--'][vg[i,j]]}" for j, a in enumerate(ALPHAS)))


if __name__ == "__main__":
    main()
