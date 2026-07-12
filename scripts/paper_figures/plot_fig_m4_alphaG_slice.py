"""M4 alpha_G slice at k_q=0.10 (Topic 4) — the CORRECTED phase diagram along pool strength, using the
LONGEST-T verdict available per cell (the T=1000 sweep mislabels delayed-runaways as survivors, so aG>=12
uses the T=5000/15000 confirm/delay/longconfirm runs). Shows the non-monotonic bound window: runaway
(aG<=8) -> delayed-runaway (aG12) -> BOUNDED (aG16, 3/4 seeds) -> overshoot delayed-runaway (aG20/24).

3 panels (§7, each an independent view of the same non-monotonic structure):
  A max sustained rate (Hz) + 120 runaway line   -> the intensity dip at aG16 (below runaway)
  B runaway onset (ms) or 'none'                  -> the delay curve + the aG16 no-runaway gap
  C final sheet-mean q_I + 0.05 floor             -> q_I preserved above floor only in the aG16 window

aG16 shows the 4-seed spread (3 bounded + seed2 delayed-runaway). Reads all
results/topic4_m4_dynamic_{sweep,confirm,delay,longconfirm,multiseed}/ summaries. res rate_E is already Hz.
Framing: this is a non-runaway bounded attractor candidate window, NOT a full seizure cycle; broad-not-localized.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "results/paper-ready-figure/fig_m4_dynamic_qi/figures"
KQ = 0.10
ALPHAS = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0, 24.0]
RUNAWAY_HZ, Q_FLOOR = 120.0, 0.05


def _load(path):
    p = RES / path / "dynamic_qi_summary.json"
    return json.load(open(p))["rows"] if p.exists() else []


def _rows_for(alpha):
    """all rows across dirs at (k_q=0.10, alpha), tagged with T; pick longest-T for the headline."""
    dirs = ["topic4_m4_dynamic_sweep", "topic4_m4_dynamic_confirm", "topic4_m4_dynamic_delay",
            "topic4_m4_dynamic_longconfirm", "topic4_m4_dynamic_multiseed/seed2",
            "topic4_m4_dynamic_multiseed/seed3", "topic4_m4_dynamic_multiseed/seed4"]
    out = []
    for d in dirs:
        for r in _load(d):
            ag = 0.0 if not r["use_SG"] else r["alpha_G"]
            if abs(r["k_q"] - KQ) < 1e-6 and abs(ag - alpha) < 1e-6:
                out.append(r)
    return out


def _bounded(r):
    return r["runaway_ms"] is None and r["max_rate_hz"] < RUNAWAY_HZ and r["q_mean_final"] > Q_FLOOR + 0.01


def _status(r):
    if r is None:
        return "none"
    if r["runaway_ms"] is not None:
        return "runaway"
    if r["T"] >= 15000 and _bounded(r):
        return "bounded"                        # long-confirmed no-runaway + q_I above floor
    return "uncertain"                          # runaway_ms None but T<15000 -> could flip (aG20 did at T=15000)


COL = {"bounded": "#2e7d32", "runaway": "#c1272d", "uncertain": "#e8873a", "none": "0.7"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    head = []                                   # longest-T headline row per alpha
    seeds16 = []                                # all aG16 rows (multi-seed)
    for a in ALPHAS:
        rows = _rows_for(a)
        if not rows:
            head.append(None); continue
        best = max(rows, key=lambda r: r["T"])  # longest T = most trustworthy verdict
        head.append(best)
        if abs(a - 16.0) < 1e-6:
            seeds16 = [r for r in rows if r["T"] >= 15000]   # T=15000 seeds only
    ax_a = np.array(ALPHAS)
    maxr = np.array([h["max_rate_hz"] if h else np.nan for h in head])
    qend = np.array([h["q_mean_final"] if h else np.nan for h in head])
    run_ms = [h["runaway_ms"] if h else None for h in head]
    stat = [_status(h) for h in head]
    col = [COL[s] for s in stat]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    fig.suptitle("M4 pool-strength slice at $k_q$=0.10 (longest-T verdict per cell): a NARROW non-monotonic "
                 "BOUND window at $\\alpha_G$=16 — a non-runaway bounded attractor candidate, not a full seizure cycle",
                 fontsize=11.5, y=1.02)

    # A max rate
    ax[0].plot(ax_a, maxr, "-", color="0.6", lw=1, zorder=1)
    ax[0].scatter(ax_a, maxr, c=col, s=60, zorder=3, edgecolors="k", linewidths=0.5)
    ax[0].axhline(RUNAWAY_HZ, color="0.35", lw=0.9, ls="--")
    ax[0].text(ax_a[-1], RUNAWAY_HZ + 8, "runaway 120 Hz", ha="right", fontsize=8, color="0.35")
    if seeds16:
        s16 = [r["max_rate_hz"] for r in seeds16]
        ax[0].scatter([16.0] * len(s16), s16, facecolors="none", edgecolors="#1f6fb2", s=110, lw=1.4,
                      zorder=4, label=f"aG16 per-seed (n={len(s16)}: 3 bound + 1 runaway)")
        ax[0].legend(fontsize=7.5, loc="upper right")
    ax[0].set_title("A  max sustained rate (Hz/neuron)", fontsize=10.5, loc="left")
    ax[0].set_xlabel("pool strength $\\alpha_G$"); ax[0].set_ylabel("Hz/neuron")

    # B runaway onset
    from matplotlib.lines import Line2D
    for a, rm, s in zip(ax_a, run_ms, stat):
        if rm is None:
            ax[1].scatter([a], [0], marker="^", c=COL[s], s=80, zorder=3, edgecolors="k", linewidths=0.5)
        else:
            ax[1].scatter([a], [rm], c=COL[s], s=60, zorder=3, edgecolors="k", linewidths=0.5)
    ax[1].text(16, 3500, "aG16:\nno runaway\n(bounded, 3/4 seed)", ha="center", fontsize=8, color="#2e7d32")
    ax[1].text(24, 1200, "aG24:\nT=5000 only\n(uncertain)", ha="center", fontsize=7.5, color="#e8873a")
    ax[1].set_title("B  runaway onset (ms); ▲=no runaway in window", fontsize=10.5, loc="left")
    ax[1].set_xlabel("pool strength $\\alpha_G$"); ax[1].set_ylabel("runaway onset (ms)")
    ax[1].legend(handles=[Line2D([], [], marker="o", ls="none", color=COL["bounded"], label="bounded (T=15000)"),
                          Line2D([], [], marker="o", ls="none", color=COL["runaway"], label="runaway"),
                          Line2D([], [], marker="o", ls="none", color=COL["uncertain"], label="uncertain (T<15000)")],
                 fontsize=7.5, loc="upper left")

    # C q_end
    ax[2].plot(ax_a, qend, "-", color="0.6", lw=1, zorder=1)
    ax[2].scatter(ax_a, qend, c=col, s=60, zorder=3, edgecolors="k", linewidths=0.5)
    ax[2].axhline(Q_FLOOR, color="0.5", lw=0.9, ls=":")
    ax[2].text(ax_a[-1], Q_FLOOR + 0.015, "$q_{min}$ floor 0.05", ha="right", fontsize=8, color="0.5")
    ax[2].set_ylim(0, 0.35)
    ax[2].set_title("C  final sheet-mean $q_I$ (above floor = not drained)", fontsize=10.5, loc="left")
    ax[2].set_xlabel("pool strength $\\alpha_G$"); ax[2].set_ylabel("sheet-mean $q_I$ (end)")

    fig.text(0.5, 0.005, "Verdict COLOR is the robust claim. Continuous values are at each cell's longest T "
             "(aG≤8: T=1000, caught mid-transient; aG12/24: T=5000; aG16/20: T=15000) — so a short-T cell's "
             "q_end/rate is not its asymptote (e.g. aG8 ran away at 770ms but T=1000 stopped before q_I fully drained).",
             ha="center", fontsize=7.3, color="0.35")
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out = OUT / "fig_m4_alphaG_slice.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); fig.savefig(OUT / "fig_m4_alphaG_slice.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    for a, h in zip(ALPHAS, head):
        if h:
            print(f"  aG={a:4}: T={h['T']:.0f} verdict={'BOUNDED' if _bounded(h) else 'runaway'} "
                  f"runaway_ms={h['runaway_ms']} max={h['max_rate_hz']}Hz q_end={h['q_mean_final']}")


if __name__ == "__main__":
    main()
