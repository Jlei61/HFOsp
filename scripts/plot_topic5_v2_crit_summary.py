#!/usr/bin/env python
"""Paper-grade Phase-2 (criticality/state layer) result figure.

Two independent questions, one panel each (CLAUDE.md §7):
  A. Does the preictal state (susceptibility rise / leading dynamic mode / avalanche
     forward flow) align with the fixed interictal HFO propagation axis? -> near zero = weak.
  B. Is the avalanche cascade a real forward flow along that axis, or self-persistence?
     Forward displacement (flow-sensitive) vs rank-coupling (self-persistence-prone).

Observed statistics only (final; null-independent). EXPLORATORY; broad/narrow never pooled.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
BASE = _ROOT / "results/topic5_ictal_recruitment/v2_criticality"
OUTDIR = BASE / "figures"
COHORT_COLOR = {"broad": "#3b6fb0", "narrow": "#c0603a"}
PRIMARY_SUSC = "line_length_rate"


def _read(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _collect(axis_set):
    """Per-leg maps (robust to a leg's CSV being absent, e.g. finals still running)."""
    base = BASE / axis_set
    susc = {r["subject"]: _f(r["K_signed_oriented"])
            for r in _read(base / "phase2_susceptibility_subject.csv")
            if r.get("feature") == PRIMARY_SUSC and r.get("status") == "ok"}
    dyn = {r["subject"]: _f(r["M_loading_spearman"])
           for r in _read(base / "phase2_dynamics_subject.csv") if r.get("status") == "ok"}
    aval = _read(base / "phase2_avalanche_subject.csv")
    fwd = {r["subject"]: _f(r["atm_forward_displacement"]) for r in aval if r.get("status") == "ok"}
    coup = {r["subject"]: _f(r["atm_rank_coupling_spearman"]) for r in aval if r.get("status") == "ok"}
    return {"K": susc, "M": dyn, "fwd": fwd, "coup": coup}


def _vals(d):
    return np.array([v for v in d.values() if np.isfinite(v)], dtype=float)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = {ax: _collect(ax) for ax in ("broad", "narrow")}

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    rng = np.random.default_rng(0)

    # ---- Panel A: signed alignment of 3 state legs to G_HFO ----
    legs = ["Susceptibility\nrise", "Leading\ndynamic mode", "Avalanche\nforward flow"]
    keys = ["K", "M", "fwd"]
    axA.axhline(0, color="0.6", lw=1.0, ls="--", zorder=0)
    for xi, key in enumerate(keys):
        for off, ax in zip((-0.16, 0.16), ("broad", "narrow")):
            vals = _vals(data[ax][key])
            if vals.size == 0:
                continue
            jit = rng.uniform(-0.06, 0.06, vals.size)
            axA.scatter(np.full(vals.size, xi + off) + jit, vals, s=34,
                        color=COHORT_COLOR[ax], alpha=0.8, edgecolor="white", linewidth=0.5, zorder=3)
            med = float(np.median(vals))
            axA.plot([xi + off - 0.11, xi + off + 0.11], [med, med],
                     color=COHORT_COLOR[ax], lw=2.6, zorder=4)
    axA.set_xticks(range(len(legs)))
    axA.set_xticklabels(legs, fontsize=9)
    axA.set_ylim(-1.02, 1.02)
    axA.set_ylabel("signed alignment to interictal HFO axis\n(+1 early end · −1 late end · 0 none)", fontsize=9)
    axA.set_title("A  Preictal state does not consistently align with the HFO axis", fontsize=10, loc="left")
    axA.text(0.5, -0.95, "cohort medians (thick bars) sit near zero → weak / no consistent projection",
             fontsize=8, style="italic", color="0.3")

    # ---- Panel B: forward flow vs self-persistence ----
    axB.axhline(0, color="0.6", lw=1.0, ls="--", zorder=0)
    for ax in ("broad", "narrow"):
        coup, fwd = data[ax]["coup"], data[ax]["fwd"]
        paired = [(coup[s], fwd[s]) for s in sorted(set(coup) & set(fwd))
                  if np.isfinite(coup[s]) and np.isfinite(fwd[s])]
        if not paired:
            continue
        x, y = np.array([p[0] for p in paired]), np.array([p[1] for p in paired])
        axB.scatter(x, y, s=44, color=COHORT_COLOR[ax], alpha=0.85,
                    edgecolor="white", linewidth=0.5, label=f"{ax} (n={len(paired)})", zorder=3)
    axB.set_xlim(-0.05, 1.0)
    axB.set_ylim(-0.4, 0.4)
    axB.set_xlabel("rank-coupling Spearman\n(high even under pure self-persistence)", fontsize=9)
    axB.set_ylabel("forward displacement\n(net flow along the HFO axis)", fontsize=9)
    axB.set_title("B  High self-persistence, ~zero forward flow", fontsize=10, loc="left")
    axB.text(0.5, 0.34, "cascades recur on the same contacts (high x)\nbut do not travel along the axis (y≈0)",
             fontsize=8, style="italic", color="0.3", ha="center")
    axB.legend(loc="lower left", fontsize=8, frameon=False)

    fig.suptitle("Topic 5 V2 Phase 2 — preictal criticality/state layer vs interictal HFO geometry "
                 "(EXPLORATORY; observed statistics)", fontsize=10.5, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUTDIR / "phase2_state_layer_alignment.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"[fig] -> {out}")
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(_ROOT))
    main()
