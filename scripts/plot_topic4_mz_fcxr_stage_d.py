"""Plot the FCXR Stage D frozen fast-branch map (parameter-scan DIAGNOSTIC, not the 4-column mechanism figure).

Two panels, each one independent question (figure discipline):
  A) persistence: longest contiguous high activity vs the failure coordinate D -> is there a persistent
     high branch anywhere? (threshold = 1000ms >> the ~12ms interictal event)
  B) excitability: end-of-run activity vs D -> does failure raise activity, and does saturation keep it bounded?

Usage: python scripts/plot_topic4_mz_fcxr_stage_d.py [branch_map.json]  (default: newest grid run)
"""
from __future__ import annotations

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                       "fast_slow_dynamics", "figures")
RUNS = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                    "fast_slow_dynamics", "runs")
UNSAT_TRANSITION_D = 0.087   # sharp cliff on the UNSATURATED slow-fast-transition line (context marker)
SLOTS = [("low", None, "#4c72b0", "o"),        # (slot, kick-index into j["kicks"] or None, color, marker)
         ("high1", 0, "#dd8452", "s"),
         ("high2", 1, "#c44e52", "^")]
SHORT = {"LOW_ONLY": "low", "METASTABLE_TRANSIENT": "metastable", "REFRACTORY_CEILING": "ceiling",
         "FINITE_HIGH_FIXED": "finite-high", "FINITE_HIGH_ORBIT": "finite-orbit", "FINITE_HIGH": "finite-high",
         "BISTABLE": "bistable", "NUMERICAL_UNSAFE": "unsafe", "UNRESOLVED": "unresolved"}


def _latest_branch_map():
    cands = sorted(glob.glob(os.path.join(RUNS, "*grid*", "branch_map.json")))
    if not cands:
        raise SystemExit("no grid branch_map.json found; run `grid --confirm-run` first")
    return cands[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _latest_branch_map()
    j = json.load(open(path))
    is_smoke = bool(j.get("smoke_only")) or float(j.get("T1", 0)) < float(j["thresholds"]["HIGH_MS"])  # P0 gate
    rows = j["base_rows"]
    D_grid = sorted(set(r["D"] for r in rows))
    per_D = {p["D"]: p for p in j["per_D"]}
    HIGH_OCC = float(j["thresholds"]["HIGH_OCC"])
    kicks = j.get("kicks", [3.0, 12.0])

    def _slabel(ki):
        return "native low (no kick)" if ki is None else f"kicked high (kick {kicks[ki]:g})"

    def series(slot, field):
        by = {r["D"]: r.get(field, np.nan) for r in rows if r["slot"] == slot}
        return np.array([by.get(D, np.nan) for D in D_grid], float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.9))

    # Panel A: persistence -- smoothed-envelope still-elevated-at-end occupancy vs D (a transient decays -> low)
    for slot, ki, c, m in SLOTS:
        axA.plot(D_grid, series(slot, "env_end_occ"), m + "-", color=c, label=_slabel(ki), ms=7, lw=1.6)
    axA.set_ylim(-0.03, 1.08)
    axA.axhline(HIGH_OCC, ls="--", color="k", lw=1.2)
    axA.text(D_grid[-1], HIGH_OCC, f"persistent-high threshold (occupancy {HIGH_OCC:.2f})", fontsize=9,
             ha="right", va="bottom")
    axA.axvline(UNSAT_TRANSITION_D, ls=":", color="0.5", lw=1.2)
    axA.text(UNSAT_TRANSITION_D, 0.60, " unsaturated\n runaway onset\n (D≈0.087)", fontsize=8, color="0.4", va="center")
    axA.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axA.set_ylabel("envelope still-high-at-end occupancy")
    axA.set_title("A. Persistence — still elevated at window end?")
    axA.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel B: end-of-run activity level vs D
    for slot, ki, c, m in SLOTS:
        axB.plot(D_grid, series(slot, "end_rate_hz"), m + "-", color=c, label=_slabel(ki), ms=7, lw=1.6)
    axB.axvline(UNSAT_TRANSITION_D, ls=":", color="0.5", lw=1.2)
    axB.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axB.set_ylabel("end-of-run mean firing rate  (Hz)")
    axB.set_title("B. Excitability — activity level (bounded)")
    axB.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # per-D verdict labels inside panel A's upper region (short forms; avoids x-ticks + title)
    for D in D_grid:
        lab = per_D.get(D, {}).get("D_label", "?")
        axA.text(D, 0.86, SHORT.get(lab, lab.lower()), rotation=90, fontsize=6.5, ha="center", va="center", color="0.45")

    verdict = j.get("verdict", "")
    short = ("SMOKE_ONLY — plumbing validation, NOT a scientific verdict" if is_smoke else
             ("CLEAN NO-GO: saturation bounds the transient, no persistent high branch" if "NO-GO" in verdict
              else verdict[:80]))
    fig.suptitle(f"FCXR-RC1 frozen fast-branch map (seed {j['seed']}, dt={j['dt']}, T1={j['T1']:.0f}ms)  —  {short}",
                 fontsize=11)
    fig.text(0.5, 0.005, "parameter-scan diagnostic — not the 4-column mechanism figure", ha="center",
             fontsize=7.5, color="0.5")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = "_SMOKE_frozen_branch_map_DO_NOT_USE" if is_smoke else "frozen_branch_map"   # never canonicalize a smoke run
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), dpi=150)
    tag = "SMOKE (T1<HIGH_MS) — NOT canonical" if is_smoke else "canonical"
    print(f"wrote {OUT_DIR}/{stem}.png  [{tag}]  (from {os.path.basename(os.path.dirname(path))})")


if __name__ == "__main__":
    main()
