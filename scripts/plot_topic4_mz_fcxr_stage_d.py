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
SLOTS = [("low", "native low (no kick)", "#4c72b0", "o"),
         ("high1", "kicked high (kick 3)", "#dd8452", "s"),
         ("high2", "kicked high (kick 6)", "#c44e52", "^")]
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
    rows = j["base_rows"]
    D_grid = sorted(set(r["D"] for r in rows))
    per_D = {p["D"]: p for p in j["per_D"]}
    HIGH_MS = float(j["thresholds"]["HIGH_MS"])

    def series(slot, field):
        by = {r["D"]: r[field] for r in rows if r["slot"] == slot}
        return np.array([by.get(D, np.nan) for D in D_grid], float)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.9))

    # Panel A: persistence (longest contiguous elevation) vs D
    for slot, label, c, m in SLOTS:
        axA.plot(D_grid, series(slot, "high_duration_ms"), m + "-", color=c, label=label, ms=7, lw=1.6)
    axA.set_ylim(bottom=-HIGH_MS * 0.03, top=HIGH_MS * 1.12)          # headroom so the threshold clears the title
    axA.axhline(HIGH_MS, ls="--", color="k", lw=1.2)
    axA.text(D_grid[-1], HIGH_MS, f"persistent-high threshold ({HIGH_MS:.0f} ms)", fontsize=9,
             ha="right", va="bottom")
    axA.axvline(UNSAT_TRANSITION_D, ls=":", color="0.5", lw=1.2)
    axA.text(UNSAT_TRANSITION_D, HIGH_MS * 0.55, " unsaturated\n runaway onset\n (D≈0.087)",
             fontsize=8, color="0.4", va="center")
    axA.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axA.set_ylabel("longest continuous high activity  (ms)")
    axA.set_title("A. Persistence — is there a high branch?")
    axA.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel B: end-of-run activity vs D
    for slot, label, c, m in SLOTS:
        axB.plot(D_grid, series(slot, "end_rate_hz"), m + "-", color=c, label=label, ms=7, lw=1.6)
    axB.axvline(UNSAT_TRANSITION_D, ls=":", color="0.5", lw=1.2)
    axB.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axB.set_ylabel("end-of-run mean firing rate  (Hz)")
    axB.set_title("B. Excitability — activity rises but stays bounded")
    axB.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # per-D verdict labels inside panel A's empty upper region (short forms; avoids x-ticks + title)
    for D in D_grid:
        lab = per_D.get(D, {}).get("D_label", "?")
        axA.text(D, HIGH_MS * 0.72, SHORT.get(lab, lab.lower()), rotation=90, fontsize=6.5,
                 ha="center", va="center", color="0.45")

    verdict = j.get("verdict", "")
    short = "CLEAN NO-GO: saturation bounds the transient, no persistent high branch" if "NO-GO" in verdict else verdict[:80]
    fig.suptitle(f"FCXR-RC1 frozen fast-branch map (seed {j['seed']}, dt={j['dt']}, T1={j['T1']:.0f}ms)  —  {short}",
                 fontsize=11)
    fig.text(0.5, 0.005, "parameter-scan diagnostic — not the 4-column mechanism figure", ha="center",
             fontsize=7.5, color="0.5")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"frozen_branch_map.{ext}"), dpi=150)
    print(f"wrote {OUT_DIR}/frozen_branch_map.png  (from {os.path.basename(os.path.dirname(path))})")


if __name__ == "__main__":
    main()
