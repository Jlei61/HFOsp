"""Plot the FCXR Stage D frozen fast-branch map (parameter-scan DIAGNOSTIC, not the 4-column mechanism figure).

Two panels, each one independent question (figure discipline):
  A) densification: whole-window above-band occupancy vs D -> does the elevated event train densify CONTINUOUSLY
     into the labelled "finite-high" region, or jump discretely? Main = median across the 3 ICs + min-max band
     (shows IC disagreement); faint = end-window occupancy (auxiliary persistence read); seed3 D=0.15 overlaid.
  B) boundedness: end-of-run activity vs D -> saturation keeps rate bounded (never runaway), and the metastable
     dropouts (rate -> 0 for the IC that decays) are seed/IC-dependent.

Usage: python scripts/plot_topic4_mz_fcxr_stage_d.py [branch_map.json]  (default: assembled workpoint map)
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


def _latest_branch_map():
    assembled = os.path.join(os.path.dirname(OUT_DIR), "branch_map.json")   # workpoint assembly output
    if os.path.exists(assembled):
        return assembled
    cands = sorted(glob.glob(os.path.join(RUNS, "*grid*", "branch_map.json")))
    if not cands:
        raise SystemExit("no branch_map.json found; run the assembler or `grid --confirm-run` first")
    return cands[-1]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else _latest_branch_map()
    j = json.load(open(path))
    is_smoke = bool(j.get("smoke_only"))
    rows = j.get("cells", [])
    D_grid = j.get("D_grid") or sorted(set(r["D"] for r in rows))
    band = float(j.get("band_hz", 0.0))
    kicks = j.get("kicks", [3.0, 12.0])
    HIGH_OCC = float(j["thresholds"]["HIGH_OCC"])
    seed3 = j.get("seed3_d015", {})

    def pick(D, slot):                                   # per (D, slot) take the longest observation window
        best = None
        for r in rows:
            if r["D"] == D and r["slot"] == slot and (best is None or r.get("T_ms", 0) > best.get("T_ms", 0)):
                best = r
        return best

    def _slabel(ki):
        return "native low (no kick)" if ki is None else f"kicked high (kick {kicks[ki]:g})"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 4.9))

    # ---- Panel A: whole-window above-band occupancy -- median + IC min-max band (continuous ramp vs jump) ----
    occ = {D: [pick(D, s)["roll_occ"] for s in ("low", "high1", "high2") if pick(D, s)] for D in D_grid}
    endo = {D: [pick(D, s)["roll_end_occ"] for s in ("low", "high1", "high2") if pick(D, s)] for D in D_grid}
    med = np.array([np.median(occ[D]) if occ[D] else np.nan for D in D_grid])
    lo = np.array([np.min(occ[D]) if occ[D] else np.nan for D in D_grid])
    hi = np.array([np.max(occ[D]) if occ[D] else np.nan for D in D_grid])
    endm = np.array([np.mean(endo[D]) if endo[D] else np.nan for D in D_grid])

    axA.fill_between(D_grid, lo, hi, color="#4c72b0", alpha=0.16, label="IC spread (min–max)")
    axA.plot(D_grid, med, "o-", color="#2a4d8f", lw=1.9, ms=6, label="median occupancy (3 ICs, whole 8 s)")
    axA.plot(D_grid, endm, ":", color="0.55", lw=1.3, label="end-window occupancy (aux)")
    if seed3:
        ys = [v["roll_occ"] for v in seed3.values()]
        axA.plot([0.15] * len(ys), ys, "D", mfc="none", mec="#c44e52", mew=1.7, ms=9,
                 label="seed3 D=0.15 (per IC)")
    axA.axhline(HIGH_OCC, ls="--", color="k", lw=1.0)
    axA.text(D_grid[0], HIGH_OCC + 0.015,
             f"operational threshold ({HIGH_OCC:.1f}) — not a dynamical breakpoint", fontsize=8, va="bottom")
    axA.axvline(UNSAT_TRANSITION_D, ls=":", color="0.6", lw=1.1)
    axA.text(UNSAT_TRANSITION_D + 0.001, 0.93, "unsat. runaway\nonset (D≈0.087)", fontsize=7.5, color="0.45", va="top")
    axA.set_ylim(-0.03, 1.05)
    axA.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axA.set_ylabel(f"fraction of window above interictal band ({band:.1f} Hz)")
    axA.set_title("A. Densification — continuous ramp, not a discrete jump")
    axA.legend(loc="upper left", fontsize=7.6, framealpha=0.9)

    # ---- Panel B: end-of-run rate -- bounded (no runaway) + seed/IC-dependent metastable dropouts ----
    for slot, ki, c, m in SLOTS:
        y = np.array([(pick(D, slot) or {}).get("end_rate_hz", np.nan) for D in D_grid], float)
        axB.plot(D_grid, y, m + "-", color=c, label=_slabel(ki), ms=6, lw=1.5)
    if seed3:
        ys = [v["end_rate_hz"] for v in seed3.values()]
        axB.plot([0.15] * len(ys), ys, "D", mfc="none", mec="0.25", mew=1.6, ms=9, label="seed3 D=0.15 (per IC)")
    axB.axvline(UNSAT_TRANSITION_D, ls=":", color="0.6", lw=1.1)
    axB.set_xlabel("failure coordinate  D  (frozen mean depletion)")
    axB.set_ylabel("end-of-run mean firing rate  (Hz)")
    axB.set_title("B. Boundedness — capped ~10–12 Hz, no runaway")
    axB.legend(loc="upper left", fontsize=7.6, framealpha=0.9)

    fig.suptitle("FCXR-RC1 frozen fast-branch map (seed1, dt=0.05) — no robust independent high branch",
                 fontsize=11.5)
    fig.text(0.5, 0.005, "parameter-scan diagnostic — not the 4-column mechanism figure", ha="center",
             fontsize=7.5, color="0.5")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = "_SMOKE_frozen_branch_map_DO_NOT_USE" if is_smoke else "frozen_branch_map"
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{stem}.{ext}"), dpi=150)
    tag = "SMOKE (T1<HIGH_MS) — NOT canonical" if is_smoke else "canonical"
    print(f"wrote {OUT_DIR}/{stem}.png  [{tag}]  (from {os.path.basename(os.path.dirname(path))})")


if __name__ == "__main__":
    main()
