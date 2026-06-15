#!/usr/bin/env python3
"""Topic 5 A-line NULL LADDER — the one figure that shows the conclusion is HIERARCHICAL.

The A-line result is NOT 'every metric beats every null'. It is a ladder of increasingly strict
nulls; each activation metric survives up to a different rung:
  rung 1  channel        removes: any random spatial structure        -> is there a COARSE axis?
  rung 2  within-shaft   removes: shaft / gross-anatomy effect        -> finer than the shaft?
  rung 3  anchor-matched removes: baseline-activity confound          -> not just 'already active'?
  rung 4  joint          removes: shaft + activity together (strict)  -> a fine, activity-free axis?
Main conclusion = the COARSE shared network axis (rung 1, all metrics, robust). The fine,
activity-independent axis (rung 4) is reached ONLY by fast-activity (HFA) — a mechanism-sensitive
probe, not the pre-registered primary. The grid makes that altitude difference legible.

Reads results/.../axis_alignment_FINAL.json (FDR q per metric x null). Cells colored by FDR pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
FINAL = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/axis_alignment_FINAL.json"
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures"

NULLS = ["channel", "within_shaft", "anchor_matched", "joint"]      # ascending stringency
NULL_LABEL = {"channel": "channel shuffle", "within_shaft": "within-shaft shuffle",
              "anchor_matched": "anchor-matched shuffle", "joint": "joint shuffle"}
NULL_REMOVES = {"channel": "any random spatial structure",
                "within_shaft": "shaft / gross-anatomy effect",
                "anchor_matched": "baseline-activity confound",
                "joint": "shaft + activity (strictest)"}
NULL_TESTS = {"channel": "is there a coarse network axis?",
              "within_shaft": "finer than the electrode shaft?",
              "anchor_matched": "not just 'already-active' contacts?",
              "joint": "a fine, activity-independent axis?"}
METRICS = ["broadband", "hfa", "ramp", "ei"]
METRIC_LABEL = {"broadband": "broadband\n(PRIMARY)", "hfa": "fast activity\n(HFA, sens.)",
                "ramp": "ramp slope\n(sens.)", "ei": "EI-like\n(expl.)"}


def _grid(table, B):
    q = {}
    for r in table:
        if r["B"] == B:
            q[(r["metric"], r["null"])] = r
    return q


def plot(B=1000):
    data = json.load(open(FINAL))
    q = _grid(data["table"], B)
    nrow, ncol = len(NULLS), len(METRICS)
    fig, ax = plt.subplots(figsize=(11.2, 6.6), constrained_layout=True)
    # rows top->bottom = strict->coarse so the 'ladder' climbs upward visually
    rows_top_down = list(reversed(NULLS))
    for ri, nul in enumerate(rows_top_down):
        y = nrow - 1 - ri
        for ci, met in enumerate(METRICS):
            rec = q.get((met, nul))
            if rec is None:
                continue
            qv = rec.get("wilcoxon_fdr_q")
            npass, n = rec.get("n_pass"), rec.get("n")
            passed = qv is not None and qv < 0.05
            face = "#2e7d32" if passed else "#e8e8e8"
            txt_c = "white" if passed else "0.45"
            ax.add_patch(Rectangle((ci, y), 1, 1, facecolor=face, edgecolor="white", lw=2))
            mark = "PASS" if passed else "—"
            ax.text(ci + 0.5, y + 0.62, mark, ha="center", va="center",
                    fontsize=10.5, fontweight="bold", color=txt_c)
            ax.text(ci + 0.5, y + 0.34, f"q={qv:.3f}\n{npass}/{n} subj", ha="center", va="center",
                    fontsize=8.0, color=txt_c)
    # metric headers
    for ci, met in enumerate(METRICS):
        ax.text(ci + 0.5, nrow + 0.12, METRIC_LABEL[met], ha="center", va="bottom",
                fontsize=10, fontweight=("bold" if met == "broadband" else "normal"))
    # null rows: left = name + removes, right = tests-for
    for ri, nul in enumerate(rows_top_down):
        y = nrow - 1 - ri
        rung = NULLS.index(nul) + 1
        ax.text(-0.12, y + 0.5, f"rung {rung}\n{NULL_LABEL[nul]}", ha="right", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(-0.12, y + 0.16, f"removes: {NULL_REMOVES[nul]}", ha="right", va="center",
                fontsize=7.6, color="0.4")
        ax.text(ncol + 0.12, y + 0.5, NULL_TESTS[nul], ha="left", va="center",
                fontsize=8.6, color="0.3", style="italic")
    ax.set_xlim(-3.0, ncol + 2.6)
    ax.set_ylim(0, nrow + 0.7)
    ax.axis("off")
    fig.suptitle("A-line null ladder — the conclusion is a HIERARCHY, not 'all metrics pass all nulls'\n"
                 "coarse shared network axis = robust everywhere (rung 1); a fine activity-independent "
                 "axis (rung 4) is reached ONLY by fast activity (HFA)",
                 fontsize=11.5, y=1.02)
    fig.text(0.5, -0.02, f"green = beats this null after BH-FDR (q<0.05), Wilcoxon over Epilepsiae "
             f"n=18; B={B} null draws · broadband is the pre-registered primary, HFA/ramp/EI are "
             f"sensitivity/exploratory", ha="center", fontsize=8.2, color="0.35")
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"axis_alignment_null_ladder_B{B}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    print(f"wrote {plot(args.B)}")


if __name__ == "__main__":
    main()
