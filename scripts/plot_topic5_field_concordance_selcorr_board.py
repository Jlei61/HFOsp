#!/usr/bin/env python3
"""Topic 5 — field-concordance SELECTION-CORRECTED board (formal pass, not a screen).

Companion to the best-only screen board, but the null repeats the best-of-N candidate selection
(max-statistic family-wise null; src/topic5_field_selcorr.py). Per subject: the observed best
candidate's real |r| vs the SELECTION-CORRECTED null (max over candidates per draw); filled =
beats the corrected null at p<0.05. This is a formal pass rate (corrected for picking the best of
N candidates), unlike the screen board.

Reads results/topic5_ictal_recruitment/axis_alignment/field_concordance_selcorr/per_subject/*.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[1]
SELC = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/field_concordance_selcorr"
OUT = _ROOT / "results/topic5_ictal_recruitment/axis_alignment/figures/field_concordance"
COL = {"bb maxAB": "#1f4e79", "HFA maxAB": "#c0392b", "bb broad": "#5b9bd5", "HFA broad": "#e8a33d"}


def _rows(dataset):
    rows = []
    for f in sorted((SELC / "per_subject").glob("*.json")):
        d = json.load(open(f))
        if d.get("status") != "ok":
            continue
        sid = d["subject_id"]
        if dataset and not sid.startswith(dataset):
            continue
        s = d["selcorr"]
        rows.append({"sid": sid, "best": s["best_candidate"], "obs": s["observed_max"],
                     "p95": s["null_max_p95"], "p": s["p_selcorr"], "pass": s["pass_selcorr"],
                     "ncand": s["n_candidates"], "reals": s.get("per_candidate_real", {})})
    rows.sort(key=lambda r: r["obs"] - r["p95"], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="", help="subject prefix filter ('' = all)")
    args = ap.parse_args()
    rows = _rows(args.dataset)
    if not rows:
        print("no selcorr rows"); return
    n = len(rows); npass = sum(r["pass"] for r in rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * n + 2.3))
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, rows):
        col = COL.get(r["best"], "0.4")
        # faint: all candidate real values (the selection pool)
        for c, v in r["reals"].items():
            ax.scatter([v], [y], s=22, facecolors="none", edgecolors=COL.get(c, "0.6"), alpha=0.45, zorder=2)
        # selection-corrected threshold = max-over-candidates null p95
        ax.plot([r["p95"], r["p95"]], [y - 0.34, y + 0.34], color="0.3", lw=1.6, zorder=2)
        # observed best (filled if beats corrected null at p<0.05)
        face = col if r["pass"] else "white"
        ax.scatter([r["obs"]], [y], s=140, facecolors=face, edgecolors=col, linewidths=1.9, zorder=4)
        ax.text(-0.012, y, r["sid"].replace("epilepsiae_", "E").replace("yuquan_", "Y-"),
                ha="right", va="center", fontsize=9,
                fontweight="bold" if r["pass"] else "normal", color="black" if r["pass"] else "0.5")
        ax.text(1.005, y, f"{r['best']}  (p={r['p']:.3f}, {r['ncand']} cand)",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8, color=col)
    ax.set_yticks([]); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("field alignment |r|   (filled = observed best beats the SELECTION-CORRECTED null "
                  "[max over candidates] at p<0.05;  black tick = that corrected-null p95)")
    ds = args.dataset or "all"
    ax.set_title(f"{ds} field concordance — SELECTION-CORRECTED (formal): {npass}/{n} pass (p<0.05)", fontsize=12)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor=c, markersize=10, label=k)
               for k, c in COL.items()]
    handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="0.3",
                           markersize=10, label="open = fails corrected null"),
                plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="0.6",
                           markersize=6, label="faint = each candidate's real |r|")]
    ax.legend(handles=handles, loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.text(0.5, 0.005, "FORMAL pass: the null repeats 'take the best of N candidates' every draw (max-statistic "
             "family-wise). Stricter than the best-only SCREEN. yuquan = eligible subjects only (broad candidates).",
             ha="center", fontsize=7.5, color="0.4")
    fig.tight_layout(rect=(0.04, 0.03, 0.82, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"field_concordance_selcorr_board_{ds}.png"
    fig.savefig(fp, dpi=140); plt.close(fig)
    print(f"[done] {npass}/{n} pass selection-corrected -> {fp}")


if __name__ == "__main__":
    main()
