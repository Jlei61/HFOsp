#!/usr/bin/env python3
"""Topic 5 — field-concordance "best-only" board (exploratory: how many subjects are concordant).

A leaner version of the OR-over-4 field-bias board prototype: instead of drawing all four
candidate nulls per subject, show ONLY each subject's BEST candidate — its real field-alignment
|r| and the channel-shuffle null (median -> p95) OF THAT BEST CANDIDATE. Filled marker = real |r|
beats its own best-candidate null p95; open = fails. Color = which of the four candidates is best.

Candidates (the A-line interictal field-bias variants, real|r| vs ictal-onset gradient):
  bb maxAB / HFA maxAB / bb broad / HFA broad   (from axis_alignment_*_{max_ab_B1000,broad_B2000}.json)

This is EXPLORATORY: "best of four" is a selection over candidates, so the per-subject pass is
NOT corrected for that selection (a formal OR claim needs the OR-selection repeated inside the
null). It answers "how many subjects show ANY field concordance" as a screen, not a verdict.
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
ALIGN = _ROOT / "results/topic5_ictal_recruitment/axis_alignment"
OUT = ALIGN / "figures/field_concordance"
CANDIDATES = [("bb maxAB", "axis_alignment_broadband_max_ab_B1000.json", "#1f4e79"),
              ("HFA maxAB", "axis_alignment_hfa_max_ab_B1000.json", "#c0392b"),
              ("bb broad", "axis_alignment_broadband_broad_B2000.json", "#5b9bd5"),
              ("HFA broad", "axis_alignment_hfa_broad_B2000.json", "#e8a33d")]


def _load():
    out = {}
    for name, fn, col in CANDIDATES:
        f = ALIGN / fn
        if not f.exists():
            out[name] = ({}, col); continue
        j = json.load(open(f))
        out[name] = ({r["subject_id"]: r for r in j.get("per_subject", []) if r.get("status") == "ok"}, col)
    return out


def _best_per_subject(data, dataset):
    subs = sorted(set().union(*[set(d) for d, _ in data.values()]))
    rows = []
    for s in subs:
        if dataset and not s.startswith(dataset):
            continue
        cands = {}
        for name, (d, col) in data.items():
            r = d.get(s)
            if r and r.get("channel_null_p95") is not None and r.get("real_median_abs_corr") is not None:
                cands[name] = {"real": float(r["real_median_abs_corr"]),
                               "p95": float(r["channel_null_p95"]),
                               "med": float(r.get("channel_null_median", np.nan)),
                               "margin": float(r["real_median_abs_corr"] - r["channel_null_p95"]),
                               "n_seizures": r.get("n_seizures"), "color": col}
        if not cands:
            continue
        best = max(cands, key=lambda k: cands[k]["margin"])
        rows.append({"subject_id": s, "best": best, **cands[best],
                     "or_pass": any(c["margin"] > 0 for c in cands.values()),
                     "best_pass": cands[best]["margin"] > 0,
                     "n_candidates": len(cands)})
    rows.sort(key=lambda r: r["margin"], reverse=True)
    return rows


def plot_board(rows, dataset):
    n = len(rows)
    n_pass = sum(r["best_pass"] for r in rows)
    fig, ax = plt.subplots(figsize=(10.5, 0.5 * n + 2.2))
    ys = np.arange(n)[::-1]
    for y, r in zip(ys, rows):
        col = r["color"]
        # null of the BEST candidate only: median -> p95 whisker + p95 threshold tick
        ax.plot([r["med"], r["p95"]], [y, y], color="0.6", lw=4, alpha=0.45, solid_capstyle="butt", zorder=1)
        ax.plot([r["p95"], r["p95"]], [y - 0.32, y + 0.32], color="0.35", lw=1.4, zorder=2)
        # real |r| point (filled = beats its own null p95)
        face = col if r["best_pass"] else "white"
        ax.scatter([r["real"]], [y], s=130, facecolors=face, edgecolors=col, linewidths=1.8, zorder=3)
        ax.text(-0.012, y, r["subject_id"].replace("epilepsiae_", "E").replace("yuquan_", "Y-"),
                ha="right", va="center", fontsize=9,
                fontweight="bold" if r["best_pass"] else "normal",
                color="black" if r["best_pass"] else "0.5")
        ax.text(1.005, y, f"{r['best']}  (Δ{r['margin']:+.2f})", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=8, color=col)
    ax.axvline(0, color="0.3", lw=0.8, ls=":")
    ax.set_yticks([]); ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("field alignment  |r|   (point = real best-candidate value;  grey bar = that candidate's "
                  "channel-null median→p95)")
    ds = dataset or "all"
    ax.set_title(f"{ds} field concordance — best-of-four, best-only (SCREEN): "
                 f"{n_pass}/{n} eligible-subset beat their own null", fontsize=12)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markeredgecolor=c,
                          markersize=10, label=nm) for nm, _, c in CANDIDATES]
    handles += [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor="0.3",
                           markersize=10, label="open = fails its own null")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    fig.text(0.5, 0.005, "EXPLORATORY SCREEN — eligible-subset count, NOT a cohort significance rate. each subject = its "
             "single best of N candidates; null shown is that candidate's. best-of-N is a SELECTION not corrected in the "
             "null → a formal OR/max pass needs 'pick best candidate' repeated inside every shuffle. (yuquan rows = the few "
             "eligible subjects, not a yuquan cohort; yuquan has only broad candidates.)",
             ha="center", fontsize=7.2, color="0.4")
    fig.tight_layout(rect=(0.04, 0.03, 0.86, 0.97))
    OUT.mkdir(parents=True, exist_ok=True)
    fp = OUT / f"field_concordance_best_only_board_{ds}.png"
    fig.savefig(fp, dpi=140); plt.close(fig)
    return fp, n_pass, n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="epilepsiae", help="subject_id prefix filter ('' = all)")
    args = ap.parse_args()
    data = _load()
    rows = _best_per_subject(data, args.dataset)
    fp, n_pass, n = plot_board(rows, args.dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"note": "best-of-N field concordance (best-only) — EXPLORATORY SCREEN, eligible-subset count NOT a cohort "
               "significance rate; best-of-N selection NOT corrected in the null (formal OR pass needs the selection "
               "repeated per shuffle). yuquan rows = eligible subjects only (broad candidates only), not a yuquan cohort.",
               "dataset": args.dataset, "n": n, "n_best_pass": n_pass,
               "rows": [{k: r[k] for k in ("subject_id", "best", "real", "p95", "margin",
                                           "best_pass", "or_pass", "n_candidates", "n_seizures")} for r in rows]},
              open(OUT / f"field_concordance_best_only_{args.dataset or 'all'}.json", "w"), indent=2)
    print(f"[done] {n_pass}/{n} beat their own best-candidate null -> {fp}")


if __name__ == "__main__":
    main()
