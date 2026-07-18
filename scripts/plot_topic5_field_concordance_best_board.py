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
import textwrap
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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


def plot_or_margin_board(rows, candidates, output_path, title_text, *,
                         xlabel=None, candidate_title=None, footer_text=None,
                         save_pdf=False, open_label="open square = fail"):
    """The original ``field_concordance_or_margin_board_prototype.png`` painter.

    This is the prototype's plotting block promoted unchanged to a callable function.  Callers
    provide only the candidate records and labels; the layout, marks and visual grammar remain
    the same as the accepted prototype.
    """
    fig = plt.figure(figsize=(10.8, max(8.2, 0.36 * len(rows) + 2.5)))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.0, 1.25], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[0, 1], sharey=ax)
    y = np.arange(len(rows))

    for i, row in enumerate(rows):
        if row["or_pass"]:
            ax.axhspan(i - 0.45, i + 0.45, color="#eef7ee", zorder=0)
            axh.axhspan(i - 0.45, i + 0.45, color="#eef7ee", zorder=0)
        else:
            ax.axhspan(i - 0.45, i + 0.45, color="#f6f6f6", zorder=0)
            axh.axhspan(i - 0.45, i + 0.45, color="#f6f6f6", zorder=0)

    for i, row in enumerate(rows):
        for candidate in candidates:
            value = row["vals"].get(candidate["name"])
            if value is not None:
                ax.plot([value["margin"], value["margin"]], [i - 0.18, i + 0.18],
                        color=candidate["color"], lw=1.5, alpha=0.28, zorder=1)
        ax.scatter(row["margin"], i, s=68,
                   color=row["color"] if row["or_pass"] else "white",
                   edgecolor=row["color"], lw=1.8, zorder=3)

    ax.axvline(0, color="0.25", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([(row["subject_id"].replace("epilepsiae_", "E")
                         .replace("yuquan_", "Y:")) for row in rows], fontsize=8.5)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if row["or_pass"] else "normal")
        tick.set_color("black" if row["or_pass"] else "0.55")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel or "best field-alignment margin: real |r| − channel-null p95")
    ax.set_title(title_text, pad=12)
    max_abs = max(abs(row["margin"]) for row in rows) + 0.04
    ax.set_xlim(-max(0.18, max_abs * 0.45), max(0.28, max_abs))
    ax.grid(axis="x", color="0.88", lw=0.8)

    labels = [candidate["name"] for candidate in candidates]
    axh.set_xlim(-0.5, len(labels) - 0.5)
    axh.set_xticks(range(len(labels)))
    axh.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    axh.tick_params(axis="y", left=False, labelleft=False)
    axh.set_title(candidate_title or "which candidate passes", fontsize=10, pad=12)
    for i, row in enumerate(rows):
        for j, candidate in enumerate(candidates):
            value = row["vals"].get(candidate["name"])
            if value is None:
                axh.scatter(j, i, marker="s", s=65, facecolor="white",
                            edgecolor="0.85", lw=0.8)
            elif value["pass"]:
                axh.scatter(j, i, marker="s", s=92, facecolor=candidate["color"],
                            edgecolor="white", lw=0.8)
            else:
                axh.scatter(j, i, marker="s", s=65, facecolor="white",
                            edgecolor=candidate["color"], lw=1.3, alpha=0.8)
    for spine in ("top", "right", "left"):
        axh.spines[spine].set_visible(False)
    axh.spines["bottom"].set_color("0.6")

    legend = [Patch(facecolor=candidate["color"], edgecolor="white", label=candidate["name"])
              for candidate in candidates]
    legend.append(Patch(facecolor="white", edgecolor="0.35", label=open_label))
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=True)
    footer = footer_text or (
        "Display rule: one main point per subject = best candidate; formal OR claims still "
        "need OR selection repeated in the null."
    )
    fig.text(0.012, 0.018, textwrap.fill(footer, width=145), fontsize=8, color="0.35")
    # Shared-y axes plus the figure-level footer trigger a known benign
    # tight_layout warning; the accepted board geometry is visually audited.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes")
        fig.tight_layout(rect=[0, 0.065, 1, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    if save_pdf:
        fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)
    return output_path


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
