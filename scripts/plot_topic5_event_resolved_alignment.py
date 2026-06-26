#!/usr/bin/env python3
"""Topic 5 event-resolved interictal axis_bias — PILOT figures (one per subject).

Spec §7 panel discipline (each panel = one independent question):
  (a) per-event mirror-invariant field alignment by class A/B (the primary M) — shows the
      WITHIN-class dispersion (the "std effect") + the A-vs-B location difference.
  (b) the 1D order-vs-activation companion (M1d) by class — a strictly more replay-flavoured
      construct, shown only as a class-level distribution (construct cross-check).
  (c) substrate density (per-event participating contacts by class) + the block-level
      class-separation summary (R2).

Reads results/topic5_ictal_recruitment/event_resolved_alignment/per_subject/*.json.
Self-contained labels (no §codes / cluster_id jargon). Exploratory secondary; the figure
title states it is NOT the A-line primary and NOT a replay claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/topic5_ictal_recruitment/event_resolved_alignment")
A_COL, B_COL = "#3b6fb6", "#c0504d"   # class A / class B (consistent across panels)


def _violin(ax, data_a, data_b, ylabel, title):
    parts = ax.violinplot([data_a, data_b], positions=[0, 1], showmedians=True, widths=0.8)
    for pc, col in zip(parts["bodies"], (A_COL, B_COL)):
        pc.set_facecolor(col); pc.set_alpha(0.55)
    for key in ("cmedians", "cmaxes", "cmins", "cbars"):
        if key in parts:
            parts[key].set_color("0.25")
    for x, d, col in ((0, data_a, A_COL), (1, data_b, B_COL)):
        jit = x + (np.random.default_rng(0).random(min(len(d), 400)) - 0.5) * 0.25
        ax.scatter(jit, np.random.default_rng(1).permutation(d)[:len(jit)], s=3, color=col, alpha=0.22, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["class A", "class B"], fontsize=9)
    ax.set_xlabel("interictal propagation class")
    ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.set_ylim(0, 1.05)

    def _iqr(d):
        return float(np.subtract(*np.percentile(d, [75, 25])))
    stats = (f"A: med {np.median(data_a):.2f}  IQR {_iqr(data_a):.2f}  n={len(data_a)}\n"
             f"B: med {np.median(data_b):.2f}  IQR {_iqr(data_b):.2f}  n={len(data_b)}")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))


def _plot_subject(res, outdir):
    sid = res["subject_id"]
    bc = res["M_by_class"]
    if "class_0" not in bc or "class_1" not in bc:
        return None
    a, b = bc["class_0"], bc["class_1"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    # (a) primary: per-event field alignment by class
    _violin(axes[0], np.array(a["aligns"]), np.array(b["aligns"]),
            "per-event field alignment  |collinearity with seizure-onset gradient|",
            f"each interictal event vs seizure-onset field\n(analyzed {res['n_events_analyzed']} ev, "
            f"usable {res['M_usable_fraction']*100:.0f}%, {res['M_n_blocks_usable']} usable blocks)")

    # (b) companion: 1D order-vs-activation (replay-adjacent)
    if a.get("m1d_aligns") and b.get("m1d_aligns"):
        _violin(axes[1], np.array(a["m1d_aligns"]), np.array(b["m1d_aligns"]),
                "per-event order-vs-activation |rank correlation|",
                "companion (within-event order vs onset strength)\n— more replay-flavoured; cross-check only")
    else:
        axes[1].text(0.5, 0.5, "companion metric\nnot eligible", ha="center", va="center")
        axes[1].set_axis_off()

    # (c) substrate density (FULL data) + separation summary (analyzed subsample)
    pa = res["participation_full"].get("class_0", {}); pb = res["participation_full"].get("class_1", {})
    axes[2].bar([0, 1], [pa.get("median_n_part", 0), pb.get("median_n_part", 0)],
                color=[A_COL, B_COL], alpha=0.7, width=0.6)
    axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["A", "B"])
    axes[2].set_ylabel("median participating contacts / event")
    axes[2].set_title("substrate density + class separation", fontsize=10)
    R2 = res.get("R2_separation", {})
    txt = (f"A median {a['median_align']:.3f} (IQR {a['iqr_align']:.3f})\n"
           f"B median {b['median_align']:.3f} (IQR {b['iqr_align']:.3f})\n"
           f"A−B Δmedian {R2.get('delta_median_obs', float('nan')):.3f}\n"
           f"  within-block-shuffle p {R2.get('delta_median_null_p')}\n"
           f"IQR ratio A/B {R2.get('disp_ratio_obs', float('nan')):.2f}"
           f" (size-matched {R2.get('size_matched_iqr_ratio', float('nan')):.2f})\n"
           f"R2 basis: {R2.get('n')} ev, {R2.get('n_blocks')} blocks\n"
           f"analyzed {res.get('n_events_analyzed')} / full {res.get('n_events_valid_full')}")
    axes[2].text(1.02, 0.5, txt, transform=axes[2].transAxes, fontsize=8.5, va="center",
                 bbox=dict(boxstyle="round", fc="0.96", ec="0.7"))

    fig.suptitle(f"{sid} — event-resolved interictal↔ictal field alignment "
                 f"(EXPLORATORY secondary; NOT the A-line primary, NOT a replay claim)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    fp = outdir / f"{sid}_event_resolved_alignment.png"
    fig.savefig(fp, dpi=130); plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    outdir = Path(args.out); figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    made = []
    for f in sorted((outdir / "per_subject").glob("*.json")):
        res = json.load(open(f))
        if res.get("status") != "ok":
            print(f"[skip] {res.get('subject_id')}: {res.get('status')}"); continue
        fp = _plot_subject(res, figdir)
        if fp:
            print(f"[fig] {fp}"); made.append(fp.name)
    print(f"[done] {len(made)} figures -> {figdir}")


if __name__ == "__main__":
    main()
