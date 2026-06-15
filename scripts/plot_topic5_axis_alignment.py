"""Topic 5 A-line — cohort figure: per-subject real |axis-alignment| vs the null p95 bands,
one panel per activation metric. Reads results/.../axis_alignment/axis_alignment_<m>_B<B>.json.
Best-effort: a failure here never blocks the sweep results (the JSONs are the source of truth).
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/topic5_ictal_recruitment/axis_alignment")
FIG = OUT / "figures"
NULLS = [("channel_null_p95", "channel (coarse)", "#888"),
         ("within_shaft_null_p95", "within-shaft (fine)", "#e07b39"),
         ("anchor_matched_null_p95", "anchor-matched (activity)", "#3b7dd8"),
         ("joint_null_p95", "joint (shaft+activity)", "#9b59b6")]


def _panel(ax, summ):
    rows = [r for r in summ["per_subject"] if r.get("status") == "ok"
            and r["dataset"] == "epilepsiae"]
    rows.sort(key=lambda r: r["real_median_abs_corr"])
    x = np.arange(len(rows))
    real = [r["real_median_abs_corr"] for r in rows]
    ax.plot(x, real, "ko", ms=5, zorder=3, label="real |corr|")
    for key, lbl, col in NULLS:
        yy = [r.get(key) for r in rows]
        if any(v is not None for v in yy):
            ax.plot(x, [np.nan if v is None else v for v in yy], "_", color=col, ms=10, mew=2, label=lbl)
    cs = summ.get("epilepsiae_primary", {})
    pa = cs.get("n_pass_anchor_matched")
    pj = cs.get("n_pass_joint")
    ax.set_title(f"{summ['activation']} (B={summ['B']}, n={cs.get('n')})\n"
                 f"beat coarse {cs.get('n_pass_channel')}/{cs.get('n')} | "
                 f"fine {cs.get('n_pass_within_shaft')}/{cs.get('n')} | "
                 f"activity {pa if pa is not None else 'NA'}/{cs.get('n')} | "
                 f"joint {pj if pj is not None else 'NA'}/{cs.get('n')}", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["subject_id"].replace("epilepsiae_", "") for r in rows], rotation=90, fontsize=6)
    ax.set_ylabel("median |corr| (real vs null p95)")
    ax.set_ylim(0, 1)


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    files = sorted(glob.glob(str(OUT / f"axis_alignment_*_B{B}.json")))
    if not files:
        print(f"no axis_alignment_*_B{B}.json found", flush=True)
        return
    summs = [json.load(open(f)) for f in files]
    n = len(summs)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.6), squeeze=False)
    for ax, s in zip(axes[0], summs):
        try:
            _panel(ax, s)
        except Exception as e:  # noqa: BLE001
            ax.text(0.5, 0.5, f"panel error: {e}", ha="center")
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Topic5 A-line: ictal activation vs interictal axis (Epilepsiae, B={B})\n"
                 "real above a null's p95 = beats that null", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / f"axis_alignment_cohort_B{B}.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
