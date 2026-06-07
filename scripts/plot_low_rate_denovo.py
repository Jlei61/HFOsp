#!/usr/bin/env python3
"""Figure for the de novo layer (LR-7): discovering the template from scratch vs reading it back.

Panel A: recovery of the full-recording propagation axis vs window event count, for four readouts
         on the SAME windows -- (1) read-back (window borrows global event labels), (2) de novo
         signed (window re-clusters and self-orients; reverse-dominated windows score negative),
         (3) de novo axis-line |.| (polarity dropped), (4) firing count. The read-back-minus-de-novo
         gap is the cost of forbidding the peek; the signed-minus-|.| gap is the polarity cost.
Panel B: per-subject de novo SIGNED advantage over firing count in low-event windows (null-corrected)
         -- is the from-scratch advantage consistent across subjects, or does discovery collapse.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
FIG = DIR / "figures"

RATE_C = "#c44e52"      # firing count
BACK_C = "#4c72b0"      # read-back (knows the template)
DENO_C = "#dd8452"      # de novo (discovers from scratch)


def main() -> int:
    per = [json.load(open(f)) for f in sorted(glob.glob(str(DIR / "per_subject_denovo" / "*.json")))]
    per = [p for p in per if not p.get("insufficient")]
    if not per:
        print("no computable subjects")
        return 1
    cohort = json.loads((DIR / "cohort_denovo.json").read_text())

    ev, back, dsig, dabs, rate = [], [], [], [], []
    for p in per:
        for w in p["windows"]:
            ev.append(w["n_events"]); back.append(w["template_repro_global"])
            dsig.append(w["template_repro_denovo_signed"]); dabs.append(w["template_repro_denovo_abs"])
            rate.append(w["rate_repro"])
    ev = np.array(ev, float); back = np.array(back, float); dsig = np.array(dsig, float)
    dabs = np.array(dabs, float); rate = np.array(rate, float)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel A: median recovery vs event-count bins (log-spaced)
    bins = np.geomspace(max(3, ev.min()), ev.max(), 9)
    centers, mback, mdsig, mdabs, mrate = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ev >= bins[i]) & (ev < bins[i + 1])
        if m.sum() < 5:
            continue
        centers.append(np.sqrt(bins[i] * bins[i + 1]))
        mback.append(np.nanmedian(back[m])); mdsig.append(np.nanmedian(dsig[m]))
        mdabs.append(np.nanmedian(dabs[m])); mrate.append(np.nanmedian(rate[m]))
    centers = np.array(centers)
    axA.plot(centers, mback, "s-", color=BACK_C, label="read-back (global template known)")
    axA.plot(centers, mdsig, "o-", color=DENO_C, label="de novo, directed (from scratch, signed)")
    axA.plot(centers, mdabs, "o--", color=DENO_C, alpha=0.6, label="de novo, axis line (|ρ|, direction ignored)")
    axA.plot(centers, mrate, "^:", color=RATE_C, alpha=0.7, label="firing count (operational reference)")
    axA.axhline(0, color="k", lw=0.7, alpha=0.5)
    axA.set_xscale("log")
    axA.set_xlabel("HFO events in window")
    axA.set_ylabel("recovery of full-recording propagation axis\n(Spearman ρ, window vs whole recording)")
    axA.set_title("Can a short quiet window re-discover the full template?\n(ground truth = whole-recording axis)")
    axA.legend(frameon=False, loc="lower left", fontsize=8.5)
    axA.set_ylim(min(-0.05, np.nanmin(mdsig) - 0.05), 1.02)
    axA.grid(alpha=0.25)

    # Panel B: per-subject de novo SIGNED null-corrected advantage over firing count (low windows)
    rows = sorted((p.get("primary_low_excess_denovo_signed"), p["dataset"]) for p in per
                  if p.get("primary_low_excess_denovo_signed") is not None
                  and p["primary_low_excess_denovo_signed"] == p["primary_low_excess_denovo_signed"])
    vals = [r[0] for r in rows]
    colors = ["#55a868" if r[1] == "epilepsiae" else "#8172b3" for r in rows]
    y = np.arange(len(vals))
    axB.barh(y, vals, color=colors)
    axB.axvline(0, color="k", lw=0.8)
    axB.set_yticks([])
    axB.set_xlabel("de novo directed recovery − null (low-event windows, per subject)\n"
                   "[primary: recovery of full-recording axis; rate shown as operational reference]")
    med = np.median(vals) if vals else float("nan")
    cagg = cohort["cohort_all"]
    p_all = cagg["wilcoxon_p_primary"]
    p_str = "<0.001" if (p_all == p_all and p_all < 0.001) else (f"{p_all:.3f}" if p_all == p_all else "n/a")
    axB.set_title(f"Short-window independent discovery: per-subject recovery\n"
                  f"(null-corrected; median {med:+.2f}, n={len(vals)}, {sum(v>0 for v in vals)}/{len(vals)} >0, p={p_str})")
    axB.legend(handles=[Patch(color="#55a868", label="epilepsiae"), Patch(color="#8172b3", label="yuquan")],
               frameon=False, loc="lower right")
    axB.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    out = FIG / "denovo_recovery_vs_event_count.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (n_subjects={len(per)}, pooled windows={ev.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
