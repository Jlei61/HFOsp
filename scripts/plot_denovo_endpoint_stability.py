#!/usr/bin/env python3
"""Figure: direction-agnostic endpoint stability (KMeans-union approach).

Panel A: median endpoint Jaccard vs window event count (log-binned)
         -- KMeans-union endpoints vs rate top-4, with IQR bands.
         Ground truth = full-recording endpoint channels.
Panel B: per-subject low-window endpoint Jaccard (absolute) sorted,
         colored by dataset. Rate topk shown as a reference line per subject.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_ROOT = Path(__file__).resolve().parents[1]
DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
FIG = DIR / "figures"

ENDPT_C = "#2ca02c"   # endpoint (KMeans-union)
RATE_C  = "#c44e52"   # rate reference


def main() -> int:
    per = [json.load(open(f)) for f in sorted(glob.glob(str(DIR / "per_subject_endpoint_stability" / "*.json")))]
    per = [p for p in per if not p.get("insufficient")]
    if not per:
        print("no subjects"); return 1
    cohort = json.loads((DIR / "cohort_endpoint_stability.json").read_text())

    # Pool all windows
    ev, ej, rj = [], [], []
    for p in per:
        for w in p.get("windows", []):
            ev.append(w["n_events"])
            ej.append(w["endpoint_jaccard"])
            rj.append(w["rate_topk_jaccard"])
    ev = np.array(ev, float); ej = np.array(ej, float); rj = np.array(rj, float)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Jaccard vs event count (log-binned) ---
    bins = np.geomspace(max(3, ev.min()), ev.max(), 9)
    cx, mej, mej_lo, mej_hi, mrj = [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ev >= bins[i]) & (ev < bins[i + 1])
        if m.sum() < 5:
            continue
        cx.append(np.sqrt(bins[i] * bins[i + 1]))
        mej.append(np.median(ej[m])); mrj.append(np.median(rj[m]))
        mej_lo.append(np.percentile(ej[m], 25)); mej_hi.append(np.percentile(ej[m], 75))
    cx = np.array(cx)

    axA.fill_between(cx, mej_lo, mej_hi, color=ENDPT_C, alpha=0.15)
    axA.plot(cx, mej, "s-", color=ENDPT_C, label="KMeans-union endpoints (de novo)")
    axA.plot(cx, mrj, "^:", color=RATE_C, alpha=0.8, label="firing count top-4 (reference)")
    axA.axhline(0, color="k", lw=0.6, alpha=0.4)
    axA.set_xscale("log")
    axA.set_xlabel("HFO events in window")
    axA.set_ylabel("Jaccard similarity to full-recording endpoints\n(first + last to fire, k=2 each end)")
    axA.set_title("Direction-agnostic endpoint recovery vs event count\n(ground truth = whole-recording endpoints)")
    axA.legend(frameon=False, fontsize=9)
    axA.set_ylim(-0.05, 1.02)
    axA.grid(alpha=0.25)

    # random chance reference line (4 channels from ~15 common)
    axA.axhline(0.15, color="gray", lw=0.8, ls="--", alpha=0.5)
    axA.text(cx[-1] * 1.05, 0.16, "random\nchance", fontsize=7, color="gray", va="bottom", ha="left")

    # --- Panel B: per-subject low-window endpoint_J (absolute), sorted ---
    rows = sorted(
        [(p["primary_low_endpoint_jaccard"],
          p["by_stratum"]["low"]["rate_topk_jaccard"],
          p["dataset"])
         for p in per
         if p.get("primary_low_endpoint_jaccard") is not None
         and p["primary_low_endpoint_jaccard"] == p["primary_low_endpoint_jaccard"]],
        key=lambda x: x[0]
    )
    vals = [r[0] for r in rows]
    rate_vals = [r[1] for r in rows]
    colors = ["#55a868" if r[2] == "epilepsiae" else "#8172b3" for r in rows]
    y = np.arange(len(vals))

    axB.barh(y, vals, color=colors, alpha=0.8, label="_nolegend_")
    axB.scatter(rate_vals, y, color=RATE_C, zorder=3, s=25, marker="|",
                linewidths=2, label="rate reference (per subject)")
    axB.axvline(0.33, color=RATE_C, lw=0.8, ls=":", alpha=0.6)  # cohort rate median
    axB.axvline(0.15, color="gray", lw=0.8, ls="--", alpha=0.5)  # random chance
    axB.set_yticks([])
    axB.set_xlabel("Endpoint Jaccard (low-event windows, per subject)\n"
                   "| = per-subject rate reference; dashed = random chance (~0.15)")
    med = np.median(vals)
    axB.set_title(f"Per-subject endpoint recovery (absolute)\n"
                  f"(median {med:.2f}, n={len(vals)}, {sum(v > 0.33 for v in vals)}/{len(vals)} > rate median)")
    axB.legend(handles=[
        Patch(color="#55a868", label="epilepsiae"),
        Patch(color="#8172b3", label="yuquan"),
    ], frameon=False, loc="lower right", fontsize=9)
    axB.grid(alpha=0.25, axis="x")
    axB.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    out = FIG / "endpoint_stability_denovo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (n_subjects={len(per)}, pooled_windows={ev.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
