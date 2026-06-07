#!/usr/bin/env python3
"""Figure for low-rate-window template stability (paper-grade, self-contained).

Panel A: reproducibility vs window HFO-event count — firing count vs propagation template;
         shows whether count degrades in quiet windows while the template holds up (crossover).
Panel B: per-subject reproducibility gap (template - firing count) in the low-event windows —
         is the low-window advantage consistent across subjects.
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

_ROOT = Path(__file__).resolve().parents[1]
DIR = _ROOT / "results" / "topic4_sef_hfo" / "low_rate_template_stability"
FIG = DIR / "figures"

RATE_C = "#c44e52"      # firing count
TMPL_C = "#4c72b0"      # propagation template


def main() -> int:
    per = [json.load(open(f)) for f in sorted(glob.glob(str(DIR / "per_subject" / "*.json")))]
    per = [p for p in per if not p.get("insufficient")]
    if not per:
        print("no computable subjects")
        return 1
    cohort = json.loads((DIR / "cohort.json").read_text())

    # pooled windows for Panel A
    ev, rate, tmpl, rate_all = [], [], [], []
    for p in per:
        for w in p["windows"]:
            ev.append(w["n_events"]); rate.append(w["rate_repro"]); tmpl.append(w["template_repro"])
            rate_all.append(w.get("rate_repro_allch", np.nan))
    ev = np.array(ev, float); rate = np.array(rate, float); tmpl = np.array(tmpl, float)
    rate_all = np.array(rate_all, float)

    FIG.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Panel A: median reproducibility vs event-count bins (log-spaced)
    bins = np.geomspace(max(5, ev.min()), ev.max(), 8)
    centers, rmed, tmed, rlo, rhi, tlo, thi, ramed = [], [], [], [], [], [], [], []
    for i in range(len(bins) - 1):
        m = (ev >= bins[i]) & (ev < bins[i + 1])
        if m.sum() < 5:
            continue
        centers.append(np.sqrt(bins[i] * bins[i + 1]))
        rmed.append(np.median(rate[m])); tmed.append(np.median(tmpl[m]))
        rlo.append(np.percentile(rate[m], 25)); rhi.append(np.percentile(rate[m], 75))
        tlo.append(np.percentile(tmpl[m], 25)); thi.append(np.percentile(tmpl[m], 75))
        ra = rate_all[m]; ramed.append(np.median(ra[np.isfinite(ra)]) if np.any(np.isfinite(ra)) else np.nan)
    centers = np.array(centers)
    axA.fill_between(centers, rlo, rhi, color=RATE_C, alpha=0.15)
    axA.fill_between(centers, tlo, thi, color=TMPL_C, alpha=0.15)
    axA.plot(centers, ramed, ":", color=RATE_C, alpha=0.6, label="firing count (all channels, incl. silent)")
    axA.plot(centers, rmed, "o-", color=RATE_C, label="firing count (common channels)")
    axA.plot(centers, tmed, "s-", color=TMPL_C, label="propagation template")
    axA.set_xscale("log")
    axA.set_xlabel("HFO events in window")
    axA.set_ylabel("reproducibility of full-recording pattern\n(Spearman, window vs whole recording)")
    axA.set_title("Stability vs sampling: where firing count degrades")
    axA.legend(frameon=False, loc="lower right")
    axA.set_ylim(min(0.0, np.min(rmed) - 0.05), 1.02)
    axA.grid(alpha=0.25)

    # Panel B: per-subject low-event-window EXCESS (null-corrected, time-structured) — honest headline
    deltas = sorted((p.get("primary_low_excess"), p["dataset"]) for p in per
                    if p.get("primary_low_excess") is not None
                    and p["primary_low_excess"] == p["primary_low_excess"])
    vals = [d[0] for d in deltas]
    colors = ["#55a868" if d[1] == "epilepsiae" else "#8172b3" for d in deltas]
    y = np.arange(len(vals))
    axB.barh(y, vals, color=colors)
    axB.axvline(0, color="k", lw=0.8)
    axB.set_yticks([])
    axB.set_xlabel("template − firing-count reproducibility, null-corrected\n(low-event windows, per subject)")
    med = np.median(vals) if vals else float("nan")
    cagg = cohort["cohort_all"]
    p_all = cagg.get("wilcoxon_p_excess_low", cagg.get("wilcoxon_p_template_gt_rate_low", float("nan")))
    p_str = "<0.001" if p_all < 0.001 else f"{p_all:.3f}"
    axB.set_title(f"Low-window advantage per subject (null-corrected)\n(median {med:+.2f}, n={len(vals)}, "
                  f"{sum(v>0 for v in vals)}/{len(vals)} >0, p={p_str})")
    from matplotlib.patches import Patch
    axB.legend(handles=[Patch(color="#55a868", label="epilepsiae"), Patch(color="#8172b3", label="yuquan")],
               frameon=False, loc="lower right")
    axB.grid(alpha=0.25, axis="x")

    fig.tight_layout()
    out = FIG / "reproducibility_vs_event_count.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}  (n_subjects={len(per)}, pooled windows={ev.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
