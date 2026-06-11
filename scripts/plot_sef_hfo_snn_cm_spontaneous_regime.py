"""3-panel spontaneous-regime diagnostic for the lesion mean x var sweep (user 2026-06-11 + review):
confirm the lesion abnormality grades spontaneous activity AND find the clean read-out band — but
gate on REAL large directional waves, not bare returned-event count, and use a TRUE inter-event
baseline (active fraction OUTSIDE event windows), not the detector's 5-50ms calibration floor.

Panels vs lesion core-mean (left = more abnormal):
  a) LARGE-directional event rate (solid; gated: >=5000 neurons, |corr|>=0.5, peak>=0.02) overlaid
     with the bare returned-event rate (dashed) — the gap = tiny flat fluctuations that must NOT
     count as readable events (narrow mean>=17.5 is almost all gap).
  b) TRUE inter-event baseline (p95 active fraction outside event windows) — clean-quiet vs
     quasi-continuous core.
  c) median event size (recruited E).
The clean read-out region is SPREAD-DEPENDENT and emerges from panel a: wide is clean ~17-17.5 then
fragments at <=16.5 (large rate drops); narrow only ignites at <=17 but then stays clean down to 16.
"""
import os
import sys
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
from run_sef_hfo_snn_cm_spontaneous_ratemap import _gated_metrics, T   # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FIG = os.path.join(OUT, "figures")


def _load():
    import re
    rows = []
    for p in glob.glob(os.path.join(OUT, "spont_RATE_*.json")):
        m = re.match(r"spont_RATE_m([0-9.]+)_(wide|narrow)_s(\d+)\.json", os.path.basename(p))
        if not m:
            continue
        g = _gated_metrics(p, T / 1000.0)
        rows.append(dict(mean=float(m.group(1)), spread=m.group(2), seed=int(m.group(3)),
                         sep_rate=g["sep_rate_per_s"], large_rate=g["large_dir_rate_per_s"],
                         true_floor=(g["true_inter_event_floor"] if g["true_inter_event_floor"] is not None else np.nan),
                         ev_size=g["median_event_size"]))
    return rows


def main():
    rows = _load()
    if not rows:
        print("no spont_RATE_*.json — run the rate-map sweep first"); return
    os.makedirs(FIG, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))
    COL = {"wide": "darkorange", "narrow": "steelblue"}

    def curve(ax, key, marker="-o"):
        for sname, col in COL.items():
            sub = [r for r in rows if r["spread"] == sname]
            ms = sorted({r["mean"] for r in sub}, reverse=True)
            ym = [float(np.nanmean([r[key] for r in sub if r["mean"] == m])) for m in ms]
            ax.plot(ms, ym, marker, color=col, label=sname)
            ax.scatter([r["mean"] for r in sub], [r[key] for r in sub], color=col, s=13, alpha=0.35)

    # a) large-directional rate (solid) + bare returned rate (dashed, faint)
    curve(axes[0], "large_rate", "-o")
    for sname, col in COL.items():
        sub = [r for r in rows if r["spread"] == sname]
        ms = sorted({r["mean"] for r in sub}, reverse=True)
        ym = [float(np.nanmean([r["sep_rate"] for r in sub if r["mean"] == m])) for m in ms]
        axes[0].plot(ms, ym, "--", color=col, alpha=0.5, lw=1.0)
    axes[0].set_ylabel("event rate (events/s)")
    axes[0].set_title("a) LARGE directional waves (solid) vs all returned (dashed)\n"
                      "gap = tiny flat fluctuations (NOT readable)", fontsize=9)
    curve(axes[1], "true_floor")
    axes[1].axhline(0.01, color="0.4", ls="--", lw=0.8)
    axes[1].set_ylabel("true inter-event baseline (p95 active frac)")
    axes[1].set_title("b) clean-quiet vs quasi-continuous core\n(active fraction OUTSIDE events)", fontsize=9)
    curve(axes[2], "ev_size")
    axes[2].axhline(5000, color="0.4", ls="--", lw=0.8)
    axes[2].set_ylabel("median event size (recruited E)")
    axes[2].set_title("c) full sheet wave (>=5000) vs fragment", fontsize=9)
    for ax in axes:
        ax.set_xlabel("lesion core mean (mV) — left = more abnormal"); ax.invert_xaxis()
    axes[0].legend(fontsize=8, title="core variance")
    fig.suptitle("Spontaneous regime vs lesion abnormality (−end lesion, L=20/d100). Clean read-out is "
                 "SPREAD-DEPENDENT: wide clean ~17–17.5 (fragments at ≤16.5); narrow ignites at ≤17, "
                 "stays clean to 16.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "spontaneous_regime.png"), dpi=140)
    plt.close(fig)
    print("wrote figures/spontaneous_regime.png")


if __name__ == "__main__":
    main()
