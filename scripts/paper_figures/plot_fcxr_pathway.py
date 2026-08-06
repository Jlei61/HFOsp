"""FCXR 2x2 pathway-attribution figure (reviewer 2026-07-20): which AMPA pathway (feedforward vs
recurrent) causes the workpoint drift? Two panels:
  A  event-profile ratio to reference per arm (events / participation / peak-rate) -> over-activation
  B  settled cap-clip fraction per arm -> which pathway clips
Reads the pathway summary.json (default: latest_pathway.json).
Output: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/figures/pathway_2x2.png
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
FIGDIR = os.path.join(OUT_ROOT, "figures")
ARM_COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}
LABELS = {"A_ff-add_rec-add": "A\nadd/add\n(ref)", "B_ff-cond_rec-add": "B\nff-cond\n(feedforward)",
          "C_ff-add_rec-cond": "C\nrec-cond\n(recurrent)", "D_ff-cond_rec-cond": "D\ncond/cond\n(NO-GO)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", default=None); args = ap.parse_args()
    rd = args.run_dir or json.load(open(os.path.join(OUT_ROOT, "latest_pathway.json")))["path"]
    s = json.load(open(os.path.join(rd, "summary.json")))
    ref = s["reference_workpoint"]
    arms = sorted(s["arms"], key=lambda a: a["label"])
    n_ref = ref["n_events"]; part_ref = 0.5 * (ref["part_lo"] + ref["part_hi"])
    peak_ref = 0.5 * (ref["act_lo"] + ref["act_hi"])

    os.makedirs(FIGDIR, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(wspace=0.25))
    x = np.arange(len(arms)); w = 0.26
    dims = [("n_returning", n_ref, "events"), ("participation_median", part_ref, "participation"),
            ("peak_rate_median_hz", peak_ref, "peak rate")]
    axA.axhspan(0.5, 2.0, color="0.85", alpha=0.6, zorder=0, label="~within-band (0.5–2×)")
    axA.axhline(1.0, color="k", lw=1.1, ls="--")
    for k, (key, rv, name) in enumerate(dims):
        vals = [((a.get(key) or np.nan) / rv if rv else np.nan) for a in arms]
        axA.bar(x + (k - 1) * w, vals, w, color=plt.cm.Greys(0.35 + 0.25 * k), label=name)
    axA.set_xticks(x); axA.set_xticklabels([LABELS.get(a["label"], a["label"]) for a in arms], fontsize=8)
    axA.set_yscale("log"); axA.set_ylabel("value / reference (×)")
    axA.set_title("A · event profile ÷ ref  (feedforward B,D over-activate)", fontsize=9.5, loc="left")
    axA.legend(fontsize=7, loc="upper left")

    clips = [100.0 * (a.get("settled_max_clip_fraction") or 0.0) for a in arms]
    cols = [ARM_COL[a["label"][0]] for a in arms]
    axB.bar(x, clips, 0.6, color=cols)
    for xi, c in zip(x, clips):
        axB.text(xi, c + 0.02, f"{c:.2f}%", ha="center", va="bottom", fontsize=8)
    axB.set_xticks(x); axB.set_xticklabels([LABELS.get(a["label"], a["label"]) for a in arms], fontsize=8)
    axB.set_ylabel("settled cap-clip fraction (%)")
    axB.set_title("B · clip  (recurrent C,D clip)", fontsize=9.5, loc="left")
    axB.set_ylim(0, max(1.0, max(clips) * 1.3))

    fig.suptitle(f"MZ-FCXR 2×2 pathway attribution (seed{s.get('seed')}, c_E={s.get('c_E')}, T={s.get('T'):.0f}ms)  —  "
                 f"{s.get('attribution')}", fontsize=11, y=1.02)
    out = os.path.join(FIGDIR, "pathway_2x2.png")
    fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)


if __name__ == "__main__":
    main()
