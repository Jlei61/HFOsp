#!/usr/bin/env python3
"""Clinician-facing overview: how completely was each patient's interictal
propagation network ablated by RF-thermocoagulation, and was the network's
'driver' (earliest-firing) site hit.

One question per figure (CLAUDE.md §7). Clinical language only, no codenames.
Input: results/template_ablation_coverage/yuquan_coverage_prep.csv
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV = "results/template_ablation_coverage/yuquan_coverage_prep.csv"
OUTDIR = "results/template_ablation_coverage/figures"

rows = [r for r in csv.DictReader(open(CSV)) if r["status"] == "ok"]
for r in rows:
    r["cov"] = float(r["template_coverage"])
    r["src"] = float(r["source_ablated_frac"]) if r["source_ablated_frac"] else 0.0
rows.sort(key=lambda r: r["cov"])

n = len(rows)
fig, ax = plt.subplots(figsize=(8.2, 0.42 * n + 1.6))
y = range(n)
covs = [r["cov"] * 100 for r in rows]
# bar colour: low coverage (network largely spared) flagged warm, else neutral
colors = ["#c2785f" if r["cov"] < 0.5 else "#8aa6a3" for r in rows]
ax.barh(list(y), covs, color=colors, height=0.62, zorder=2)

# driver-site marker at x just past bar end
for i, r in enumerate(rows):
    filled = r["src"] >= 0.999
    ax.scatter(102, i, s=46, marker="o",
               facecolor=("#3b3b3b" if filled else "white"),
               edgecolor="#3b3b3b", linewidth=1.2, zorder=3, clip_on=False)

ax.set_yticks(list(y))
ax.set_yticklabels([r["subject"] for r in rows], fontsize=9)
ax.set_xlim(0, 100)
ax.set_xlabel("Interictal propagation network ablated by RF-thermocoagulation (%)",
              fontsize=10)
ax.set_title("Per-patient ablation coverage of the interictal propagation network\n"
             "Yuquan (n=%d) — pre-surgical SEEG templates vs RF-TC contacts" % n,
             fontsize=11)
ax.axvline(50, color="#bbbbbb", ls="--", lw=0.8, zorder=1)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(length=0)

# annotate the spared-network cases (clinical hypothesis: higher recurrence risk)
for i, r in enumerate(rows):
    if r["cov"] < 0.5:
        ax.text(covs[i] + 4, i, "network largely spared", va="center",
                fontsize=7.5, color="#c2785f")

legend = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#3b3b3b",
           markeredgecolor="#3b3b3b", markersize=8, label="driver site ablated"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#3b3b3b", markersize=8, label="driver site spared"),
]
ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=8.5,
          bbox_to_anchor=(1.0, -0.02))
fig.text(0.01, 0.005,
         "Predictor candidate for surgical outcome (pending follow-up). "
         "Not a result: contrast exists, outcome association untested.",
         fontsize=7, color="#888888")
fig.tight_layout(rect=(0, 0.03, 1, 1))

os.makedirs(OUTDIR, exist_ok=True)
out = os.path.join(OUTDIR, "yuquan_coverage_overview.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
