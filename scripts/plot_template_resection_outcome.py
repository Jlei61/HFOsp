#!/usr/bin/env python3
"""Track E1 pre-outcome figures: coverage landscape + per-metric contrast.

Outcome labels are absent, so NO outcome figures here — only the predictor side:
  fig 1: per-subject treated-fraction across the 6 network targets (the landscape)
  fig 2: per-metric coverage spread (which target has discriminative contrast vs saturates)

Reads results/template_resection_outcome/yuquan_template_resection_metrics.csv.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results/template_resection_outcome")
FIGDIR = os.path.join(OUT, "figures")

# (csv column, paper-grade readable label)
METRICS = [
    ("template_endpoint_coverage", "Template\nendpoint"),
    ("early_end_coverage", "Early end\n(source)"),
    ("shared_endpoint_core_coverage", "Shared endpoint\ncore"),
    ("clinical_network_coverage", "Clinical network\n(origin+spread)"),
    ("hfo_rate_topk_sozsize_coverage", "HFO-rate\n(size-matched)"),
    ("clinical_soz_coverage", "Clinical SOZ"),
]


def _f(s):
    return np.nan if s == "" else float(s)


def load():
    rows = list(csv.DictReader(open(os.path.join(OUT, "yuquan_template_resection_metrics.csv"))))
    return rows


def fig_landscape(rows):
    # sort subjects by template_endpoint_coverage desc; NA-template subjects last
    def key(r):
        v = _f(r["template_endpoint_coverage"])
        return (np.isnan(v), -(0 if np.isnan(v) else v))
    rows = sorted(rows, key=key)
    subj = [r["subject"] for r in rows]
    disc = [r["discordant_candidate"] == "True" for r in rows]
    M = np.array([[_f(r[c]) for c, _ in METRICS] for r in rows])

    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    cmap = plt.cm.viridis.copy()
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    # NA cells -> light gray overlay
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="0.85"))
                ax.text(j, i, "NA", ha="center", va="center", fontsize=7, color="0.4")
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([lab for _, lab in METRICS], fontsize=8)
    ax.set_yticks(range(len(subj)))
    ax.set_yticklabels(subj, fontsize=8)
    # mark discordant subjects in red
    for i, d in enumerate(disc):
        if d:
            ax.get_yticklabels()[i].set_color("crimson")
            ax.get_yticklabels()[i].set_fontweight("bold")
    ax.set_title("Treated fraction of each network target, per Yuquan subject\n"
                 "(RF-thermocoagulation; resection unknown / not text-extractable)\n"
                 "red label = discordant candidate · outcome labels pending",
                 fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("fraction treated", fontsize=8)
    ax.axvline(2.5, color="white", lw=1.5)  # separate template metrics | baselines
    fig.tight_layout()
    p = os.path.join(FIGDIR, "coverage_landscape.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_contrast(rows):
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    data = [[_f(r[c]) for r in rows if not np.isnan(_f(r[c]))] for c, _ in METRICS]
    positions = range(len(METRICS))
    bp = ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                    medianprops=dict(color="black"), showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(to_rgba("steelblue", 0.35))
    # overlay individual subjects (jitter)
    rng = np.linspace(-0.12, 0.12, 7)
    for j, (c, _) in enumerate(METRICS):
        vals = [_f(r[c]) for r in rows]
        xs = [j + rng[i % len(rng)] for i, v in enumerate(vals) if not np.isnan(v)]
        ys = [v for v in vals if not np.isnan(v)]
        ax.scatter(xs, ys, s=16, color="navy", alpha=0.6, zorder=3)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([lab for _, lab in METRICS], fontsize=8)
    ax.set_ylabel("fraction treated", fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.axhline(1.0, color="0.6", ls=":", lw=1)
    ax.set_title("Which target has discriminative contrast vs saturates at full coverage\n"
                 "(SOZ / early-end ≈ saturated → cannot ever separate outcome; "
                 "network / HFO / template-endpoint carry spread)", fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "coverage_contrast_by_target.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    rows = load()
    p1 = fig_landscape(rows)
    p2 = fig_contrast(rows)
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
