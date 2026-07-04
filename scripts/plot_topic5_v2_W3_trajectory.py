"""Topic5 V2 Phase-1-v2 / W3 Task 3.2 — peri-ictal scaffold-score TRAJECTORY figure ("when?").

Pure post-processing of the Task 3.2 CSVs (phase1_v2_alignment_trajectory.csv +
phase1_v2_trajectory_contrasts.csv): NO new nulls / simulation / alignment. Two INDEPENDENT
scientific questions -> two panels (CLAUDE.md §7):

  Panel A  cohort scaffold-score trajectory across peri-ictal time (far pre-ictal -> early post-onset).
           Does the scaffold-score rise pre -> post, or is it flat (already present pre-ictally)?
  Panel B  the 3 per-subject paired contrasts with the subject-level sign-flip p (star if p<0.05).
           Is any rise statistically supported at the subject level (unit = subject)?

EEG-onset anchor = PRIMARY (solid); clinical-onset anchor = SENSITIVITY (dashed). narrow + broad
pools. Style: figure_style_guide §0 -- tight axes, one shared legend per panel, English only (no
CJK, no internal codenames). Descriptive candidate-scaffold tier: the figure shows trajectory shape
+ sign-flip support, NOT criticality / mechanism.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_topic5_v2_trajectory import (  # noqa: E402
    BIN_ORDER, BIN_CENTERS, CONTRASTS, subject_bin_wide, cohort_trajectory)

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/topic5_ictal_recruitment/v2_band_scan"
FIGDIR = V2 / "figures"
TRAJ_CSV = V2 / "phase1_v2_alignment_trajectory.csv"
CON_CSV = V2 / "phase1_v2_trajectory_contrasts.csv"

POOL_N = {"narrow": 20, "broad": 17}
# Okabe-Ito colorblind-safe pool palette (matches the W2 phenotype figure).
POOL_C = {"narrow": "#CC79A7", "broad": "#009E73"}
# EEG = primary (dark/solid), clin = sensitivity (light/dashed).
ANCHOR_C = {"eeg": "#0072B2", "clin": "#E69F00"}
ANCHOR_LABEL = {"eeg": "EEG onset (primary)", "clin": "clinical onset (sensitivity)"}
REGION_LABEL = {"far_pre": "far\npre-ictal", "mid_pre": "mid\npre-ictal",
                "near_pre": "near\npre-ictal", "peri_onset": "peri-\nonset",
                "early_post": "early\npost-onset"}
CONTRAST_LABEL = {"near_pre_minus_far_pre": "near pre  -  far pre",
                  "post_minus_far_pre": "early post  -  far pre",
                  "post_minus_near_pre": "early post  -  near pre"}


def _wide(traj, anchor, pool):
    sub = traj[(traj.anchor == anchor) & (traj.pool == pool)]
    return subject_bin_wide(sub[["subject", "epoch_region", "subject_bin"]])


# ---------------------------------------------------------------------------
# Panel A — cohort scaffold-score trajectory (one facet per pool)
# ---------------------------------------------------------------------------
def _panel_A(ax, traj, pool, show_xlabel):
    xs = np.array([BIN_CENTERS[b] for b in BIN_ORDER])
    for anchor, dx, ls, mk, fill in (("eeg", -1.3, "-", "o", True), ("clin", +1.3, "--", "s", False)):
        coh = cohort_trajectory(_wide(traj, anchor, pool)).set_index("epoch_region").loc[BIN_ORDER]
        med = coh["cohort_median"].to_numpy()
        lo = med - coh["q25"].to_numpy()
        hi = coh["q75"].to_numpy() - med
        c = ANCHOR_C[anchor]
        ax.errorbar(xs + dx, med, yerr=[lo, hi], ls=ls, color=c, lw=1.9,
                    marker=mk, ms=6.5, mfc=(c if fill else "white"), mec=c, mew=1.4,
                    capsize=3, elinewidth=1.1, alpha=0.95, zorder=4,
                    label=ANCHOR_LABEL[anchor])
    ax.axvline(0.0, color="black", lw=1.3, ls=":", zorder=2)
    ax.text(0.0, 0.985, "EEG onset", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.25", style="italic")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{REGION_LABEL[b]}\n({int(BIN_CENTERS[b]):+d}s)" for b in BIN_ORDER], fontsize=8.5)
    ax.set_xlim(xs.min() - 8, xs.max() + 8)
    ax.set_ylabel(f"{pool}  (n={POOL_N[pool]})\nscaffold alignment score", fontsize=10)
    ax.grid(alpha=0.2)
    ax.tick_params(axis="y", labelsize=9)
    if show_xlabel:
        ax.set_xlabel("peri-ictal region  (bin center, seconds relative to EEG onset)", fontsize=10)
    ax.legend(loc="lower right", fontsize=8.3, frameon=True, framealpha=0.92)


# ---------------------------------------------------------------------------
# Panel B — the 3 paired contrasts + subject-level sign-flip significance
# ---------------------------------------------------------------------------
def _panel_B(ax, con):
    # 4 series per contrast group: (pool, anchor). eeg = solid, clin = hatched.
    series = [("narrow", "eeg"), ("narrow", "clin"), ("broad", "eeg"), ("broad", "clin")]
    offs = np.array([+0.30, +0.10, -0.10, -0.30])
    h = 0.18
    names = [n for n, _, _ in CONTRASTS]
    ybase = {n: len(names) - i for i, n in enumerate(names)}   # first contrast on top
    for name in names:
        for (pool, anchor), off in zip(series, offs):
            r = con[(con.contrast == name) & (con.pool == pool) & (con.anchor == anchor)].iloc[0]
            val, p = float(r.cohort_median_diff), float(r.p_signflip)
            y = ybase[name] + off
            hatch = "////" if anchor == "clin" else None
            ax.barh(y, val, height=h, color=POOL_C[pool], edgecolor="black", linewidth=0.5,
                    hatch=hatch, alpha=0.9 if anchor == "eeg" else 0.6, zorder=3)
            if np.isfinite(p) and p < 0.05:
                ax.text(val + (0.004 if val >= 0 else -0.004), y, "*",
                        ha="left" if val >= 0 else "right", va="center",
                        fontsize=13, fontweight="bold", color="black", zorder=5)
    ax.axvline(0.0, color="black", lw=1.2, zorder=2)
    ax.set_yticks([ybase[n] for n in names])
    ax.set_yticklabels([CONTRAST_LABEL[n] for n in names], fontsize=9.5)
    ax.set_ylim(0.45, len(names) + 0.6)
    ax.set_xlabel("cohort median of per-subject paired difference\n"
                  "(scaffold score;  > 0 = higher toward onset)", fontsize=9.5)
    ax.grid(alpha=0.22, axis="x")
    ax.tick_params(axis="x", labelsize=9)
    for i in range(len(names) - 1):
        ax.axhline(len(names) - i - 0.5, color="0.85", lw=0.8, zorder=1)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=POOL_C["narrow"], ec="k", alpha=0.9),
               plt.Rectangle((0, 0), 1, 1, fc=POOL_C["broad"], ec="k", alpha=0.9),
               plt.Rectangle((0, 0), 1, 1, fc="0.7", ec="k", alpha=0.9),
               plt.Rectangle((0, 0), 1, 1, fc="0.7", ec="k", hatch="////", alpha=0.6),
               plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="k",
                          markeredgecolor="k", markersize=12, linestyle="none")]
    labels = ["narrow (n=20)", "broad (n=17)", "EEG onset (solid)",
              "clinical onset (hatched)", "sign-flip p < 0.05"]
    ax.legend(handles, labels, loc="upper right", fontsize=8.0, frameon=True, framealpha=0.92, ncol=1)


def make_figure():
    traj = pd.read_csv(TRAJ_CSV, dtype={"subject": str})
    con = pd.read_csv(CON_CSV)

    fig = plt.figure(figsize=(14.6, 7.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.02], height_ratios=[1, 1],
                          hspace=0.30, wspace=0.30)
    axA0 = fig.add_subplot(gs[0, 0])
    axA1 = fig.add_subplot(gs[1, 0], sharex=axA0)
    axB = fig.add_subplot(gs[:, 1])

    _panel_A(axA0, traj, "narrow", show_xlabel=False)
    _panel_A(axA1, traj, "broad", show_xlabel=True)
    axA0.set_title("A  Cohort scaffold-score trajectory across peri-ictal time",
                   fontsize=11.5, loc="left", fontweight="bold")
    # coverage caveat (brief): far pre-ictal bin is thinner for large EEG-clinical-gap seizures.
    axA1.text(0.005, -0.42, "far pre-ictal bin is thinner: large EEG-clinical-gap seizures map their "
              "EEG -100..-60s beyond the -130s cache.\n~7% of windows (every bin) score over 5 of 7 "
              "primary bands -- the two 80-250 Hz HFA bands drop together there.\nMarkers = cohort "
              "median; bars = subject IQR (q25-q75).  EEG anchor offset left, clinical anchor offset right.",
              transform=axA1.transAxes, fontsize=7.6, color="0.35", va="top")

    _panel_B(axB, con)
    axB.set_title("B  Is any pre->post rise supported at the subject level? (sign-flip, unit = subject)",
                  fontsize=11.5, loc="left", fontweight="bold")

    fig.suptitle("Topic 5 interictal-HFO / peri-ictal energy-field alignment: peri-ictal scaffold "
                 "trajectory  -  high & roughly flat pre-ictally, modest onset-associated rise "
                 "(clearer under the EEG anchor)", fontsize=12, y=0.985)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / "phase1_v2_W3_trajectory.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    make_figure()
