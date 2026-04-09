#!/usr/bin/env python3
"""
Spatial modulation PR-1: visualization.

Generates:
  1. soz_vs_nonsoz_lag1r_paired.png — paired median iei_lag1_r
  2. soz_vs_nonsoz_deadtime_paired.png — paired median iei_p02 + iei_median
  3. event_rate_vs_lag1r_scatter.png — confound diagnostic
  4. perchannel_lag1r_by_soz.png — per-channel boxplot by SOZ label

Usage:
    python scripts/plot_spatial_modulation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RESULTS_DIR = Path("results/spatial_modulation")
FIG_DIR = RESULTS_DIR / "soz_comparison" / "figures"


def load_data():
    csv_path = RESULTS_DIR / "soz_comparison" / "cohort_soz_vs_nonsoz.csv"
    stats_path = RESULTS_DIR / "soz_comparison" / "cohort_statistics.json"

    df = pd.read_csv(csv_path)
    with open(stats_path) as f:
        cohort = json.load(f)
    return df, cohort


def plot_paired_metric(df, cohort, metric, ylabel, title, filename,
                       invert_y=False, min_group=3):
    """Paired dot-line plot: SOZ median vs non-SOZ median per subject."""
    paired = cohort.get("paired_data", [])
    if not paired:
        return

    soz_vals = []
    nonsoz_vals = []
    subj_labels = []
    for row in paired:
        s = row.get(f"soz_median_{metric}")
        n = row.get(f"nonsoz_median_{metric}")
        if s is not None and n is not None:
            soz_vals.append(s)
            nonsoz_vals.append(n)
            subj_labels.append(row["subject"][:6])

    if not soz_vals:
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    for i in range(len(soz_vals)):
        ax.plot([0, 1], [soz_vals[i], nonsoz_vals[i]], "o-", color="0.5",
                alpha=0.6, markersize=6, zorder=2)

    ax.errorbar(0, np.median(soz_vals), fmt="D", color="red",
                markersize=10, zorder=3, label="SOZ median")
    ax.errorbar(1, np.median(nonsoz_vals), fmt="D", color="blue",
                markersize=10, zorder=3, label="non-SOZ median")

    test = cohort.get("tests", {}).get(metric, {})
    p_val = test.get("wilcoxon_p", 1.0)
    n_subj = test.get("n_subjects", 0)
    n_greater = test.get("n_soz_greater", 0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["SOZ", "non-SOZ"])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nWilcoxon p={p_val:.4f}, n={n_subj}, SOZ>{n_greater}/{n_subj}")
    if invert_y:
        ax.invert_yaxis()
    ax.legend(loc="best")

    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_event_rate_scatter(df):
    """Scatter: event_rate vs iei_lag1_r, colored by SOZ label."""
    clean = df[(df["artifact_suspect"] == False) & df["iei_lag1_r"].notna()].copy()
    if clean.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    soz = clean[clean["soz_label"] == "soz"]
    nonsoz = clean[clean["soz_label"] == "non_soz"]

    ax.scatter(soz["event_rate"], soz["iei_lag1_r"],
               c="red", alpha=0.5, s=20, label=f"SOZ (n={len(soz)})", zorder=3)
    ax.scatter(nonsoz["event_rate"], nonsoz["iei_lag1_r"],
               c="blue", alpha=0.5, s=20, label=f"non-SOZ (n={len(nonsoz)})", zorder=2)

    ax.set_xlabel("Event rate (events/hour)")
    ax.set_ylabel("IEI lag-1 serial correlation")
    ax.set_xscale("log")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Event rate vs serial correlation\n(confound diagnostic)")
    ax.legend(loc="best")

    soz_rates = soz["event_rate"].values
    nonsoz_rates = nonsoz["event_rate"].values
    overlap_lo = max(soz_rates.min() if len(soz_rates) else 0,
                     nonsoz_rates.min() if len(nonsoz_rates) else 0)
    overlap_hi = min(soz_rates.max() if len(soz_rates) else 1e6,
                     nonsoz_rates.max() if len(nonsoz_rates) else 1e6)
    if overlap_lo < overlap_hi:
        ax.axvspan(overlap_lo, overlap_hi, alpha=0.08, color="green",
                   label=f"overlap [{overlap_lo:.0f}, {overlap_hi:.0f}]")
        ax.legend(loc="best")

    fig.tight_layout()
    out = FIG_DIR / "event_rate_vs_lag1r_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_perchannel_boxplot(df, metric, ylabel, title, filename):
    """Boxplot of per-channel metric by SOZ label, faceted by subject."""
    clean = df[(df["artifact_suspect"] == False) & df[metric].notna()].copy()
    if clean.empty:
        return

    subjects = sorted(clean["subject"].unique())
    n = len(subjects)
    fig, axes = plt.subplots(1, n, figsize=(max(2.5 * n, 8), 5), sharey=True)
    if n == 1:
        axes = [axes]

    for i, subj in enumerate(subjects):
        ax = axes[i]
        sub = clean[clean["subject"] == subj]
        soz_vals = sub[sub["soz_label"] == "soz"][metric].dropna().values
        nonsoz_vals = sub[sub["soz_label"] == "non_soz"][metric].dropna().values

        data = []
        labels = []
        if len(soz_vals):
            data.append(soz_vals)
            labels.append("SOZ")
        if len(nonsoz_vals):
            data.append(nonsoz_vals)
            labels.append("nSOZ")

        if data:
            bp = ax.boxplot(data, tick_labels=labels, widths=0.6)
        ax.set_title(subj[:8], fontsize=8)
        if i == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / filename
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_detrend_fraction_paired(df, cohort):
    """Paired plot for detrend_fraction — SOZ has less slow drift?"""
    plot_paired_metric(
        df, cohort,
        metric="detrend_fraction",
        ylabel="Detrend fraction",
        title="Detrend fraction: SOZ vs non-SOZ",
        filename="soz_vs_nonsoz_detrend_fraction_paired.png",
    )


def write_readme(cohort):
    """Generate figures/README.md in Chinese."""
    tests = cohort.get("tests", {})
    n_valid = cohort.get("n_valid_subjects", 0)

    lines = [
        "# 空间调制 PR-1 图表说明\n",
        f"> 有效配对被试数：{n_valid}（Yuquan-only，k=0.0 relaxed refine）\n",
        "",
    ]

    fig_descs = [
        ("soz_vs_nonsoz_lag1r_paired.png",
         "IEI lag-1 serial correlation 的被试内 SOZ vs non-SOZ 配对对比。",
         "如果 SOZ 系统性更高，表明 SOZ 网络有独立的慢状态记忆效应。"),
        ("soz_vs_nonsoz_deadtime_paired.png",
         "IEI 2nd percentile（dead-time proxy）的被试内配对对比。",
         "如果 SOZ 的 dead-time 更短，与 SOZ 兴奋性更高/恢复更快 的假说一致。"),
        ("event_rate_vs_lag1r_scatter.png",
         "每个通道的事件率 vs IEI serial correlation 散点图。红色=SOZ，蓝色=non-SOZ。"
         "绿色阴影标示两组事件率的重叠区间。",
         "这是推断前的**门控诊断图**。只有在两组的事件率支撑域有实质重叠时，"
         "serial correlation 的差异才可归因于 SOZ 属性而非事件率混淆。"),
        ("perchannel_lag1r_by_soz.png",
         "每个被试内，SOZ 和 non-SOZ 通道的 IEI lag-1 serial correlation 箱线图。",
         "显示被试间异质性和被试内 SOZ/non-SOZ 分布重叠程度。"),
        ("soz_vs_nonsoz_detrend_fraction_paired.png",
         "去趋势分数（detrend_fraction）的配对对比。Detrend fraction 表示慢漂移占原始 "
         "serial correlation 的比例。",
         "如果 SOZ 的 detrend fraction 更低，说明 SOZ 的 serial correlation 更多来自"
         "短程网络依赖而非全局慢漂移——即 SOZ 有自主调制。"),
    ]

    for fname, desc, focus in fig_descs:
        lines.append(f"### {fname}\n")
        lines.append(desc + "\n")
        lines.append(f"**关注点**：{focus}\n")
        lines.append("")

    # Add key statistics
    lines.append("### 统计摘要\n")
    lines.append("| 指标 | median_diff | Wilcoxon p | n | SOZ>nonSOZ |")
    lines.append("|------|-----------|------------|---|-----------|")
    for metric in ("iei_lag1_r", "iei_detrended_r", "detrend_fraction",
                   "iei_p02", "iei_median", "event_rate"):
        t = tests.get(metric, {})
        if t:
            lines.append(
                f"| {metric} | {t['median_diff']:.4f} | {t['wilcoxon_p']:.4f} "
                f"| {t['n_subjects']} | {t['n_soz_greater']}/{t['n_subjects']} |"
            )
    lines.append("")
    lines.append("### 注意事项\n")
    lines.append("- chengshuai 的 k=0.0 通道全部是 SOZ，无有效配对")
    lines.append("- zhangjiaqi 的全部通道 CV > 10（极端 bursty），被质控排除")
    lines.append("- Epilepsiae 的 gpu.npz 全部是损坏桩（216 bytes），per-channel 分析不可用")
    lines.append("- 当前分析为 **Yuquan-only 探索性分析**")
    lines.append("")

    readme_path = FIG_DIR / "README.md"
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved {readme_path}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df, cohort = load_data()

    # 1. Paired lag1_r
    plot_paired_metric(
        df, cohort,
        metric="iei_lag1_r",
        ylabel="Median IEI lag-1 r",
        title="IEI serial correlation: SOZ vs non-SOZ",
        filename="soz_vs_nonsoz_lag1r_paired.png",
    )

    # 2. Paired dead-time (iei_p02)
    plot_paired_metric(
        df, cohort,
        metric="iei_p02",
        ylabel="Median IEI 2nd percentile (s)",
        title="Dead-time proxy: SOZ vs non-SOZ",
        filename="soz_vs_nonsoz_deadtime_paired.png",
    )

    # 3. Paired iei_median
    plot_paired_metric(
        df, cohort,
        metric="iei_median",
        ylabel="Median IEI (s)",
        title="IEI median: SOZ vs non-SOZ",
        filename="soz_vs_nonsoz_iei_median_paired.png",
    )

    # 4. Event rate scatter
    plot_event_rate_scatter(df)

    # 5. Per-channel boxplot
    plot_perchannel_boxplot(
        df, "iei_lag1_r",
        ylabel="IEI lag-1 r",
        title="Per-channel serial correlation by SOZ label",
        filename="perchannel_lag1r_by_soz.png",
    )

    # 6. Detrend fraction paired
    plot_detrend_fraction_paired(df, cohort)

    # README
    write_readme(cohort)

    print("Done.")


if __name__ == "__main__":
    main()
