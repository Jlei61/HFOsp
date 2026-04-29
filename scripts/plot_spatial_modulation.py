#!/usr/bin/env python3
"""
Spatial modulation PR-1 (Yuquan SOZ vs non-SOZ) + PR-2 (Epilepsiae i/l/e) visualization.

PR-1 (Yuquan, default):
  1. soz_vs_nonsoz_lag1r_paired.png — paired median iei_lag1_r
  2. soz_vs_nonsoz_deadtime_paired.png — paired median iei_p02 + iei_median
  3. event_rate_vs_lag1r_scatter.png — confound diagnostic
  4. perchannel_lag1r_by_soz.png — per-channel boxplot by SOZ label

PR-2 (Epilepsiae, --dataset epilepsiae):
  1. ep_three_tier_<metric>_box.png — per-metric i/l/e boxplots across cohort
  2. ep_three_tier_<metric>_paired.png — subject-level i/l/e medians dot-line
  3. ep_three_tier_summary_table.csv — per-metric Wilcoxon + Bonferroni + monotonicity
  Output goes to results/spatial_modulation/figures/epilepsiae/.

Usage:
    python scripts/plot_spatial_modulation.py                       # Yuquan PR-1
    python scripts/plot_spatial_modulation.py --dataset epilepsiae  # PR-2
"""

from __future__ import annotations

import argparse
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
EP_FIG_DIR = RESULTS_DIR / "figures" / "epilepsiae"


def _label_col(df: pd.DataFrame) -> str:
    """Return the region/SOZ column name in the CSV (handles old `soz_label` and new `region_label`)."""
    if "region_label" in df.columns:
        return "region_label"
    if "soz_label" in df.columns:
        return "soz_label"
    raise KeyError("Neither 'region_label' nor 'soz_label' found in DataFrame")


def load_data():
    csv_path = RESULTS_DIR / "soz_comparison" / "cohort_soz_vs_nonsoz.csv"
    stats_path = RESULTS_DIR / "soz_comparison" / "cohort_statistics.json"

    df = pd.read_csv(csv_path)
    with open(stats_path) as f:
        cohort = json.load(f)
    return df, cohort


def load_data_epilepsiae():
    csv_path = RESULTS_DIR / "soz_comparison" / "epilepsiae" / "cohort_i_l_e.csv"
    stats_path = RESULTS_DIR / "soz_comparison" / "epilepsiae" / "cohort_three_tier_statistics.json"

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

    label_col = _label_col(clean)
    soz = clean[clean[label_col] == "soz"]
    nonsoz = clean[clean[label_col] == "non_soz"]

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

    label_col = _label_col(clean)
    for i, subj in enumerate(subjects):
        ax = axes[i]
        sub = clean[clean["subject"] == subj]
        soz_vals = sub[sub[label_col] == "soz"][metric].dropna().values
        nonsoz_vals = sub[sub[label_col] == "non_soz"][metric].dropna().values

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


EP_METRIC_CONFIG = [
    ("iei_detrended_r",  "IEI lag-1 r (detrended)",     "去趋势 lag-1 序列相关"),
    ("detrend_fraction", "Detrend fraction",             "去趋势分数（高 = 慢漂移占比大）"),
    ("iei_median",       "IEI median (s)",               "事件间隔中位数"),
    ("iei_p02",          "IEI 2nd percentile (s)",       "Dead-time 末梢"),
    ("iei_lag1_r",       "IEI lag-1 r (raw)",            "原始 lag-1 序列相关"),
    ("iei_cv",           "IEI CV",                        "IEI 离散程度"),
    ("event_rate",       "Event rate (events/hour)",     "事件率"),
]
EP_REGIONS = [("i", "SOZ (i)", "tab:red"),
              ("l", "Lesion (l)", "tab:orange"),
              ("e", "Extra-focal (e)", "tab:blue")]


def plot_epilepsiae_three_tier_box(df: pd.DataFrame, metric: str, ylabel: str, out: Path) -> None:
    """Per-metric channel-level boxplot grouped by region (i/l/e)."""
    label_col = _label_col(df)
    clean = df[(df["artifact_suspect"] == False) & df[metric].notna()].copy()
    if clean.empty:
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    data = []
    labels = []
    colors = []
    for region, name, color in EP_REGIONS:
        vals = clean[clean[label_col] == region][metric].values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(f"{name}\nn={len(vals)}")
        colors.append(color)
    if not data:
        plt.close(fig)
        return
    bp = ax.boxplot(data, tick_labels=labels, widths=0.6, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.5)
    # Strip plot per region
    for k, vals in enumerate(data):
        x = np.full(len(vals), k + 1) + np.random.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x, vals, color=colors[k], alpha=0.5, s=14, zorder=2)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Epilepsiae per-channel {metric}\n(i/l/e relaxed-refine, k=0.0)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def plot_epilepsiae_three_tier_paired(cohort: dict, metric: str, ylabel: str, out: Path) -> None:
    """Subject-level i/l/e dot-line paired plot."""
    paired = cohort.get("paired_data", [])
    if not paired:
        return
    rows = []
    for r in paired:
        i = r.get(f"i_median_{metric}")
        l = r.get(f"l_median_{metric}")
        e = r.get(f"e_median_{metric}")
        if i is not None and l is not None and e is not None:
            rows.append((r["subject"], float(i), float(l), float(e)))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for subj, i, l, e in rows:
        ax.plot([0, 1, 2], [i, l, e], "o-", color="0.6", alpha=0.5, markersize=5, zorder=2)
    medians = np.median([(i, l, e) for _, i, l, e in rows], axis=0)
    ax.plot([0, 1, 2], medians, "D-", color="black", markersize=10, lw=2, zorder=3, label="cohort median")
    test = cohort.get("tests", {}).get(metric, {})
    direction = test.get("direction", "?")
    mono = test.get("monotonicity", {}) or {}
    parts = []
    pt = test.get("pair_tests", {})
    for name in ("i_vs_l", "i_vs_e", "l_vs_e"):
        pr = pt.get(name)
        if pr:
            p = pr.get("wilcoxon_p_bonferroni", pr.get("wilcoxon_p_two_sided", np.nan))
            parts.append(f"{name}:p={p:.3f}")
    mono_str = ""
    if mono.get("binomial_p_one_sided") is not None:
        mono_str = f"  mono={mono['n_monotonic']}/{mono['n_subjects']} p={mono['binomial_p_one_sided']:.3f}"
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([r[1] for r in EP_REGIONS])
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric} (dir={direction})\n{' '.join(parts)}{mono_str}", fontsize=10)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def write_summary_table_epilepsiae(cohort: dict, out: Path) -> pd.DataFrame:
    rows = []
    for metric, t in cohort.get("tests", {}).items():
        if "skipped" in t:
            rows.append({"metric": metric, "skipped": t["skipped"]})
            continue
        pt = t.get("pair_tests", {})
        mono = t.get("monotonicity") or {}
        rows.append({
            "metric": metric,
            "direction": t.get("direction"),
            "n_subjects": t.get("n_subjects"),
            "i_median_cohort": t.get("i_median_cohort"),
            "l_median_cohort": t.get("l_median_cohort"),
            "e_median_cohort": t.get("e_median_cohort"),
            "p_il_bonf": pt.get("i_vs_l", {}).get("wilcoxon_p_bonferroni"),
            "p_ie_bonf": pt.get("i_vs_e", {}).get("wilcoxon_p_bonferroni"),
            "p_le_bonf": pt.get("l_vs_e", {}).get("wilcoxon_p_bonferroni"),
            "n_mono": mono.get("n_monotonic"),
            "frac_mono": mono.get("fraction_monotonic"),
            "p_mono_binomial": mono.get("binomial_p_one_sided"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Saved {out}")
    return df


def write_readme_epilepsiae(cohort: dict) -> None:
    n_valid = cohort.get("n_valid_subjects", 0)
    subjects = cohort.get("subjects", [])
    tests = cohort.get("tests", {})
    lines = [
        "# 空间调制 PR-2 — Epilepsiae i/l/e 三层梯度图表说明\n",
        f"> 有效配对被试数：{n_valid}（含 i / l / e 各 ≥3 通道；共 {len(subjects)} 个：{', '.join(subjects)}）\n",
        f"> 数据合同：新 pipeline `results/hfo_detection/<subject>/`（legacy gpu_npz 全 stub，详见 `docs/archive/topic3/epilepsiae_artifact_census_2026-04-27.md`）\n",
        "",
        "## 图说",
        "",
    ]
    for metric, ylabel, ch_desc in EP_METRIC_CONFIG:
        lines.append(f"### ep_three_tier_{metric}_box.png\n")
        lines.append(f"每通道 {ch_desc} 按 i (SOZ) / l (lesion) / e (extra-focal) 分组的箱线图（散点为单通道值）。\n")
        lines.append(f"**关注点**：i 与 e 之间 {ch_desc} 是否呈预期方向梯度；l 是否在两者之间。\n")
        lines.append("")
        lines.append(f"### ep_three_tier_{metric}_paired.png\n")
        lines.append(f"被试内 i / l / e 的中位 {ch_desc} 配对折线图，黑色钻石为 cohort 中位线。\n")
        t = tests.get(metric, {})
        direction = t.get("direction", "?")
        lines.append(f"**关注点**：方向 = `{direction}`；按预期 i→e 应当单调（greater = i 高 / less = i 低）。"
                     "标题展示 Bonferroni 后三对 paired Wilcoxon p 与单调 sign-test p（null=1/6）。\n")
        lines.append("")
    lines.append("## 统计摘要表\n")
    lines.append("详细数值见 `ep_three_tier_summary_table.csv`。各列含义：")
    lines.append("- `p_il_bonf` / `p_ie_bonf` / `p_le_bonf`：三对 paired Wilcoxon Bonferroni 校正 p（按 metric 方向 one-sided 或 two-sided）")
    lines.append("- `n_mono` / `frac_mono` / `p_mono_binomial`：被试 (i, l, e) 中位严格单调（按方向）的计数 + 比例 + binomtest（null=1/6）单边 p")
    lines.append("- `event_rate` 列出三层 cohort 中位数和两两比较，但**不**做单调假设检验（已知 SOZ confound）")
    lines.append("")
    lines.append("## 注意")
    lines.append("- 当方向为 `two-sided` 时（`iei_lag1_r` / `iei_cv`），单调列为空。")
    lines.append("- focus_rel 缺失或单一区域 < 3 通道的 subject 自动剔除（见 `cohort_three_tier_statistics.json::subjects` 列表）。")
    lines.append("")
    readme = EP_FIG_DIR / "README.md"
    readme.parent.mkdir(parents=True, exist_ok=True)
    readme.write_text("\n".join(lines))
    print(f"Saved {readme}")


def main_yuquan():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, cohort = load_data()
    plot_paired_metric(df, cohort, metric="iei_lag1_r",
                       ylabel="Median IEI lag-1 r",
                       title="IEI serial correlation: SOZ vs non-SOZ",
                       filename="soz_vs_nonsoz_lag1r_paired.png")
    plot_paired_metric(df, cohort, metric="iei_p02",
                       ylabel="Median IEI 2nd percentile (s)",
                       title="Dead-time proxy: SOZ vs non-SOZ",
                       filename="soz_vs_nonsoz_deadtime_paired.png")
    plot_paired_metric(df, cohort, metric="iei_median",
                       ylabel="Median IEI (s)",
                       title="IEI median: SOZ vs non-SOZ",
                       filename="soz_vs_nonsoz_iei_median_paired.png")
    plot_event_rate_scatter(df)
    plot_perchannel_boxplot(df, "iei_lag1_r", ylabel="IEI lag-1 r",
                            title="Per-channel serial correlation by SOZ label",
                            filename="perchannel_lag1r_by_soz.png")
    plot_detrend_fraction_paired(df, cohort)
    write_readme(cohort)
    print("Done.")


def main_epilepsiae():
    EP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    df, cohort = load_data_epilepsiae()
    for metric, ylabel, _ in EP_METRIC_CONFIG:
        plot_epilepsiae_three_tier_box(
            df, metric, ylabel,
            EP_FIG_DIR / f"ep_three_tier_{metric}_box.png",
        )
        plot_epilepsiae_three_tier_paired(
            cohort, metric, ylabel,
            EP_FIG_DIR / f"ep_three_tier_{metric}_paired.png",
        )
    write_summary_table_epilepsiae(cohort, EP_FIG_DIR / "ep_three_tier_summary_table.csv")
    write_readme_epilepsiae(cohort)
    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae"], default="yuquan")
    args = parser.parse_args()
    if args.dataset == "yuquan":
        main_yuquan()
    else:
        main_epilepsiae()


if __name__ == "__main__":
    main()
