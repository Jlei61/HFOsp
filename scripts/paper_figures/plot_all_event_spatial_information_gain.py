#!/usr/bin/env python3
"""Plot the no-hard-QC all-event Timing versus Timing+Space comparison."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TIMING_COLOR = "#5B7894"
SPACE_COLOR = "#C75D3A"
SHUFFLED_COLOR = "#999999"
TEXT = "#202020"
LIGHT = "#D0D0D0"
DEFAULT_ANALYSIS = (
    ROOT / "results/interictal_propagation_masked/"
    "spatial_information_gain_all_events"
)
DEFAULT_OUT = (
    ROOT / "results/paper-ready-figure/"
    "supp_all_event_spatial_information_gain/figures"
)


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=8, width=0.7, length=3)


def _violin_box(ax: plt.Axes, values: np.ndarray, x: float, color: str) -> None:
    violin = ax.violinplot(
        values,
        positions=[x],
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.20)
        body.set_linewidth(0.7)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    ax.plot([x, x], [q1, q3], color=color, lw=4.3, solid_capstyle="butt", zorder=4)
    ax.scatter([x], [median], s=17, color="white", edgecolor=color, lw=0.8, zorder=5)


def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                **row,
                "timing_only_score": float(row["timing_only_score"]),
                "timing_plus_space_score": float(row["timing_plus_space_score"]),
                "spatial_information_gain": float(row["spatial_information_gain"]),
                "train_spatial_shuffle_null_hybrid_median": float(
                    row["train_spatial_shuffle_null_hybrid_median"]
                ),
                "n_events_used_for_clustering": int(row["n_events_used_for_clustering"]),
                "n_direction_estimable": int(row["n_direction_estimable"]),
            })
    return rows


def plot(analysis_root: Path, out_dir: Path) -> dict:
    rows = _read_rows(analysis_root / "subject_spatial_information_gain.csv")
    payload = json.loads(
        (analysis_root / "spatial_information_gain_summary.json").read_text()
    )
    stats = payload["cohort_statistics"]
    train_null_stats = payload["training_spatial_shuffle_control"]["primary"]
    timing = np.asarray([row["timing_only_score"] for row in rows], float)
    shuffled_space = np.asarray([
        row["train_spatial_shuffle_null_hybrid_median"] for row in rows
    ], float)
    space = np.asarray([row["timing_plus_space_score"] for row in rows], float)
    gain = space - timing

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(7.15, 3.15), gridspec_kw={"width_ratios": [1.05, 1.35]}
    )
    order = np.argsort(gain)
    for index in order:
        left.plot(
            [0, 1, 2],
            [timing[index], shuffled_space[index], space[index]],
            color=LIGHT,
            lw=0.55,
            zorder=1,
        )
    _violin_box(left, timing, 0, TIMING_COLOR)
    _violin_box(left, shuffled_space, 1, SHUFFLED_COLOR)
    _violin_box(left, space, 2, SPACE_COLOR)
    left.scatter(
        np.zeros(len(timing)), timing, s=13, color=TIMING_COLOR,
        edgecolor="white", lw=0.35, zorder=5,
    )
    left.scatter(
        np.ones(len(shuffled_space)), shuffled_space, s=13,
        color=SHUFFLED_COLOR, edgecolor="white", lw=0.35, zorder=5,
    )
    left.scatter(
        np.full(len(space), 2.0), space, s=13, color=SPACE_COLOR,
        edgecolor="white", lw=0.35, zorder=5,
    )
    top = float(max(timing.max(), shuffled_space.max(), space.max()) + 0.035)
    left.plot([0, 0, 2, 2], [top, top + 0.015, top + 0.015, top], color=TEXT, lw=0.8)
    left.text(
        1.0, top + 0.020,
        _stars(float(stats["paired_wilcoxon_greater_p"])),
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
    null_top = top + 0.058
    left.plot(
        [1, 1, 2, 2],
        [null_top, null_top + 0.015, null_top + 0.015, null_top],
        color=TEXT, lw=0.8,
    )
    left.text(
        1.5, null_top + 0.020,
        _stars(float(train_null_stats["paired_wilcoxon_real_greater_p"])),
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )
    left.set_xticks(
        [0, 1, 2], ["Timing", "Shuffled\nspace", "Timing +\nspace"]
    )
    left.set_ylabel("Held-out direction score", fontsize=9)
    left.set_xlim(-0.48, 2.48)
    left.set_ylim(
        min(-0.05, float(min(timing.min(), shuffled_space.min(), space.min()) - 0.04)),
        null_top + 0.070,
    )
    _style(left)

    ordered = sorted(rows, key=lambda row: float(row["spatial_information_gain"]))
    y = np.arange(len(ordered))
    gains = np.asarray([float(row["spatial_information_gain"]) for row in ordered])
    right.hlines(y, 0, gains, color=LIGHT, lw=0.7)
    right.scatter(gains, y, s=18, color=SPACE_COLOR, edgecolor="white", lw=0.4, zorder=3)
    right.axvline(0, color="#777777", lw=0.75, ls=(0, (3, 2)))
    median = float(stats["median_gain"])
    ci_lo, ci_hi = map(float, stats["median_gain_bootstrap_ci95"])
    summary_y = len(ordered) + 0.8
    right.errorbar(
        median, summary_y,
        xerr=np.asarray([[median - ci_lo], [ci_hi - median]]),
        fmt="D", ms=4.2, color=TEXT, mfc="white", capsize=2.2, lw=1.0,
    )
    right.set_yticks([])
    right.set_ylim(-1.0, len(ordered) + 2.0)
    right.set_xlabel("Δ direction score (+space − Timing)", fontsize=9)
    _style(right)
    fig.tight_layout(w_pad=2.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "all_event_spatial_information_gain"
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metadata = {
        "schema_id": "all_event_spatial_information_gain_figure_v2",
        "analysis_root": str(analysis_root),
        "n": len(rows),
        "statistics": stats,
        "training_spatial_shuffle_control": train_null_stats,
        "n_events_used_for_clustering": int(sum(
            row["n_events_used_for_clustering"] for row in rows
        )),
        "n_direction_estimable_for_possible_scoring": int(sum(
            row["n_direction_estimable"] for row in rows
        )),
        "hard_event_qc_used": False,
        "missing_spatial_view": "masked; event retained through timing",
        "training_spatial_null": (
            "finite directions shuffled within training fold and recording block; "
            "missingness fixed; Hybrid refitted"
        ),
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    (out_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        "# 全事件空间信息增益\n\n"
        "### all_event_spatial_information_gain.png / .pdf\n\n"
        "左侧比较同一患者的 Timing、训练折内按记录块打乱方向后重拟合的 "
        "Shuffled space，以及 Timing+Space held-out direction score；右侧显示"
        "逐患者的 Timing+Space 相对 Timing 增益和中位数 95% bootstrap CI。全部间期事件均参与聚类，"
        "方向无法数学定义的事件只缺失空间视图，不从 timing template 中删除。\n\n"
        "**关注点**：方向得分只在两种模型共同可评分的最小方向可估事件上计算；"
        "这不会改变两种聚类所使用的全事件总体。\n",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(json.dumps(plot(args.analysis_root, args.out_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
