#!/usr/bin/env python3
"""Render the frozen dual-core Node pathway-factorization panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_ood_node_pathways.json"
ARM_IDS = (
    "frozen_dualcore_node",
    "frozen_dualcore_ee",
    "frozen_dualcore_etoi",
    "frozen_dualcore_both",
)
ARM_LABELS = ("Node", "+EE", "+E-to-I", "+EE+EI")
ARM_COLORS = ("#777777", "#d7892f", "#2a9d8f", "#202020")
MODE_COLORS = ("#c43c39", "#277da1")


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.2,
        "ytick.labelsize": 6.2,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _bootstrap(values: np.ndarray, seed: int, draws: int = 4096) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(int(seed))
    sampled = np.mean(
        values[rng.integers(0, len(values), size=(draws, len(values)))], axis=1,
    )
    return (
        float(np.mean(values)), float(np.quantile(sampled, 0.05)),
        float(np.quantile(sampled, 0.95)),
    )


def _paired_significant(left: np.ndarray, right: np.ndarray, seed: int) -> bool:
    left, right = np.asarray(left, float), np.asarray(right, float)
    finite = np.isfinite(left) & np.isfinite(right)
    delta = left[finite] - right[finite]
    if len(delta) < 3:
        return False
    rng = np.random.default_rng(int(seed))
    sampled = np.mean(
        delta[rng.integers(0, len(delta), size=(4096, len(delta)))], axis=1,
    )
    low, high = np.quantile(sampled, (0.05, 0.95))
    return bool(low > 0 or high < 0)


def _bracket(axis, left: int, right: int, y: float, height: float) -> None:
    axis.plot(
        [left, left, right, right], [y, y + height, y + height, y],
        color="#555555", lw=0.65, clip_on=False,
    )
    axis.text(
        0.5 * (left + right), y + 1.25 * height, "*", ha="center",
        va="bottom", fontsize=7.5, color="#333333", clip_on=False,
    )


def _metric_arrays(analysis: dict) -> dict[str, np.ndarray]:
    arms = analysis["pathway_arms"]
    seeds = sorted(map(int, arms[ARM_IDS[0]]["per_network"]))
    output = {}
    for mode in (0, 1):
        output[f"Mode {mode + 1} share (%)"] = np.asarray([
            [
                100.0 * arms[arm]["per_network"][str(seed)][
                    "mode_fraction_in_support"
                ][mode]
                for arm in ARM_IDS
            ]
            for seed in seeds
        ], float)
    output["KMeans match (%)"] = np.asarray([
        [
            100.0 * arms[arm]["per_network"][str(seed)]["natural_kmeans"].get(
                "direction_balanced_alignment", np.nan,
            )
            for arm in ARM_IDS
        ]
        for seed in seeds
    ], float)
    output["OOD (%)"] = np.asarray([
        [
            100.0 * arms[arm]["per_network"][str(seed)]["ood_all_returned"]
            for arm in ARM_IDS
        ]
        for seed in seeds
    ], float)
    return output


def render(config_path: Path, output_dir: Path) -> dict:
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    analysis_path = root / "final_analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis.get("status") != "DUAL_CORE_OOD_NODE_AND_PATHWAYS_ANALYZED":
        raise RuntimeError("final dual-core analysis is incomplete")
    metrics = _metric_arrays(analysis)
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.55))
    statistics = {}
    for metric_index, (axis, (title, values)) in enumerate(zip(axes, metrics.items())):
        x = np.arange(len(ARM_IDS))
        for row in values:
            finite = np.isfinite(row)
            axis.plot(x[finite], row[finite], color="#b8b8b8", lw=0.45, alpha=0.5)
            axis.scatter(
                x[finite], row[finite], s=7, facecolor="white", edgecolor="#8e8e8e",
                linewidth=0.35, alpha=0.8, zorder=2,
            )
        summaries = []
        for arm_index, color in enumerate(ARM_COLORS):
            mean, low, high = _bootstrap(
                values[:, arm_index], 20260830 + 20 * metric_index + arm_index,
            )
            summaries.append({"mean": mean, "q05": low, "q95": high})
            axis.errorbar(
                arm_index, mean, yerr=[[mean - low], [high - mean]], fmt="o",
                ms=4.0, color=color, ecolor=color, elinewidth=1.0,
                capsize=2.0, capthick=0.8, zorder=4,
            )
        significant = []
        y_top = 92.0
        for arm_index in range(1, len(ARM_IDS)):
            is_significant = _paired_significant(
                values[:, arm_index], values[:, 0],
                20260930 + 20 * metric_index + arm_index,
            )
            significant.append(is_significant)
            if is_significant:
                _bracket(axis, 0, arm_index, y_top, 1.8)
                y_top += 7.0
        statistics[title] = {
            "equal_network": summaries,
            "paired_90pct_interval_excludes_zero_vs_node": significant,
        }
        axis.set_title(title, loc="left", fontweight="bold", pad=3.0)
        if metric_index < 2:
            axis.title.set_color(MODE_COLORS[metric_index])
        axis.set_xticks(x, ARM_LABELS, rotation=28, ha="right", rotation_mode="anchor")
        axis.set_ylim(0, 112)
        axis.tick_params(length=2.2, width=0.55, pad=1.5)
        axis.spines["left"].set_color("#777777")
        axis.spines["bottom"].set_color("#777777")
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.25, top=0.91, wspace=0.42)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "dual_core_node_pathway_factorization"
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    fig.savefig(
        stem.with_suffix(".pdf"), facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    plt.close(fig)
    metadata = {
        "status": "DUAL_CORE_NODE_PATHWAY_FIGURE_RENDERED",
        "arm_order": list(ARM_IDS),
        "arm_labels": list(ARM_LABELS),
        "statistics": statistics,
        "star_rule": "paired network bootstrap 90% interval excludes zero",
        "primary_endpoint": "OOD_all_returned",
        "kmeans_population": "formal-clean events inside frozen patient support",
        "mode_labels_have_no_pathological_meaning": True,
        "analysis": str(analysis_path.relative_to(ROOT)),
        "outputs": [
            str(stem.with_suffix(".png").relative_to(ROOT)),
            str(stem.with_suffix(".pdf").relative_to(ROOT)),
        ],
    }
    (output_dir / "dual_core_node_pathway_factorization_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (output_dir / "README.md").write_text(
        "### dual_core_node_pathway_factorization.png\n"
        "冻结双 core Node 后，在相同 12 张网络上比较 Node、+EE、+E-to-I 和 +EE+EI。"
        "灰线连接同一网络，彩色点为 equal-network mean，误差条为 90% network bootstrap；"
        "星号只表示相对 Node 的配对区间排除 0。\n\n"
        "**关注点**：OOD 是主端点；模式占比和 natural KMeans 只用于判断连接重分配改变了"
        "双模式可达性还是仅改变事件产率。\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir or (
        ROOT / config["output_root"] / "pathway/figures"
    )
    metadata = render(args.config.resolve(), output.resolve())
    print(json.dumps({
        "status": metadata["status"], "outputs": metadata["outputs"],
    }, indent=2))


if __name__ == "__main__":
    main()
