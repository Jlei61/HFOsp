#!/usr/bin/env python3
"""Compact paper-style diagnostic for held-out event and mode-contrast R2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "interictal_variance_capture/variance_capture.json"
)
OUTPUT = INPUT.parent / "figures/interictal_variance_capture"
ORDER = (
    "node_baseline",
    "joint_04_ee_only",
    "joint_04_etoi_only",
    "joint_04_control",
)
LABELS = ("Node", "+E-E", "+E-to-I", "+EE+EI")
COLORS = ("#808080", "#D98B35", "#1D9A8A", "#252525")


def render(input_path: Path, output_stem: Path) -> None:
    data = json.loads(input_path.read_text())
    patient_r2 = float(data["patient"]["patient_train_k2_r2_on_heldout"])
    rows = [data["arms"][arm] for arm in ORDER]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    ax = axes[0]
    values = np.asarray([
        row["components"]["all"]["model_r2_on_patient_heldout"] for row in rows
    ])
    intervals = np.asarray([
        [row["hierarchical_bootstrap"]["r2"]["q05"],
         row["hierarchical_bootstrap"]["r2"]["q95"]]
        for row in rows
    ])
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.axhline(patient_r2, color="#B92C35", lw=1.2, ls="--")
    ax.bar(x, values, width=0.62, color=COLORS, edgecolor="none")
    ax.errorbar(
        x, values, yerr=np.vstack([values - intervals[:, 0], intervals[:, 1] - values]),
        fmt="none", ecolor="#202020", elinewidth=0.9, capsize=2.2,
    )
    ax.set(
        xticks=x, xticklabels=LABELS,
        ylabel=r"Held-out event-cloud $R^2$",
        title="Complete event distribution",
    )
    ax.text(
        0.98, patient_r2, f"patient K=2  {patient_r2:.3f}",
        color="#B92C35", fontsize=7.2, ha="right", va="bottom",
        transform=ax.get_yaxis_transform(),
    )

    ax = axes[1]
    raw = np.asarray([
        row["between_mode_contrast"]["heldout_raw_contrast_r2"] for row in rows
    ])
    scaled = np.asarray([
        row["between_mode_contrast"]["heldout_scale_calibrated_contrast_r2"]
        for row in rows
    ])
    scaled_intervals = np.asarray([
        [row["hierarchical_bootstrap"]["between_mode_contrast"]
         ["heldout_scale_calibrated_contrast_r2"]["q05"],
         row["hierarchical_bootstrap"]["between_mode_contrast"]
         ["heldout_scale_calibrated_contrast_r2"]["q95"]]
        for row in rows
    ])
    width = 0.30
    ax.axhline(0.0, color="#555555", lw=0.8)
    ax.bar(x - width / 2, raw, width=width, color=COLORS, alpha=0.38,
           edgecolor="none", label="raw")
    ax.bar(x + width / 2, scaled, width=width, color=COLORS,
           edgecolor="none", label="train-scaled")
    ax.errorbar(
        x + width / 2, scaled,
        yerr=np.vstack([scaled - scaled_intervals[:, 0],
                        scaled_intervals[:, 1] - scaled]),
        fmt="none", ecolor="#202020", elinewidth=0.9, capsize=2.2,
    )
    ax.set(
        xticks=x, xticklabels=LABELS,
        ylabel=r"Held-out TA-TB contrast $R^2$",
        title="Between-mode geometry",
        ylim=(0.0, max(0.72, float(np.max(scaled_intervals)) + 0.04)),
    )
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=7.5)
        axis.title.set_fontsize(9.2)
        axis.title.set_fontweight("bold")
        axis.set_axisbelow(True)
        axis.yaxis.grid(True, color="#E4E4E4", lw=0.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    (output_stem.parent / "README.md").write_text(
        f"""### {output_stem.name}.png

左图用患者训练集全局均值作为零基线，比较患者自身 K=2 原型和四个冻结模型臂对患者留出 recording blocks 中完整事件云的解释率。右图只比较 TA-TB 模式差异向量；浅色为模型原始幅值，实色只允许在患者训练集拟合一个非负整体尺度，再在留出数据评分。误差线为 recording-block 与 network 两层 bootstrap 的 5-95% 区间。

**关注点**：模式差异几何可部分恢复，不等于完整事件分布被解释；负的 event-cloud R2 表示模型原型不如患者训练集的单一全局均值，不应解释成负百分比方差。
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    render(args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
