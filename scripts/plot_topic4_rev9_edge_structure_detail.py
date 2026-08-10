"""Plot the 12-network rev9 edge structural sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SUMMARY = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/edge_structure_detail/"
    "edge_structure_detail_summary.json")


def _panel_label(axis, label):
    axis.text(-0.15, 1.08, label, transform=axis.transAxes,
              fontsize=10, fontweight="bold", va="top")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-dir")
    args = parser.parse_args()
    payload = json.loads(Path(args.summary).read_text())
    out_dir = Path(args.out_dir or Path(args.summary).parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [value.replace("component_", "C").replace("background", "BG")
              for value in payload["labels"]]
    summaries = payload["summaries"]
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)

    axis = axes[0]
    change = 100.0 * (np.asarray(summaries["flow_ratio"]["estimate"]) - 1.0)
    limit = max(1.0, float(np.nanmax(np.abs(change))))
    image = axis.imshow(change, cmap="RdBu_r", vmin=-limit, vmax=limit)
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, f"{change[row, column]:+.1f}",
                      ha="center", va="center", fontsize=6.5)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("source group")
    axis.set_ylabel("target group")
    axis.set_title("incoming weight is redistributed", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    colorbar.set_label("paired median change (%)")
    _panel_label(axis, "a")

    axis = axes[1]
    estimate = np.asarray(summaries["group_outgoing_ratio"]["estimate"])
    low, high = np.asarray(summaries["group_outgoing_ratio"]["interval_95"])
    x = np.arange(len(labels))
    axis.axhline(1.0, color="#777777", linewidth=0.7)
    axis.errorbar(x, estimate, yerr=[estimate - low, high - estimate],
                  fmt="o", color="#0072B2", capsize=2.5)
    axis.set_xticks(x, labels)
    axis.set_ylabel("new / old source outgoing influence")
    axis.set_title("source columns are not conserved", loc="left", fontweight="bold")
    _panel_label(axis, "b")

    axis = axes[2]
    estimate = np.asarray(summaries["group_target_delay_delta_ms"]["estimate"])
    low, high = np.asarray(summaries["group_target_delay_delta_ms"]["interval_95"])
    axis.axhline(0.0, color="#777777", linewidth=0.7)
    axis.errorbar(x, estimate, yerr=[estimate - low, high - estimate],
                  fmt="o", color="#D55E00", capsize=2.5)
    axis.set_xticks(x, labels)
    axis.set_ylabel("weighted incoming delay change (ms)")
    axis.set_title("delay labels stay fixed", loc="left", fontweight="bold")
    _panel_label(axis, "c")

    stem = out_dir / "rev9_edge_structure_detail"
    fig.savefig(stem.with_suffix(".png"), dpi=260, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    (out_dir / "README.md").write_text(
        "### rev9_edge_structure_detail.png\n\n"
        "这张图解释冻结 alpha=0.75 在 12 张 factorial 网络中实际改变了什么。左图按 target 行、source 列展示 component/background 之间的 E->E 权重流变化；中图显示 source outgoing influence，右图显示固定 delay labels 下的加权输入延迟变化。\n\n"
        "component membership 定义为 h 乘 raw Gaussian responsibility，background 为 1-h，因此远场尾部不会被强行归入某个 Gaussian。\n\n"
        "**关注点**：每个 target 的 incoming-E 总量不变，但 source column influence 和 component-pair flow 会变；这应称 field-assortative redistribution，而不是新 topology 或严格局部 core。\n")
    print(json.dumps({"status": "REV9_EDGE_STRUCTURE_DETAIL_FIGURE_COMPLETE",
                      "figure": str(stem.with_suffix('.png'))}, indent=2))


if __name__ == "__main__":
    main()
