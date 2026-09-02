#!/usr/bin/env python3
"""Compact diagnostic for the rev21 topology-by-dynamics seed audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
COLORS = {
    "topology": "#C44E52",
    "dynamics": "#4C72B0",
    "interaction": "#8C8C8C",
}


def _heatmap(ax, values, title, fmt, cmap, topology, dynamics):
    values = np.asarray(values, float)
    image = ax.imshow(values, cmap=cmap, aspect="auto")
    midpoint = 0.5 * (float(np.nanmin(values)) + float(np.nanmax(values)))
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            color = "white" if values[row, col] > midpoint else "black"
            ax.text(col, row, format(values[row, col], fmt), ha="center",
                    va="center", fontsize=6.5, color=color)
    ax.set_title(title, fontsize=8.8, fontweight="bold")
    ax.set_xticks(range(len(dynamics)), [str(v) for v in dynamics], fontsize=6.7)
    ax.set_yticks(range(len(topology)), [str(v) for v in topology], fontsize=6.7)
    ax.tick_params(length=0)
    ax.set_xlabel("dynamics seed", fontsize=7.5)
    ax.set_ylabel("topology seed", fontsize=7.5)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=ARTIFACT_ROOT / (
            "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
            "seed_audit/seed_factorization_audit.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ARTIFACT_ROOT / (
            "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
            "seed_audit/figures"
        ),
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if payload.get("status") != "REV21_SEED_FACTORIZATION_COMPLETE":
        raise RuntimeError("seed factorization audit is incomplete")
    topology = payload["topology_seeds"]
    dynamics = payload["dynamics_seeds"]
    matrices = payload["endpoint_matrices"]

    fig = plt.figure(figsize=(7.15, 4.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.08))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
    specs = [
        ("training_complete_distribution", "Complete distribution", ".2f", "magma_r"),
        ("two_template_alignment", "Two-template alignment", ".2f", "viridis"),
        ("ood_all_returned", "OOD fraction", ".2f", "magma_r"),
        ("returned_family_count", "Returned families", ".0f", "viridis"),
    ]
    for label, ax, (key, title, fmt, cmap) in zip("ABCD", axes, specs):
        image = _heatmap(
            ax, matrices[key], title, fmt, cmap, topology, dynamics,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        ax.text(-0.35, 1.15, label, transform=ax.transAxes, fontsize=12,
                fontweight="bold", va="top")

    ax = fig.add_subplot(grid[:, 2])
    order = [
        "training_complete_distribution", "two_template_alignment",
        "ood_all_returned", "returned_family_count",
    ]
    labels = ["Distribution", "Template", "OOD", "Event count"]
    variance = payload["variance_decomposition"]
    left = np.zeros(len(order), float)
    for name, field in [
        ("topology", "variance_share_topology"),
        ("dynamics", "variance_share_dynamics"),
        ("interaction", "variance_share_residual_interaction"),
    ]:
        values = np.asarray([variance[key][field] for key in order], float)
        ax.barh(labels, values, left=left, height=0.58,
                color=COLORS[name], label=name)
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of crossed variance", fontsize=7.8)
    ax.tick_params(labelsize=7.0, length=2.5)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, -0.09))
    ax.set_title("Seed-source decomposition", fontsize=8.8, fontweight="bold")
    ax.text(-0.40, 1.10, "E", transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "seed-factorization-audit"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    readme = """### seed-factorization-audit.png

四张矩阵逐格展示固定双 core + Joint=1.25 底物在 3 个网络拓扑 seed 和 4 个动力学 seed 下的间期端点；每个格子都是独立的 topology-by-dynamics 单元。右侧把总变异拆成 topology、dynamics 和两者交互，避免把 pooled events 当独立重复。

**关注点**：12/12 单元都能自然形成两簇且两簇均出现；具体一致性仍受 topology 与 dynamics 交互影响，因此后续必须做多 seed 确认。
"""
    (args.output_dir / "README.md").write_text(readme)
    print(json.dumps({
        "status": "REV21_SEED_FACTORIZATION_FIGURE_COMPLETE",
        "png": str(stem.with_suffix(".png")),
        "pdf": str(stem.with_suffix(".pdf")),
    }, indent=2))


if __name__ == "__main__":
    main()
