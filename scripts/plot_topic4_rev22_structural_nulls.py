#!/usr/bin/env python3
"""Render the frozen rev22 structural-control aggregate without recomputation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ENDPOINTS = (
    ("D_support", "Support"), ("D_order", "Order"),
    ("D_time_ms", "Timing"), ("recall", "Recall"),
    ("kmeans_alignment", "KMeans"), ("ood", "OOD"),
)
LABELS = {
    "r180": "Rotate 180 deg", "r90": "Rotate 90 deg",
    "matched_norm_row_1": "Random EE pattern",
    "matched_norm_row_2": "Random E-to-I pattern",
    "merged_midpoint_core": "Merged Node field",
    "random_two_core_centers": "Random two-core field",
    "exact_dual_anchor": "Continuous dual field",
    "v62_density_t050": "Continuous single field",
}
BLUE, RED, GRAY = "#2878B5", "#C43C39", "#888888"


def _save(fig, root: Path, stem: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"{stem}.{suffix}", dpi=300 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _contrast_map(record: dict) -> dict:
    contrast = record.get("contrast") or record.get("within_isotropic_full_minus_M0000") or {}
    return contrast.get("endpoints") or {}


def _effect_grid(axes, rows: list[dict], row_labels: list[str], *, title: str) -> None:
    for column, (endpoint, label) in enumerate(ENDPOINTS):
        ax = axes[column]
        values = []
        for y, row in enumerate(rows):
            effect = _contrast_map(row).get(endpoint) or {}
            value, lo, hi = effect.get("delta"), effect.get("lo"), effect.get("hi")
            if value is None or lo is None or hi is None:
                continue
            values.extend([lo, hi])
            significant = lo > 0 or hi < 0
            ax.errorbar(float(value), y,
                        xerr=[[float(value) - float(lo)], [float(hi) - float(value)]],
                        fmt="o", ms=3.5, color=RED if significant else BLUE,
                        ecolor=RED if significant else BLUE, elinewidth=0.8, capsize=1.5)
            if significant:
                ax.text(float(hi), y - 0.23, "*", fontsize=7, ha="left", va="center")
        ax.axvline(0, color=GRAY, lw=0.7, ls="--")
        ax.set_title(label, fontsize=7, pad=3)
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=6)
        if column == 0:
            ax.set_yticks(range(len(rows)), row_labels, fontsize=6)
            ax.set_ylabel(title, fontsize=7)
        else:
            ax.set_yticks([])
        ax.set_xlabel("effect", fontsize=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.aggregate.read_text(encoding="utf-8"))
    if payload.get("schema_id") != "topic4_rev22_dci_structural_null_aggregate_v1":
        raise RuntimeError("unexpected structural aggregate schema")
    contrasts = payload["contrasts"]
    fixed = [row for row in contrasts
             if row["family"] == "fixed_topology" and row["condition"] == "full"]
    node = [row for row in contrasts if row["family"] == "node_blocking_factor"]
    iso = [row for row in contrasts
           if row["family"] == "isotropic_graph" and row["condition"] == "full"]
    if len(fixed) != 6 or len(node) != 2 or len(iso) != 1:
        raise RuntimeError("structural aggregate does not contain the frozen 6+2+1 contrasts")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(8.0, 4.8))
    outer = fig.add_gridspec(3, 1, height_ratios=(2.6, 1.2, 0.75), hspace=0.55)
    panels = []
    for row_index, (rows, labels, title) in enumerate((
        (fixed, [LABELS[r["candidate_id"].split("sn_", 1)[1].rsplit("_full", 1)[0]]
                 for r in fixed], "A  Matched structural null"),
        (node, [LABELS[r["candidate_id"].split("sn_node_", 1)[1].rsplit("_full", 1)[0]]
                for r in node], "B  Node blocking factor"),
        (iso, ["AR=1 topology"], "C  Within isotropic topology"),
    )):
        sub = outer[row_index].subgridspec(1, len(ENDPOINTS), wspace=0.35)
        axes = [fig.add_subplot(sub[0, i]) for i in range(len(ENDPOINTS))]
        _effect_grid(axes, rows, labels, title=title)
        panels.extend(axes)
    fig.suptitle("Structural sensitivity of the frozen connectivity result",
                 fontsize=9, fontweight="bold", y=0.995)
    _save(fig, args.out, "rev22_dci_structural_controls")
    readme = args.out / "README.md"
    readme.write_text(
        "### rev22_dci_structural_controls\n"
        "六列依次展示 held-out support、order、timing、recall、KMeans alignment 和 OOD。"
        "A 比较完整模型与六个同种子结构对照，B 检查连接效应在两个历史 Node 场下是否保留，"
        "C 是重建 AR=1 拓扑内部 full 相对 M0000 的配对变化；AR=1 与原拓扑之间不作配对推断。"
        "红色星号表示 90% 配对区间不跨 0，正值均表示前者更好。\n\n"
        "**关注点**：改善是否依赖原始 Node 位置、learned row、采样各向异性，及其是否跨 Node 场保持。\n",
        encoding="utf-8")


if __name__ == "__main__":
    main()
