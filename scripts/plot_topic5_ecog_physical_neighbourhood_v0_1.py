#!/usr/bin/env python3
"""Paper-ready two-panel summary of ECoG physical-neighbour learning and lesion tests."""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import coordinate_array  # noqa: E402


RESULT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")
COLORS = {
    "TRUE_GRID": "#2B6CB0",
    "WRONG_GRID": "#D97706",
    "DEGREE_RANDOM": "#7A5195",
    "SUFFIX_SHUFFLED": "#737373",
    "958": "#B2182B",
    "1084": "#2166AC",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def load_graph(subject: str, graph_id: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    path = RESULT / f"graphs/{subject}/four_neighbour/{graph_id}.npz"
    with np.load(path, allow_pickle=False) as data:
        names = [str(value) for value in data["channel_names"].tolist()]
        xy = np.asarray(data["coordinates"], dtype=float)
        mask = np.asarray(data["mask"], dtype=bool)
    return names, xy, mask


def draw_graph(ax: plt.Axes, graph_id: str, label: str, color: str) -> None:
    names, xy, mask = load_graph("958", graph_id)
    _, _, truth = load_graph("958", "TRUE_GRID")
    center = names.index("GE5")
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if truth[first, second]:
                ax.plot(
                    [xy[first, 1], xy[second, 1]], [7 - xy[first, 0], 7 - xy[second, 0]],
                    color="#B8B8B8", lw=0.45, alpha=0.72, zorder=0,
                )
    neighbours = np.flatnonzero(mask[:, center] | mask[center])
    for neighbour in neighbours:
        ax.plot(
            [xy[center, 1], xy[neighbour, 1]], [7 - xy[center, 0], 7 - xy[neighbour, 0]],
            color=color, lw=1.7, alpha=0.95, zorder=2,
        )
    ax.scatter(xy[:, 1], 7 - xy[:, 0], s=5.5, c="#555555", edgecolors="none", zorder=3)
    ax.scatter(xy[neighbours, 1], 7 - xy[neighbours, 0], s=14, c=color, edgecolors="white", linewidths=0.3, zorder=4)
    ax.scatter([xy[center, 1]], [7 - xy[center, 0]], s=23, c="#C62828", edgecolors="white", linewidths=0.5, zorder=5)
    ax.set_title(label, pad=1.5, color=color, fontweight="bold")
    ax.set_xlim(-0.45, 7.45); ax.set_ylim(-0.45, 7.45); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def jitter_points(ax: plt.Axes, x: float, values: np.ndarray, color: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, len(values))
    ax.scatter(x + jitter, values, s=11, color=color, alpha=0.72, edgecolors="white", linewidths=0.25, zorder=3)
    median = float(np.median(values))
    ax.plot([x - 0.20, x + 0.20], [median, median], color="#111111", lw=1.6, zorder=4)


def draw_training_effects(ax: plt.Axes, subject: str) -> None:
    graph = [row for row in read_csv(RESULT / "summary/GRAPH_LEVEL_EFFECTS.csv") if row["subject"] == subject]
    units = [row for row in read_csv(RESULT / "summary/TRAINING_UNIT_RESULTS.csv") if row["subject"] == subject]
    patient = next(row for row in read_csv(RESULT / "summary/PATIENT_RESULTS.csv") if row["subject"] == subject)
    values = {
        "WRONG_GRID": -np.asarray([float(row["true_minus_control_nll"]) for row in graph if row["family"] == "WRONG_GRID"]),
        "DEGREE_RANDOM": -np.asarray([float(row["true_minus_control_nll"]) for row in graph if row["family"] == "DEGREE_RANDOM"]),
    }
    true_by_seed = {
        int(row["seed_index"]): float(row["test_contact_nll"])
        for row in units if row["family"] == "TRUE_GRID"
    }
    suffix = np.asarray([
        float(row["test_contact_nll"]) - true_by_seed[int(row["seed_index"])]
        for row in units if row["family"] == "SUFFIX_SHUFFLED"
    ])
    values["SUFFIX_SHUFFLED"] = suffix
    labels = ("wrong\npositions", "random\nneighbours", "shuffled\nevent endings")
    families = ("WRONG_GRID", "DEGREE_RANDOM", "SUFFIX_SHUFFLED")
    for index, family in enumerate(families):
        current = values[family]
        if len(current) >= 10:
            violin = ax.violinplot(current, positions=[index], widths=0.68, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(COLORS[family]); body.set_edgecolor("none"); body.set_alpha(0.20)
        jitter_points(ax, float(index), current, COLORS[family], 20260816 + index + int(subject))
    ax.axhline(0, color="#777777", ls="--", lw=0.8, zorder=0)
    p = float(patient["true_vs_wrong_grid_exact_p_lower"])
    if p <= 0.05:
        ymax = max(float(np.max(value)) for value in values.values())
        ax.text(0, ymax + 0.06 * max(0.1, abs(ymax)), "*", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(3), labels)
    ax.set_title(f"E{subject}", fontweight="bold", pad=2)
    ax.set_ylabel("extra test loss vs true grid\n(nats per next contact)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=2)


def draw_lesion_schematic(ax: plt.Axes, dispersed: bool) -> None:
    names, xy, truth = load_graph("958", "TRUE_GRID")
    if dispersed:
        # Illustrative directed local edges. Formal controls are regenerated for
        # every patch and seed with the frozen degree/weight matching contract.
        edges = [(0, 1), (7, 6), (24, 16), (39, 31), (49, 48), (62, 54), (18, 10), (46, 45)]
        title = "same number elsewhere"
        patch_nodes: list[int] = []
        color = "#777777"
    else:
        patch_nodes = [names.index(name) for name in ("GD4", "GD5", "GE4", "GE5")]
        involved = np.zeros(len(names), dtype=bool); involved[patch_nodes] = True
        # Stored recurrent masks index [target, source]. Draw only outside-to-patch
        # directed edges used by the final first-entry necessity estimand.
        edges = [
            (target, source) for target in range(len(names)) for source in range(len(names))
            if truth[target, source] and involved[target] and not involved[source]
        ]
        title = "edges entering one 2×2 area"
        color = "#C62828"
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            if truth[first, second]:
                ax.plot([xy[first, 1], xy[second, 1]], [7 - xy[first, 0], 7 - xy[second, 0]], color="#C5C5C5", lw=0.4)
    for target, source in edges:
        start = (xy[source, 1], 7 - xy[source, 0])
        end = (xy[target, 1], 7 - xy[target, 0])
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=5.2,
            color=color, lw=1.15, alpha=0.95, shrinkA=2.0, shrinkB=2.0,
        ))
    ax.scatter(xy[:, 1], 7 - xy[:, 0], s=5, c="#555555", edgecolors="none", zorder=3)
    if patch_nodes:
        ax.scatter(xy[patch_nodes, 1], 7 - xy[patch_nodes, 0], s=18, c="#F4A6A6", edgecolors="#C62828", linewidths=0.6, zorder=4)
    ax.set_title(title, pad=1.5, color=color, fontweight="bold")
    ax.set_xlim(-0.45, 7.45); ax.set_ylim(-0.45, 7.45); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)


def draw_dose_response(ax: plt.Axes, subject: str) -> None:
    rows = [
        row for row in read_csv(RESULT / "summary_inbound/INBOUND_ENTRY_PATCH_RESULTS.csv")
        if row["subject"] == subject
    ]
    patient = next(
        row for row in read_csv(RESULT / "summary_inbound/INBOUND_ENTRY_PATIENT_RESULTS.csv")
        if row["subject"] == subject
    )
    retained = np.asarray([100, 75, 50, 0], dtype=float)
    curves = np.asarray([
        [0.0, float(row["entry_damage_contrast_dose_0.75_median_seed"]),
         float(row["entry_damage_contrast_dose_0.5_median_seed"]),
         float(row["entry_damage_contrast_dose_0_median_seed"])]
        for row in rows
    ])
    color = COLORS[subject]
    for curve in curves:
        ax.plot(retained, curve, color=color, alpha=0.12, lw=0.55)
    median = np.median(curves, axis=0)
    q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
    ax.fill_between(retained, q25, q75, color=color, alpha=0.20, lw=0)
    ax.plot(retained, median, color=color, lw=2.0, marker="o", ms=3.3, mfc="white", mec=color)
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    p = float(patient["stratified_randomization_p_one_sided"])
    if p <= 0.05:
        ax.text(0, median[-1], " *", color=color, fontsize=11, va="center", ha="left")
    ax.set_xlim(105, -5)
    ax.set_xticks([100, 75, 50, 0])
    ax.set_title(f"E{subject}", fontweight="bold", pad=2)
    ax.set_xlabel("incoming local-edge strength retained (%)")
    ax.set_ylabel("extra first-entry test loss\nvs removing dispersed edges")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    style()
    fig = plt.figure(figsize=(7.2, 7.8), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.16, left=0.115, right=0.978, top=0.955, bottom=0.075)

    panel_a = outer[0].subgridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.28)
    top = panel_a[0].subgridspec(1, 4, wspace=0.28)
    draw_graph(fig.add_subplot(top[0, 0]), "TRUE_GRID", "actual neighbours", COLORS["TRUE_GRID"])
    draw_graph(fig.add_subplot(top[0, 1]), "WRONG_GRID_00", "wrong positions", COLORS["WRONG_GRID"])
    draw_graph(fig.add_subplot(top[0, 2]), "DEGREE_RANDOM_00", "random neighbours", COLORS["DEGREE_RANDOM"])
    draw_graph(fig.add_subplot(top[0, 3]), "TRUE_GRID", "same grid,\nshuffled endings", COLORS["SUFFIX_SHUFFLED"])
    training = panel_a[1].subgridspec(1, 2, wspace=0.24)
    training_958 = fig.add_subplot(training[0, 0])
    training_1084 = fig.add_subplot(training[0, 1])
    draw_training_effects(training_958, "958")
    draw_training_effects(training_1084, "1084")
    training_limits = (
        min(training_958.get_ylim()[0], training_1084.get_ylim()[0]),
        max(training_958.get_ylim()[1], training_1084.get_ylim()[1]),
    )
    training_958.set_ylim(training_limits); training_1084.set_ylim(training_limits)
    training_1084.set_ylabel("")

    panel_b = outer[1].subgridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.28)
    bottom = panel_b[0].subgridspec(1, 2, wspace=0.20)
    draw_lesion_schematic(fig.add_subplot(bottom[0, 0]), dispersed=False)
    draw_lesion_schematic(fig.add_subplot(bottom[0, 1]), dispersed=True)
    dose = panel_b[1].subgridspec(1, 2, wspace=0.24)
    dose_958 = fig.add_subplot(dose[0, 0])
    dose_1084 = fig.add_subplot(dose[0, 1])
    draw_dose_response(dose_958, "958")
    draw_dose_response(dose_1084, "1084")
    dose_limits = (
        min(dose_958.get_ylim()[0], dose_1084.get_ylim()[0]),
        max(dose_958.get_ylim()[1], dose_1084.get_ylim()[1]),
    )
    dose_958.set_ylim(dose_limits); dose_1084.set_ylim(dose_limits)
    dose_1084.set_ylabel("")

    fig.text(0.008, 0.985, "A", fontsize=11, fontweight="bold", va="top")
    fig.text(0.008, 0.505, "B", fontsize=11, fontweight="bold", va="top")
    output = RESULT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "topic5_ecog_physical_neighbourhood_v0_1"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    metadata = {
        "schema": "topic5_ecog_physical_neighbourhood_figure_v0.1",
        "panel_A": "Pre-training fixed graph constraints and held-out next-contact loss relative to the actual physical grid.",
        "panel_B": "Post-training attenuation of directed local edges entering a contiguous 2x2 area versus matched dispersed directed edges, evaluated on held-out first-entry decisions.",
        "statistics": "Stars use the frozen one-sided graph-tail test in A and the patch-by-seed focal-label randomization in B.",
    }
    (output / "FIGURE_METADATA.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "README.md").write_text(
        "### topic5_ecog_physical_neighbourhood_v0_1\n\n"
        "Panel A 上排画四种在训练开始前就固定的网络/标签条件（同一个中心触点被允许直接影响谁）；下排两位患者各一张，每个点是一张预先冻结的对照图在未见 block 上相对真实网格增加的下一触点损失。正值表示真实上下左右近邻更利于学习；星号只在真实网格对 31 张位置打乱网格的单侧 exact 图尾检验 p ≤ 0.05 时出现。\n\n"
        "Panel B 上排画训练完成后的直接干预：只削弱从区域外进入一块连续 2×2 区域的有向局部入边（红），并与相同边数、端点度数和训练后权重相近、完全避开该区域的分散有向边比较（灰，示意）。下排只评价未见事件中此前从未招募过该区域、下一步首次进入的决策；每条浅线是一块合格区域在三个 seed 上的中位响应，粗线和色带是患者内中位数与四分位范围。正值才表示真实入边具有选择性必要性；两位患者均无星号。\n\n"
        "**关注点**：A 回答真实物理邻接是否帮助从头学习；B 独立回答训练好的真实网格模型是否在线依赖连续局部连接。两位患者分别报告，不能当作大队列效应。\n"
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
