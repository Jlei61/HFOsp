#!/usr/bin/env python3
"""Plot the six-panel bounded-negative persistent path-mode RNN figure."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = (
    ROOT
    / "results/topic5_structured_axis_graph/"
    "screen_persistent_path_mode_v0_9/analysis/bounded_negative"
)
OUTPUT_ROOT = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_persistent_path_mode_bounded_negative/figures"
)
STEM = "fig6_persistent_path_mode_bounded_negative"

BLUE = "#3A6EA5"
RUST = "#B45F3C"
TEAL = "#2A8C82"
PURPLE = "#8064A2"
GREY = "#777777"
LIGHT_GREY = "#D9D9D9"
DARK = "#222222"
DATASET_COLORS = {"epilepsiae": BLUE, "yuquan": RUST}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.75, length=2.5)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=color,
        facecolor=mcolors.to_rgba(color, 0.10),
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=5.7,
        color=DARK,
    )


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=GREY,
        )
    )


def _panel_a(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    _box(axis, (0.02, 0.34), 0.18, 0.32, "Contact\nranks", BLUE)
    _box(axis, (0.29, 0.55), 0.18, 0.23, "Train-only\npath bases", TEAL)
    _box(axis, (0.29, 0.20), 0.18, 0.23, "Observed\nprefix", PURPLE)
    _box(axis, (0.57, 0.34), 0.18, 0.32, "Fixed path\n+ dir.", RUST)
    _box(axis, (0.83, 0.34), 0.15, 0.32, "Next\n+ rollout", BLUE)
    _arrow(axis, (0.20, 0.50), (0.29, 0.65))
    _arrow(axis, (0.20, 0.50), (0.29, 0.31))
    _arrow(axis, (0.47, 0.66), (0.57, 0.54))
    _arrow(axis, (0.47, 0.31), (0.57, 0.46))
    _arrow(axis, (0.75, 0.50), (0.83, 0.50))
    axis.text(
        0.66,
        0.15,
        "Self-supervised; no seizure labels",
        ha="center",
        va="center",
        color=GREY,
        fontsize=6.2,
    )
    axis.set_title("Persistent path-mode RNN", loc="left", pad=4)


def _panel_b(axis: plt.Axes, prior: pd.DataFrame) -> None:
    k = prior["mode_count"].to_numpy(int)
    series = [
        ("split_half_cosine", "Split-half stability", BLUE, "o"),
        ("heldout_reconstruction", "Held-out reconstruction", TEAL, "s"),
        ("mode_distinctness", "Mode distinctness", RUST, "^"),
    ]
    work = prior.copy()
    work["mode_distinctness"] = 1.0 - work["within_patient_mode_cosine"]
    work.loc[work.mode_count == 1, "mode_distinctness"] = np.nan
    for column, label, color, marker in series:
        axis.plot(
            k,
            work[column],
            marker=marker,
            ms=4,
            lw=1.2,
            color=color,
            label=label,
        )
    axis.set_xticks(k)
    axis.set_xlabel("Number of path modes (K)")
    axis.set_ylabel("Train-only score")
    axis.set_ylim(0.30, 1.02)
    axis.legend(frameon=False, loc="lower left", handlelength=1.4)
    axis.set_title("Stable path bases", loc="left", pad=4)
    _finish_axis(axis)


def _panel_c(axis: plt.Axes, nll: pd.DataFrame) -> None:
    row_order = ["no_history", "weight_shuffle", "merged_path", "mode_shuffle"]
    labels = ["No history", "Weight shuffle", "Single path", "Mode shuffle"]
    medians = (
        nll.groupby(["baseline", "mode_count"])["nll_benefit"]
        .median()
        .unstack()
        .reindex(row_order)
        .reindex(columns=[1, 2, 3, 4])
    )
    values = medians.to_numpy(float)
    limit = max(0.03, float(np.nanmax(np.abs(values))))
    image = axis.imshow(
        values,
        aspect="auto",
        cmap="RdBu_r",
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if abs(value) > 0.55 * limit else DARK,
                )
            else:
                axis.text(column, row, "—", ha="center", va="center", color=GREY)
    axis.set_xticks(range(4), [1, 2, 3, 4])
    axis.set_yticks(range(4), labels)
    axis.set_xlabel("Number of path modes (K)")
    axis.set_title("Local next-contact benefit", loc="left", pad=4)
    axis.text(
        1.0,
        -0.23,
        "positive = lower held-out NLL",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=GREY,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    colorbar = axis.figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
    colorbar.ax.tick_params(labelsize=5.6, width=0.6, length=2)
    colorbar.outline.set_linewidth(0.6)


def _panel_d(axis: plt.Axes, node: pd.DataFrame) -> None:
    for dataset, color in DATASET_COLORS.items():
        selected = node.dataset == dataset
        axis.scatter(
            node.loc[selected, "observed_participation"],
            node.loc[selected, "generated_participation"],
            s=19,
            marker="o",
            facecolor=mcolors.to_rgba(color, 0.65),
            edgecolor="white",
            linewidth=0.3,
        )
        axis.scatter(
            node.loc[selected, "observed_mean_rank"],
            node.loc[selected, "generated_mean_rank"],
            s=20,
            marker="^",
            facecolor="none",
            edgecolor=color,
            linewidth=0.8,
        )
    axis.plot([0, 1], [0, 1], color=GREY, lw=0.7, ls="--")
    participation_r = node[
        ["observed_participation", "generated_participation"]
    ].corr().iloc[0, 1]
    rank_r = node[["observed_mean_rank", "generated_mean_rank"]].corr().iloc[0, 1]
    axis.text(
        0.03,
        0.97,
        f"participation  r = {participation_r:.2f}\nmean rank      r = {rank_r:.2f}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=5.9,
    )
    axis.scatter([], [], marker="o", color=GREY, s=17, label="Participation")
    axis.scatter(
        [], [], marker="^", facecolor="none", edgecolor=GREY, s=18, label="Mean rank"
    )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Observed contact statistic")
    axis.set_ylabel("Generated contact statistic")
    axis.legend(frameon=False, loc="lower right", handletextpad=0.3)
    axis.set_title("Contact statistics (K = 2)", loc="left", pad=4)
    _finish_axis(axis)


def _panel_e(
    axis: plt.Axes,
    trajectory: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> None:
    for (_, _), values in trajectory.groupby(["subject", "seed"]):
        values = values.sort_values("prefix_fraction")
        axis.plot(
            values.prefix_fraction,
            values.normalized_component_entropy,
            color=PURPLE,
            alpha=0.16,
            lw=0.65,
        )
    summary = trajectory.groupby("prefix_fraction")[
        "normalized_component_entropy"
    ].agg(["median", "quantile"])
    x = summary.index.to_numpy(float)
    median = summary["median"].to_numpy(float)
    lower = trajectory.groupby("prefix_fraction")[
        "normalized_component_entropy"
    ].quantile(0.25).reindex(x).to_numpy(float)
    upper = trajectory.groupby("prefix_fraction")[
        "normalized_component_entropy"
    ].quantile(0.75).reindex(x).to_numpy(float)
    axis.fill_between(x, lower, upper, color=mcolors.to_rgba(PURPLE, 0.16))
    axis.plot(x, median, color=PURPLE, lw=1.5)
    axis.axhline(1.0, color=GREY, lw=0.7, ls="--")
    axis.set_xlim(0, 1)
    axis.set_ylim(0.93, 1.003)
    axis.set_xlabel("Observed fraction of event")
    axis.set_ylabel("Normalized mode entropy")
    axis.set_title("Path identity is ambiguous", loc="left", pad=4)
    axis.text(
        0.03,
        0.06,
        "1 = uniform over path × direction",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=GREY,
    )
    inset = axis.inset_axes([0.64, 0.19, 0.30, 0.34])
    information = identifiability.groupby("mode_count")[
        "posterior_information_fraction"
    ].median()
    inset.plot(
        information.index,
        information.values,
        color=RUST,
        marker="o",
        ms=3,
        lw=1.0,
    )
    inset.set_xticks([1, 2, 3, 4])
    inset.set_ylim(0, 0.055)
    inset.set_title("Information", fontsize=5.8, pad=2)
    inset.tick_params(labelsize=5.2, width=0.5, length=2)
    inset.spines[["top", "right"]].set_visible(False)
    _finish_axis(axis)


def _panel_f(axis: plt.Axes, gate: pd.DataFrame) -> None:
    rows = [
        ("comparison", "no_history", "No history"),
        ("comparison", "merged_path", "K=1 path"),
        ("comparison", "mode_shuffle", "Mode-shuff."),
        ("comparison", "weight_shuffle", "Weight-shuff."),
        ("lesion", "graph", "Graph lesion"),
        ("lesion", "inhibition", "Inhib. lesion"),
        ("lesion", "direction_removal_mean", "Direction lesion"),
        ("lesion", "mode_collapse", "Collapse modes"),
        ("lesion", "drop_dominant_mode", "Dominant lesion"),
    ]
    values = np.full((len(rows), 4), np.nan)
    annotations: dict[tuple[int, int], str] = {}
    pass_cells = []
    for row_index, (section, key, _) in enumerate(rows):
        for mode_count in range(1, 5):
            selected = gate[
                (gate.section == section)
                & (gate.row == key)
                & (gate.mode_count == mode_count)
            ]
            if selected.empty:
                continue
            record = selected.iloc[0]
            precedence = int(record.precedence_n_better)
            whole = int(record.whole_path_n_better)
            values[row_index, mode_count - 1] = min(precedence, whole) / 9.0
            annotations[(row_index, mode_count - 1)] = f"{precedence}|{whole}"
            if bool(record.both_metric_gate_pass):
                pass_cells.append((row_index, mode_count - 1))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "gate", ["#F2F2F2", "#D8B8A8", RUST]
    )
    axis.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    for (row, column), label in annotations.items():
        axis.text(column, row, label, ha="center", va="center", fontsize=5.7)
    for row, column in pass_cells:
        axis.add_patch(
            plt.Rectangle(
                (column - 0.47, row - 0.47),
                0.94,
                0.94,
                fill=False,
                edgecolor=TEAL,
                linewidth=1.2,
            )
        )
    axis.axhline(3.5, color="white", lw=2.0)
    axis.set_xticks(range(4), [1, 2, 3, 4])
    axis.set_yticks(range(len(rows)), [item[2] for item in rows])
    axis.set_xlabel("Number of path modes (K)")
    axis.set_title("Global path gate failed", loc="left", pad=4)
    axis.text(
        1.0,
        -0.21,
        "counts: precedence | whole-path wins (of 9)\ngreen: individual check passed",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=GREY,
    )
    axis.tick_params(length=0, axis="y", labelsize=5.7, pad=1)
    axis.tick_params(length=0, axis="x")
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    _style()
    paths = {
        "prior": INPUT_ROOT / "path_mode_prior_summary.csv",
        "nll": INPUT_ROOT / "local_transition_nll_benefits.csv",
        "node": INPUT_ROOT / "node_distribution_k2.csv",
        "trajectory": INPUT_ROOT / "posterior_state_trajectory_k2.csv",
        "identifiability": INPUT_ROOT / "mode_identifiability.csv",
        "gate": INPUT_ROOT / "hard_gate_matrix.csv",
        "summary": INPUT_ROOT / "bounded_negative_summary.json",
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    summary = json.loads(paths["summary"].read_text())
    if summary["hard_gate_pass"] or summary["ictal_target_read"]:
        raise RuntimeError("bounded-negative or ictal seal failed")

    prior = pd.read_csv(paths["prior"])
    nll = pd.read_csv(paths["nll"])
    node = pd.read_csv(paths["node"])
    trajectory = pd.read_csv(paths["trajectory"])
    identifiability = pd.read_csv(paths["identifiability"])
    gate = pd.read_csv(paths["gate"])

    fig = plt.figure(figsize=(7.25, 5.15))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.98, 0.93, 1.12],
        height_ratios=[0.92, 1.08],
        wspace=0.66,
        hspace=0.54,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    _panel_a(axes[0])
    _panel_b(axes[1], prior)
    _panel_c(axes[2], nll)
    _panel_d(axes[3], node)
    _panel_e(axes[4], trajectory, identifiability)
    _panel_f(axes[5], gate)
    for label, axis in zip("ABCDEF", axes):
        axis.text(
            -0.18,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(
            OUTPUT_ROOT / f"{STEM}.{extension}",
            dpi=350 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    metadata = {
        "figure": STEM,
        "result_tier": "bounded_negative",
        "n_pilot_patients": 3,
        "n_seeds": 3,
        "n_runs": 117,
        "mode_counts": [1, 2, 3, 4],
        "diagnostic_mode_count": 2,
        "hard_gate_pass": False,
        "formal_34x3_started": False,
        "ictal_target_read": False,
        "inputs": {
            key: {"path": str(path), "sha256": _hash(path)}
            for key, path in paths.items()
        },
    }
    (OUTPUT_ROOT / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    (OUTPUT_ROOT / "README.md").write_text(
        f"""### {STEM}.png

六个 panel 依次回答这条 RNN 线是否形成了可解释的证据链。A 定义模型：仅用间期 contact-rank 事件，在训练集内构造患者特异的路径基，并假定每个事件由一条固定路径及方向驱动。B 显示路径基在数据切半后较稳定，但对未见事件的重建有限。C 显示模型能改善局部的下一触点预测，却没有从多路径身份本身获得稳定收益。D 显示生成事件可以部分恢复触点参与概率，但难以恢复触点的平均先后位置。E 显示即使观察到事件后段，模型对路径身份仍接近均匀不确定。F 汇总相对强对照和结构消融的全局门：只有单个消融单元通过，K=1–4 均未满足完整门。

**关注点**：这是一项预先限定的 3 位患者 × 3 seeds pilot。它支持“间期事件包含可学习的局部触点统计”，但不支持“单条持续隐路径能够稳定生成完整传播活动”。因此未启动 34×3 正式训练，也未读取发作期目标；该结果只能作为 bounded-negative 模型检验，不能作为间期到发作期的正向机制证据。
"""
    )


if __name__ == "__main__":
    main()
