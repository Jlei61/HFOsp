#!/usr/bin/env python3
"""Render the six-panel structured rank-RNN paper figure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


COLORS = {
    "participation_mae": "#2A9D8F",
    "rank_wasserstein": "#6A4C93",
    "data": "#B2182B",
    "null": "#8C8C8C",
    "state": "#D97706",
    "inhibition": "#2166AC",
    "entropy": "#5B5B5B",
}
METRIC_LABELS = {
    "participation_mae": "Participation",
    "rank_wasserstein": "Rank distribution",
}


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _clean(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, pad=2)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _draw_model(ax: plt.Axes) -> None:
    ax.set_title("Structured self-supervised generator", loc="left", pad=5)
    x = np.linspace(0.10, 0.88, 7)
    for index, position in enumerate(x):
        color = mpl.colormaps["viridis"](index / (len(x) - 1))
        ax.add_patch(
            Circle(
                (position, 0.58),
                0.035,
                facecolor=color,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )
        )
    for left, right in zip(x[:-1], x[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (left + 0.025, 0.62),
                (right - 0.025, 0.62),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=1.0,
                color="#B2182B",
            )
        )
        ax.add_patch(
            FancyArrowPatch(
                (right - 0.025, 0.54),
                (left + 0.025, 0.54),
                arrowstyle="-|>",
                mutation_scale=7,
                linewidth=1.0,
                color="#2166AC",
            )
        )
    ax.text(0.49, 0.73, "patient-specific path modes", ha="center")
    ax.text(0.49, 0.43, "contact state", ha="center")
    inhibition = Circle(
        (0.49, 0.23),
        0.055,
        facecolor="#D9E8F5",
        edgecolor="#2166AC",
        linewidth=0.9,
    )
    ax.add_patch(inhibition)
    ax.text(0.49, 0.23, "I", color="#2166AC", ha="center", va="center")
    ax.add_patch(
        FancyArrowPatch(
            (0.49, 0.29),
            (0.49, 0.40),
            arrowstyle="-[",
            mutation_scale=7,
            linewidth=0.9,
            color="#2166AC",
        )
    )
    ax.text(0.07, 0.03, "observed rank prefix", ha="left")
    ax.text(0.93, 0.03, "next set / STOP", ha="right")
    ax.add_patch(
        FancyArrowPatch(
            (0.25, 0.08),
            (0.72, 0.08),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.9,
            color="#4B4B4B",
        )
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.85)
    ax.axis("off")


def _identity_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(color)
    x, y, color = x[finite], y[finite], color[finite]
    low = float(min(np.min(x), np.min(y)))
    high = float(max(np.max(x), np.max(y)))
    padding = max((high - low) * 0.08, 0.02)
    ax.plot(
        [low - padding, high + padding],
        [low - padding, high + padding],
        color="#999999",
        linewidth=0.8,
        zorder=0,
    )
    ax.scatter(
        x,
        y,
        c=color,
        cmap="viridis",
        vmin=0,
        vmax=1,
        s=24,
        edgecolor="white",
        linewidth=0.4,
        zorder=2,
    )
    ax.set(
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        xlim=(low - padding, high + padding),
        ylim=(low - padding, high + padding),
    )
    _clean(ax)


def _draw_example(
    spec,
    nodes: pd.DataFrame,
    *,
    subject: str = "epilepsiae_1146",
) -> None:
    current = nodes[nodes.subject.eq(subject)].copy()
    if current.empty:
        raise RuntimeError(f"paper example missing: {subject}")
    columns = [
        "observed_participation",
        "generated_participation",
        "observed_mean_rank",
        "generated_mean_rank",
        *[f"observed_rank_bin_{index}" for index in range(10)],
        *[f"generated_rank_bin_{index}" for index in range(10)],
    ]
    current = (
        current.groupby(["subject", "contact_name"], as_index=False)[columns]
        .median()
        .sort_values("observed_mean_rank")
    )
    sub = spec.subgridspec(
        1, 2, width_ratios=(0.90, 1.10), wspace=0.45
    )
    left = plt.subplot(sub[0, 0])
    _identity_scatter(
        left,
        current.observed_participation.to_numpy(float),
        current.generated_participation.to_numpy(float),
        current.observed_mean_rank.to_numpy(float),
        xlabel="Observed",
        ylabel="Generated",
        title="Participation",
    )
    heatmap = sub[0, 1].subgridspec(
        2,
        2,
        width_ratios=(1.0, 0.07),
        hspace=0.23,
        wspace=0.08,
    )
    observed_ax = plt.subplot(heatmap[0, 0])
    generated_ax = plt.subplot(heatmap[1, 0])
    color_ax = plt.subplot(heatmap[:, 1])
    observed = current[
        [f"observed_rank_bin_{index}" for index in range(10)]
    ].to_numpy(float)
    generated = current[
        [f"generated_rank_bin_{index}" for index in range(10)]
    ].to_numpy(float)
    vmax = float(max(np.nanmax(observed), np.nanmax(generated), 1e-6))
    image = observed_ax.imshow(
        observed,
        aspect="auto",
        origin="upper",
        extent=(0, 1, len(current), 0),
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    generated_ax.imshow(
        generated,
        aspect="auto",
        origin="upper",
        extent=(0, 1, len(current), 0),
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    tick_step = max(1, len(current) // 5)
    tick_index = np.arange(0, len(current), tick_step)
    for ax, title in (
        (observed_ax, "Observed ranks"),
        (generated_ax, "Generated ranks"),
    ):
        ax.set_title(title, fontsize=7.2, pad=2)
        ax.set_yticks(
            tick_index + 0.5,
            current.contact_name.iloc[tick_index],
            fontsize=5.8,
        )
        ax.tick_params(length=0, pad=1)
    observed_ax.tick_params(labelbottom=False)
    generated_ax.set_xlabel("Normalized rank · early → late")
    generated_ax.set_xticks([0, 0.5, 1])
    colorbar = plt.colorbar(image, cax=color_ax)
    colorbar.set_label("P(rank | active)", fontsize=6.2, labelpad=2)
    colorbar.ax.tick_params(labelsize=5.8, length=2)
    _panel_label(left, "B")


def _benefit_panel(
    ax: plt.Axes,
    patient: pd.DataFrame,
    stats: pd.DataFrame,
    *,
    group_column: str,
    order: list[str],
    labels: list[str],
    title: str,
    letter: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    rng = np.random.default_rng(20260726)
    offsets = {"participation_mae": -0.13, "rank_wasserstein": 0.13}
    markers = {"participation_mae": "o", "rank_wasserstein": "D"}
    for x_index, group in enumerate(order):
        for metric in ("participation_mae", "rank_wasserstein"):
            current = patient[
                patient[group_column].eq(group) & patient.metric.eq(metric)
            ]
            values = current.seed_median_benefit.to_numpy(float)
            x = (
                x_index
                + offsets[metric]
                + rng.uniform(-0.045, 0.045, len(values))
            )
            ax.scatter(
                x,
                values,
                s=10,
                alpha=0.45,
                color=COLORS[metric],
                marker=markers[metric],
                linewidth=0,
            )
            median = float(np.median(values))
            ax.plot(
                [
                    x_index + offsets[metric] - 0.08,
                    x_index + offsets[metric] + 0.08,
                ],
                [median, median],
                color=COLORS[metric],
                linewidth=2.2,
            )
            stat = stats[
                stats[group_column].eq(group) & stats.metric.eq(metric)
            ]
            if len(stat) == 1 and bool(stat.iloc[0]["pass"]):
                ymax = float(np.nanmax(values))
                ax.text(
                    x_index + offsets[metric],
                    ymax,
                    "*",
                    color=COLORS[metric],
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=5)
    _clean(ax)
    handles = [
        mpl.lines.Line2D(
            [], [], color=COLORS[metric], marker=markers[metric], linestyle="",
            label=METRIC_LABELS[metric], markersize=4
        )
        for metric in ("participation_mae", "rank_wasserstein")
    ]
    if show_legend:
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=2,
            handletextpad=0.3,
            columnspacing=0.8,
        )
    _panel_label(ax, letter)


def _draw_dynamics(spec, cohort: pd.DataFrame) -> None:
    metrics = (
        ("posterior_entropy_normalized", "Path entropy", COLORS["entropy"]),
        (
            "posterior_weighted_excitation",
            "Excitation",
            COLORS["state"],
        ),
        (
            "posterior_weighted_inhibition",
            "Inhibition",
            COLORS["inhibition"],
        ),
    )
    sub = spec.subgridspec(3, 1, hspace=0.12)
    axes = []
    for index, (metric, label, color) in enumerate(metrics):
        ax = plt.subplot(sub[index, 0])
        current = cohort[cohort.metric.eq(metric)].sort_values("progress_bin")
        x = current.progress_bin.to_numpy(float)
        ax.fill_between(
            x,
            current.q25.to_numpy(float),
            current.q75.to_numpy(float),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(x, current["median"], color=color, linewidth=1.5)
        ax.set_ylabel(label, labelpad=2)
        ax.set_xlim(0, 1)
        if index < 2:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Normalized event progress")
        _clean(ax)
        axes.append(ax)
    axes[0].set_title("Internal state across event rank", loc="left", pad=5)
    _panel_label(axes[0], "E")


def _draw_readout(ax: plt.Axes, analysis: Path, gate: dict) -> str:
    summary_path = analysis / "ictal_readout_summary.json"
    if gate["formal_interictal_gate_pass"]:
        if not summary_path.exists():
            raise RuntimeError(
                "interictal gate passed but frozen ictal readout is missing"
            )
        subjects = pd.read_csv(
            analysis / "ictal_readout_patient_statistics.csv"
        )
        order = [
            "intact",
            "no_history",
            "graph_lesion",
            "mode_collapse_lesion",
        ]
        labels = ["Intact", "No history", "No graph", "One path"]
        rng = np.random.default_rng(20260726)
        for index, condition in enumerate(order):
            current = subjects[subjects.condition.eq(condition)]
            x_null = index - 0.08 + rng.uniform(-0.025, 0.025, len(current))
            x_data = index + 0.08 + rng.uniform(-0.025, 0.025, len(current))
            for xn, xd, null, data in zip(
                x_null,
                x_data,
                current.rho_channel_shuffle_median,
                current.rho_data,
            ):
                ax.plot(
                    [xn, xd],
                    [null, data],
                    color="#C9C9C9",
                    linewidth=0.5,
                    zorder=0,
                )
            ax.scatter(
                x_null,
                current.rho_channel_shuffle_median,
                s=10,
                facecolor="white",
                edgecolor=COLORS["null"],
                linewidth=0.6,
                zorder=2,
            )
            ax.scatter(
                x_data,
                current.rho_data,
                s=11,
                color=COLORS["data"],
                linewidth=0,
                zorder=3,
            )
            ax.plot(
                [index + 0.01, index + 0.15],
                [np.median(current.rho_data)] * 2,
                color=COLORS["data"],
                linewidth=2.1,
            )
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(order)), labels, rotation=25, ha="right")
        ax.set_ylabel("Clinical-onset field ρ")
        ax.set_title("Frozen early-ictal readout", loc="left", pad=5)
        _clean(ax)
        handles = [
            mpl.lines.Line2D(
                [], [], marker="o", linestyle="", color=COLORS["data"],
                label="Observed", markersize=4
            ),
            mpl.lines.Line2D(
                [], [], marker="o", linestyle="", markerfacecolor="white",
                markeredgecolor=COLORS["null"], label="Channel shuffle",
                markersize=4
            ),
        ]
        ax.legend(handles=handles, frameon=False, loc="best")
        return "frozen_clinical_onset_readout"

    labels = [
        "Distribution vs controls",
        "Structure required",
        "Clinical target",
    ]
    passed = [
        bool(gate["comparison_gate_pass"]),
        bool(gate["structure_gate_pass"]),
        None,
    ]
    y = np.arange(3)[::-1]
    for position, status in zip(y, passed):
        if status is None:
            color, marker, text = "#B0B0B0", "s", "sealed"
        elif status:
            color, marker, text = "#2A9D8F", "o", "pass"
        else:
            color, marker, text = "#C65D4B", "X", "fail"
        ax.scatter(0.15, position, s=55, color=color, marker=marker)
        ax.text(0.25, position, text, va="center", color=color)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, 2.7)
    ax.set_xticks([])
    ax.set_title("Pre-registered decision", loc="left", pad=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="y", length=0)
    return "target_sealed_after_interictal_gate_failure"


def _write_readme(path: Path, readout_mode: str) -> None:
    if readout_mode == "frozen_clinical_onset_readout":
        final = (
            "F 检验冻结的间期节点分布能否跨患者读出 clinical onset 后 "
            "0–10 s、1–150 Hz 静态能量场，并以 all-contact channel shuffle "
            "为主随机基准。"
        )
    else:
        final = (
            "F 显示预注册门的最终状态；纯间期门未通过时，发作期目标保持封存，"
            "不允许用发作结果挽救模型。"
        )
    path.write_text(
        "### fig6_structured_rank_rnn.png / fig6_structured_rank_rnn.pdf\n\n"
        "A 说明患者特异路径图、触点状态和共享抑制如何组成结构化循环生成器。"
        "同一层的 B 用 E1146 heldout20 展示自由生成事件能否恢复触点参与概率和完整 rank distribution。"
        "C 在 34 人中比较 intact 与四个主对照，D 用结构损伤检验改善是否真的依赖"
        "路径图或多路径结构。E 汇总事件 rank 推进时的路径后验、兴奋和抑制状态；"
        f"同一解释层的 {final}\n\n"
        "**关注点**：先看 C 的两个触点级主终点是否同时过门，再看 D 的结构损伤和 "
        "E 的内部状态是否给出一致解释；只有这些成立时才解释 F 的跨状态 readout。\n"
    )


def _output_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=(
            ROOT
            / "results/paper-ready-figure/fig6_structured_rank_rnn/figures"
        ),
    )
    args = parser.parse_args()
    analysis = args.root.resolve() / "analysis"
    gate = json.loads((analysis / "formal_gate_summary.json").read_text())
    if gate.get("status") != "complete":
        raise RuntimeError("formal analysis is incomplete")
    nodes = pd.read_csv(analysis / "intact_k2_contact_distributions.csv")
    comparison_patient = pd.read_csv(
        analysis / "comparison_patient_seed_medians.csv"
    )
    comparison_stats = pd.read_csv(
        analysis / "comparison_primary_statistics.csv"
    )
    lesion_patient = pd.read_csv(
        analysis / "lesion_patient_seed_medians.csv"
    )
    lesion_stats = pd.read_csv(
        analysis / "lesion_primary_statistics.csv"
    )
    dynamics = pd.read_csv(
        analysis / "intact_k2_internal_dynamics_cohort.csv"
    )

    _style()
    fig = plt.figure(figsize=(13.4, 7.7))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        bottom=0.09,
        top=0.96,
        wspace=0.42,
        hspace=0.42,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    _draw_model(ax_a)
    _panel_label(ax_a, "A")
    _draw_example(grid[0, 1], nodes)
    ax_c = fig.add_subplot(grid[0, 2])
    _benefit_panel(
        ax_c,
        comparison_patient,
        comparison_stats,
        group_column="baseline",
        order=["no_history", "merged_path", "weight_shuffle", "mode_shuffle"],
        labels=["No history", "Single path", "Weight shuffle", "Mode shuffle"],
        title="Heldout node-distribution benefit",
        letter="C",
        ylabel="Error reduction\n(reference − intact)",
        show_legend=True,
    )
    ax_d = fig.add_subplot(grid[1, 0])
    _benefit_panel(
        ax_d,
        lesion_patient,
        lesion_stats,
        group_column="lesion",
        order=[
            "graph",
            "mode_collapse",
            "inhibition",
            "drop_forward",
            "drop_reverse",
            "drop_dominant_mode",
        ],
        labels=[
            "No graph",
            "One path",
            "No inhibition",
            "No forward",
            "No reverse",
            "No dominant",
        ],
        title="Structural necessity",
        letter="D",
        ylabel="Error increase\n(lesion − intact)",
        show_legend=False,
    )
    _draw_dynamics(grid[1, 1], dynamics)
    ax_f = fig.add_subplot(grid[1, 2])
    readout_mode = _draw_readout(ax_f, analysis, gate)
    _panel_label(ax_f, "F")

    out = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    png = out / "fig6_structured_rank_rnn.png"
    pdf = out / "fig6_structured_rank_rnn.pdf"
    fig.savefig(png, dpi=400, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    _write_readme(out / "README.md", readout_mode)
    metadata = {
        "status": "complete",
        "contract": "topic5_persistent_path_mode_rnn_v1_0",
        "formal_interictal_gate_pass": bool(
            gate["formal_interictal_gate_pass"]
        ),
        "panel_f": readout_mode,
        "example_subject": "epilepsiae_1146",
        "outputs": {
            "png": _output_path(png),
            "pdf": _output_path(pdf),
        },
    }
    (out / "fig6_structured_rank_rnn_summary.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
