#!/usr/bin/env python3
"""Render the paper-ready Stage-A interictal operator result.

The figure is intentionally limited to the completed target-free engineering
screen. It does not read seizure targets and it does not make a formal
three-seed gate claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
SELECTION_DIR = (
    ROOT
    / "results"
    / "topic5_interictal_operator_static_readout"
    / "stage_a"
    / "hidden_size_selection_seed20260724"
)
SCREEN_DIR = (
    ROOT
    / "results"
    / "topic5_interictal_operator_static_readout"
    / "stage_a"
    / "screen_h64_seed20260724"
)
OUT = (
    ROOT
    / "results"
    / "paper-ready-figure"
    / "fig6_stagea_operator_screen"
    / "figures"
)

SELECTION_CELLS = SELECTION_DIR / "hidden_size_inner_validation_cells.csv"
SELECTION_SUMMARY = SELECTION_DIR / "hidden_size_one_se_summary.csv"
SELECTION_JSON = SELECTION_DIR / "hidden_size_selection.json"
CELL_METRICS = SCREEN_DIR / "stage_a_cell_metrics.csv"
SUBJECT_METRICS = SCREEN_DIR / "stage_a_subject_metrics.csv"
GATE_JSON = SCREEN_DIR / "stage_a_gate_summary.json"

BLUE = "#3F6F9F"
RUST = "#A35E48"
PURPLE = "#5B4B8A"
GOLD = "#D7A73E"
INK = "#252525"
MID_GREY = "#777777"
LIGHT_GREY = "#D6D6D6"
PALE_BLUE = "#EAF1F7"
PALE_RUST = "#F6ECE8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def _load_and_validate() -> dict[str, object]:
    sources = [
        SELECTION_CELLS,
        SELECTION_SUMMARY,
        SELECTION_JSON,
        CELL_METRICS,
        SUBJECT_METRICS,
        GATE_JSON,
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing accepted Stage-A artifacts: {missing}")

    selection_cells = pd.read_csv(SELECTION_CELLS)
    selection_summary = pd.read_csv(SELECTION_SUMMARY)
    cell_metrics = pd.read_csv(CELL_METRICS)
    subject_metrics = pd.read_csv(SUBJECT_METRICS)
    selection = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    gate = json.loads(GATE_JSON.read_text(encoding="utf-8"))

    _require_columns(
        selection_cells,
        {"subject", "seed", "hidden_size", "best_inner_validation_loss"},
        SELECTION_CELLS,
    )
    _require_columns(
        selection_summary,
        {
            "hidden_size",
            "mean_inner_validation_loss",
            "se_inner_validation_loss",
            "n_cells",
        },
        SELECTION_SUMMARY,
    )
    _require_columns(
        cell_metrics,
        {
            "subject",
            "seed",
            "hidden_size",
            "strongest_static_nll_name",
            "strongest_static_concordance_name",
        },
        CELL_METRICS,
    )
    gain_columns = {
        "subject",
        "n_seeds",
        "next_gain_vs_static",
        "suffix_gain_vs_static",
        "next_gain_vs_rank_shuffle",
        "suffix_gain_vs_rank_shuffle",
    }
    _require_columns(subject_metrics, gain_columns, SUBJECT_METRICS)

    expected_sizes = {32, 64}
    observed_sizes = set(selection_cells["hidden_size"].astype(int))
    if observed_sizes != expected_sizes:
        raise ValueError(
            f"hidden-size comparison must be {sorted(expected_sizes)}, got {sorted(observed_sizes)}"
        )
    count_by_size = selection_cells.groupby("hidden_size").size().to_dict()
    if count_by_size != {32: 13, 64: 13}:
        raise ValueError(f"expected 13 matched cells per size, got {count_by_size}")
    paired = selection_cells.pivot(
        index=["subject", "seed"],
        columns="hidden_size",
        values="best_inner_validation_loss",
    )
    if paired.shape != (13, 2) or paired.isna().any().any():
        raise ValueError(f"hidden-size cells are not a complete matched 13 x 2 grid: {paired.shape}")

    if selection.get("selected_hidden_size") != 64:
        raise ValueError("accepted target-free selection is no longer h64")
    if selection.get("heldout_last20_metrics_read") is not False:
        raise ValueError("selection artifact reports held-out metric access")
    if selection.get("ictal_target_opened") is not False:
        raise ValueError("selection artifact reports ictal target access")
    if selection.get("n_matched_subject_seed_cells_per_size") != 13:
        raise ValueError("selection artifact no longer contains 13 matched cells per size")

    if gate.get("n_patients") != 13 or gate.get("seed_count_min") != 1:
        raise ValueError("paper-ready screen requires the accepted 13-patient one-seed result")
    if gate.get("hidden_size") != 64:
        raise ValueError("screen artifact does not use the selected h64 model")
    if gate.get("formal_gate_eligible") is not False:
        raise ValueError("this renderer is scoped to the non-formal engineering screen")
    if gate.get("ictal_target_opened") is not False:
        raise ValueError("screen artifact reports ictal target access")
    if gate.get("scientific_status") != "pilot_only_not_formal":
        raise ValueError("unexpected Stage-A scientific status")
    if subject_metrics["subject"].nunique() != 13 or len(subject_metrics) != 13:
        raise ValueError("patient-level table must contain exactly 13 unique patients")
    if set(subject_metrics["n_seeds"].astype(int)) != {1}:
        raise ValueError("patient-level table is no longer a one-seed screen")
    if cell_metrics["subject"].nunique() != 13 or len(cell_metrics) != 13:
        raise ValueError("cell table must contain exactly 13 one-seed cells")
    if set(cell_metrics["hidden_size"].astype(int)) != {64}:
        raise ValueError("cell table contains a hidden size other than h64")

    for metric in (
        "next_gain_vs_static",
        "suffix_gain_vs_static",
        "next_gain_vs_rank_shuffle",
        "suffix_gain_vs_rank_shuffle",
    ):
        observed = float(np.median(subject_metrics[metric].to_numpy(float)))
        accepted = float(gate["metrics"][metric]["patient_median"])
        if not np.isclose(observed, accepted, atol=1e-12):
            raise ValueError(f"{metric} median differs between CSV and accepted gate JSON")

    return {
        "selection_cells": selection_cells,
        "selection_summary": selection_summary,
        "selection": selection,
        "cell_metrics": cell_metrics,
        "subject_metrics": subject_metrics.sort_values("subject").reset_index(drop=True),
        "gate": gate,
        "sources": sources,
    }


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        color=INK,
    )


def _rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.9,
    radius: float = 0.025,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MID_GREY,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color=color,
            connectionstyle=connectionstyle,
            transform=ax.transAxes,
        )
    )


def _draw_task_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_title(
        "What can one interictal-event prefix predict?",
        loc="left",
        pad=9,
        fontsize=9.2,
        fontweight="bold",
    )

    ax.text(
        0.02,
        0.83,
        "Masked contact-rank prefix",
        transform=ax.transAxes,
        fontsize=7.8,
        color=INK,
        ha="left",
    )
    token_specs = [
        (0.02, "{C2, C5}", PURPLE),
        (0.19, "{C3}", "#3B7E8D"),
        (0.34, "…", GOLD),
    ]
    for x, label, color in token_specs:
        _rounded_box(
            ax,
            (x, 0.58),
            0.125,
            0.16,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            radius=0.025,
        )
        ax.text(
            x + 0.0625,
            0.66,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.4,
            color="white",
            fontweight="bold",
        )
    _arrow(ax, (0.15, 0.66), (0.185, 0.66))
    _arrow(ax, (0.32, 0.66), (0.335, 0.66))

    _rounded_box(
        ax,
        (0.49, 0.54),
        0.16,
        0.24,
        facecolor=PALE_BLUE,
        edgecolor=BLUE,
        linewidth=1.15,
        radius=0.035,
    )
    ax.text(
        0.57,
        0.67,
        "GRU",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.0,
        color=BLUE,
        fontweight="bold",
    )
    _arrow(ax, (0.45, 0.66), (0.485, 0.66), color=BLUE)
    _arrow(
        ax,
        (0.61, 0.80),
        (0.53, 0.80),
        color=BLUE,
        connectionstyle="arc3,rad=0.55",
    )

    output_specs = [
        (0.73, 0.69, "next set\nor STOP", PALE_BLUE, BLUE),
        (0.73, 0.39, "remaining\nparticipation\n+ suffix rank", PALE_RUST, RUST),
    ]
    for x, y, label, face, edge in output_specs:
        _rounded_box(
            ax,
            (x, y - 0.09),
            0.25,
            0.18,
            facecolor=face,
            edgecolor=edge,
            linewidth=1.0,
            radius=0.03,
        )
        ax.text(
            x + 0.125,
            y,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.0,
            color=edge,
            fontweight="bold",
        )
    _arrow(ax, (0.655, 0.67), (0.725, 0.69), color=BLUE)
    _arrow(ax, (0.645, 0.59), (0.725, 0.42), color=RUST)

    _rounded_box(
        ax,
        (0.13, 0.10),
        0.74,
        0.17,
        facecolor="#F6F6F6",
        edgecolor=LIGHT_GREY,
        linewidth=0.8,
        radius=0.025,
    )
    ax.text(
        0.50,
        0.185,
        "Recurrence is confined to one IED",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        color=INK,
        fontweight="bold",
    )
    ax.text(
        0.50,
        0.125,
        "no inter-event interval or seizure input",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.1,
        color=MID_GREY,
    )
    _panel_label(ax, "A")


def _draw_selection_panel(
    ax: plt.Axes,
    cells: pd.DataFrame,
    summary: pd.DataFrame,
    selection: dict[str, object],
) -> None:
    paired = (
        cells.pivot(
            index=["subject", "seed"],
            columns="hidden_size",
            values="best_inner_validation_loss",
        )
        .sort_index()
        .astype(float)
    )
    rng = np.random.default_rng(20260725)
    for _, row in paired.iterrows():
        jitter = float(rng.uniform(-0.025, 0.025))
        ax.plot(
            [0 + jitter, 1 + jitter],
            [row[32], row[64]],
            color="#BFC4C8",
            linewidth=0.75,
            alpha=0.8,
            zorder=1,
        )
        ax.scatter(
            [0 + jitter, 1 + jitter],
            [row[32], row[64]],
            s=17,
            c=["#9CA5AB", BLUE],
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )

    summary = summary.set_index("hidden_size")
    for x, hidden_size, color in ((0, 32, "#6F777C"), (1, 64, BLUE)):
        mean = float(summary.loc[hidden_size, "mean_inner_validation_loss"])
        se = float(summary.loc[hidden_size, "se_inner_validation_loss"])
        ax.errorbar(
            x,
            mean,
            yerr=se,
            fmt="D",
            markersize=4.8,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.55,
            capsize=3,
            linewidth=1.4,
            zorder=4,
        )

    threshold = float(selection["one_se_threshold"])
    ax.axhline(threshold, color=RUST, linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    ax.text(
        0.98,
        threshold + 0.0012,
        "one-SE threshold",
        color=RUST,
        fontsize=6.8,
        ha="right",
        va="bottom",
    )
    ax.text(
        0.97,
        0.94,
        "h=64 selected",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.7,
        fontweight="bold",
        color=BLUE,
    )
    ax.text(
        0.97,
        0.85,
        "ictal target sealed",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=MID_GREY,
    )
    ax.text(
        0.03,
        0.06,
        "diamonds: mean ± SE",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        color=MID_GREY,
    )
    ax.set_xticks([0, 1], ["h=32", "h=64"])
    ax.set_xlim(-0.28, 1.28)
    values = paired.to_numpy().ravel()
    padding = max(0.006, 0.12 * (values.max() - values.min()))
    ax.set_ylim(values.min() - padding, values.max() + padding)
    ax.set_ylabel("Inner-validation loss\n(lower is better)")
    ax.set_title(
        "Capacity selected without seizure labels",
        loc="left",
        pad=9,
        fontsize=9.2,
        fontweight="bold",
    )
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6, zorder=0)
    _panel_label(ax, "B")


def _draw_gain_panel(
    ax: plt.Axes,
    subject: pd.DataFrame,
    gate: dict[str, object],
    *,
    metric: str,
    shuffle_metric: str,
    color: str,
    title: str,
    ylabel: str,
    panel_label: str,
) -> None:
    values = subject[metric].to_numpy(float)
    rng = np.random.default_rng(20260725 + (1 if panel_label == "C" else 2))
    jitter = rng.uniform(-0.105, 0.105, size=len(values))

    ax.boxplot(
        values,
        positions=[0],
        widths=0.30,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.12, "edgecolor": color, "linewidth": 0.9},
        medianprops={"color": color, "linewidth": 0.0},
        whiskerprops={"color": color, "linewidth": 0.8},
        capprops={"color": color, "linewidth": 0.8},
        zorder=1,
    )
    ax.scatter(
        jitter,
        values,
        s=29,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.92,
        zorder=3,
    )

    metric_summary = gate["metrics"][metric]
    median = float(metric_summary["patient_median"])
    low, high = (float(value) for value in metric_summary["bootstrap_95ci"])
    ax.errorbar(
        0.25,
        median,
        yerr=np.asarray([[median - low], [high - median]]),
        fmt="s",
        markersize=5.0,
        color=INK,
        markerfacecolor="white",
        markeredgewidth=1.1,
        capsize=3.2,
        linewidth=1.4,
        zorder=4,
    )

    positive = int(np.sum(values > 0))
    ax.text(
        0.04,
        0.95,
        f"{positive}/13 patients > 0",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.4,
        color=INK,
    )
    ax.text(
        0.96,
        0.95,
        f"median {median:+.3f}\n95% CI [{low:+.3f}, {high:+.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
        color=INK,
    )
    if int(np.sum(subject[shuffle_metric].to_numpy(float) > 0)) != 13:
        raise ValueError(f"{shuffle_metric} no longer has the accepted 13/13 positive sanity result")

    ax.axhline(0, color="#555555", linestyle=(0, (3, 2)), linewidth=0.9, zorder=0)
    ax.set_xlim(-0.32, 0.48)
    data_low = min(float(values.min()), low, 0.0)
    data_high = max(float(values.max()), high, 0.0)
    span = max(data_high - data_low, 0.01)
    ax.set_ylim(data_low - 0.12 * span, data_high + 0.23 * span)
    ax.set_xticks([0, 0.25], ["patients", "median\n[95% CI]"])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", pad=9, fontsize=9.2, fontweight="bold")
    ax.tick_params(axis="x", length=0, labelsize=7.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#ECECEC", linewidth=0.6, zorder=0)
    _panel_label(ax, panel_label)


def _write_readme(gate: dict[str, object]) -> None:
    next_summary = gate["metrics"]["next_gain_vs_static"]
    suffix_summary = gate["metrics"]["suffix_gain_vs_static"]
    readme = f"""# Figure 6 Stage-A interictal operator screen

### fig6_stagea_operator_screen.png

**A，单事件学习任务。** 输入是一个间期群体事件内部的 masked contact-rank
prefix，递归只沿该事件的 recruitment pseudo-time 展开。模型同时预测下一招募
contact set / STOP，以及剩余参与和 direct suffix rank；不输入 IEI、跨事件历史或
发作数据。

**B，无发作标签的容量选择。** 13 个完全匹配的 patient-by-seed folds 上，h64 的
inner-validation loss 低于 h32，并按冻结的 one-SE 规则选为 Stage-A 模型。该选择
没有读取 held-out last-20% 指标，也没有打开 ictal target。

**C，局部下一招募。** 每个点是一名患者，纵轴为 selected h64 相对每名患者最强
非循环对照（unordered DeepSets 或 matched feed-forward）的 next-set NLL 增益，
正值表示 h64 更好。患者中位数为 {next_summary['patient_median']:+.5f}，
bootstrap 95% CI [{next_summary['bootstrap_95ci'][0]:+.5f},
{next_summary['bootstrap_95ci'][1]:+.5f}]；10/13 患者为正。

**D，完整剩余顺序。** 同样的患者级比较用于 direct suffix-rank concordance。
中位数为 {suffix_summary['patient_median']:+.5f}，bootstrap 95% CI
[{suffix_summary['bootstrap_95ci'][0]:+.5f},
{suffix_summary['bootstrap_95ci'][1]:+.5f}]；只有 6/13 患者为正，未提供超越强
非循环对照的方向证据。两个任务相对 participation-preserving within-event
rank-shuffle 都是 13/13 正向，说明模型学到了真实顺序信息，但这不能替代强静态
对照。

**关注点**：当前结果支持“单个间期事件前缀含有可学习的局部下一招募信息”，但
不支持“GRU latent state 已恢复完整传播路径”，也不允许进入发作早期 readout。
这是 13 人、单 seed 的 cheap-first 工程筛查，不是冻结合同要求的三 seed 正式检验。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    artifacts = _load_and_validate()
    selection_cells = artifacts["selection_cells"]
    selection_summary = artifacts["selection_summary"]
    selection = artifacts["selection"]
    subject = artifacts["subject_metrics"]
    gate = artifacts["gate"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    figure = plt.figure(figsize=(7.15, 5.55), facecolor="white")
    grid = figure.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.985,
        bottom=0.095,
        top=0.955,
        wspace=0.33,
        hspace=0.48,
        height_ratios=[0.92, 1.08],
    )
    ax_a = figure.add_subplot(grid[0, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 0])
    ax_d = figure.add_subplot(grid[1, 1])

    _draw_task_panel(ax_a)
    _draw_selection_panel(ax_b, selection_cells, selection_summary, selection)
    _draw_gain_panel(
        ax_c,
        subject,
        gate,
        metric="next_gain_vs_static",
        shuffle_metric="next_gain_vs_rank_shuffle",
        color=BLUE,
        title="Local next-set gain is positive",
        ylabel="Next-set NLL gain\n(control − GRU)",
        panel_label="C",
    )
    _draw_gain_panel(
        ax_d,
        subject,
        gate,
        metric="suffix_gain_vs_static",
        shuffle_metric="suffix_gain_vs_rank_shuffle",
        color=RUST,
        title="Direct suffix rank adds no gain",
        ylabel="Suffix concordance gain\n(GRU − control)",
        panel_label="D",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#707070",
            markeredgecolor="white",
            markersize=5,
            label="patient",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="-",
            color=INK,
            markerfacecolor="white",
            markersize=4.5,
            label="median and bootstrap 95% CI",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=2,
        frameon=False,
        fontsize=7.1,
        handlelength=1.5,
        columnspacing=1.8,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "fig6_stagea_operator_screen"
    figure.savefig(stem.with_suffix(".png"), dpi=400)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)

    sources = artifacts["sources"]
    metadata = {
        "figure": "fig6_stagea_operator_screen",
        "scientific_status": "paper-ready rendering of a one-seed engineering screen",
        "formal_gate_claim": False,
        "ictal_target_opened": False,
        "selected_hidden_size": int(selection["selected_hidden_size"]),
        "selection_rule": selection["selection_rule"],
        "n_patients": int(gate["n_patients"]),
        "n_seeds": 1,
        "panels": {
            "A": "single-interictal-event prefix task; recurrence is within event only",
            "B": "target-free h32 versus h64 one-SE model selection",
            "C": "patient-level next-set NLL gain over strongest non-recurrent comparator",
            "D": "patient-level direct suffix-rank concordance gain over strongest non-recurrent comparator",
        },
        "metrics": gate["metrics"],
        "positive_patient_count": {
            column: int(np.sum(subject[column].to_numpy(float) > 0))
            for column in (
                "next_gain_vs_static",
                "suffix_gain_vs_static",
                "next_gain_vs_rank_shuffle",
                "suffix_gain_vs_rank_shuffle",
            )
        },
        "strongest_nonrecurrent_winner_counts": {
            "next_set_nll": {
                str(key): int(value)
                for key, value in artifacts["cell_metrics"][
                    "strongest_static_nll_name"
                ].value_counts().items()
            },
            "suffix_concordance": {
                str(key): int(value)
                for key, value in artifacts["cell_metrics"][
                    "strongest_static_concordance_name"
                ].value_counts().items()
            },
        },
        "claim_boundary": [
            "supports learnable local next-recruitment information within a single interictal event",
            "does not establish direct full-suffix recovery beyond strong non-recurrent controls",
            "does not support seizure prediction, ictal replay, or mechanism claims",
        ],
        "source_files": {
            str(path.relative_to(ROOT)): {"sha256": _sha256(path)} for path in sources
        },
    }
    (stem.parent / f"{stem.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(gate)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.parent / f"{stem.name}_metadata.json")
    print(OUT / "README.md")


if __name__ == "__main__":
    main()
