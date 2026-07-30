#!/usr/bin/env python3
"""Build the six-panel structured graph-RNN scientific gate figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PRIOR_ROOT = (
    ROOT
    / "results/topic5_structured_axis_graph/transition_skeleton_prior_v0_7"
)
ANALYSIS_ROOT = (
    ROOT
    / "results/topic5_structured_axis_graph/"
    "cohort_screen_transition_skeleton_v0_7_1/analysis"
)
OUTPUT_ROOT = (
    ROOT
    / "results/paper-ready-figure/fig_structured_graph_rnn_gate"
)
COLORS = {"epilepsiae": "#2166AC", "yuquan": "#B66A2B"}
CONTROL_ORDER = ["rank0", "shuffled_paths", "patient_paths"]
CONTROL_LABELS = ["No history", "Shuffled", "Patient paths"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.7,
            "ytick.labelsize": 6.7,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(width=0.8, length=3)


def _paired_panel(
    axis: plt.Axes,
    patient: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    q_value: float,
) -> None:
    pivot = patient.pivot(index="subject", columns="control", values=metric)
    x = np.arange(3)
    for subject, values in pivot.iterrows():
        dataset = patient.loc[patient.subject == subject, "dataset"].iloc[0]
        y = [values[item] for item in CONTROL_ORDER]
        axis.plot(x, y, color=COLORS[dataset], alpha=0.20, lw=0.6)
        axis.scatter(
            x,
            y,
            color=COLORS[dataset],
            alpha=0.58,
            s=8,
            edgecolor="none",
        )
    median = np.asarray(
        [pivot[item].median() for item in CONTROL_ORDER], float
    )
    axis.plot(x, median, color="#202020", lw=1.5, zorder=4)
    axis.scatter(x, median, color="#202020", s=18, zorder=5)
    axis.set_xticks(x, CONTROL_LABELS, rotation=20)
    axis.set_ylabel(ylabel)
    axis.set_ylim(bottom=0)
    axis.set_title(title, loc="left", pad=4)
    axis.text(
        0.98,
        0.98,
        f"q vs shuffle = {q_value:.3g}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
    )
    _finish_axis(axis)


def main() -> None:
    _style()
    prior = pd.read_csv(PRIOR_ROOT / "transition_skeleton_audit.csv")
    null = pd.read_csv(
        PRIOR_ROOT / "transition_skeleton_weight_null_audit.csv"
    )
    patient = pd.read_csv(ANALYSIS_ROOT / "patient_median_metrics.csv")
    stats = pd.read_csv(ANALYSIS_ROOT / "paired_statistics.csv")
    gate = json.loads((ANALYSIS_ROOT / "gate_summary.json").read_text())
    inventory = pd.read_csv(ANALYSIS_ROOT / "run_inventory.csv")
    if inventory.ictal_target_read.any() or gate["ictal_target_read"]:
        raise RuntimeError("ictal target seal failed")

    def q_for(metric: str) -> float:
        row = stats.loc[
            (stats.baseline == "shuffled_paths")
            & (stats.metric == metric)
        ]
        if len(row) != 1:
            raise RuntimeError(f"missing statistic: {metric}")
        return float(row.wilcoxon_q_directional.iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.55))

    axis = axes[0, 0]
    rng = np.random.default_rng(20260726)
    for index, dataset in enumerate(["epilepsiae", "yuquan"]):
        values = prior.loc[
            prior.dataset == dataset, "split_half_skeleton_cosine"
        ].to_numpy(float)
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            color=COLORS[dataset],
            s=14,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.25,
        )
        axis.plot(
            [index - 0.18, index + 0.18],
            [np.median(values)] * 2,
            color="#202020",
            lw=1.5,
        )
    axis.axhline(0.8, color="#888888", lw=0.7, ls="--")
    axis.set_xticks([0, 1], ["Epilepsiae", "Yuquan"])
    axis.set_ylim(0.75, 1.01)
    axis.set_ylabel("Split-half graph cosine")
    axis.set_title("Stable patient paths", loc="left", pad=4)
    _finish_axis(axis)

    axis = axes[0, 1]
    for dataset in ["epilepsiae", "yuquan"]:
        selected = null.dataset == dataset
        axis.scatter(
            null.loc[selected, "heldout_nll_null_median"],
            null.loc[selected, "heldout_nll_true_weights"],
            color=COLORS[dataset],
            s=17,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.25,
            label=dataset.capitalize(),
        )
    limits = [
        0.0,
        1.03
        * float(
            max(
                null.heldout_nll_null_median.max(),
                null.heldout_nll_true_weights.max(),
            )
        ),
    ]
    axis.plot(limits, limits, color="#777777", lw=0.8, ls="--")
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Weight-shuffle NLL")
    axis.set_ylabel("Patient-path NLL")
    axis.set_title("Edge-weight specificity", loc="left", pad=4)
    axis.legend(frameon=False, loc="upper left")
    _finish_axis(axis)

    _paired_panel(
        axes[0, 2],
        patient,
        "heldout_event_nll",
        "Held-out next-contact NLL",
        "One-step prediction",
        q_for("heldout_event_nll"),
    )
    _paired_panel(
        axes[1, 0],
        patient,
        "participation_mae",
        "Participation MAE",
        "Contact participation",
        q_for("participation_mae"),
    )
    _paired_panel(
        axes[1, 1],
        patient,
        "precedence_mae",
        "Pairwise precedence MAE",
        "Fine propagation order",
        q_for("precedence_mae"),
    )
    _paired_panel(
        axes[1, 2],
        patient,
        "path_sliced_wasserstein",
        "Whole-path distance",
        "Event-level paths",
        q_for("path_sliced_wasserstein"),
    )

    for label, axis in zip("ABCDEF", axes.flat):
        axis.text(
            -0.18,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=9,
            fontweight="bold",
            va="top",
        )
    fig.tight_layout(w_pad=1.4, h_pad=1.45)
    figures = OUTPUT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "pdf"]:
        fig.savefig(
            figures / f"structured_graph_rnn_gate.{extension}",
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    (figures / "README.md").write_text(
        """### structured_graph_rnn_gate.png

六个 panel 构成结构化 RNN 的完整科学门。A 检查患者多路径图能否在 train80 内稳定复现；B 检查真实边权是否比同密度权重打乱更能解释 heldout 转移；C 检查 RNN 是否用这些路径改善下一触点预测；D 检查生成事件的触点参与分布；E 与 F 分别检验精细先后和完整事件路径。

**关注点**：A/B 证明结构先验有真实信息，C/D 说明模型是否利用该信息，E/F 才决定模型是否足以连接到发作早期静态能量场。每条细线是一位患者的三 seed 中位数，黑线为 cohort median；q 值是患者级、相对同密度路径打乱的方向性 Wilcoxon 检验并经过全指标 FDR。
"""
    )
    metadata = {
        "figure": "structured_graph_rnn_gate",
        "n_patients": int(patient.subject.nunique()),
        "n_seeds": int(gate["n_seeds"]),
        "controls": CONTROL_ORDER,
        "primary_gate_pass": bool(gate["primary_gate_pass"]),
        "ictal_target_read": False,
        "inputs": {
            "prior_audit": str(
                PRIOR_ROOT / "transition_skeleton_audit.csv"
            ),
            "weight_null": str(
                PRIOR_ROOT / "transition_skeleton_weight_null_audit.csv"
            ),
            "patient_metrics": str(
                ANALYSIS_ROOT / "patient_median_metrics.csv"
            ),
            "statistics": str(ANALYSIS_ROOT / "paired_statistics.csv"),
        },
    }
    (figures / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
