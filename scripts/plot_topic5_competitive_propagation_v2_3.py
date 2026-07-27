#!/usr/bin/env python3
"""Render the patient-first paper-ready v2.3 formal result figure."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
FORMAL = BASE / "formal"
FIGURES = BASE / "figures"
RUST = "#A35E48"
BLUE = "#4477AA"
GREEN = "#228833"
PURPLE = "#AA3377"
GRAY = "#777777"
LIGHT = "#D9D9D9"


def benefit(
    pivot: pd.DataFrame, baseline: str, model: str
) -> np.ndarray:
    return (pivot[baseline] - pivot[model]).to_numpy(float)


def strip(
    axis: plt.Axes,
    arrays: list[np.ndarray],
    labels: list[str],
    colors: list[str],
) -> None:
    generator = np.random.default_rng(20260727)
    for index, (values, color) in enumerate(zip(arrays, colors)):
        jitter = generator.uniform(-0.11, 0.11, size=len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=15,
            facecolor=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.78,
            zorder=2,
        )
        median = float(np.median(values))
        axis.plot(
            [index - 0.18, index + 0.18],
            [median, median],
            color="black",
            linewidth=1.7,
            zorder=3,
        )
    axis.axhline(0.0, color="#888888", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(len(labels)), labels)
    axis.tick_params(axis="x", rotation=20)
    axis.spines[["top", "right"]].set_visible(False)
    axis.margins(x=0.18)


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.13,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def draw_model(axis: plt.Axes) -> None:
    axis.set_xlim(-0.25, 5.1)
    axis.set_ylim(-1.3, 1.45)
    axis.axis("off")
    x = np.arange(5, dtype=float)
    for first in range(4):
        axis.plot(
            [x[first], x[first + 1]],
            [0, 0],
            color="#9E9E9E",
            linewidth=2.0,
            zorder=1,
        )
        axis.add_patch(
            FancyArrowPatch(
                (x[first] + 0.18, 0.08),
                (x[first + 1] - 0.18, 0.08),
                arrowstyle="-|>",
                mutation_scale=8,
                color="#B0B0B0",
                linewidth=0.7,
            )
        )
        axis.add_patch(
            FancyArrowPatch(
                (x[first + 1] - 0.18, -0.08),
                (x[first] + 0.18, -0.08),
                arrowstyle="-|>",
                mutation_scale=8,
                color="#B0B0B0",
                linewidth=0.7,
            )
        )
    colors = [RUST, "#D6A594", "#C8C8C8", "#C8C8C8", "#C8C8C8"]
    axis.scatter(
        x,
        np.zeros_like(x),
        s=155,
        c=colors,
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    axis.annotate(
        "observed source",
        xy=(0, 0.1),
        xytext=(0.05, 0.75),
        ha="left",
        fontsize=7.5,
        arrowprops={"arrowstyle": "-", "color": RUST, "linewidth": 0.8},
    )
    axis.text(
        2.0,
        1.2,
        "shared symmetric axis scaffold",
        ha="center",
        fontsize=8.2,
        fontweight="bold",
    )
    axis.text(
        2.0,
        -0.68,
        r"$P_{t+1}=\rho_P P_t+W^Sx_t$",
        ha="center",
        fontsize=8.2,
        color=BLUE,
    )
    axis.text(
        2.0,
        -1.08,
        r"$C_{t+1}=\rho_C C_t+W^Sx_t$",
        ha="center",
        fontsize=8.2,
        color=PURPLE,
    )


def main() -> None:
    status = json.loads(
        (FORMAL / "FORMAL_GATE_STATUS.json").read_text(encoding="utf-8")
    )
    patient = pd.read_csv(FORMAL / "patient_model_metrics.csv")
    comparisons = pd.read_csv(FORMAL / "claim_comparisons.csv").set_index(
        "comparison"
    )
    recovery = pd.read_csv(FORMAL / "benefit_recovery.csv")
    runs = pd.read_csv(FORMAL / "formal_run_inventory.csv")
    if status.get("target_values_read") or patient.subject.nunique() != 22:
        raise SystemExit("formal figure input is not sealed n=22")
    pivot = patient.pivot(
        index="subject", columns="model", values="heldout_categorical_nll"
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 5.35))
    plt.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.92,
        wspace=0.42,
        hspace=0.58,
    )

    draw_model(axes[0, 0])
    axes[0, 0].set_title(
        "Structured recurrent model",
        fontsize=8.6,
        loc="left",
        pad=4,
    )

    values_a = benefit(
        pivot, "node_bias_categorical", "axis_two_state_source_full"
    )
    strip(axes[0, 1], [values_a], ["Full model"], [RUST])
    axes[0, 1].set_ylabel("Held-out NLL benefit")
    axes[0, 1].set_title(
        "Next-contact prediction",
        fontsize=8.6,
        loc="left",
        pad=4,
    )
    row = comparisons.loc["claim_A_predictive_adequacy"]
    axes[0, 1].text(
        0.98,
        0.96,
        f"{int(row.n_positive)}/{int(row.n_patients)} positive\n"
        f"q={row.bh_fdr_q:.2g}",
        transform=axes[0, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
    )

    history_arrays = [
        benefit(
            pivot,
            "axis_instantaneous_no_history",
            "axis_two_state_source_full",
        ),
        benefit(
            pivot,
            "axis_one_state_no_competition",
            "axis_two_state_source_full",
        ),
    ]
    strip(
        axes[0, 2],
        history_arrays,
        ["History", "Competition"],
        [BLUE, PURPLE],
    )
    axes[0, 2].set_ylabel("Held-out NLL benefit")
    axes[0, 2].set_title(
        "History and competition",
        fontsize=8.6,
        loc="left",
        pad=4,
    )

    structure_arrays = [
        benefit(
            pivot,
            "local_isotropic_two_state",
            "axis_two_state_source_full",
        ),
        benefit(
            pivot,
            "local_isotropic_two_state",
            "axis_two_state_no_source",
        ),
        benefit(
            pivot,
            "axis_two_state_no_source",
            "axis_two_state_source_full",
        ),
    ]
    strip(
        axes[1, 0],
        structure_arrays,
        ["Axis bundle", "Matched axis", "Source term"],
        [GREEN, BLUE, PURPLE],
    )
    axes[1, 0].set_ylabel("Held-out NLL benefit")
    axes[1, 0].set_title(
        "Axis and source increments",
        fontsize=8.6,
        loc="left",
        pad=4,
    )

    axis = axes[1, 1]
    x = recovery.ordered_markov_over_node_benefit.to_numpy(float)
    y = recovery.full_over_node_benefit.to_numpy(float)
    axis.scatter(
        x,
        y,
        s=22,
        color=RUST,
        edgecolor="white",
        linewidth=0.4,
        alpha=0.82,
    )
    limits = np.asarray(
        [
            min(0.0, float(np.min(x)), float(np.min(y))),
            max(float(np.max(x)), float(np.max(y))),
        ]
    )
    padding = 0.06 * max(1.0e-6, limits[1] - limits[0])
    limits += np.asarray([-padding, padding])
    axis.plot(limits, limits, color="#999999", linestyle="--", linewidth=0.8)
    axis.axhline(0, color="#BBBBBB", linewidth=0.6)
    axis.axvline(0, color="#BBBBBB", linewidth=0.6)
    axis.set_xlim(limits)
    axis.set_ylim(limits)
    axis.set_xlabel("Ordered-history Markov benefit")
    axis.set_ylabel("Structured RNN benefit")
    axis.set_title(
        "Recovery of empirical transition signal",
        fontsize=8.6,
        loc="left",
        pad=4,
    )
    axis.spines[["top", "right"]].set_visible(False)
    ratio = status.get("median_benefit_recovery_ratio")
    axis.text(
        0.04,
        0.96,
        "median recovery="
        + ("N/A" if ratio is None else f"{100 * ratio:.0f}%"),
        transform=axis.transAxes,
        va="top",
        fontsize=7.0,
    )

    axis = axes[1, 2]
    allowed = bool(status["latent_state_analysis_allowed"])
    if allowed:
        full = runs[runs.model == "axis_two_state_source_full"].copy()
        per_patient = full.groupby("subject", as_index=False).agg(
            gain_propagation=("gain_propagation", "mean"),
            gain_competition=("gain_competition", "mean"),
        )
        ratio_values = (
            per_patient.gain_competition
            / per_patient.gain_propagation.clip(lower=1.0e-8)
        ).to_numpy(float)
        strip(
            axis,
            [ratio_values],
            ["Competition / drive"],
            [PURPLE],
        )
        axis.axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
        axis.set_ylabel("Learned gain ratio")
        axis.set_title(
            "Delayed competition in the fitted state",
            fontsize=8.6,
            loc="left",
            pad=4,
        )
    else:
        axis.axis("off")
        labels = [
            ("Predictive", status["claim_A_predictive_adequacy"]),
            ("History", status["claim_B_history_vs_instantaneous"]),
            ("Competition", status["claim_B_competition_vs_one_state"]),
            ("Axis", status["claim_C_matched_axis_increment"]),
            ("Source", status["claim_D_source_conditioned_direction"]),
        ]
        for index, (label, state) in enumerate(labels):
            y0 = 0.86 - index * 0.175
            passed = state == "PASS"
            box = FancyBboxPatch(
                (0.07, y0 - 0.055),
                0.82,
                0.105,
                boxstyle="round,pad=0.02",
                facecolor="#DDEBDD" if passed else "#F2D9D5",
                edgecolor=GREEN if passed else RUST,
                linewidth=0.8,
                transform=axis.transAxes,
            )
            axis.add_patch(box)
            axis.text(
                0.13,
                y0,
                label,
                transform=axis.transAxes,
                va="center",
                fontsize=8.0,
            )
            axis.text(
                0.82,
                y0,
                state,
                transform=axis.transAxes,
                ha="right",
                va="center",
                fontsize=7.5,
                fontweight="bold",
            )
        axis.set_title(
            "Interpretation gates",
            fontsize=8.6,
            loc="left",
            pad=4,
        )

    for label, axis in zip("ABCDEF", axes.flat):
        panel_label(axis, label)
        axis.tick_params(labelsize=7)
        axis.title.set_fontweight("bold")
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / "competitive_propagation_rnn_formal.png"
    pdf = FIGURES / "competitive_propagation_rnn_formal.pdf"
    fig.savefig(png, dpi=400, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    metadata = {
        "figure": png.name,
        "n_patients": 22,
        "n_seeds": 3,
        "latent_state_analysis_allowed": allowed,
        "formal_gate_status": status,
        "target_values_read": False,
        "inputs": [
            "formal/patient_model_metrics.csv",
            "formal/claim_comparisons.csv",
            "formal/benefit_recovery.csv",
            "formal/formal_run_inventory.csv",
        ],
    }
    (FIGURES / "competitive_propagation_rnn_formal.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    readme = (
        "### competitive_propagation_rnn_formal.png\n\n"
        "A 给出唯一允许的对称 scaffold、source 与 propagation/competition "
        "状态；B–D 依次检验模型是否可预测、历史状态是否必要，以及 axis/source "
        "项是否提供匹配增益。E 比较可解释模型恢复了多少 empirical "
        "ordered-history Markov 信号；F 仅在 A–C 与 matched-axis safeguard "
        "通过时显示状态参数，否则如实显示停止门。\n\n"
        "**关注点**：所有点均为 patient-first heldout20 结果；模型不读取 A/B、"
        "SOZ 或发作期 target，图中正向 benefit 表示前一个模型的 NLL 更低。\n"
    )
    (FIGURES / "README.md").write_text(readme, encoding="utf-8")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
