#!/usr/bin/env python3
"""Build the paper-facing transition-decomposition diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_interictal_transition_decomposition_v0_1"
FIGURES = OUT / "figures"


def comparison_values(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    name: str,
) -> tuple[np.ndarray, pd.Series]:
    row = comparisons.set_index("comparison").loc[name]
    table = metrics[
        metrics["analysis_scope"] == row["analysis_scope"]
    ].pivot(index="subject", columns="model", values="heldout_next_nll")
    values = (
        table[str(row["baseline"])] - table[str(row["model"])]
    ).to_numpy(float)
    return values, row


def draw_group(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    *,
    color: str | np.ndarray,
    label: str,
    qvalue: float,
    n: int,
) -> None:
    offsets = np.linspace(-0.11, 0.11, len(values))
    ax.scatter(
        x + offsets,
        values,
        s=30,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
        zorder=2,
    )
    ax.plot(
        [x - 0.23, x + 0.23],
        [np.median(values)] * 2,
        color="black",
        lw=1.7,
        zorder=3,
    )
    ax.text(
        x,
        0.985,
        f"n={n}, q={qvalue:.2g}",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=7.4,
    )


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.axhline(0, color="#777777", ls="--", lw=0.9, zorder=0)
    ax.set_title(title, loc="left", fontsize=10.5, pad=8)
    ax.set_ylabel("Held-out NLL benefit", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


def main() -> None:
    metrics = pd.read_csv(OUT / "patient_model_metrics.csv")
    comparisons = pd.read_csv(OUT / "cohort_comparisons.csv")
    operator = pd.read_csv(OUT / "operator_component_metrics.csv").set_index(
        "subject"
    )
    cross_metrics = pd.read_csv(OUT / "cross_shaft_prefix_metrics.csv")
    cross_status = json.loads(
        (OUT / "CROSS_SHAFT_STATUS.json").read_text(encoding="utf-8")
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    palette = {
        "blue": "#4477AA",
        "green": "#228833",
        "rose": "#CC6677",
        "orange": "#EE7733",
        "purple": "#AA3377",
    }

    ax = axes[0, 0]
    for x, name, label, color in (
        (0, "directed_logit_over_node", "Markov\n> node", palette["blue"]),
        (
            1,
            "directed_beyond_local_geometry",
            "Markov\n> local geometry",
            palette["green"],
        ),
    ):
        values, row = comparison_values(metrics, comparisons, name)
        draw_group(
            ax,
            x,
            values,
            color=color,
            label=label,
            qvalue=float(row["bh_fdr_q"]),
            n=int(row["n_patients"]),
        )
    cross = cross_metrics.pivot(
        index="subject",
        columns="model",
        values="heldout_cross_shaft_conditional_nll",
    )
    cross_values = (
        cross["same_shaft_plus_distance"]
        - cross["directed_logit_markov"]
    ).to_numpy(float)
    draw_group(
        ax,
        2,
        cross_values,
        color=palette["rose"],
        label="Cross-shaft Markov\n> local geometry",
        qvalue=float(cross_status["bh_fdr_q"]),
        n=int(cross_status["n_patients"]),
    )
    ax.set_xticks(
        [0, 1, 2],
        [
            "Markov\n> node",
            "Markov\n> local geometry",
            "Cross-shaft Markov\n> local geometry",
        ],
    )
    style_axis(ax, "A  Transition signal beyond marginal hazard and geometry")

    ax = axes[0, 1]
    for x, name, label, color in (
        (0, "symmetric_over_node", "Symmetric\n> node", palette["blue"]),
        (
            1,
            "skew_increment_over_symmetric",
            "Skew increment\n> symmetric",
            palette["green"],
        ),
    ):
        values, row = comparison_values(metrics, comparisons, name)
        draw_group(
            ax,
            x,
            values,
            color=color,
            label=label,
            qvalue=float(row["bh_fdr_q"]),
            n=int(row["n_patients"]),
        )
    ax.set_xticks([0, 1], ["Symmetric\n> node", "Skew increment\n> symmetric"])
    style_axis(ax, "B  Effective transition residual is predominantly symmetric")

    ax = axes[1, 0]
    axis_values, axis_row = comparison_values(
        metrics, comparisons, "axis_beyond_local_geometry"
    )
    physical = metrics[metrics["analysis_scope"] == "physical_axis_n22"].pivot(
        index="subject", columns="model", values="heldout_next_nll"
    )
    axis_by_subject = (
        physical["same_shaft_plus_distance"]
        - physical["physical_axis_residual"]
    )
    axis_colors = np.asarray(
        [
            palette["blue"]
            if operator.loc[subject, "axis_excess_coefficient"] > 0
            else palette["orange"]
            for subject in axis_by_subject.index
        ]
    )
    draw_group(
        ax,
        0,
        axis_by_subject.to_numpy(float),
        color=axis_colors,
        label="Axis residual\n> local geometry",
        qvalue=float(axis_row["bh_fdr_q"]),
        n=int(axis_row["n_patients"]),
    )
    source_values, source_row = comparison_values(
        metrics, comparisons, "source_direction_beyond_axis"
    )
    draw_group(
        ax,
        1,
        source_values,
        color=palette["purple"],
        label="Source modulation\n> axis residual",
        qvalue=float(source_row["bh_fdr_q"]),
        n=int(source_row["n_patients"]),
    )
    ax.set_xticks(
        [0, 1],
        [
            "Axis residual\n> local geometry",
            "Source modulation\n> axis residual",
        ],
    )
    ax.scatter([], [], color=palette["blue"], label="axis coefficient > 0")
    ax.scatter([], [], color=palette["orange"], label="axis coefficient < 0")
    ax.legend(
        frameon=False,
        fontsize=7.2,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.88),
    )
    style_axis(ax, "C  Axis-aligned anisotropy and source-conditioned modulation")

    ax = axes[1, 1]
    for x, name, label, color in (
        (
            0,
            "last_rank_over_source_only",
            "Last rank\n> source only",
            palette["blue"],
        ),
        (
            1,
            "ordered_history_over_last_rank",
            "Ordered history\n> last rank",
            palette["green"],
        ),
    ):
        values, row = comparison_values(metrics, comparisons, name)
        draw_group(
            ax,
            x,
            values,
            color=color,
            label=label,
            qvalue=float(row["bh_fdr_q"]),
            n=int(row["n_patients"]),
        )
    ax.set_xticks([0, 1], ["Last rank\n> source only", "Ordered history\n> last rank"])
    style_axis(ax, "D  Ordered multi-step history adds predictive information")

    fig.tight_layout(w_pad=2.0, h_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURES / "transition_signal_decomposition_paper_ready.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "transition_signal_decomposition_paper_ready.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print("transition decomposition paper-ready figure written")


if __name__ == "__main__":
    main()
