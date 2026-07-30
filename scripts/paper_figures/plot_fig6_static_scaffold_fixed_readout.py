#!/usr/bin/env python3
"""Paper-ready Figure 6 candidate for the fixed static-scaffold validation."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESULT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
INTERNAL = ROOT / "results/topic5_rnn_internal_state_reduction"
OUT = (
    ROOT
    / "results/paper-ready-figure"
    / "fig6_static_contact_topography"
    / "figures"
)
MODEL_COLORS = {
    "raw_train80_participation": "#6F6F6F",
    "best_validation_regularized_participation": "#2878B5",
    "static_contact_hazard": "#76B7B2",
    "last_set_first_order": "#59A14F",
    "rank_shuffle_gru": "#F28E2B",
    "full_history_gru": "#C43C39",
    "teacher_forced_full_gru": "#8C6BB1",
}
DISPLAY = {
    "raw_train80_participation": "raw rate",
    "best_validation_regularized_participation": "best regularized",
    "static_contact_hazard": "static hazard",
    "last_set_first_order": "first-order",
    "rank_shuffle_gru": "rank-shuffle",
    "full_history_gru": "full GRU",
    "teacher_forced_full_gru": "teacher-forced",
}


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.5,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.65, length=2.5)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.16,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def jitter_points(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    *,
    width: float = 0.10,
    seed: int,
    size: float = 10,
) -> None:
    rng = np.random.default_rng(seed)
    ax.scatter(
        x + rng.uniform(-width, width, len(values)),
        values,
        s=size,
        facecolor=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=3,
    )
    median = float(np.median(values))
    q1, q3 = np.quantile(values, [0.25, 0.75])
    ax.plot([x, x], [q1, q3], color="black", lw=1.1, zorder=4)
    ax.plot([x - 0.13, x + 0.13], [median, median], color="black", lw=1.4, zorder=4)


def schematic(ax: plt.Axes) -> None:
    ax.set_axis_off()
    boxes = [
        (0.02, 0.57, 0.24, 0.23, "interictal\nrank events", "#E9F1F7"),
        (0.37, 0.57, 0.24, 0.23, "ordered GRU\nor static\nestimator", "#F5E8E6"),
        (0.72, 0.57, 0.24, 0.23, "one fixed\ncontact\ntopography", "#E9F3EA"),
        (0.72, 0.13, 0.24, 0.23, "early-ictal\n1–150 Hz\nenergy", "#EDE8F3"),
        (0.37, 0.13, 0.24, 0.23, "signed +\norientation-free\nreadouts", "#F3EFE2"),
    ]
    for x, y, w, h, text, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            facecolor=color,
            edgecolor="#555555",
            linewidth=0.75,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    arrows = [
        ((0.26, 0.685), (0.37, 0.685)),
        ((0.61, 0.685), (0.72, 0.685)),
        ((0.84, 0.57), (0.84, 0.36)),
        ((0.72, 0.245), (0.61, 0.245)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#555555",
            )
        )
    ax.plot([0.03, 0.96], [0.47, 0.47], color="#999999", lw=0.8, ls="--")
    ax.text(
        0.03,
        0.43,
        "target opened only after field freeze",
        ha="left",
        va="top",
        color="#666666",
        fontsize=6.7,
    )
    ax.set_title("Frozen cross-state test", loc="left", fontweight="bold")


def order_sensitivity(ax: plt.Axes) -> None:
    perturbation = pd.read_csv(
        INTERNAL / "interictal_order_full_vs_rank_shuffle.csv"
    )
    formal = pd.read_csv(
        ROOT
        / "results/topic5_interictal_rank_distribution/runs"
        / "formal_multiseed_20260725_v1/patient_seed_collapsed_summary.csv"
    )
    formal_root = (
        ROOT
        / "results/topic5_interictal_rank_distribution/runs"
        / "formal_multiseed_20260725_v1"
    )
    rank_shuffle_rows = []
    for seed in (20260725, 20260726, 20260727):
        for path in sorted(
            (formal_root / f"seed_{seed}").glob("*/heldout_metrics.csv")
        ):
            metric = pd.read_csv(path).set_index("control").heldout_event_nll
            rank_shuffle_rows.append(
                {
                    "subject": path.parent.name,
                    "gain": float(
                        metric["rank_shuffle_gru"]
                        - metric["full_history_gru"]
                    ),
                }
            )
    rank_shuffle_gain = (
        pd.DataFrame(rank_shuffle_rows)
        .groupby("subject")
        .gain.mean()
        .to_numpy(float)
    )
    value_sets = [
        formal["ordered_history_nll_gain"].to_numpy(float),
        rank_shuffle_gain,
        perturbation.loc[
            (perturbation.metric == "nll_loss")
            & (perturbation.order_perturbation == "shuffle"),
            "full_minus_rank_shuffle_sensitivity",
        ].to_numpy(float),
    ]
    colors = ("#777777", "#2878B5", "#C43C39")
    for index, values in enumerate(value_sets):
        jitter_points(
            ax,
            index,
            values,
            colors[index],
            width=0.12,
            seed=20260728 + index,
            size=11,
        )
        p = wilcoxon(values, alternative="greater").pvalue
        ax.text(
            index,
            np.max(values) + 0.006,
            f"{np.count_nonzero(values > 0)}/34\nP={p:.1e}",
            ha="center",
            va="bottom",
            fontsize=6.3,
        )
    lower = min(0.0, min(np.min(values) for values in value_sets)) - 0.01
    upper = max(np.max(values) for values in value_sets) + 0.035
    ax.set_ylim(lower, upper)
    ax.axhline(0, color="#777777", lw=0.75, ls="--")
    ax.set_xticks(
        [0, 1, 2],
        [
            "full vs best\nnonrecurrent",
            "full vs trained\nrank-shuffle",
            "NLL cost when\norder is shuffled",
        ],
    )
    ax.set_ylabel("NLL difference")
    ax.set_title(
        "Order benefit depends on the comparator",
        loc="left",
        fontweight="bold",
    )
    clean_axis(ax)


def signed_readout(ax: plt.Axes, combined: pd.DataFrame) -> None:
    models = [
        "raw_train80_participation",
        "best_validation_regularized_participation",
        "rank_shuffle_gru",
        "full_history_gru",
        "teacher_forced_full_gru",
    ]
    subset = combined.loc[
        (combined.null_mode == "all_contact")
        & combined.eligible
        & combined.model.isin(models)
    ]
    for index, model in enumerate(models):
        values = subset.loc[
            subset.model == model, "observed_signed_rho"
        ].to_numpy(float)
        jitter_points(
            ax,
            index,
            values,
            MODEL_COLORS[model],
            seed=20260800 + index,
        )
    ax.axhline(0, color="#777777", lw=0.75, ls="--")
    ax.set_xticks(range(len(models)), [DISPLAY[model] for model in models], rotation=35, ha="right")
    ax.set_ylabel("signed Spearman ρ")
    ax.set_title("Fixed positive direction is not supported", loc="left", fontweight="bold")
    clean_axis(ax)


def morphology_nulls(ax: plt.Axes, combined: pd.DataFrame) -> None:
    models = [
        "raw_train80_participation",
        "best_validation_regularized_participation",
        "rank_shuffle_gru",
        "full_history_gru",
    ]
    nulls = [
        ("all_contact", "all-contact"),
        ("within_shaft_circular", "within-shaft"),
        ("geometry_smooth_rbf", "smooth-field"),
    ]
    offsets = np.linspace(-0.27, 0.27, len(models))
    for null_index, (null_mode, label) in enumerate(nulls):
        for model_index, model in enumerate(models):
            values = combined.loc[
                (combined.model == model)
                & (combined.null_mode == null_mode)
                & combined.eligible,
                "absolute_margin",
            ].to_numpy(float)
            if not len(values):
                continue
            x = null_index + offsets[model_index]
            median = np.median(values)
            lo, hi = np.quantile(values, [0.25, 0.75])
            ax.plot([x, x], [lo, hi], color=MODEL_COLORS[model], lw=1.1)
            ax.scatter(
                [x],
                [median],
                s=24,
                facecolor=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    ax.axhline(0, color="#777777", lw=0.75, ls="--")
    ax.set_xticks(range(len(nulls)), [label for _, label in nulls])
    ax.set_ylabel("|ρ| minus spatial-null median")
    ax.set_title("Orientation-free morphology survives spatial nulls", loc="left", fontweight="bold")
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color=MODEL_COLORS[model],
            lw=1.2,
            markersize=4,
            label=DISPLAY[model],
        )
        for model in models
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right")
    clean_axis(ax)


def gru_increment(
    ax: plt.Axes,
    phase1: pd.DataFrame,
    phase2: pd.DataFrame,
    phase3: pd.DataFrame,
) -> None:
    controls = [
        ("raw_train80_participation", phase2),
        ("best_validation_regularized_participation", phase2),
        ("static_contact_hazard", phase1),
        ("last_set_first_order", phase1),
        ("rank_shuffle_gru", phase1),
        ("teacher_forced_full_gru", phase3),
    ]
    full = phase1.loc[
        (phase1.model == "full_history_gru")
        & (phase1.null_mode == "all_contact")
        & phase1.eligible
    ].set_index("subject")
    for index, (control, frame) in enumerate(controls):
        right = frame.loc[
            (frame.model == control)
            & (frame.null_mode == "all_contact")
            & frame.eligible
        ].set_index("subject")
        common = full.index.intersection(right.index)
        values = (
            full.loc[common, "absolute_margin"]
            - right.loc[common, "absolute_margin"]
        ).to_numpy(float)
        jitter_points(
            ax,
            index,
            values,
            MODEL_COLORS[control],
            seed=20260900 + index,
            width=0.11,
            size=9,
        )
    ax.axhline(0, color="#777777", lw=0.75, ls="--")
    ax.set_xticks(
        range(len(controls)),
        [DISPLAY[control] for control, _ in controls],
        rotation=35,
        ha="right",
    )
    ax.set_ylabel("full-GRU increment in |ρ| margin")
    ax.set_title("No GRU-specific static gain", loc="left", fontweight="bold")
    clean_axis(ax)


def confound_sensitivity(ax: plt.Axes, confound: pd.DataFrame) -> None:
    blocks = [
        ("within_shaft_position", "shaft position"),
        ("geometry_pc1", "geometry PC1"),
        ("soz_indicator", "SOZ"),
        ("baseline_band_power", "baseline power"),
    ]
    models = [
        "raw_train80_participation",
        "full_history_gru",
    ]
    offsets = (-0.10, 0.10)
    valid_blocks = [
        item for item in blocks if item[0] in set(confound.confound_block)
    ]
    plotted_values = []
    for block_index, (block, label) in enumerate(valid_blocks):
        block_top = -np.inf
        block_n = []
        for model_index, model in enumerate(models):
            values = confound.loc[
                (confound.confound_block == block)
                & (confound.model == model)
                & confound.eligible,
                "absolute_margin",
            ].to_numpy(float)
            if not len(values):
                continue
            plotted_values.append(values)
            x = block_index + offsets[model_index]
            median = np.median(values)
            lo, hi = np.quantile(values, [0.25, 0.75])
            block_top = max(block_top, float(hi))
            block_n.append(len(values))
            ax.plot([x, x], [lo, hi], color=MODEL_COLORS[model], lw=1.2)
            ax.scatter(
                [x],
                [median],
                s=27,
                facecolor=MODEL_COLORS[model],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        if block_n:
            n_label = (
                f"n={block_n[0]}"
                if len(set(block_n)) == 1
                else "n=" + "/".join(map(str, block_n))
            )
            ax.text(
                block_index,
                block_top + 0.025,
                n_label,
                ha="center",
                va="bottom",
                fontsize=5.8,
                color="#555555",
            )
    ax.axhline(0, color="#777777", lw=0.75, ls="--")
    if plotted_values:
        lower = min(0.0, min(np.min(values) for values in plotted_values)) - 0.05
        upper = max(np.max(values) for values in plotted_values) + 0.12
        ax.set_ylim(lower, upper)
    ax.set_xticks(
        range(len(valid_blocks)),
        [label for _, label in valid_blocks],
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("partial |ρ| minus residual-null")
    ax.set_title("One-confound-at-a-time sensitivity", loc="left", fontweight="bold")
    handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            color=MODEL_COLORS[model],
            lw=1.2,
            markersize=4,
            label=DISPLAY[model],
        )
        for model in models
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    clean_axis(ax)


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    phase1 = pd.read_csv(RESULT / "phase1_existing_fields_patient_metrics.csv")
    phase2 = pd.read_csv(RESULT / "phase2_regularized_baseline_patient_metrics.csv")
    phase3 = pd.read_csv(RESULT / "phase3_teacher_forced_patient_metrics.csv")
    confound = pd.read_csv(RESULT / "phase4_contact_confound_partial_scores.csv")
    combined = pd.concat([phase1, phase2, phase3], ignore_index=True)

    fig = plt.figure(figsize=(10.8, 6.9))
    grid = fig.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.985,
        bottom=0.10,
        top=0.955,
        wspace=0.34,
        hspace=0.43,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(2) for column in range(3)]
    schematic(axes[0])
    order_sensitivity(axes[1])
    signed_readout(axes[2], combined)
    morphology_nulls(axes[3], combined)
    gru_increment(axes[4], phase1, phase2, phase3)
    confound_sensitivity(axes[5], confound)
    for letter, ax in zip("ABCDEF", axes):
        panel_letter(ax, letter)

    stem = OUT / "fig6_static_contact_topography"
    fig.savefig(stem.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)

    metadata = {
        "figure": "fig6_static_contact_topography",
        "title": (
            "Interictal contact topography shows static cross-state "
            "correspondence and order information without a GRU-specific "
            "static-field increment"
        ),
        "status": "supplementary_boundary_control",
        "scientific_panels": {
            "A": "target-sealed fixed-field cross-state contract",
            "B": (
                "formal heldout full-vs-rank-shuffle gain alongside matched "
                "order-perturbation sensitivity, 34 patients"
            ),
            "C": "pre-specified positive signed early-ictal readout, 16 patients",
            "D": (
                "orientation-free morphology under all-contact, within-shaft, "
                "and smooth-field nulls; absolute correlation permits similar "
                "or reversed contact ordering"
            ),
            "E": "full-GRU static increment over target-free and history controls",
            "F": (
                "one-confound-at-a-time partial-rank morphology versus residual "
                "permutation null; not multivariable causal adjustment"
            ),
        },
        "claim_boundary": (
            "True event order improves heldout NLL over a separately trained "
            "rank-shuffle GRU, but the full-history GRU does not outperform "
            "the strongest nonrecurrent prefix comparator; "
            "interictal participation and early-ictal energy show "
            "orientation-free contact-topography correspondence that is also "
            "present in nonrecurrent fields; absolute correlation does not "
            "establish positive replay, a physical axis or dynamic seizure "
            "propagation."
        ),
        "target_reused_not_independent_confirmation": True,
        "primary_signed_direction_supported": False,
        "ordered_history_vs_rank_shuffle_supported": True,
        "ordered_history_static_increment_established": False,
        "inputs": [
            str((INTERNAL / "interictal_order_full_vs_rank_shuffle.csv").relative_to(ROOT)),
            str((RESULT / "phase1_existing_fields_patient_metrics.csv").relative_to(ROOT)),
            str((RESULT / "phase2_regularized_baseline_patient_metrics.csv").relative_to(ROOT)),
            str((RESULT / "phase3_teacher_forced_patient_metrics.csv").relative_to(ROOT)),
            str((RESULT / "phase4_contact_confound_partial_scores.csv").relative_to(ROOT)),
        ],
    }
    stem.with_name(stem.name + "_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    readme = OUT / "README.md"
    readme.write_text(
        "### fig6_static_contact_topography.png\n\n"
        "A–F 依次展示冻结的跨状态合同、间期顺序敏感性、预设 signed 读出、"
        "orientation-free 空间形态、GRU 相对简单对照的增量，以及单混杂敏感性。"
        "真实顺序相对独立训练的 rank-shuffle GRU 有 heldout NLL 增益，但 full-history "
        "GRU 没有超过最佳 nonrecurrent prefix 对照；"
        "间期参与率与发作早期能量沿同一患者特异 contact organization 变化，但患者间可同序"
        "也可逆序，且该对应不能归因于 GRU 特有动力学或物理病理轴。\n\n"
        "**关注点**：Panel C 的固定正方向不成立；Panel D 的 `|ρ|` 允许同序或逆序，"
        "不代表 positive replay；Panel E 必须据零线判断 GRU-specific increment；"
        "Panel F 一次只控制一个 covariate，不是多变量因果校正。\n",
        encoding="utf-8",
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
