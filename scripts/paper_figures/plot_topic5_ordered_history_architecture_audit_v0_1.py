#!/usr/bin/env python3
"""Render the paper-ready ordered-history architecture audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    strict_clinical_inventory,
)


ANALYSIS = ROOT / "results/topic5_ordered_history_architecture_audit/analysis"
INTERVENTION = (
    ROOT
    / "results/topic5_ordered_history_architecture_audit/interventions/"
    "selected_history_interventions_20260729"
)
BASELINE = (
    ROOT
    / "results/topic5_static_scaffold_fixed_readout_validation/"
    "target_free_baselines/per_subject"
)
OUT = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_ordered_history_architecture_audit/figures"
)
REPRESENTATIVE = "epilepsiae_1146"
SEEDS = (20260725, 20260726, 20260727)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.15,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def centered_rank(values: np.ndarray) -> np.ndarray:
    x = rankdata(np.asarray(values, float))
    return x - x.mean()


def zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, float)
    scale = np.std(x)
    return (x - np.mean(x)) / scale if scale > 0 else np.zeros_like(x)


def strip_summary(
    ax,
    values: dict[str, np.ndarray],
    *,
    colors: dict[str, str],
    ylabel: str,
    zero: bool = True,
) -> None:
    rng = np.random.default_rng(20260729)
    for index, (label, data) in enumerate(values.items()):
        x = index + rng.uniform(-0.13, 0.13, len(data))
        ax.scatter(
            x,
            data,
            s=10,
            color=colors[label],
            alpha=0.38,
            linewidth=0,
            rasterized=True,
        )
        median = float(np.median(data))
        ax.plot([index - 0.22, index + 0.22], [median, median], color="black", lw=1.5)
    if zero:
        ax.axhline(0, color="#777777", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(range(len(values)), list(values), rotation=28, ha="right")
    ax.set_ylabel(ylabel)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    style()
    architecture = pd.read_csv(
        ANALYSIS / "patient_seed_collapsed_architecture_metrics.csv"
    )
    intervention = pd.read_csv(ANALYSIS / "patient_history_intervention_costs.csv")
    intervention_summary = json.loads(
        (ANALYSIS / "HISTORY_INTERVENTION_SUMMARY.json").read_text()
    )
    early = pd.read_csv(ANALYSIS / "early_ictal_conditional_patient_metrics.csv")
    early_summary = json.loads(
        (ANALYSIS / "EARLY_ICTAL_CONDITIONAL_SUMMARY.json").read_text()
    )
    selection = json.loads((ANALYSIS / "ARCHITECTURE_SUMMARY.json").read_text())
    selected = selection["target_blind_best_non_gru"]["control"]

    nll = architecture.pivot(
        index="subject", columns="control", values="heldout_event_nll"
    )
    best_low_rank = min(
        [f"low_rank_r{rank}" for rank in (0, 1, 2, 4)],
        key=lambda label: float(np.median(nll[label])),
    )
    candidates = [
        "static_contact_hazard",
        "unordered_prefix",
        "linear_state",
        "vanilla_rnn",
        "full_history_gru",
        best_low_rank,
    ]
    labels = {
        "static_contact_hazard": "Static",
        "unordered_prefix": "Unordered",
        "linear_state": "Linear state",
        "vanilla_rnn": "Rate RNN",
        "full_history_gru": "GRU",
        best_low_rank: best_low_rank.replace("low_rank_r", "Low-rank r="),
    }
    palette = {
        "Static": "#8C8C8C",
        "Unordered": "#2A9D8F",
        "Linear state": "#457B9D",
        "Rate RNN": "#F4A261",
        "GRU": "#7B2CBF",
        labels[best_low_rank]: "#4D908E",
    }

    fig = plt.figure(figsize=(11.2, 6.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.15])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes

    # A — exact scientific task.
    ax_a.set_axis_off()
    boxes = [
        (0.02, 0.68, 0.27, 0.18, "Static contact\nscaffold", "#D9D9D9"),
        (0.36, 0.68, 0.27, 0.18, "Unordered\nprefix set", "#BFE3DC"),
        (0.70, 0.68, 0.27, 0.18, "Ordered rank-set\nhistory", "#BFD7EA"),
        (0.36, 0.25, 0.27, 0.18, "Event-indexed\nstate", "#E7D4F4"),
        (0.70, 0.25, 0.27, 0.18, "Next contact /\nSTOP", "#F7D6BF"),
    ]
    for x, y, w, h, text, color in boxes:
        ax_a.add_patch(
            plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#444444", lw=0.8)
        )
        ax_a.text(x + w / 2, y + h / 2, text, ha="center", va="center")
    for start, end in [
        ((0.29, 0.77), (0.36, 0.77)),
        ((0.63, 0.77), (0.70, 0.77)),
        ((0.835, 0.68), (0.50, 0.43)),
        ((0.63, 0.34), (0.70, 0.34)),
    ]:
        ax_a.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1})
    ax_a.text(
        0.02,
        0.04,
        "State resets at each group event\n(rank step ≠ real time)",
        color="#555555",
        va="bottom",
    )
    panel_label(ax_a, "A")

    # B — conditional information across architectures.
    gains = {
        labels[candidate]: (
            nll["unordered_prefix"] - nll[candidate]
        ).to_numpy(float)
        for candidate in candidates
    }
    strip_summary(
        ax_b,
        gains,
        colors=palette,
        ylabel="Held-out NLL gain vs unordered",
    )
    ax_b.set_title("Architecture audit (best low-rank shown)")
    ax_b.text(
        0.98,
        0.97,
        "Family-wise support: linear only (1/7)",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    panel_label(ax_b, "B")

    # C — matched order null.
    selected_shuffle = f"{selected}_rank_shuffle"
    order_values = {
        "Selected\nrecurrence": (
            nll[selected_shuffle] - nll[selected]
        ).to_numpy(float),
        "GRU": (
            nll["rank_shuffle_gru"] - nll["full_history_gru"]
        ).to_numpy(float),
    }
    strip_summary(
        ax_c,
        order_values,
        colors={"Selected\nrecurrence": "#B2182B", "GRU": "#7B2CBF"},
        ylabel="True-order NLL gain vs rank shuffle",
    )
    ax_c.set_title("Matched within-event order null")
    panel_label(ax_c, "C")

    # D — explicit state interventions.
    selected_cost = intervention.loc[
        intervention.model.eq("selected_ordered")
        & intervention.metric.eq("heldout_event_balanced_nll")
    ]
    ordered_interventions = [
        "reverse_prefix",
        "drop_earliest",
        "reset_after_rank_0",
        "reset_after_rank_1",
        "reset_after_rank_2",
    ]
    intervention_values = {
        {
            "reverse_prefix": "Reverse",
            "drop_earliest": "Drop first",
            "reset_after_rank_0": "Reset after 1",
            "reset_after_rank_1": "Reset after 2",
            "reset_after_rank_2": "Reset after 3",
        }[name]: selected_cost.loc[
            selected_cost.intervention.eq(name), "nll_cost_vs_ordered"
        ].to_numpy(float)
        for name in ordered_interventions
    }
    strip_summary(
        ax_d,
        intervention_values,
        colors={label: "#B2182B" for label in intervention_values},
        ylabel="Intervention NLL cost",
    )
    ax_d.set_title("Readout-relevant history interventions")
    retention = intervention_summary["readout_relevant_local_memory"][
        "readout_retention_median"
    ]["median"]
    ax_d.text(
        0.98,
        0.96,
        f"Local readout retention = {retention:.2f}\n(rank-step units)",
        transform=ax_d.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    panel_label(ax_d, "D")

    # E — conditional early-ictal morphology.
    conditional = early.loc[
        early.conditioning.eq("static_plus_unordered")
        & early.field.isin(["selected_ordered", "selected_rank_shuffle"])
        & early.eligible
    ]
    wide = conditional.pivot(
        index="subject", columns="field", values="absolute_margin"
    ).dropna()
    for _, row in wide.iterrows():
        ax_e.plot(
            [0, 1],
            [row.selected_rank_shuffle, row.selected_ordered],
            color="#B0B0B0",
            lw=0.7,
            alpha=0.7,
        )
    ax_e.scatter(
        np.zeros(len(wide)),
        wide.selected_rank_shuffle,
        color="#8C8C8C",
        s=18,
        zorder=3,
    )
    ax_e.scatter(
        np.ones(len(wide)),
        wide.selected_ordered,
        color="#B2182B",
        s=18,
        zorder=3,
    )
    ax_e.axhline(0, color="#777777", lw=0.8, ls="--")
    ax_e.set_xticks([0, 1], ["Rank shuffle", "True order"])
    ax_e.set_ylabel("|partial r| margin vs contact shuffle")
    ax_e.set_title("Early-ictal increment beyond static + unordered")
    paired_early = next(
        row
        for row in early_summary["paired_comparisons"]
        if row["conditioning"] == "static_plus_unordered"
        and row["metric"] == "absolute_rho"
        and row["right"] == "selected_rank_shuffle"
    )
    ax_e.text(
        0.98,
        0.97,
        (
            f"Δmedian = {paired_early['median']:.3f}; "
            f"P = {paired_early['wilcoxon_greater_p']:.3f}\n"
            "conditional increment not established"
        ),
        transform=ax_e.transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
        color="#555555",
    )
    panel_label(ax_e, "E")

    # F — fixed representative, never selected from target performance.
    subject = REPRESENTATIVE
    arrays = []
    unordered_arrays = []
    names = None
    for seed in SEEDS:
        with np.load(
            INTERVENTION / f"seed_{seed}" / subject / "teacher_forced_fields.npz",
            allow_pickle=False,
        ) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            if names is None:
                names = current
            arrays.append(
                np.asarray(data["selected_ordered_union_participation"], float)
            )
            unordered_arrays.append(
                np.asarray(data["unordered_prefix_union_participation"], float)
            )
    ordered = np.median(np.row_stack(arrays), axis=0)
    unordered = np.median(np.row_stack(unordered_arrays), axis=0)
    with np.load(BASELINE / f"{subject}.npz", allow_pickle=False) as data:
        raw = np.asarray(data["raw_train80_participation"], float)
        regularized = np.asarray(
            data["best_validation_regularized_participation"], float
        )
    keep, target, _ = load_target(
        subject, strict_clinical_inventory()[subject], names
    )
    covariates = np.column_stack(
        [centered_rank(raw[keep]), centered_rank(regularized[keep]), centered_rank(unordered[keep])]
    )
    design = np.column_stack([np.ones(len(keep)), covariates])
    ordered_rank = centered_rank(ordered[keep])
    ordered_residual = ordered_rank - design @ (
        np.linalg.pinv(design) @ ordered_rank
    )
    heat = np.row_stack(
        [
            zscore(regularized[keep]),
            zscore(unordered[keep]),
            zscore(ordered_residual),
            zscore(np.median(target, axis=0)),
        ]
    )
    limit = float(np.max(np.abs(heat)))
    image = ax_f.imshow(
        heat,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    ax_f.set_yticks(
        range(4),
        ["Static", "Unordered", "Ordered residual", "Early-ictal"],
    )
    ax_f.set_xticks(range(len(keep)), names[keep], rotation=65, ha="right")
    ax_f.set_title("E1146 frozen contact fields")
    colorbar = fig.colorbar(image, ax=ax_f, fraction=0.046, pad=0.03)
    colorbar.set_label("Standardized field")
    panel_label(ax_f, "F")

    stem = "fig6_ordered_history_architecture_audit"
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "PAPER_READY_CANDIDATE",
        "selected_architecture": selected,
        "best_low_rank_displayed": best_low_rank,
        "n_interictal_patients": 34,
        "n_early_ictal_patients": 16,
        "n_early_ictal_seizures": 106,
        "cross_architecture_status": "NOT_ESTABLISHED_1_OF_7",
        "early_ictal_conditional_increment_status": early_summary[
            "conditional_early_ictal_increment_status"
        ],
        "representative_subject": REPRESENTATIVE,
        "representative_selection": "pre-existing manuscript representative, not target-selected",
        "panels": {
            "A": "exact within-event task and temporal semantics",
            "B": "patient-level NLL increment over unordered prefix by architecture",
            "C": "matched within-event rank-order null",
            "D": "history intervention cost",
            "E": "conditional early-ictal morphology beyond static and unordered fields",
            "F": "representative frozen fields and readout-relevant ordered residual",
        },
        "claim_boundary": (
            "event-indexed predictive history coordinate; early-ictal target is "
            "reused and static; no biological time constant or independent "
            "manifold validation"
        ),
    }
    (output / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    (output / "README.md").write_text(
        f"""### {stem}.png

这张六面板图依次回答：模型究竟学习哪一种事件内历史、这种顺序增量是否跨架构存在、是否超过匹配的 rank-shuffle、显式删除/重排历史是否损伤预测、以及冻结后的有序残差能否在静态结构和无序前缀之外对应 clinical-onset 后 `[0,10] s` 的早期发作能量场。F 固定使用既有论文代表病例 E1146，不按本轮 target 表现挑选。

**关注点**：B/C/D 的统计单位均为患者；B 中 7 个预注册递归家族只有 linear-state 通过 family-wise inference；E 是复用 16 人 106 次发作 target 的条件性静态场检验且增量未建立，不是独立验证；整图中的 state 仅指事件 rank 索引上的现象学状态。

### {stem}.pdf

与 PNG 内容和数据版本完全相同的矢量版本，用于论文排版。

**关注点**：排版时保持六个 panel 的科学分工，不把 F 的代表病例扩写为 cohort 或机制证据。
""",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
