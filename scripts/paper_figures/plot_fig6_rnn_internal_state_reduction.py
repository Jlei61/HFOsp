#!/usr/bin/env python3
"""Paper-ready Figure 6 candidate for the GRU internal-state reduction."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    ordinary_model_fields,
    strict_clinical_inventory,
)


BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
OUT = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_rnn_internal_state_reduction/figures"
)
FULL = "#B4543C"
SHUFFLE = "#4C78A8"
GRAY = "#8A8A8A"
DARK = "#202020"


def panel_label(axis, label: str) -> None:
    axis.text(
        -0.14,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
    )


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sd = values.std()
    return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)


def draw_schematic(axis) -> None:
    axis.axis("off")
    boxes = [
        (0.02, 0.57, 0.25, 0.20, "Rank-set\nsequence"),
        (0.37, 0.57, 0.25, 0.20, "GRU\nstate"),
        (0.72, 0.57, 0.25, 0.20, "Next\ncontact"),
        (0.37, 0.13, 0.25, 0.20, "Frozen\nPCs"),
        (0.72, 0.13, 0.25, 0.20, "Ictal\nfield"),
    ]
    for x, y, w, h, text in boxes:
        color = "#F3E3DE" if "GRU" in text or "low-D" in text else "#EEF2F6"
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#777777",
            facecolor=color,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=6.0,
        )
    arrows = [
        ((0.27, 0.68), (0.37, 0.68)),
        ((0.62, 0.68), (0.72, 0.68)),
        ((0.50, 0.57), (0.50, 0.34)),
        ((0.62, 0.23), (0.72, 0.23)),
    ]
    for start, end in arrows:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color="#555555",
                transform=axis.transAxes,
            )
        )
    axis.text(
        0.02,
        0.02,
        "Interictal directions were target-blind\nIctal residual read-back is exploratory",
        transform=axis.transAxes,
        fontsize=5.7,
        color="#555555",
    )


def spectrum_payload():
    rows = []
    for path in sorted(
        (BASE / "interictal/cells").glob("seed_*/**/hidden_states.npz")
    ):
        subject = path.parent.name
        seed = path.parent.parent.name
        with np.load(path, allow_pickle=False) as data:
            for control in ("full_history_gru", "rank_shuffle_gru"):
                eig = np.asarray(data[f"{control}_pca_eigenvalues"], float)
                cumulative = np.cumsum(eig) / eig.sum()
                for index, value in enumerate(cumulative, start=1):
                    rows.append(
                        {
                            "subject": subject,
                            "seed": seed,
                            "control": control,
                            "component": index,
                            "cumulative": value,
                        }
                    )
    frame = pd.DataFrame(rows)
    collapsed = (
        frame.groupby(["subject", "control", "component"], as_index=False)
        .cumulative.median()
    )
    return collapsed


def draw_spectrum(axis, inventory: dict) -> None:
    frame = spectrum_payload()
    for control, color, label in (
        ("full_history_gru", FULL, "Ordered GRU"),
        ("rank_shuffle_gru", SHUFFLE, "Rank-shuffled GRU"),
    ):
        group = frame.loc[frame.control == control]
        wide = group.pivot(index="subject", columns="component", values="cumulative")
        x = wide.columns.to_numpy(int)
        values = wide.to_numpy(float)
        axis.plot(x, np.nanmedian(values, axis=0), color=color, lw=1.8, label=label)
        axis.fill_between(
            x,
            np.nanquantile(values, 0.25, axis=0),
            np.nanquantile(values, 0.75, axis=0),
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    axis.axhline(0.9, color="#B0B0B0", lw=0.8, ls="--")
    axis.set_xlim(1, 12)
    axis.set_ylim(0.45, 1.01)
    axis.set_xticks([1, 2, 4, 8, 12])
    axis.set_xlabel("Hidden components")
    axis.set_ylabel("Cumulative variance")
    axis.legend(frameon=False, fontsize=6.5, loc="lower right")
    full_rank = inventory["cohort_metrics"][
        "full_history_gru__effective_rank"
    ]["median"]
    shuffle_rank = inventory["cohort_metrics"][
        "rank_shuffle_gru__effective_rank"
    ]["median"]
    axis.text(
        0.98,
        0.49,
        f"effective rank\n{full_rank:.2f} vs {shuffle_rank:.2f}",
        ha="right",
        va="bottom",
        transform=axis.transAxes,
        fontsize=6.4,
        color="#555555",
    )


def jitter_points(axis, values, x, color, width=0.06, seed=0, **kwargs):
    rng = np.random.default_rng(seed)
    axis.scatter(
        x + rng.uniform(-width, width, len(values)),
        values,
        s=10,
        color=color,
        alpha=0.72,
        linewidth=0,
        **kwargs,
    )


def draw_stability(axis) -> None:
    frame = pd.read_csv(BASE / "interictal_stability_metrics.csv")
    chosen = frame.loc[
        (frame.k == -1)
        & frame.comparison.isin(
            ("cross_seed_raw_cka", "cross_seed_residual_cka")
        )
    ]
    collapsed = (
        chosen.groupby(["subject", "control", "comparison"], as_index=False)
        .value.median()
    )
    positions = [0, 1, 2, 3]
    specs = [
        ("full_history_gru", "cross_seed_raw_cka", FULL, "Ordered\nraw"),
        ("full_history_gru", "cross_seed_residual_cka", FULL, "Ordered\nresidual"),
        ("rank_shuffle_gru", "cross_seed_raw_cka", SHUFFLE, "Shuffled\nraw"),
        (
            "rank_shuffle_gru",
            "cross_seed_residual_cka",
            SHUFFLE,
            "Shuffled\nresidual",
        ),
    ]
    for x, (control, comparison, color, _) in zip(positions, specs):
        values = collapsed.loc[
            (collapsed.control == control)
            & (collapsed.comparison == comparison),
            "value",
        ].to_numpy(float)
        jitter_points(axis, values, x, color, seed=100 + x)
        axis.plot([x - 0.17, x + 0.17], [np.median(values)] * 2, color=DARK, lw=1.3)
    axis.set_xticks(positions, [spec[-1] for spec in specs], fontsize=6.2)
    axis.set_ylim(0.35, 1.02)
    axis.set_ylabel("Cross-seed CKA")
    axis.axhline(0.5, color="#CCCCCC", lw=0.7, ls=":")


def draw_order(axis, sensitivity: dict) -> None:
    frame = pd.read_csv(BASE / "interictal_order_perturbation_metrics.csv")
    frame = frame.loc[
        (frame.order_perturbation == "shuffle")
        & (frame.prefix_bin == "all")
        & (frame.metric == "nll_loss")
    ]
    collapsed = (
        frame.groupby(["subject", "control"], as_index=False).value.median()
    )
    wide = collapsed.pivot(index="subject", columns="control", values="value")
    for _, row in wide.iterrows():
        axis.plot(
            [0, 1],
            [row.full_history_gru, row.rank_shuffle_gru],
            color="#D6D6D6",
            lw=0.55,
            zorder=0,
        )
    jitter_points(
        axis,
        wide.full_history_gru.to_numpy(float),
        0,
        FULL,
        seed=201,
    )
    jitter_points(
        axis,
        wide.rank_shuffle_gru.to_numpy(float),
        1,
        SHUFFLE,
        seed=202,
    )
    axis.plot(
        [-0.18, 0.18],
        [wide.full_history_gru.median()] * 2,
        color=DARK,
        lw=1.4,
    )
    axis.plot(
        [0.82, 1.18],
        [wide.rank_shuffle_gru.median()] * 2,
        color=DARK,
        lw=1.4,
    )
    axis.axhline(0, color="#777777", lw=0.8)
    axis.set_xticks([0, 1], ["Ordered GRU", "Rank-shuffled\nGRU"], fontsize=6.4)
    axis.set_ylabel("Order-shuffle NLL penalty")
    metric = sensitivity["metrics"][
        "order_shuffle__nll_loss__full_minus_rank_shuffle"
    ]
    axis.text(
        0.98,
        0.96,
        f"paired Δ={metric['median']:.3f}\n"
        f"P={metric['wilcoxon_p']:.1e}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
    )


def representative_payload(subject="epilepsiae_1146"):
    inventory = strict_clinical_inventory()
    names, _ = ordinary_model_fields(subject)
    keep, target, _ = load_target(subject, inventory[subject], names)
    joined_names = names[keep]
    target_field = np.median(
        np.row_stack([rankdata(row) for row in target]), axis=0
    )
    frame = pd.read_csv(BASE / "interictal_direction_contact_fields.csv")
    frame = frame.loc[
        (frame.subject == subject)
        & (frame.direction_type == "pca")
        & (frame.direction_index == 1)
        & np.isclose(frame.amplitude_sd, 0.5)
        & (frame.event_half == "all")
    ]
    fields = {}
    for control in ("full_history_gru", "rank_shuffle_gru"):
        seed_fields = []
        for _, seed in frame.loc[frame.control == control].groupby("seed_dir"):
            seed = seed.sort_values("contact_index")
            field = seed.probability_contrast.to_numpy(float)
            participation = seed.train80_participation.to_numpy(float)
            design = np.column_stack([np.ones(len(field)), participation])
            coefficient, *_ = np.linalg.lstsq(design, field, rcond=None)
            seed_fields.append((field - design @ coefficient)[keep])
        fields[control] = np.median(np.row_stack(seed_fields), axis=0)
    order = np.argsort(target_field)
    return (
        joined_names[order],
        zscore(target_field[order]),
        zscore(fields["full_history_gru"][order]),
        zscore(fields["rank_shuffle_gru"][order]),
    )


def draw_representative(axis) -> None:
    names, target, full, shuffled = representative_payload()
    x = np.arange(len(names))
    axis.plot(x, target, color=DARK, lw=1.7, label="Early-ictal energy")
    axis.plot(x, full, color=FULL, lw=1.4, marker="o", ms=2.5, label="Ordered PC1")
    axis.plot(
        x,
        shuffled,
        color=SHUFFLE,
        lw=1.2,
        marker="o",
        ms=2.2,
        label="Shuffled PC1",
    )
    axis.axhline(0, color="#CCCCCC", lw=0.7)
    axis.set_xlim(-0.4, len(x) - 0.6)
    axis.set_xticks(x, [str(index + 1) for index in x], fontsize=5.7)
    axis.set_xlabel("Contacts ordered by ictal energy")
    axis.set_ylabel("Standardized field")
    axis.legend(frameon=False, fontsize=6.0, loc="upper left")
    axis.text(
        0.98,
        0.04,
        "Illustrative; contacts target-ordered",
        transform=axis.transAxes,
        ha="right",
        fontsize=5.8,
        color="#666666",
    )


def draw_transfer(axis) -> None:
    frame = pd.read_csv(BASE / "early_ictal_fixed_readback_patient_metrics.csv")
    frame = frame.loc[
        (frame.seizure_split == "all")
        &
        (frame.field == "probability_contrast_residual_participation")
        & frame.model.str.contains("_pca")
    ]
    positions = []
    labels = []
    values_by_position = []
    colors = []
    xpos = 0
    for metric, metric_label in (
        ("all_contact_margin", "All-contact"),
        ("within_shaft_margin", "Within-shaft"),
    ):
        for direction_index in (1, 2):
            full = f"internal_full_history_gru_pca{direction_index}"
            shuffled = f"internal_rank_shuffle_gru_pca{direction_index}"
            wide = frame.pivot(index="subject", columns="model", values=metric)
            difference = (wide[full] - wide[shuffled]).dropna().to_numpy(float)
            positions.append(xpos)
            short = "All" if metric == "all_contact_margin" else "Shaft"
            labels.append(f"{short}\nPC{direction_index}")
            values_by_position.append(difference)
            colors.append(FULL if metric == "all_contact_margin" else "#A66A57")
            xpos += 1
        xpos += 0.45
    for index, (x, values, color) in enumerate(
        zip(positions, values_by_position, colors)
    ):
        jitter_points(axis, values, x, color, seed=300 + index)
        axis.plot([x - 0.17, x + 0.17], [np.median(values)] * 2, color=DARK, lw=1.35)
    axis.axhline(0, color="#666666", lw=0.8)
    axis.set_xticks(positions, labels, fontsize=5.4)
    axis.set_ylabel("Ordered − shuffled\ntransfer margin")
    comparisons = pd.read_csv(
        BASE / "early_ictal_internal_full_vs_rank_shuffle.csv"
    )
    chosen = comparisons.loc[
        (comparisons.field == "probability_contrast_residual_participation")
        & comparisons.direction_type.eq("pca")
        & comparisons.metric.isin(("all_contact_margin", "within_shaft_margin"))
    ]
    text = []
    for metric in ("all_contact_margin", "within_shaft_margin"):
        group = chosen.loc[chosen.metric == metric].sort_values("direction_index")
        q = group.direction_family_bh_fdr_q.to_numpy(float)
        text.append(
            ("all" if metric == "all_contact_margin" else "shaft")
            + f" q={q[0]:.3f}/{q[1]:.3f}"
        )
    axis.text(
        0.98,
        0.98,
        "exploratory; target reused\n" + "\n".join(text),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
    )


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=6.4, length=2.5, width=0.7)
    axis.xaxis.label.set_size(7)
    axis.yaxis.label.set_size(7)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((BASE / "INTERICTAL_SUMMARY.json").read_text())
    sensitivity = json.loads(
        (BASE / "INTERICTAL_SENSITIVITY_SUMMARY.json").read_text()
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.15))
    draw_schematic(axes[0, 0])
    draw_spectrum(axes[0, 1], inventory)
    draw_stability(axes[0, 2])
    draw_order(axes[1, 0], sensitivity)
    draw_representative(axes[1, 1])
    draw_transfer(axes[1, 2])
    for axis, label in zip(axes.flat, "ABCDEF"):
        panel_label(axis, label)
        if axis is not axes[0, 0]:
            style_axis(axis)
    fig.subplots_adjust(
        left=0.07, right=0.99, bottom=0.09, top=0.96, wspace=0.43, hspace=0.48
    )
    png = OUT / "fig6_rnn_internal_state_reduction.png"
    pdf = OUT / "fig6_rnn_internal_state_reduction.pdf"
    fig.savefig(png, dpi=400, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    metadata = {
        "contract": "topic5_rnn_internal_state_reduction_v0_1",
        "status": "supplementary_exploratory_candidate",
        "n_interictal_subjects": 34,
        "n_seeds": 3,
        "strict_early_ictal": {
            "n_patients": 16,
            "n_seizures": 106,
            "time_reference": "clinical_onset",
            "window_sec": [0, 10],
            "band_hz": [1, 150],
        },
        "representative_subject_private": "epilepsiae_1146",
        "representative_selection": "canonical patient, not selected by transfer effect",
        "field_aggregation": "event-first",
        "primary_null": "coherent all-contact permutation",
        "anatomical_sensitivity": "coherent within-shaft permutation",
        "key_metrics": {
            "ordered_gru_effective_rank": inventory["cohort_metrics"][
                "full_history_gru__effective_rank"
            ],
            "rank_shuffle_effective_rank": inventory["cohort_metrics"][
                "rank_shuffle_gru__effective_rank"
            ],
            "order_shuffle_full_minus_rank_nll": sensitivity["metrics"][
                "order_shuffle__nll_loss__full_minus_rank_shuffle"
            ],
            "transfer_comparison_csv": (
                "results/topic5_rnn_internal_state_reduction/"
                "early_ictal_internal_full_vs_rank_shuffle.csv"
            ),
        },
        "target_opening_context": (
            "exploratory mechanism decomposition on the same strict target "
            "already opened in v2.5; participation residualization was added "
            "after target inspection and is not an independent confirmation"
        ),
        "primary_manuscript_claim": (
            "interictal ordered-history sensitivity only; the early-ictal "
            "internal-state read-back is exploratory and should not be the "
            "primary static-scaffold evidence"
        ),
        "sources": [
            "results/topic5_rnn_internal_state_reduction/INTERICTAL_SUMMARY.json",
            "results/topic5_rnn_internal_state_reduction/INTERICTAL_SENSITIVITY_SUMMARY.json",
            "results/topic5_rnn_internal_state_reduction/INTERICTAL_EVENTFIRST_FIELD_SUMMARY.json",
            "results/topic5_rnn_internal_state_reduction/EARLY_ICTAL_READBACK_SUMMARY.json",
        ],
    }
    (OUT / "fig6_rnn_internal_state_reduction_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    readme = """### fig6_rnn_internal_state_reduction.png

这张六联图依次展示自监督任务、hidden state 的低维性、跨 seed 稳定性、prefix 顺序扰动、代表患者的 contact field，以及严格 clinical-onset early-ictal read-back。图中把普通有序 GRU 与 rank-shuffle GRU 分开，并在 transfer 前去除 interictal participation 的线性成分。

**关注点**：有序 GRU 对历史顺序的依赖明显强于 rank-shuffle control，这是本图较可靠的间期结果。PC1/PC2 的 event-first residual contact fields 在 all-contact 和 within-shaft null 下显示额外 early-ictal 对应，但 residualization 是查看同一 target 后补充的机制拆解，只能作为探索性结果；本图因此定位为补充图候选，不作为静态 scaffold 的主证据。

### fig6_rnn_internal_state_reduction.pdf

PDF 与 PNG 使用相同数据、布局和统计，只用于论文排版和矢量输出。

**关注点**：精确效应量、置信区间、校正后 q 值和分母记录在 metadata 及主结果 CSV/JSON 中；图中 q 值不是独立确认性 P 值。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
