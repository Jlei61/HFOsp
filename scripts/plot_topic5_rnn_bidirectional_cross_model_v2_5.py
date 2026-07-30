#!/usr/bin/env python3
"""Render v2.5 RNN bidirectionality and cross-model static-transfer figures."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    FIELDS,
    RANK_ROOT,
    SEED_DIRS,
    TARGET_ROOT,
    V24,
    load_target,
    strict_clinical_inventory,
)


BASE = ROOT / "results/topic5_rnn_bidirectional_cross_model_audit_v2_5"
FIGURES = BASE / "figures"
EXAMPLES = ("epilepsiae_1084", "epilepsiae_958", "epilepsiae_1096")
MODEL_COLORS = {
    "empirical_rank_distribution": "#343434",
    "full_history_gru": "#B2182B",
    "static_contact_hazard": "#9E9E9E",
    "rank_shuffle_gru": "#D8A03D",
    "structured_full": "#2166AC",
    "structured_no_history": "#67A9CF",
    "structured_local_isotropic": "#92C5DE",
    "structured_node_only": "#BDBDBD",
}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def median_conditional_hist(subject: str, control: str, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    names = None
    arrays = []
    for seed_dir in SEED_DIRS:
        frame = pd.read_csv(
            RANK_ROOT / seed_dir / subject / "contact_rank_distributions.csv"
        )
        subset = frame.loc[frame.control == control].sort_values("contact_index")
        current = subset.contact_name.astype(str).to_numpy()
        if names is None:
            names = current
        elif not np.array_equal(names, current):
            raise RuntimeError(f"{subject}: contact order drifted")
        arrays.append(
            np.column_stack(
                [
                    subset[f"{prefix}_rank_bin_{index}"].to_numpy(float)
                    for index in range(10)
                ]
            )
        )
    return names, np.median(np.stack(arrays), axis=0)


def median_structured_hist(subject: str) -> tuple[np.ndarray, np.ndarray]:
    names = None
    arrays = []
    for seed in (17, 29, 43):
        with np.load(
            V24 / "representations/per_seed" / f"{subject}_seed{seed}.npz",
            allow_pickle=False,
        ) as data:
            current = np.asarray(data["contact_names"]).astype(str)
            joint = np.asarray(data["full_fixed_axis"], dtype=np.float64)
        if names is None:
            names = current
        elif not np.array_equal(names, current):
            raise RuntimeError(f"{subject}: structured contact order drifted")
        participation = 1.0 - joint[:, :1]
        conditional = np.divide(
            joint[:, 1:],
            participation,
            out=np.zeros_like(joint[:, 1:]),
            where=participation > 0,
        )
        arrays.append(conditional)
    return names, np.median(np.stack(arrays), axis=0)


def representative_figure() -> None:
    inventory = strict_clinical_inventory()
    configure()
    fig, axes = plt.subplots(
        len(EXAMPLES),
        4,
        figsize=(8.4, 6.8),
        gridspec_kw={"width_ratios": [1, 1, 1, 0.28], "wspace": 0.16, "hspace": 0.32},
    )
    image = None
    energy_image = None
    for row, subject in enumerate(EXAMPLES):
        names, empirical = median_conditional_hist(
            subject, "empirical_rank_distribution", "observed"
        )
        gru_names, gru = median_conditional_hist(
            subject, "full_history_gru", "predicted"
        )
        structured_names, structured = median_structured_hist(subject)
        if not (
            np.array_equal(names, gru_names)
            and np.array_equal(names, structured_names)
        ):
            raise RuntimeError(f"{subject}: representative contact mismatch")
        keep, target, _ = load_target(subject, inventory[subject], names)
        names = names[keep]
        empirical = empirical[keep]
        gru = gru[keep]
        structured = structured[keep]
        energy = np.nanmedian(target, axis=0)
        energy = (rankdata(energy) - 1.0) / max(len(energy) - 1.0, 1.0)
        centers = (np.arange(10) + 0.5) / 10.0
        order = np.argsort(empirical @ centers)
        names = names[order]
        empirical = empirical[order]
        gru = gru[order]
        structured = structured[order]
        energy = energy[order]
        for column, (title, values) in enumerate(
            (
                ("Empirical interictal", empirical),
                ("Full-history GRU", gru),
                ("Structured RNN", structured),
            )
        ):
            ax = axes[row, column]
            image = ax.imshow(
                values,
                aspect="auto",
                interpolation="nearest",
                cmap="viridis",
                vmin=0,
                vmax=max(
                    np.quantile(empirical, 0.98),
                    np.quantile(gru, 0.98),
                    np.quantile(structured, 0.98),
                ),
            )
            if row == 0:
                ax.set_title(title, fontweight="bold")
            ax.set_yticks(np.arange(len(names)))
            ax.set_yticklabels(names if column == 0 else [])
            ax.set_xticks([0, 4.5, 9])
            ax.set_xticklabels(["early", "mid", "late"] if row == len(EXAMPLES) - 1 else [])
            if column == 0:
                ax.set_ylabel(subject.replace("epilepsiae_", "E"))
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
        ax_energy = axes[row, 3]
        energy_image = ax_energy.imshow(
            energy[:, None],
            aspect="auto",
            interpolation="nearest",
            cmap="Blues",
            norm=colors.Normalize(vmin=0.0, vmax=1.0),
        )
        if row == 0:
            ax_energy.set_title("Early-ictal\nenergy", fontweight="bold")
        ax_energy.set_xticks([])
        ax_energy.set_yticks([])
        for spine in ax_energy.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    fig.text(0.47, 0.02, "Normalized within-event rank", ha="center")
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.63])
    fig.colorbar(image, cax=cbar_ax, label="Conditional rank probability")
    energy_cbar_ax = fig.add_axes([0.965, 0.18, 0.012, 0.63])
    fig.colorbar(
        energy_image,
        cax=energy_cbar_ax,
        label="Within-patient energy rank",
        ticks=[0, 1],
    )
    fig.suptitle(
        "Contact-level rank distributions and clinical-onset static energy",
        y=0.995,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.07, top=0.94)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"representative_rank_distribution_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def cohort_figure() -> None:
    bidirectional = pd.read_csv(BASE / "bidirectional_patient_metrics.csv")
    static = pd.read_csv(BASE / "static_transfer_patient_metrics.csv")
    configure()
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 2.8), gridspec_kw={"wspace": 0.48})

    ax = axes[0]
    x = np.arange(len(bidirectional))
    negative = bidirectional.negative_source_inward_displacement.to_numpy(float)
    positive = bidirectional.positive_source_inward_displacement.to_numpy(float)
    for index in x:
        ax.plot([0, 1], [negative[index], positive[index]], color="#BDBDBD", lw=0.6)
    ax.scatter(np.zeros_like(x), negative, s=12, color="#2166AC", zorder=3)
    ax.scatter(np.ones_like(x), positive, s=12, color="#B2182B", zorder=3)
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xticks([0, 1], ["negative-side\nsource", "positive-side\nsource"])
    ax.set_ylabel("Inward displacement")
    ax.set_title("Both source sides", fontweight="bold")

    ax = axes[1]
    values = bidirectional.selected_axis_displacement_margin.to_numpy(float)
    ax.scatter(np.zeros(len(values)), values, s=16, color="#4D4D4D", alpha=0.8)
    ax.boxplot(
        [values],
        positions=[0],
        widths=0.35,
        showfliers=False,
        boxprops={"color": "#222222"},
        medianprops={"color": "#B2182B", "lw": 1.4},
    )
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xticks([0], ["selected −\ncandidate median"])
    ax.set_ylabel("Direction-specific margin")
    ax.set_title("Axis specificity", fontweight="bold")

    common_models = (
        "empirical_rank_distribution",
        "full_history_gru",
        "structured_full",
    )
    common = static.loc[static.model.isin(common_models)].pivot(
        index="subject", columns="model", values="observed_max_abs_rho"
    ).dropna()
    ax = axes[2]
    model_x = np.arange(3)
    for _, row in common.iterrows():
        ax.plot(
            model_x,
            [row[model] for model in common_models],
            color="#C7C7C7",
            lw=0.7,
            zorder=1,
        )
    for index, model in enumerate(common_models):
        values = common[model].to_numpy(float)
        ax.scatter(
            np.full(len(values), index),
            values,
            s=16,
            color=MODEL_COLORS[model],
            zorder=3,
        )
        ax.plot(
            [index - 0.18, index + 0.18],
            [np.median(values), np.median(values)],
            color="#111111",
            lw=1.5,
            zorder=4,
        )
    ax.set_xticks(model_x, ["Empirical", "Full-history\nGRU", "Structured\nRNN"])
    ax.set_ylabel("Absolute static similarity")
    ax.set_ylim(0, 1)
    ax.set_title("Clinical-onset transfer", fontweight="bold")

    display_models = (
        "static_contact_hazard",
        "rank_shuffle_gru",
        "full_history_gru",
        "structured_node_only",
        "structured_full",
    )
    labels = (
        "Static",
        "Rank-\nshuffle",
        "Full\nGRU",
        "Node\nonly",
        "Structured\nRNN",
    )
    ax = axes[3]
    for index, (model, label) in enumerate(zip(display_models, labels)):
        values = static.loc[
            static.model == model, "all_contact_margin"
        ].to_numpy(float)
        jitter = np.linspace(-0.08, 0.08, len(values))
        ax.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=12,
            color=MODEL_COLORS[model],
            alpha=0.85,
        )
        ax.plot(
            [index - 0.2, index + 0.2],
            [np.median(values), np.median(values)],
            color="#111111",
            lw=1.4,
        )
    ax.axhline(0, color="#777777", ls="--", lw=0.8)
    ax.set_xticks(
        np.arange(len(labels)),
        labels,
        rotation=22,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("Observed − all-contact null")
    ax.set_title("Static scaffold above null", fontweight="bold")
    fig.suptitle(
        "RNN audit: bilateral geometry and source-free early-ictal readout",
        y=1.02,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.23, top=0.84)
    for suffix in ("png", "pdf"):
        fig.savefig(
            FIGURES / f"cohort_bidirectional_static_transfer_summary.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def write_readme() -> None:
    text = """### cohort_bidirectional_static_transfer_summary.png

四个 panel 依次展示同一冻结方向在两侧 source 事件中的位移、该方向相对候选方向的特异性、共同 11 人中经验分布/普通 GRU/结构化 RNN 的发作早期静态相似度，以及主要模型和对照相对全通道打乱 null 的 margin。两侧位移为正不等于方向特异；第二个 panel 专门检验这一替代解释。

**关注点**：普通和结构化模型都保留静态 contact scaffold，但结构化轴和 source 项没有显示独立增量。

### representative_rank_distribution_comparison.png

三名预先固定患者逐行展示真实 held-out 间期 rank distribution、普通 full-history GRU、结构化 RNN，以及 strict clinical-onset 后 `[0,10] s` 的 1–150 Hz 静态能量。每行 contact ordering 只由真实间期平均 rank 冻结，不读取发作 target 排序。

**关注点**：比较模型是否复现每个 contact 的完整 rank 分布，并观察这种分布与发作早期静态能量的形态关系；该图不是逐触点 replay 证据。
"""
    (FIGURES / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    representative_figure()
    cohort_figure()
    write_readme()


if __name__ == "__main__":
    main()
