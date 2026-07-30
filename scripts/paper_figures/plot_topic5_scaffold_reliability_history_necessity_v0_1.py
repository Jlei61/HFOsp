#!/usr/bin/env python3
"""Plot the target-blind scaffold reliability and history-necessity figure."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
DATASET_COLORS = {"epilepsiae": "#6A3D9A", "yuquan": "#188A8A"}


def _bootstrap_curve(
    frame: pd.DataFrame, *, value: str, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    counts = np.sort(frame.event_count.unique())
    median = []
    lower = []
    upper = []
    rng = np.random.default_rng(int(seed))
    for count in counts:
        values = frame.loc[frame.event_count == count, value].to_numpy(float)
        draws = values[
            rng.integers(0, len(values), size=(10000, len(values)))
        ]
        medians = np.median(draws, axis=1)
        median.append(np.median(values))
        lower.append(np.quantile(medians, 0.025))
        upper.append(np.quantile(medians, 0.975))
    return counts, np.asarray(median), np.asarray(lower), np.asarray(upper)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.7,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(axis, label: str) -> None:
    axis.text(
        -0.13,
        1.075,
        label,
        transform=axis.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-root",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_scaffold_reliability_history_necessity/static_reliability_v0_1",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_scaffold_reliability_history_necessity/history_runs_v0_1",
    )
    parser.add_argument(
        "--matched-shuffle-root",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_scaffold_reliability_history_necessity/history3_rank_shuffle_runs_v0_1",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_scaffold_reliability_history_necessity/figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reliability = pd.read_csv(args.static_root / "patient_reliability.csv")
    saturation_draw = pd.read_csv(
        args.static_root / "event_count_saturation.csv"
    )
    saturation = (
        saturation_draw.groupby(
            ["subject", "dataset", "event_count"], as_index=False
        )
        .spearman_rho.median()
    )
    patient = pd.read_csv(
        args.history_root / "patient_seed_collapsed_nll.csv"
    )
    static_summary = json.loads((args.static_root / "summary.json").read_text())
    history_summary = json.loads(
        (args.history_root / "history_necessity_summary.json").read_text()
    )
    matched_patient = pd.read_csv(
        args.matched_shuffle_root / "patient_seed_collapsed_comparison.csv"
    )
    matched_summary = json.loads(
        (
            args.matched_shuffle_root
            / "history3_rank_shuffle_summary.json"
        ).read_text()
    )
    patient = patient.merge(
        matched_patient[
            [
                "subject",
                "rank_shuffle_history3_nll",
                "ordered_gain",
            ]
        ],
        on="subject",
        validate="one_to_one",
    )

    _style()
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.15))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    eligible = reliability[reliability.structured_null_eligible].copy()
    rng = np.random.default_rng(20260728)
    for row in eligible.itertuples():
        color = DATASET_COLORS[row.dataset]
        jitter = rng.normal(0, 0.018)
        values = [
            row.structured_null_median_rho,
            row.train80_heldout20_spearman_rho,
        ]
        ax_a.plot(
            [0 + jitter, 1 + jitter],
            values,
            color=color,
            alpha=0.23,
            linewidth=0.7,
            zorder=1,
        )
        ax_a.scatter(
            [0 + jitter, 1 + jitter],
            values,
            s=10,
            color=color,
            alpha=0.72,
            linewidths=0,
            zorder=2,
        )
    ax_a.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax_a.set_xticks([0, 1], ["Within-shaft\nnull", "Train80 →\nheldout20"])
    ax_a.set_ylabel("Contact-field Spearman ρ")
    ax_a.set_title("Static contact scaffold is reproducible", loc="left")
    ax_a.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgewidth=0,
                markersize=4.5,
                label=dataset.capitalize(),
            )
            for dataset, color in DATASET_COLORS.items()
        ],
        frameon=False,
        loc="upper left",
    )
    ax_a.text(
        0.03,
        0.05,
        (
            f"eligible n={len(eligible)}\n"
            f"median excess={static_summary['structured_null_excess']['median']:.2f}"
        ),
        transform=ax_a.transAxes,
        va="bottom",
    )
    _panel_label(ax_a, "A")

    for subject, frame in saturation.groupby("subject"):
        ax_b.plot(
            frame.event_count,
            frame.spearman_rho,
            color="#B5B5B5",
            alpha=0.22,
            linewidth=0.55,
        )
    counts, median, lower, upper = _bootstrap_curve(
        saturation, value="spearman_rho", seed=20260728
    )
    ax_b.fill_between(counts, lower, upper, color="#356AA0", alpha=0.18)
    ax_b.plot(counts, median, color="#245A8D", linewidth=2, marker="o", ms=3)
    for count in counts:
        n_patient = saturation.loc[
            saturation.event_count == count, "subject"
        ].nunique()
        ax_b.text(
            count,
            0.04,
            f"{n_patient}",
            ha="center",
            va="bottom",
            fontsize=6.5,
            color="#555555",
        )
    ax_b.set_xscale("log")
    ax_b.set_ylim(0, 1.03)
    ax_b.set_xlabel("Interictal events used")
    ax_b.set_ylabel("Subsample → heldout20 ρ")
    ax_b.set_title("Field estimation saturates with event count", loc="left")
    ax_b.text(
        0.03,
        0.08,
        "numbers: patients available",
        transform=ax_b.transAxes,
        fontsize=6.5,
        color="#555555",
    )
    _panel_label(ax_b, "B")

    condition_order = [
        "last_set_first_order",
        "history_1_gru",
        "history_2_gru",
        "history_3_gru",
        "rank_shuffle_history3_nll",
        "full_history_gru",
    ]
    condition_labels = [
        "First-order",
        "H1",
        "H2",
        "H3",
        "H3 shuffle",
        "All",
    ]
    centered = patient[condition_order].sub(
        patient.history_1_gru, axis=0
    )
    x = np.arange(len(condition_order))
    for index, row in patient.iterrows():
        color = DATASET_COLORS[row.dataset]
        ax_c.plot(
            x,
            centered.loc[index].to_numpy(float),
            color=color,
            alpha=0.20,
            linewidth=0.6,
        )
    medians = centered.median(axis=0).to_numpy(float)
    ax_c.plot(x, medians, color="#111111", marker="o", ms=3.5, linewidth=1.8)
    ax_c.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax_c.set_xticks(x, condition_labels, rotation=18, ha="right")
    ax_c.set_ylabel("Heldout NLL − H=1 NLL")
    ax_c.set_title("Heldout prediction across history depths", loc="left")
    ax_c.text(
        0.03,
        0.04,
        "lower is better; n=34",
        transform=ax_c.transAxes,
        fontsize=7,
    )
    _panel_label(ax_c, "C")

    contrast_names = [
        "gain_history2_over_history1",
        "gain_history3_over_history2",
        "gain_full_over_history3",
        "ordered_gain",
    ]
    contrast_labels = [
        "H2 over H1",
        "H3 over H2",
        "All over H3",
        "H3 over shuffle",
    ]
    for position, (name, label) in enumerate(
        zip(contrast_names, contrast_labels)
    ):
        values = patient[name].to_numpy(float)
        jitter = rng.normal(0, 0.045, len(values))
        colors = [DATASET_COLORS[value] for value in patient.dataset]
        ax_d.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=11,
            c=colors,
            alpha=0.58,
            linewidths=0,
        )
        statistic = (
            {
                "median": matched_summary["median_ordered_gain"],
                "ci95": matched_summary["ordered_gain_ci95"],
            }
            if name == "ordered_gain"
            else history_summary["contrasts"][name]
        )
        median_value = statistic["median"]
        ci = statistic["ci95"]
        ax_d.errorbar(
            position,
            median_value,
            yerr=[
                [median_value - ci[0]],
                [ci[1] - median_value],
            ],
            fmt="o",
            color="#111111",
            markersize=4,
            capsize=3,
            linewidth=1.4,
            zorder=4,
        )
    ax_d.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
    ax_d.set_xticks(
        np.arange(len(contrast_labels)),
        contrast_labels,
        rotation=16,
        ha="right",
    )
    ax_d.set_ylabel("Heldout NLL gain")
    ax_d.set_title("Incremental and order-matched contrasts", loc="left")
    ax_d.text(
        0.03,
        0.96,
        "positive favours first model",
        transform=ax_d.transAxes,
        fontsize=6.7,
        va="top",
        color="#555555",
    )
    _panel_label(ax_d, "D")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.margins(x=0.04)
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.115,
        top=0.94,
        wspace=0.34,
        hspace=0.40,
    )

    stem = "topic5_scaffold_reliability_history_necessity_v0_1"
    png = args.output_dir / f"{stem}.png"
    pdf = args.output_dir / f"{stem}.pdf"
    figure.savefig(png, dpi=400, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "static_root": str(args.static_root),
        "history_root": str(args.history_root),
        "matched_shuffle_root": str(args.matched_shuffle_root),
        "n_patients": int(len(patient)),
        "n_structured_null_eligible": int(len(eligible)),
        "static_summary": static_summary,
        "history_summary": history_summary,
        "matched_shuffle_summary": matched_summary,
        "ictal_target_read": False,
    }
    (args.output_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=True)
    )
    (args.output_dir / "README.md").write_text(
        f"""### {stem}.png

四个 panel 分别检验静态 contact scaffold 的 train/heldout 可重复性、估计该场所需的事件量、不同历史深度的 heldout NLL，以及有限历史增量与架构匹配的顺序打乱对照。图中所有统计均以患者为单位，并且没有读取任何发作期 target。

**关注点**：A/B 支持的是静态患者内 contact topography；C/D 只判断 ordered recurrent history 是否提供独立增量，不能解释为病理轴或生物 latent state。

### {stem}.pdf

与 PNG 内容一致的矢量版本，用于论文排版和细节核对。

**关注点**：正式引用数值以同目录 metadata 和上游 CSV/JSON 为准。
"""
    )
    print(json.dumps({"png": str(png), "pdf": str(pdf)}))


if __name__ == "__main__":
    main()
