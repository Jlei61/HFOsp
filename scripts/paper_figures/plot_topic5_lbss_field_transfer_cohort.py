#!/usr/bin/env python3
"""Paper-ready cohort summary for interictal-field recovery and ictal transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


L3 = "L3_LOCAL_PLUS_LEARNED_LR"
SHUFFLE = "C_L3_ORDER_SHUFFLED"
BLUE = "#477DA6"
RED = "#B24A52"
NULL = "#A9ADB1"
NEGATIVE = "#8FA3B2"
PAIR = "#CAD0D4"
BLACK = "#222222"


def _wilcoxon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (np.abs(values) > 1e-12)]
    if values.size == 0:
        return float("nan")
    return float(wilcoxon(values, zero_method="wilcox", method="auto").pvalue)


def _median_bar(ax: plt.Axes, x: float, values: np.ndarray) -> None:
    med = float(np.nanmedian(values))
    ax.plot([x - 0.15, x + 0.15], [med, med], color=BLACK, lw=2.0,
            solid_capstyle="butt", zorder=6)


def _significance_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _comparison_bracket(
    ax: plt.Axes, x0: float, x1: float, y: float, height: float, label: str
) -> None:
    ax.plot([x0, x0, x1, x1], [y, y + height, y + height, y],
            color=BLACK, lw=0.85, clip_on=False, zorder=7)
    ax.text((x0 + x1) / 2, y + height + 0.012, label, color=BLACK,
            ha="center", va="bottom", fontsize=10.0, fontweight="bold")


def _style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3.5)


def _plot_interictal(ax: plt.Axes, field: pd.DataFrame) -> dict:
    true = field[field["arm"] == L3].set_index("subject").sort_index()
    shuffled = field[field["arm"] == SHUFFLE].set_index("subject").reindex(true.index)
    if len(true) != 21 or shuffled.isna().all(axis=1).any():
        raise RuntimeError("expected 21 patient-matched L3/order-shuffle field rows")

    endpoints = [
        ("canonical_empirical_r", 0.0, 0.72, "Full field"),
        ("seed_removed_empirical_r", 2.0, 2.72, "Start removed"),
    ]
    rng = np.random.default_rng(20260811)
    stats = {}
    source_rows = []
    for col, x0, x1, label in endpoints:
        s = shuffled[col].to_numpy(float)
        t = true[col].to_numpy(float)
        jitter = rng.uniform(-0.045, 0.045, len(t))
        for idx, (sv, tv, j) in enumerate(zip(s, t, jitter)):
            ax.plot([x0 + j, x1 + j], [tv, sv], color=PAIR, lw=0.65,
                    alpha=0.72, zorder=1)
            source_rows.append({
                "subject": true.index[idx], "endpoint": label,
                "order_shuffle_r": sv, "true_order_r": tv,
            })
        ax.scatter(x0 + jitter, t, s=25, color=BLUE, edgecolor="white",
                   linewidth=0.4, zorder=4)
        ax.scatter(x1 + jitter, s, s=23, color=NULL, edgecolor="white",
                   linewidth=0.35, zorder=3)
        _median_bar(ax, x0, t)
        _median_bar(ax, x1, s)
        stats[label] = {
            "n": int(len(t)),
            "n_true_positive": int(np.sum(t > 0)),
            "n_true_better_than_shuffle": int(np.sum(t > s)),
            "true_median_r": float(np.median(t)),
            "shuffle_median_r": float(np.median(s)),
            "paired_median_gain": float(np.median(t - s)),
            "paired_wilcoxon_p": _wilcoxon(t - s),
        }

    ax.axhline(0, color="#777777", lw=0.75, ls="--", zorder=0)
    ax.set_xlim(-0.42, 3.14)
    ax.set_ylim(-0.68, 1.14)
    ax.set_xticks([0.0, 0.72, 2.0, 2.72],
                  ["RNN", "Shuffle", "RNN", "Shuffle"])
    ax.set_ylabel("Interictal field similarity (Spearman r)")
    ax.text(0.36, -0.17, "Full field", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9.2)
    ax.text(2.36, -0.17, "Start excluded", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=9.2)
    _comparison_bracket(
        ax, 0.0, 0.72, 0.925, 0.025,
        _significance_label(stats["Full field"]["paired_wilcoxon_p"]),
    )
    _comparison_bracket(
        ax, 2.0, 2.72, 0.925, 0.025,
        _significance_label(stats["Start removed"]["paired_wilcoxon_p"]),
    )
    ax.text(0.0, 1.075,
            f"{stats['Full field']['n_true_positive']}/{stats['Full field']['n']} > 0",
            color=BLUE, ha="center", va="bottom", fontsize=9.2)
    ax.text(2.0, 1.075,
            f"{stats['Start removed']['n_true_positive']}/{stats['Start removed']['n']} > 0",
            color=BLUE, ha="center", va="bottom", fontsize=9.2)
    _style_axis(ax)
    return {"statistics": stats, "source_rows": source_rows}


def _plot_ictal(ax: plt.Axes, early: pd.DataFrame) -> dict:
    rows = early[
        early["primary"].astype(bool)
        & (early["family"] == "intact")
        & (early["arm"] == L3)
        & (early["endpoint"] == "canonical_full")
    ].copy().sort_values("subject")
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 primary early-ictal patients, found {len(rows)}")

    null = rows["all_contact_null_median"].to_numpy(float)
    observed = rows["observed"].to_numpy(float)
    margin = observed - null
    rng = np.random.default_rng(20260812)
    jitter = rng.uniform(-0.045, 0.045, len(rows))
    for n, o, j, d in zip(null, observed, jitter, margin):
        color = RED if d > 0 else NEGATIVE
        ax.plot([j, 0.72 + j], [o, n], color=color, lw=0.8, alpha=0.72, zorder=1)
    ax.scatter(jitter, observed, s=28,
               color=np.where(margin > 0, RED, NEGATIVE),
               edgecolor="white", linewidth=0.45, zorder=4)
    ax.scatter(0.72 + jitter, null, s=25, color=NULL, edgecolor="white",
               linewidth=0.4, zorder=3)
    significant = rows["all_contact_p"].to_numpy(float) < 0.05
    _median_bar(ax, 0.0, observed)
    _median_bar(ax, 0.72, null)

    p = _wilcoxon(margin)
    n_above = int(np.sum(margin > 0))
    ax.set_xlim(-0.40, 1.12)
    ax.set_ylim(0.39, 1.14)
    ax.set_xticks([0.0, 0.72], ["Frozen RNN\nfield", "Channel-shuffle\nnull"])
    ax.set_ylabel("Early-ictal field similarity (max |r|)")
    _comparison_bracket(ax, 0.0, 0.72, 1.045, 0.022, _significance_label(p))
    ax.text(0.72, 1.005, f"{n_above}/{len(rows)} above null", color=RED,
            ha="center", va="top", fontsize=9.2)
    _style_axis(ax)

    source = rows[[
        "subject", "n_seizures", "n_contacts_min", "observed",
        "all_contact_null_median", "all_contact_margin", "all_contact_p",
        "within_shaft_margin",
    ]].copy()
    return {
        "statistics": {
            "n_patients": int(len(rows)),
            "n_raw_observed_positive": int(np.sum(observed > 0)),
            "raw_observed_median": float(np.median(observed)),
            "n_above_all_contact_null": n_above,
            "null_relative_margin_median": float(np.median(margin)),
            "paired_wilcoxon_p": p,
            "n_individual_all_contact_p_lt_0_05": int(np.sum(significant)),
            "n_above_within_shaft_null": int(np.sum(rows["within_shaft_margin"] > 0)),
        },
        "source_rows": source.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path,
                        default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("results/paper-ready-figure/fig6_lbss_field_transfer/figures"))
    args = parser.parse_args()
    result_root = args.result_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    field = pd.read_csv(result_root / "model_field_patient_metrics.csv")
    early = pd.read_csv(result_root / "early_ictal" / "early_ictal_per_patient_condition.csv")

    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelsize": 10.8, "xtick.labelsize": 9.3, "ytick.labelsize": 9.3,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.50, 3.75),
                             gridspec_kw={"width_ratios": [1.34, 0.86]})
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.25, top=0.94, wspace=0.46)
    interictal = _plot_interictal(axes[0], field)
    ictal = _plot_ictal(axes[1], early)
    axes[0].text(-0.17, 1.08, "A", transform=axes[0].transAxes, fontsize=13,
                 fontweight="bold", va="top", ha="left")
    axes[1].text(-0.22, 1.08, "B", transform=axes[1].transAxes, fontsize=13,
                 fontweight="bold", va="top", ha="left")

    stem = output_root / "topic5_lbss_interictal_to_ictal_cohort"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, facecolor="white")
    plt.close(fig)

    pd.DataFrame(interictal["source_rows"]).to_csv(
        output_root / "source_interictal_field_recovery.csv", index=False)
    pd.DataFrame(ictal["source_rows"]).to_csv(
        output_root / "source_early_ictal_transfer.csv", index=False)
    metadata = {
        "contract": "topic5_lbss_interictal_to_ictal_cohort_v0_1",
        "interictal": interictal["statistics"],
        "early_ictal": ictal["statistics"],
        "early_ictal_target": "clinical onset 0-10 s, 1-150 Hz broadband energy",
        "primary_null": "synchronized all-contact label shuffle",
        "interpretation": (
            "Interictal field recovery is cohort-wide, whereas raw cross-state "
            "correspondence is not equivalent to null-relative transfer."
        ),
    }
    (output_root / "FIGURE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    (output_root / "README.md").write_text(
        "### topic5_lbss_interictal_to_ictal_cohort.png\n\n"
        "Panel A 以患者为单位比较真实顺序 LBSS-RNN 与 order-shuffle 模型生成场和经验间期场的相关；"
        "full field 与去除第一 rank 的 field 分开显示。Panel B 将同一患者的 frozen RNN field "
        "与 synchronized all-contact shuffle null 配对。括号上的星号表示患者级 paired Wilcoxon："
        "* P<0.05，** P<0.01，*** P<0.001，ns 表示 P≥0.05；精确 P 值保存在 metadata。\n\n"
        "**关注点**：间期 field 在 cohort 中稳定恢复，但 early-ictal raw 相似度不能等同为所有患者都超过空间零假设。\n"
    )
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
