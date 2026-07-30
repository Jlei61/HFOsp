#!/usr/bin/env python
"""Render the contract-corrected Figure 6 Fit1/Fit2/RNN intermediate sheet."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig6_state_conditioned_predictor import (
    BLUE,
    GREY,
    LIGHT,
    RED,
    TEAL,
    panel_label,
    subject_aliases,
)

FIT_ROOT = ROOT / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150"
ANALYSIS = ROOT / "results/topic5_state_conditioned_predictor/fit2_rnn_final_analysis"
OUT = (
    ROOT
    / "results/paper-ready-figure/fig6_state_conditioned_predictor/"
    "fit2_rnn/figures"
)


def load_json(path: Path):
    return json.loads(path.read_text())


def draw_contract(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 3)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((2, 1.15), 34, 0.62, color=TEAL, alpha=0.18, lw=0))
    ax.text(19, 1.46, "12 h interictal prefix", ha="center", va="center")
    ax.add_patch(plt.Rectangle((49, 1.15), 25, 0.62, color=BLUE, alpha=0.18, lw=0))
    ax.text(61.5, 1.46, "event history\n−65 to −5 min", ha="center", va="center")
    ax.add_patch(
        plt.Rectangle(
            (74, 1.15), 7, 0.62, facecolor="none", edgecolor=GREY, hatch="//", lw=0.7
        )
    )
    ax.text(73.0, 2.08, "5-min cutoff", ha="right", color=GREY)
    ax.axvline(81, ymin=0.25, ymax=0.75, color=RED, lw=1.2)
    ax.text(83.0, 2.40, "clinical onset", color=RED, ha="center", va="top")
    ax.add_patch(plt.Rectangle((81, 1.15), 13, 0.62, color=RED, alpha=0.15, lw=0))
    ax.text(87.5, 1.46, "BB 1–150\n0–10 s", ha="center", va="center")
    ax.annotate(
        "prefix-only scaffold",
        xy=(36, 1.45),
        xytext=(38, 2.45),
        arrowprops=dict(arrowstyle="-|>", lw=0.7, color=GREY),
        ha="center",
        color=GREY,
    )
    ax.annotate(
        "frozen recurrent core",
        xy=(62, 1.15),
        xytext=(62, 0.28),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=BLUE),
        ha="center",
        color=BLUE,
    )
    ax.text(
        2,
        2.72,
        "Leakage-controlled state-conditioned readout",
        fontweight="bold",
        fontsize=9,
    )


def draw_scaffold_pair(ax, mode: str, title: str, aliases: dict[str, str]):
    table = pd.read_csv(
        FIT_ROOT / mode / f"fig6_{mode}_clinical_onset_scaffold_subject.csv"
    )
    table = table[table.group_id.astype(str) == "strict_broadband"].copy()
    table = table.sort_values("margin").reset_index(drop=True)
    y = np.arange(len(table))
    for i, row in enumerate(table.itertuples()):
        ax.plot(
            [row.channel_null_median, row.data],
            [i, i],
            color=LIGHT,
            lw=0.9,
            zorder=1,
        )
    ax.scatter(
        table.channel_null_median,
        y,
        color=GREY,
        s=15,
        label="channel-shuffle",
        zorder=2,
    )
    ax.scatter(table.data, y, color=RED, s=20, label="observed", zorder=3)
    ax.set_yticks(y, [aliases[s] for s in table.subject], fontsize=5.4)
    ax.set_xlabel("scaffold expression, maxAB")
    summary = load_json(
        FIT_ROOT / mode / f"fig6_{mode}_clinical_onset_scaffold_summary.json"
    )
    strict = next(
        row
        for row in summary["cohort_statistics"]
        if row["group_id"] == "strict_broadband"
    )
    ax.text(
        0.02,
        0.98,
        (
            f"{strict['n_data_gt_null']}/{strict['n_subjects']} patients\n"
            f"median Δ={strict['margin_median']:.3f}, "
            f"P={strict['wilcoxon_one_sided_data_gt_null_p']:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=6.2,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=1.4),
    )
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")


def draw_attrition(ax):
    labels = ["parent\ncohort", "prefix\nscaffold", "strict BB\ntarget", "history\neligible"]
    patients = np.array([17, 13, 13, 9])
    seizures = np.array([167, 115, 71, 11])
    x = np.arange(len(labels))
    bars = ax.bar(x, patients, color=[GREY, TEAL, RED, BLUE], alpha=0.82, width=0.62)
    for bar, n_event in zip(bars, seizures):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.45,
            f"{int(bar.get_height())} P\n{n_event} sz",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("patients")
    ax.set_ylim(0, 20)
    ax.set_title("Prospective-style attrition", loc="left", fontsize=9, fontweight="bold")


def draw_dynamic_increment(
    ax,
    subject: pd.DataFrame,
    aliases: dict[str, str],
    verdict: dict,
):
    table = subject.sort_values("rnn_increment_over_history_baseline").reset_index(
        drop=True
    )
    y = np.arange(len(table))
    for i, row in enumerate(table.itertuples()):
        ax.plot(
            [min(row.static_mae, row.history_baseline_mae, row.rnn_mae),
             max(row.static_mae, row.history_baseline_mae, row.rnn_mae)],
            [i, i],
            color=LIGHT,
            lw=1,
        )
    ax.scatter(
        table.static_mae,
        y,
        color=TEAL,
        marker="^",
        s=20,
        label="static scaffold",
    )
    ax.scatter(
        table.history_baseline_mae,
        y,
        color=GREY,
        s=20,
        label="nested history baseline",
    )
    ax.scatter(table.rnn_mae, y, color=BLUE, s=22, label="frozen-core RNN")
    ax.set_yticks(y, [aliases[s] for s in table.subject], fontsize=5.6)
    ax.set_xlabel("held-out absolute error")
    ax.set_xlim(left=0, right=max(0.19, float(table[
        ["static_mae", "history_baseline_mae", "rnn_mae"]
    ].to_numpy().max()) * 1.62))
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    ci_low, ci_high = verdict["patient_bootstrap_95ci"]
    pairing = verdict["history_pairing_null"]
    ax.text(
        0.98,
        0.98,
        (
            "MAE improvement (history − RNN)\n"
            f"median={verdict['rnn_minus_history_baseline_mae_improvement_median']:.3f}; "
            f"95% CI [{ci_low:.3f}, {ci_high:.3f}]\n"
            f"{verdict['n_subjects_rnn_better_history_baseline']}/"
            f"{verdict['n_subjects']} better; pairing "
            f"P={pairing['empirical_p_observed_lower']:.2f} "
            f"(n={pairing['n_eligible_subjects']})\n"
            "Gate 2: FAIL"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        color=GREY,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=1.5),
    )
    ax.set_title(
        "Held-out dynamic readout", loc="left", fontsize=9, fontweight="bold"
    )


def draw_pretext(ax, pretext: pd.DataFrame, verdict: dict):
    chosen = pretext.sort_values(["rank", "subject", "seed"]).reset_index(drop=True)
    ax.axhline(0, color=GREY, lw=0.7, ls="--")
    jitter = np.zeros(len(chosen), float)
    for rank, indices in chosen.groupby("rank").groups.items():
        positions = np.asarray(list(indices), int)
        jitter[positions] = np.linspace(-0.14, 0.14, len(positions))
    scatter = ax.scatter(
        chosen["rank"].to_numpy(float) + jitter[: len(chosen)],
        chosen.shuffle_minus_true,
        c=chosen["rank"],
        cmap="viridis",
        vmin=0,
        vmax=4,
        s=28,
        edgecolors="white",
        linewidths=0.4,
        alpha=0.82,
    )
    med = float(verdict["event_order_shuffle_minus_true_pretext_loss_median"])
    ax.axhline(med, color=BLUE, lw=1.2)
    n_rank0 = int(np.sum(pretext["rank"].to_numpy(int) == 0))
    ax.text(
        0.98,
        0.04,
        (
            f"order: {verdict['n_subjects_true_order_better']}/"
            f"{verdict['n_subjects']} patients, median={med:.3f}\n"
            f"one-SE selected rank 0 in {n_rank0}/{len(pretext)} folds"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.2,
    )
    ax.set(
        xlabel="selected recurrent rank",
        ylabel="shuffle-trained loss − true-order loss",
        xticks=np.arange(5),
    )
    ax.set_title(
        "Temporal-order control", loc="left", fontsize=9, fontweight="bold"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    analysis = args.analysis if args.analysis.is_absolute() else ROOT / args.analysis
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    subject = pd.read_csv(analysis / "subject_level_metrics.csv")
    pretext = pd.read_csv(analysis / "selected_rank_pretext_order_control.csv")
    verdict = load_json(analysis / "gate2_verdict.json")
    fit1_subject = pd.read_csv(
        FIT_ROOT / "fit1/fig6_fit1_clinical_onset_scaffold_subject.csv"
    )
    fit1_subject = fit1_subject[
        fit1_subject.group_id.astype(str) == "strict_broadband"
    ]
    aliases = subject_aliases(fit1_subject.subject.astype(str))
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2, 6.25), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[0.78, 1.24, 1.05])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]
    draw_contract(axes[0])
    draw_attrition(axes[1])
    draw_scaffold_pair(
        axes[2], "fit1", "Full-record scaffold benchmark", aliases
    )
    draw_scaffold_pair(
        axes[3], "fit2", "Prefix-only scaffold retention", aliases
    )
    draw_dynamic_increment(axes[4], subject, aliases, verdict)
    draw_pretext(axes[5], pretext, verdict)
    for label, ax in zip("ABCDEF", axes):
        panel_label(ax, label)
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)
    stem = out / "fig6_fit2_rnn_intermediate"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "contract": verdict["contract"],
        "status": "intermediate_fit1_fit2_and_frozen_core_loso",
        "gate2": verdict,
        "panels": {
            "A": "clinical-onset BB150 leakage-controlled task",
            "B": "cohort and event attrition",
            "C": "accepted full-record strict-BB scaffold benchmark",
            "D": "prefix-only strict-BB scaffold retention",
            "E": "subject-first frozen-core RNN vs nested history baseline",
            "F": "selected-rank true-order vs shuffle-trained pretext loss",
        },
    }
    stem.with_name(stem.name + "_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    history_ci = verdict["patient_bootstrap_95ci"]
    pairing_p = verdict["history_pairing_null"]["empirical_p_observed_lower"]
    n_rank0 = int(np.sum(pretext["rank"].to_numpy(int) == 0))
    readme = (
        "# Figure 6 contract-corrected intermediate result\n\n"
        "### fig6_fit2_rnn_intermediate.png / .pdf / .svg\n\n"
        "A 固定 clinical-onset、BB 1–150、prefix-only 与发作前历史窗口；B 展示从"
        " accepted parent cohort 到可训练 history-target pair 的 attrition；C 为 full-record"
        " 静态基准复现；D 为只替换 prefix scaffold 后的保留结果；E 比较 frozen-core RNN 与"
        " outer-training-only 选择的简单历史 baseline；F 检查真实事件顺序是否优于独立训练的"
        " order-shuffle core，并显示 one-SE 选择的 recurrent rank。\n\n"
        "**关注点**：C/D 回答静态 scaffold 是否成立；E/F 才回答发作前事件历史是否提供"
        f"患者级动态增量。本轮 Gate 2 未通过：RNN 相对最佳简单历史 baseline 的 95% CI "
        f"[{history_ci[0]:.3f}, {history_ci[1]:.3f}] 跨 0，exact history-pairing "
        f"P={pairing_p:.2f}；one-SE 在 {len(pretext)} 个 outer×seed 中 {n_rank0} 次选择 rank 0。"
        "seed 只在患者内汇总，不作为统计样本。\n"
    )
    (out / "README.md").write_text(readme)
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
