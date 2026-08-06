#!/usr/bin/env python3
"""Paper-facing audit figure for causal history to early-ictal transfer v0.2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEVELOPMENT = "epilepsiae_1146"
COLORS = {
    "static": "#6B7280",
    "unordered": "#4C78A8",
    "ewma": "#E08B3E",
    "history": "#8B5FBF",
    "control": "#A7A9AC",
    "target": "#B54A4A",
}


def _jitter(n: int, seed: int, width: float = 0.10) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _strip(
    ax: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    zero: bool = True,
) -> None:
    for index, (value, color) in enumerate(zip(values, colors)):
        finite = np.asarray(value, float)
        finite = finite[np.isfinite(finite)]
        x = np.full(len(finite), index, float) + _jitter(len(finite), 20260802 + index)
        ax.scatter(x, finite, s=22, color=color, alpha=0.68, edgecolor="white", linewidth=0.35)
        if len(finite):
            median = float(np.median(finite))
            ax.plot([index - 0.23, index + 0.23], [median, median], color="#202124", lw=2.2)
    if zero:
        ax.axhline(0, color="#9AA0A6", lw=0.9, ls="--", zorder=0)
    ax.set_xticks(range(len(labels)), labels)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelrotation=22)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=13, fontweight="bold")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("results/topic5_history_rnn_direct_early_ictal_transfer_v0_2"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / "DIRECT_TRANSFER_SUMMARY.json").read_text())
    patient = pd.read_csv(root / "direct_transfer_patient_metrics.csv")
    patient = patient.loc[patient.subject != DEVELOPMENT].copy()
    pairing = pd.read_csv(root / "state_seizure_pairing_metrics.csv")
    pairing = pairing.loc[pairing.subject != DEVELOPMENT].copy()
    residual = pd.read_csv(root / "seizure_specific_residual_metrics.csv")
    residual = residual.loc[residual.subject != DEVELOPMENT].copy()
    headroom = pd.read_csv(root / "target_headroom_metrics.csv")
    headroom = headroom.loc[
        (headroom.subject != DEVELOPMENT) & (headroom.n_seizures >= 2)
    ].copy()
    channel_null = pd.read_csv(
        root / "direct_transfer_channel_null_patient_metrics.csv"
    )
    channel_null = channel_null.loc[
        channel_null.subject != DEVELOPMENT
    ].copy()

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.2), constrained_layout=True)

    # A: exact scientific task and nesting.
    ax = axes[0, 0]
    ax.axis("off")
    boxes = [
        (0.00, 0.62, 0.30, 0.23, "Frozen static\ninterictal scaffold", COLORS["static"]),
        (0.35, 0.62, 0.30, 0.23, "Causal unordered\nhistory", COLORS["unordered"]),
        (0.70, 0.62, 0.30, 0.23, "Time-aware state\nEWMA or RNN", COLORS["history"]),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                                   facecolor=color, alpha=0.13, edgecolor=color, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                transform=ax.transAxes, color="#202124", fontsize=7.6)
    ax.text(0.325, 0.735, "+", transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="#5F6368")
    ax.text(0.675, 0.735, "+", transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="#5F6368")
    ax.add_patch(plt.Rectangle((0.20, 0.16), 0.60, 0.23, transform=ax.transAxes,
                               facecolor=COLORS["target"], alpha=0.12,
                               edgecolor=COLORS["target"], lw=1.3))
    ax.text(0.50, 0.275, "Subsequent early-ictal contact energy field\nclinical onset [0,10] s · 1–150 Hz",
            ha="center", va="center", transform=ax.transAxes)
    for start_x in (0.155, 0.495, 0.835):
        ax.annotate("", xy=(0.50, 0.40), xytext=(start_x, 0.60),
                    xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="#5F6368", lw=0.9))
    denominator = summary["contact_denominator"]
    ax.text(
        0.50,
        0.02,
        "History sealed at onset − 10 min; target-patient LOSO\n"
        f"Field = {denominator['min_contacts_per_patient']}–"
        f"{denominator['max_contacts_per_patient']} scaffold contacts per patient\n"
        f"(median {denominator['median_contacts_per_patient']:.0f}), not the full montage",
        ha="center", transform=ax.transAxes, color="#5F6368", fontsize=7.4)
    ax.set_title("Direct cross-state prediction task")
    _panel_label(ax, "A")

    # B: target reliability/headroom.
    ax = axes[0, 1]
    _strip(
        ax,
        [headroom.pairwise_target_rho.to_numpy(),
         headroom.leave_one_seizure_out_mean_oracle_rho.to_numpy()],
        ["Seizure–seizure", "LOSO patient\nmean oracle"],
        [COLORS["control"], COLORS["target"]],
    )
    ax.set_ylabel("Contact-wise Spearman ρ")
    ax.set_title(f"Early-ictal target headroom (n={len(headroom)})")
    _panel_label(ax, "B")

    # C: absolute held-out spatial predictability against the paper's null.
    ax = axes[0, 2]
    absolute_models = ["M0", "M1", "E2", "EM", "R2"]
    _strip(
        ax,
        [
            channel_null.loc[
                channel_null.model == model,
                "margin_vs_channel_null_median",
            ].to_numpy()
            for model in absolute_models
        ],
        ["Static", "Unordered", "EWMA-2h", "Multi-horizon", "RNN"],
        [COLORS["static"], COLORS["unordered"], COLORS["ewma"], COLORS["ewma"], COLORS["history"]],
    )
    ax.set_ylabel("Observed ρ − all-contact null median")
    ax.set_title("Absolute spatial readout vs channel shuffle")
    _panel_label(ax, "C")

    # D: added value above the static+unordered baseline.
    ax = axes[1, 0]
    increments = [
        "rho_increment_E0p5_minus_M1", "rho_increment_E2_minus_M1",
        "rho_increment_E6_minus_M1", "rho_increment_EM_minus_M1",
        "rho_increment_R2_minus_M1",
    ]
    _strip(
        ax,
        [patient[column].to_numpy() for column in increments],
        ["EWMA\n0.5h", "EWMA\n2h", "EWMA\n6h", "Multi-\nhorizon", "RNN"],
        [COLORS["ewma"]] * 4 + [COLORS["history"]],
    )
    ax.set_ylabel("Δρ over static + unordered")
    rnn_p = summary["primary_R2_minus_M1"]["one_sided_wilcoxon_p"]
    ax.set_title(
        f"Increment over static + unordered (RNN P={rnn_p:.3g})", fontsize=9
    )
    _panel_label(ax, "D")

    # E: chronology/time-placement controls.
    ax = axes[1, 1]
    controls = [
        "rho_true_R2_minus_order_shuffle",
        "rho_true_R2_minus_zero_state",
        "rho_true_E2_minus_time_shuffle",
        "rho_true_EM_minus_time_shuffle",
    ]
    _strip(
        ax,
        [patient[column].to_numpy() for column in controls],
        ["RNN order", "RNN state", "EWMA-2h\ntime slot", "Multi-horizon\ntime slot"],
        [COLORS["history"], COLORS["history"], COLORS["ewma"], COLORS["ewma"]],
    )
    ax.set_ylabel("True history − matched shuffle (Δρ)")
    ax.set_title("Does temporal placement matter?")
    _panel_label(ax, "E")

    # F: within-patient seizure specificity and residual field.
    ax = axes[1, 2]
    pairing_values = [
        pairing.loc[pairing.model == model, "correct_minus_wrong"].to_numpy()
        for model in ("E2", "EM", "R2")
    ]
    _strip(
        ax,
        pairing_values,
        ["EWMA-2h", "Multi-horizon", "RNN"],
        [COLORS["ewma"], COLORS["ewma"], COLORS["history"]],
    )
    ax.set_ylabel("Correct − wrong seizure pairing (Δρ)")
    residual_n = residual.subject.nunique()
    residual_text = ", ".join(
        f"{model}={residual.loc[residual.model == model, 'median_residual_rho'].median():.2f}"
        for model in ("E2", "EM", "R2")
    )
    ax.text(0.04, 0.03, f"Seizure-specific residual (n={residual_n}): {residual_text}",
            transform=ax.transAxes, fontsize=7.0, color="#5F6368", va="bottom")
    ax.set_title(f"Within-patient seizure specificity (n={pairing.subject.nunique()})")
    _panel_label(ax, "F")

    fig.suptitle(
        "Direct test: causal interictal history → subsequent early-ictal spatial field",
        fontsize=14,
        fontweight="bold",
    )
    out = root / "figures"
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "topic5_history_to_early_ictal_direct_transfer_v0_2"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    readme = f"""### {stem.name}.png

六联图直接检验 causal interictal history 能否预测随后 clinical onset `[0,10] s`、`1–150 Hz` 的 early-ictal contact energy field。A 定义嵌套预测合同；B 给出同一患者不同发作之间的 target 可重复性上限；C 使用每患者 5000 次 all-contact channel shuffle 报告绝对空间预测 margin；D 检验时间状态相对 static+unordered baseline 的增量；E 用匹配置换判断真实时间放置是否必要；F 将正确 history–seizure 配对和 seizure-specific residual 作为次级分析。

主统计均排除开发病例 `{DEVELOPMENT}`。HistoryRNN checkpoint 使用 `{summary['history_checkpoint_cycles']}` 个 target-blind history training cycles；训练预算不由 early-ictal target 选择。图中相关性只支持预测关联，不建立间期活动因果塑造发作网络的结论。

**空间分母**：每个患者的"能量场"只覆盖 `{denominator['min_contacts_per_patient']}–{denominator['max_contacts_per_patient']}` 个触点（中位 `{denominator['median_contacts_per_patient']:.0f}`），即冻结间期骨架与 rank 数据集按名字精确取交后的集合，不是完整 SEEG 蒙太奇。所有 Spearman、channel shuffle 与患者级对比都建立在这个分母上，因此阳性与阴性都是粗分辨率的：6 个触点的患者只有 720 种不同的通道排列。

**E 面板对照口径**：顺序对照把 last event 之前的**整段** causal prefix 重新分配到既有时间槽（不是只打乱最近的一小段），因此 E 的 RNN order 列是对预注册对照的完整实现。

**关注点**：Primary 需要 C 的 RNN 场超过 all-contact null，同时 D 的 RNN 增量超过 M1、E 的真实顺序优于 matched shuffle。F 只回答更严格的 seizure-specific 问题，阴性不反向否定患者级 early-ictal field transfer。
"""
    (out / "README.md").write_text(readme, encoding="utf-8")
    print(stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
