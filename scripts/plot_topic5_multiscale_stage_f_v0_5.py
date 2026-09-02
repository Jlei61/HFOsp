#!/usr/bin/env python3
"""Nature-style target-free mechanism checkpoint figure for v0.5 Stage F."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
RED = "#b43b48"
BLUE = "#336c96"
GOLD = "#b98b44"
GREY = "#8c9397"


def add_panel_letter(axis, label: str) -> None:
    axis.text(-0.18, 1.08, label, transform=axis.transAxes, fontsize=14,
              fontweight="bold", va="top", ha="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    mechanism = pd.read_csv(out / "mechanism/MECHANISM_PER_PATIENT.csv")
    attenuation = pd.read_csv(out / "ATTENUATION_PER_PATIENT_DOSE.csv")
    mode_flow = pd.read_csv(out / "mechanism/MODE_FLOW_ATTENUATION_PER_PATIENT.csv")
    gain = pd.read_csv(out / "GAIN_ADJUSTED_PER_PATIENT.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.8,
    })
    figure, axes = plt.subplots(1, 4, figsize=(14.0, 3.15), gridspec_kw={"wspace": 0.82})

    pivot = mechanism.pivot(index="subject", columns="arm", values="median_G3")
    x = pivot["L2M_MACRO_MATCHED_RANDOM_LR"].to_numpy()
    y = pivot["L3_LOCAL_PLUS_LEARNED_LR"].to_numpy()
    lower, upper = np.nanmin(np.r_[x, y]), np.nanmax(np.r_[x, y])
    pad = max(0.02, 0.08 * (upper - lower))
    axes[0].plot([lower - pad, upper + pad], [lower - pad, upper + pad],
                 color="#b7bbbd", lw=0.9, ls="--")
    axes[0].scatter(x, y, s=27, color=RED, edgecolors="white", linewidths=0.4)
    axes[0].set(xlabel="Matched nonlocal  $G_3$", ylabel="Selected nonlocal  $G_3$",
                xlim=(lower - pad, upper + pad), ylim=(lower - pad, upper + pad))

    true_suffix = mechanism[["A_true_vs_suffix_r", "B_true_vs_suffix_r"]].mean(axis=1)
    true_l2m = mechanism[["A_true_vs_l2m_r", "B_true_vs_l2m_r"]].mean(axis=1)
    for index, values in enumerate((true_suffix, true_l2m)):
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[1].scatter(index + jitter, values, s=16, color=GREY, alpha=0.78)
        axes[1].plot([index - 0.18, index + 0.18], [np.nanmedian(values)] * 2,
                     color=(RED if index == 0 else GOLD), lw=2.2)
    axes[1].set_xticks((0, 1), ("Suffix null", "Matched\nnonlocal"))
    axes[1].set_ylabel("Endpoint similarity")

    labels = {
        "L1_ADDED": ("Nearby", BLUE), "L2M_ADDED": ("Matched", GOLD),
        "L3_ADDED": ("Selected", RED), "L3_MATCHED_LOCAL": ("Local", GREY),
    }
    for target, (label, color) in labels.items():
        group = attenuation.loc[
            (attenuation.target == target) & attenuation.inferential_eligible.astype(bool)
        ]
        n_patients = int(group.subject.nunique())
        summary = group.groupby("alpha").distal_selectivity.agg(["median", "sem"]).reset_index()
        axes[2].plot(summary.alpha, summary["median"], marker="o", ms=3.5, lw=1.5,
                     color=color, label=f"{label} (n={n_patients})")
        axes[2].fill_between(summary["alpha"], summary["median"] - summary["sem"],
                             summary["median"] + summary["sem"], color=color, alpha=0.13,
                             linewidth=0)
    axes[2].axhline(0, color="#777777", lw=0.7, ls="--")
    axes[2].set(xlabel="Edge attenuation", ylabel="Distal damage difference")
    handles, legend_labels = axes[2].get_legend_handles_labels()

    mode_patient = mode_flow.loc[
        (mode_flow.condition != "MATCHED_RANDOM") | mode_flow.random_match_eligible.astype(bool)
    ].groupby(["subject", "condition"], as_index=False).distal_selectivity.mean()
    color_by_condition = {"SAME_MODE": RED, "CROSS_MODE": BLUE, "MATCHED_RANDOM": GREY}
    order = tuple(
        condition for condition in ("SAME_MODE", "CROSS_MODE", "MATCHED_RANDOM")
        if int((mode_patient.condition == condition).sum()) > 0
    )
    for index, condition in enumerate(order):
        color = color_by_condition[condition]
        values = mode_patient.loc[mode_patient.condition == condition, "distal_selectivity"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[3].scatter(index + jitter, values, s=16, color="#a1a6a9", alpha=0.78)
        axes[3].plot([index - 0.18, index + 0.18], [np.nanmedian(values)] * 2,
                     color=color, lw=2.2)
    axes[3].axhline(0, color="#777777", lw=0.7, ls="--")
    tick_label = {"SAME_MODE": "Same", "CROSS_MODE": "Cross", "MATCHED_RANDOM": "Matched\nrandom"}
    axes[3].set_xticks(range(len(order)), [tick_label[value] for value in order])
    axes[3].set_ylabel("Mode-selective\ndistal damage")

    for label, axis in zip("ABCD", axes):
        axis.spines[["top", "right"]].set_visible(False)
        add_panel_letter(axis, label)
    figure.subplots_adjust(top=0.80)
    figure.legend(handles, legend_labels, frameon=False, fontsize=8.7,
                  handlelength=1.4, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.66, 0.995), borderaxespad=0)
    stem = out / "figures/stage_f_v0_5_target_free_mechanism"
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight",
                       facecolor="white")
    plt.close(figure)
    readme = out / "figures/README.md"
    section = (
        "\n### stage_f_v0_5_target_free_mechanism.png\n\n"
        "A 比较 macro-matched random nonlocal 与 task-selected nonlocal 的 held-out trajectory finite-horizon gain，判断结构效应是否只是整体放大差异。B 显示 L3 effective endpoint pattern 与 suffix null、L2m 的患者级相似性。C 是四类 arm-specific edge attenuation 的 distal-selective damage dose response；D 比较 TA/TB same-mode、cross-mode 与 matched-random flow bundles 的损害；无法形成合格 matched-random draw 时不绘制空类别。\n\n"
        "**关注点**：本图完全 target-free；只有 selected nonlocal 在匹配 gain/边数后呈现 distal-selective、mode-selective 损害，才能支持 functional shortcut organization。\n"
    )
    existing = readme.read_text() if readme.exists() else ""
    if "### stage_f_v0_5_target_free_mechanism.png" not in existing:
        with readme.open("a") as stream:
            stream.write(section)


if __name__ == "__main__":
    main()
