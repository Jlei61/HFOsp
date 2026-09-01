#!/usr/bin/env python3
"""Render the hand dual-core versus continuous-field comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_d6_natural_kmeans import (  # noqa: E402
    best_binary_alignment,
    natural_kmeans,
    normalize_event_ranks,
)


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_vs_free_field.json"
COLORS = {
    "hand_dual_core": "#df8f2d",
    "continuous_free_field": "#168b7a",
    "A": "#c43c39",
    "B": "#2675a8",
}
LABELS = {
    "hand_dual_core": "Hand dual core",
    "continuous_free_field": "Continuous free field",
}


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def _field_grid(positions, values):
    axis = np.linspace(0.0, 20.0, 120)
    xx, yy = np.meshgrid(axis, axis)
    zz = griddata(positions, values, (xx, yy), method="linear")
    nearest = griddata(positions, values, (xx, yy), method="nearest")
    return xx, yy, np.where(np.isfinite(zz), zz, nearest)


def _panel_label(ax, label, *, x=-0.12, y=1.14, color="black"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", color=color)


def _profiles(ranks, labels):
    normalized = normalize_event_ranks(ranks) * 14.0
    output = []
    for mode in (0, 1):
        values = normalized[labels == mode]
        count = np.sum(np.isfinite(values), axis=0)
        mean = np.divide(
            np.nansum(values, axis=0), count,
            out=np.full(values.shape[1], np.nan), where=count > 0,
        )
        output.append(mean)
    return np.asarray(output)


def _comparison_figure(config, summary, data, contract, outdir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.2, "axes.linewidth": 0.65,
        "xtick.major.width": 0.55, "ytick.major.width": 0.55,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False,
    })
    fig = plt.figure(figsize=(7.5, 5.15))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.1], hspace=0.43)
    top_grid = outer[0].subgridspec(
        1, 4, width_ratios=[1.0, 1.0, 1.28, 1.28], wspace=0.48,
    )
    bottom_grid = outer[1].subgridspec(
        1, 2, width_ratios=[1.02, 1.30], wspace=0.78,
    )
    axes = [fig.add_subplot(top_grid[0, index]) for index in range(4)]
    bottom = [fig.add_subplot(bottom_grid[0, 0]), fig.add_subplot(bottom_grid[0, 1])]
    names = np.asarray([row["contact_name"] for row in contract["contacts"]])
    contact_xy = np.asarray([row["sheet_xy_mm"] for row in contract["contacts"]])
    centers = np.asarray(config["manual_dual_core"]["centers_mm"], float)

    for index, arm in enumerate(("hand_dual_core", "continuous_free_field")):
        positions = data[f"{arm}_positions_E"]
        h = data[f"{arm}_h"]
        _, _, zz = _field_grid(positions, h)
        image = axes[index].imshow(
            zz, origin="lower", extent=(0, 20, 0, 20), cmap="viridis",
            vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="equal",
        )
        axes[index].plot(contact_xy[:, 0], contact_xy[:, 1], "w-", lw=0.6, alpha=0.8)
        axes[index].scatter(contact_xy[:, 0], contact_xy[:, 1], s=9,
                            facecolor="white", edgecolor="black", linewidth=0.35)
        if arm == "hand_dual_core":
            axes[index].scatter(centers[:, 0], centers[:, 1], s=28, marker="x",
                                color="#df3b2f", linewidth=1.1)
        axes[index].set_title(LABELS[arm], fontsize=7.1, fontweight="bold", pad=4)
        axes[index].set_xlabel("sheet x (mm)")
        axes[index].set_ylabel("sheet y (mm)" if index == 0 else "")
        axes[index].set_xticks([0, 10, 20]); axes[index].set_yticks([0, 10, 20])
        _panel_label(
            axes[index], chr(ord("A") + index), x=0.03, y=0.97, color="white",
        )
    colorbar_ax = inset_axes(
        axes[1], width="4%", height="42%", loc="upper right", borderpad=0.8,
    )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([0, 1])
    colorbar.ax.tick_params(labelsize=5, length=1.5)
    colorbar.set_label("h", rotation=0, labelpad=-8, y=1.12, fontsize=5.5)

    patient_profiles = _profiles(
        data["patient_heldout_ranks"], data["patient_heldout_labels"],
    )
    order = np.arange(len(names))
    for mode, ax in enumerate(axes[2:4]):
        mode_name = "A" if mode == 0 else "B"
        for arm in ("hand_dual_core", "continuous_free_field"):
            formal = data[f"{arm}_formal_kmeans"].astype(bool)
            profiles = _profiles(
                data[f"{arm}_ranks"][formal], data[f"{arm}_labels"][formal],
            )
            ax.plot(profiles[mode], order, color=COLORS[arm], lw=1.3,
                    marker="o", ms=2.2, label=LABELS[arm])
        ax.plot(patient_profiles[mode], order, color=COLORS[mode_name], lw=1.1,
                ls="--", label=f"Patient {mode_name}")
        ax.invert_yaxis()
        ax.set_yticks(order)
        ax.set_yticklabels(names if mode == 0 else [], fontsize=5.3)
        ax.set_xlim(-0.4, 14.4)
        ax.set_xlabel("mean rank (0 = first)")
        ax.set_title(f"Mode {mode_name} rank profile", fontsize=7.1,
                     fontweight="bold")
        ax.legend(fontsize=4.8, loc="upper left", handlelength=1.5)
        _panel_label(ax, chr(ord("C") + mode), x=-0.13, y=1.13)

    primary = summary["per_network"][str(
        config["search"]["comparison"]["primary_events_per_mode_per_network"]
    )]
    paired = summary["paired_comparisons"][str(
        config["search"]["comparison"]["primary_events_per_mode_per_network"]
    )]
    ax = bottom[0]
    endpoints = ["recruitment", "precedence", "profile", "event_cloud"]
    x = np.arange(len(endpoints))
    for seed in config["search"]["confirmation_network_seeds"]:
        values = []
        valid = True
        for endpoint in endpoints:
            hand = primary["hand_dual_core"][str(seed)].get(endpoint)
            free = primary["continuous_free_field"][str(seed)].get(endpoint)
            if hand is None or free is None:
                valid = False
                break
            values.append([hand, free])
        if valid:
            values = np.asarray(values)
            for endpoint_index in range(len(endpoints)):
                ax.plot(
                    [x[endpoint_index] - 0.10, x[endpoint_index] + 0.10],
                    values[endpoint_index], color="#bcbcbc", lw=0.55, alpha=0.75,
                )
                ax.scatter(x[endpoint_index] - 0.10, values[endpoint_index, 0],
                           s=8, color=COLORS["hand_dual_core"], zorder=3)
                ax.scatter(x[endpoint_index] + 0.10, values[endpoint_index, 1],
                           s=8, color=COLORS["continuous_free_field"], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Recruitment", "Precedence", "Profile", "Event cloud"],
        rotation=18, ha="right", fontsize=5.5,
    )
    ax.set_ylabel("patient-floor excess (lower is better)")
    ax.set_title("Paired network distribution error", fontsize=7.5,
                 fontweight="bold")
    ax.scatter([], [], color=COLORS["hand_dual_core"], label="Hand dual core")
    ax.scatter([], [], color=COLORS["continuous_free_field"], label="Continuous free field")
    ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    _panel_label(ax, "E", x=-0.13, y=1.13)

    ax = bottom[1]
    forest_endpoints = [
        "weak_mode_score", "recruitment", "precedence", "profile",
        "event_cloud", "mode_proportion_js", "ood_fraction_returned_readable",
    ]
    labels = ["Weak mode", "Recruitment", "Precedence", "Profile",
              "Event cloud", "Mode share", "OOD"]
    y = np.arange(len(forest_endpoints))
    for row_index, endpoint in enumerate(forest_endpoints):
        row = paired[endpoint]
        if row is None:
            continue
        value = row["mean_continuous_minus_hand"]
        low, high = row["network_bootstrap_q05"], row["network_bootstrap_q95"]
        ax.plot([low, high], [row_index, row_index], color="#444444", lw=1.1)
        ax.scatter(value, row_index, s=23, color=(
            COLORS["continuous_free_field"] if value < 0 else COLORS["hand_dual_core"]
        ), edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0.0, color="#777777", lw=0.7, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.5); ax.invert_yaxis()
    ax.set_xlabel("continuous - hand error")
    ax.set_title("Paired effect (90% network bootstrap)", fontsize=7.5,
                 fontweight="bold")
    _panel_label(ax, "F", x=-0.08, y=1.13)
    fig.suptitle(
        "Hand dual cores versus a continuous patient-fitted Node field",
        fontsize=8.2, fontweight="bold", y=0.985,
    )
    _save(fig, outdir / "dual_core_vs_free_field_explanatory_power")
    plt.close(fig)


def _kmeans_figure(config, summary, data, contract, outdir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.2,
                         "axes.linewidth": 0.65})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.7),
                             gridspec_kw={"width_ratios": [1.8, 1.0]})
    names = np.asarray([row["contact_name"] for row in contract["contacts"]])
    patient = _profiles(data["patient_heldout_ranks"], data["patient_heldout_labels"])
    metadata = {}
    for row_index, arm in enumerate(("hand_dual_core", "continuous_free_field")):
        formal = data[f"{arm}_formal_kmeans"].astype(bool)
        ranks = data[f"{arm}_ranks"][formal]
        directions = data[f"{arm}_labels"][formal]
        result = natural_kmeans(ranks, directions, random_state=20260821 + row_index)
        if result["status"] != "OK":
            raise RuntimeError(f"pooled KMeans is not evaluable for {arm}")
        valid = result["valid_event_mask"]
        ranks = ranks[valid]
        directions = directions[valid]
        alignment = best_binary_alignment(result["cluster_labels"], directions)
        mapped = alignment["mapped_labels"]
        chosen = []
        for mode in (0, 1):
            index = np.flatnonzero(mapped == mode)
            take = min(100, len(index))
            chosen.extend(index[np.linspace(0, len(index) - 1, take).round().astype(int)])
        chosen = np.asarray(chosen, int)
        display = normalize_event_ranks(ranks[chosen]) * 14.0
        axes[row_index, 0].imshow(
            display.T, aspect="auto", interpolation="nearest", cmap="viridis",
            norm=Normalize(0, 14), origin="upper",
        )
        split = int(np.sum(mapped[chosen] == 0))
        axes[row_index, 0].axvline(split - 0.5, color="white", lw=1.0)
        axes[row_index, 0].set_yticks(np.arange(len(names)))
        axes[row_index, 0].set_yticklabels(names)
        axes[row_index, 0].set_ylabel(LABELS[arm], fontweight="bold")
        axes[row_index, 0].set_xlabel("natural KMeans events (A | B)")
        axes[row_index, 0].set_title(
            f"K=2: purity {result['direction_purity']:.2f}, "
            f"balanced {result['direction_balanced_alignment']:.2f}",
            fontsize=7.2,
        )
        profiles = _profiles(ranks, mapped)
        y = np.arange(len(names))
        for mode, mode_name in ((0, "A"), (1, "B")):
            axes[row_index, 1].plot(
                profiles[mode], y, color=COLORS[mode_name], lw=1.4,
                label=f"Model {mode_name}",
            )
            axes[row_index, 1].plot(
                patient[mode], y, color=COLORS[mode_name], lw=1.0, ls="--",
                label=f"Patient {mode_name}",
            )
        axes[row_index, 1].invert_yaxis()
        axes[row_index, 1].set_yticks(y); axes[row_index, 1].set_yticklabels([])
        axes[row_index, 1].set_xlim(-0.4, 14.4)
        axes[row_index, 1].set_xlabel("mean rank (0 = first)")
        axes[row_index, 1].legend(fontsize=5.3, ncol=2, loc="best")
        metadata[arm] = {
            key: value for key, value in result.items()
            if key not in {"valid_event_mask", "cluster_labels"}
        }
    axes[0, 0].text(-0.08, 1.10, "A", transform=axes[0, 0].transAxes,
                    fontsize=10, fontweight="bold")
    axes[0, 1].text(-0.12, 1.10, "B", transform=axes[0, 1].transAxes,
                    fontsize=10, fontweight="bold")
    fig.suptitle("Unsupervised two-mode structure under matched Node budgets",
                 fontsize=9, fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, outdir / "dual_core_vs_free_field_kmeans")
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = ROOT / config["output_root"]
    summary = json.loads((root / "comparison_summary.json").read_text())
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    with np.load(root / "comparison_plot_data.npz", allow_pickle=False) as loaded:
        data = {key: np.asarray(loaded[key]) for key in loaded.files}
    outdir = root / "figures"
    _comparison_figure(config, summary, data, contract, outdir)
    kmeans = _kmeans_figure(config, summary, data, contract, outdir)
    metadata = {
        "status": "TOPIC4_DUAL_CORE_VS_FREE_FIELD_FIGURES_COMPLETE",
        "comparison_summary": str(root / "comparison_summary.json"),
        "kmeans_pooled_descriptive": kmeans,
        "claim_boundary": config["claim_boundary"],
    }
    (outdir / "figure_metadata.json").write_text(json.dumps(
        metadata, indent=2, default=lambda value: value.tolist()
        if isinstance(value, np.ndarray) else value,
    ) + "\n")
    (outdir / "README.md").write_text(
        "### dual_core_vs_free_field_explanatory_power.png\n"
        "这张图在完全相同的 Node 预算、12 张网络、背景噪声和事件读出下，直接比较手放双 core 与连续自由场。上排给出场本身及两个患者方向的 rank profile；下排以网络为独立单位展示 shaft-aware 分布误差和配对差。\n\n"
        "**关注点**：先看连续场相对手放双 core 的配对区间是否跨 0，再看改善来自 recruitment、precedence、profile 还是 event cloud，不能只看一个总分。\n\n"
        "### dual_core_vs_free_field_kmeans.png\n"
        "这张图沿用 Fig.4 的 KMeans 语法：每个场只使用 returned、双杆、patient-support 内事件，无监督分成两簇，再与冻结的患者方向比较。虚线是患者 held-out prototype，实线是模型簇 profile。\n\n"
        "**关注点**：K=2 是否稳定存在与其是否贴近患者是两件事；同时比较簇纯度、balanced alignment 和逐触点 profile。\n"
    )
    print(json.dumps({"status": metadata["status"], "figures": str(outdir)}, indent=2))


if __name__ == "__main__":
    main()
