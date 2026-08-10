"""Plot the exploratory rev9 Node-to-Edge alpha calibration diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


DEFAULT_SUMMARY = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "node_edge_calibration/edge_alpha_selection_summary.json")
DEFAULT_OUT = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9/"
    "node_edge_calibration/figures")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()


def _panel_label(axis, label):
    axis.text(-0.14, 1.08, label, transform=axis.transAxes,
              fontsize=10, fontweight="bold", va="top")


def _set_alpha_ticks(axis, alpha):
    axis.set_xticks(alpha)
    axis.set_xticklabels([f"{value:g}" for value in alpha], rotation=45,
                         ha="right", fontsize=6.5)


def _scatter_match(axis, rows, feature_index, title, label):
    colors = {"field_component": "#D55E00", "matched_off_field": "#0072B2"}
    labels = {"field_component": "field component", "matched_off_field": "off-field control"}
    values = []
    for role in ("field_component", "matched_off_field"):
        selected = [row for row in rows if row["role"] == role]
        x = np.asarray([row["node_scalars"][feature_index] for row in selected], float)
        y = np.asarray([row["edge_scalars"][feature_index] for row in selected], float)
        valid = np.isfinite(x) & np.isfinite(y)
        axis.scatter(x[valid], y[valid], s=22, alpha=0.82, color=colors[role],
                     edgecolor="white", linewidth=0.35, label=labels[role])
        values.extend(x[valid].tolist())
        values.extend(y[valid].tolist())
    values = np.asarray(values, float)
    lower, upper = float(values.min()), float(values.max())
    pad = max(0.05 * (upper - lower), 1e-4)
    axis.plot([lower - pad, upper + pad], [lower - pad, upper + pad],
              color="#777777", linewidth=0.8, linestyle="--")
    axis.set_xlim(lower - pad, upper + pad)
    axis.set_ylim(lower - pad, upper + pad)
    all_x = np.asarray([row["node_scalars"][feature_index] for row in rows], float)
    all_y = np.asarray([row["edge_scalars"][feature_index] for row in rows], float)
    valid = np.isfinite(all_x) & np.isfinite(all_y)
    rho = spearmanr(all_x[valid], all_y[valid]).statistic if valid.sum() >= 3 else np.nan
    axis.text(0.04, 0.94, f"rho={rho:.2f} | n={valid.sum()}",
              transform=axis.transAxes, va="top", fontsize=7.5)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlabel(f"Node {label}")
    axis.set_ylabel(f"Edge {label}")
    axis.axhline(0.0, color="#BBBBBB", linewidth=0.5, zorder=0)
    axis.axvline(0.0, color="#BBBBBB", linewidth=0.5, zorder=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    rows = sorted(summary["summaries"], key=lambda row: row["alpha"])
    alpha = np.asarray([row["alpha"] for row in rows], float)
    selected_alpha = float(summary["selection"]["alpha_star"])
    selected_pairs = [row for row in summary["pair_rows"]
                      if row["alpha"] == selected_alpha
                      and row["paired_eligible"] and row["pair_loss"] is not None]

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.5), constrained_layout=True)

    axis = axes[0, 0]
    j_cal = np.asarray([row["J_cal"] for row in rows])
    axis.plot(alpha, j_cal, color="#222222", marker="o", linewidth=1.4)
    chosen = int(np.flatnonzero(np.isclose(alpha, selected_alpha))[0])
    axis.scatter(alpha[chosen], j_cal[chosen], s=72, facecolor="#009E73",
                 edgecolor="black", linewidth=0.7, zorder=4)
    axis.set(xlabel="edge strength alpha", ylabel="J_cal")
    _set_alpha_ticks(axis, alpha)
    axis.set_title("response-matched reference", loc="left", fontweight="bold")
    _panel_label(axis, "a")

    axis = axes[0, 1]
    components = np.column_stack((
        [row["response_loss_median"] for row in rows],
        [row["missing_pair_penalty"] for row in rows],
        [0.5 * row["off_field_loss_median"] for row in rows],
        [0.25 * row["baseline_shift_median"] for row in rows],
    ))
    component_labels = ("paired response", "missing pairs", "off-field", "sham baseline")
    component_colors = ("#4C78A8", "#B8B8B8", "#F2CF5B", "#E45756")
    bottom = np.zeros(len(alpha))
    for values, name, color in zip(components.T, component_labels, component_colors):
        axis.bar(alpha, values, width=0.13, bottom=bottom, color=color,
                 edgecolor="white", linewidth=0.35, label=name)
        bottom += values
    axis.set(xlabel="edge strength alpha", ylabel="objective contribution")
    _set_alpha_ticks(axis, alpha)
    axis.set_title("objective decomposition", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(axis, "b")

    axis = axes[0, 2]
    paired = np.asarray([row["n_paired"] for row in rows])
    denominator = int(rows[0]["n_node_eligible"])
    axis.bar(alpha, paired, width=0.13, color="#56B4E9", edgecolor="white")
    axis.axhline(denominator, color="#555555", linewidth=0.8, linestyle="--")
    axis.set_ylim(0, denominator + 3)
    axis.set(xlabel="edge strength alpha", ylabel=f"paired units (of {denominator})")
    _set_alpha_ticks(axis, alpha)
    axis.set_title("primary-window coverage", loc="left", fontweight="bold")
    _panel_label(axis, "c")

    axis = axes[1, 0]
    ratio_min = np.asarray([row["structure"]["edge_ratio"]["min"] for row in rows])
    ratio_p01 = np.asarray([row["structure"]["edge_ratio"]["p01"] for row in rows])
    ratio_p99 = np.asarray([row["structure"]["edge_ratio"]["p99"] for row in rows])
    ratio_max = np.asarray([row["structure"]["edge_ratio"]["max"] for row in rows])
    axis.fill_between(alpha, ratio_min, ratio_max, color="#BBBBBB", alpha=0.25,
                      label="min-max")
    axis.fill_between(alpha, ratio_p01, ratio_p99, color="#CC79A7", alpha=0.5,
                      label="1st-99th percentile")
    axis.axhline(1.0, color="#333333", linewidth=0.8)
    axis.axhline(0.25, color="#999999", linewidth=0.6, linestyle=":")
    axis.axhline(4.0, color="#999999", linewidth=0.6, linestyle=":")
    axis.set_yscale("log")
    axis.set(xlabel="edge strength alpha", ylabel="new / original edge weight")
    _set_alpha_ticks(axis, alpha)
    axis.set_title("structural redistribution", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left")
    _panel_label(axis, "d")

    _scatter_match(axes[1, 1], selected_pairs, 0, "source response", "slope")
    _panel_label(axes[1, 1], "e")
    _scatter_match(axes[1, 2], selected_pairs, 1, "downstream response", "slope")
    axes[1, 2].legend(frameon=False, loc="lower right")
    _panel_label(axes[1, 2], "f")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "rev9_edge_alpha_calibration"
    fig.savefig(stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    metadata = dict(
        status="REV9_EDGE_ALPHA_CALIBRATION_DIAGNOSTIC",
        scientific_role=(
            "Exploratory local-response calibration diagnostic; alpha_star is "
            "not Node-Edge equivalence or patient validation"),
        alpha_star=selected_alpha,
        selection=summary["selection"],
        n_node_eligible=denominator,
        n_paired_at_alpha_star=len(selected_pairs),
        source=dict(path=args.summary, sha256=_sha256(args.summary)),
        git_commit=_git_commit(),
    )
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    (out_dir / "README.md").write_text(
        "### rev9_edge_alpha_calibration.png\n\n"
        "这是一张 rev9 Node-Edge 局部响应校准诊断图。上排依次显示总目标函数、目标函数分解和逐窗配对覆盖；下排显示 E->E 权重重分配范围，以及冻结 alpha 下 source/downstream slope 的 Node-Edge 配对。\n\n"
        "绿色点只表示用于后续四臂比较的 response-matched reference，不表示两种机制等效；精确数值和全部 pair rows 见同级 selection summary JSON。\n\n"
        "**关注点**：alpha 的优势来自哪一项、paired coverage 是否下降，以及散点是否真正贴近 identity line。\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
