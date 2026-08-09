"""Render the Fig4B KMeans companion for the frozen rev8 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


ROOT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
PROFILES = f"{ROOT}/joint_confirmation_rev8/final_event_profiles.npz"
CONFIRM = f"{ROOT}/joint_confirmation_rev8/final_confirmation.json"
OUT = "results/paper-ready-figure/fig4_data_driven_core_field_rev8/figures"
MODE_COLORS = ("#c43c39", "#277da1")
VERDICT_LABELS = {
    "RIGID_TEMPLATE_MATCH_NOT_BEATEN": "fails rigid-mode benchmark",
}


def _sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _verdict_label(verdict):
    return VERDICT_LABELS.get(str(verdict), str(verdict).lower().replace("_", " "))


def _normalized_rank_matrix(ranks):
    ranks = np.asarray(ranks, float)
    output = np.full_like(ranks, np.nan)
    for column in range(ranks.shape[1]):
        finite = np.isfinite(ranks[:, column])
        if finite.sum() < 2:
            continue
        values = ranks[finite, column]
        output[finite, column] = (
            values - values.min()) / max(float(np.ptp(values)), 1.0)
    return output


def _event_order(curves, labels):
    # Stable within-mode order by axial slope makes the heatmap inspectable
    # without using patient labels to change KMeans assignments.
    x = np.arange(curves.shape[1], dtype=float)
    slopes = np.asarray([np.polyfit(x, curve, 1)[0] for curve in curves])
    return np.lexsort((slopes, labels))


def _plot_heatmap(ax, norm_ranks, labels, names, order):
    shown = np.ma.masked_invalid(norm_ranks[:, order])
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#d7d7d7")
    image = ax.imshow(shown, aspect="auto", origin="upper",
                      interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    ordered_labels = labels[order]
    split = int(np.sum(ordered_labels == 0))
    if 0 < split < len(order):
        ax.axvline(split - 0.5, color="#b22222", lw=1.4)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("final unseen-network events")
    ax.set_ylabel("contact along shared axis")
    ax.set_title("clustered event heatmap", fontsize=11.5,
                 fontweight="bold", pad=7)
    ax.text(split / 2 if split else 0, -1.15, f"mode A  n={split}",
            color=MODE_COLORS[0], ha="center", va="bottom", fontsize=9)
    ax.text(split + (len(order) - split) / 2, -1.15,
            f"mode B  n={len(order) - split}", color=MODE_COLORS[1],
            ha="center", va="bottom", fontsize=9)
    return image


def _plot_rank_distribution(ax, norm_ranks, names):
    positions, values = [], []
    for row in range(len(names)):
        finite = norm_ranks[row, np.isfinite(norm_ranks[row])]
        if len(finite):
            positions.append(row); values.append(finite)
    if values:
        violin = ax.violinplot(
            values, positions=positions, vert=False, widths=0.78,
            showmeans=False, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#6f6f6f"); body.set_edgecolor("none"); body.set_alpha(0.45)
        violin["cmedians"].set_color("#222222"); violin["cmedians"].set_linewidth(1.1)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("within-event rank")
    ax.set_title("rank distribution", fontsize=11.5,
                 fontweight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)


def _profile_stats(norm_ranks, selected):
    values = np.asarray(norm_ranks, float)[:, np.asarray(selected, bool)]
    mean = np.full(values.shape[0], np.nan)
    std = np.full(values.shape[0], np.nan)
    for row in range(values.shape[0]):
        finite = values[row, np.isfinite(values[row])]
        if len(finite):
            mean[row] = finite.mean()
            std[row] = finite.std()
    return mean, std


def _plot_profiles(ax, norm_ranks, labels, names):
    y = np.arange(len(names))
    for mode in (0, 1):
        selected = labels == mode
        mean, std = _profile_stats(norm_ranks, selected)
        finite = np.isfinite(mean)
        ax.fill_betweenx(y[finite], (mean - std)[finite], (mean + std)[finite],
                         color=MODE_COLORS[mode], alpha=0.15, lw=0)
        ax.plot(mean[finite], y[finite], "-o", color=MODE_COLORS[mode],
                lw=2.0, ms=4.2, label=f"mode {chr(65 + mode)} (n={selected.sum()})")
    ax.set_xlim(-0.08, 1.08); ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_yticks(np.arange(len(names))); ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("mean normalized rank")
    ax.set_title("cluster rank profile", fontsize=11.5,
                 fontweight="bold", pad=7)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def _plot_matrix(ax, matrix, valid):
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    for row in range(2):
        for column in range(2):
            value = float(matrix[row, column])
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if abs(value) > 0.58 else "#222222")
    ax.set_xticks((0, 1)); ax.set_xticklabels(("patient A", "patient B"), fontsize=9)
    ax.set_yticks((0, 1)); ax.set_yticklabels(("model A", "model B"), fontsize=9)
    ax.set_title("model vs patient", fontsize=11.5,
                 fontweight="bold", pad=7)
    color = "#2a9d55" if valid else "#c23b33"
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(color); spine.set_linewidth(2.2)
    ax.text(0.5, -0.24, "valid" if valid else "FAIL", color=color,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold")
    return image


def _write_readme(out_dir):
    path = os.path.join(out_dir, "README.md")
    existing = open(path).read() if os.path.exists(path) else "# Fig. 4 data-driven core-field rev8.1\n\n"
    entry = """### fig4b_data_driven_core_field_kmeans

这张图使用与 Fig4A 完全相同的最终 unseen-network 事件池。事件先在冻结的 patient-training embedding 中独立做 KMeans=2，再与两个病人训练模式做 Hungarian 匹配；最右矩阵是模型模式与病人模式的 Spearman 一致性，红框表示至少一项正式门未通过。

**关注点**：两簇是否都有足够事件、两条逐触点 profile 是否真正不同，以及 2×2 矩阵是否呈现正对角、负交叉并达到 rigid-control 基准。

"""
    if "### fig4b_data_driven_core_field_kmeans" not in existing:
        with open(path, "w") as handle:
            handle.write(existing + entry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", default=PROFILES)
    parser.add_argument("--confirmation", default=CONFIRM)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    confirmation = json.load(open(args.confirmation))
    if confirmation["event_profiles"]["sha256"] != _sha256(args.profiles):
        raise RuntimeError("confirmation/event-profile hash mismatch")
    arrays = np.load(args.profiles)
    names = [str(value) for value in arrays["contact_names"]]
    ranks = np.asarray(arrays["model_rank_matrix"], float)
    labels = np.asarray(arrays["model_labels"], int)
    curves = np.asarray(arrays["model_curves"], float)
    norm_ranks = _normalized_rank_matrix(ranks)
    order = _event_order(curves, labels)
    candidate = confirmation["candidates"][0]
    consistency = candidate["confirm"]["kmeans_data_consistency"]
    matrix = np.asarray(consistency["similarity_matrix"], float)
    gates = candidate["confirm"]["gates"]
    matrix_valid = bool(
        gates["two_cluster_support"]
        and gates["kmeans_matrix_sign_consistent"]
        and gates["beats_rigid_template_match"])

    fig = plt.figure(figsize=(18.8, 5.0), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(2.9, 0.08, 1.0, 1.15, 1.15),
        left=0.055, right=0.985, bottom=0.17, top=0.86, wspace=0.28)
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])
    ax_dist = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[0, 3])
    ax_matrix = fig.add_subplot(grid[0, 4])
    image = _plot_heatmap(ax_heat, norm_ranks, labels, names, order)
    colorbar = fig.colorbar(image, cax=ax_cbar)
    colorbar.set_ticks((0, 1)); colorbar.set_ticklabels(("first", "last"))
    colorbar.set_label("within-event rank", fontsize=9)
    _plot_rank_distribution(ax_dist, norm_ranks, names)
    _plot_profiles(ax_profile, norm_ranks, labels, names)
    matrix_image = _plot_matrix(ax_matrix, matrix, matrix_valid)
    matrix_cbar = ax_matrix.inset_axes([1.08, 0.0, 0.06, 1.0])
    fig.colorbar(matrix_image, cax=matrix_cbar).set_label("Spearman ρ", fontsize=9)
    verdict = candidate["confirm"]["verdict"]
    fig.suptitle(
        f"KMeans modes against patient data  |  {_verdict_label(verdict)}",
        fontsize=13.0, fontweight="bold", y=0.985)
    fig.text(
        0.985, 0.035,
        f"matched mean={consistency['matched_mean']:.3f}   "
        f"contrast={consistency['matrix_contrast']:.3f}   "
        f"clusters={consistency['cluster_counts'][0]}/{consistency['cluster_counts'][1]}",
        ha="right", va="bottom", fontsize=9.5, color="0.30")

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(args.out, "fig4b_data_driven_core_field_kmeans")
    fig.savefig(stem + ".png", dpi=220, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem + ".pdf", facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    metadata = dict(
        figure="Fig4B data-driven core-field KMeans consistency",
        plotting_only=True,
        input_profiles=dict(path=args.profiles, sha256=_sha256(args.profiles)),
        input_confirmation=dict(path=args.confirmation, sha256=_sha256(args.confirmation)),
        n_events=int(len(labels)), cluster_counts=np.bincount(labels, minlength=2).tolist(),
        similarity_matrix=matrix.tolist(), matched_mean=consistency["matched_mean"],
        matrix_contrast=consistency["matrix_contrast"],
        gates=gates, verdict=verdict, matrix_valid=matrix_valid,
        permutation_p_values=None,
        permutation_note=(
            "No p-values: the 31 interpolated profile positions are autocorrelated; "
            "acceptance uses support, sign structure, rigid benchmark, and held-out distances."),
    )
    with open(stem + "_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)
    _write_readme(args.out)
    print(f"wrote {stem}.png / .pdf / _metadata.json")


if __name__ == "__main__":
    main()
