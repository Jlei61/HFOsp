#!/usr/bin/env python3
"""Render Figure 6 panel I as a standalone, fully labelled figure.

The upper matrices use the same frozen E1146 response and held-out transition
contracts as the accepted R5 candidate.  The lower panel reproduces the
patient-level observed-versus-within-shaft-null comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_operator_r5 as r5  # noqa: E402
from analyze_topic5_patch_operator_v0_2 import (  # noqa: E402
    contact_space_operator,
    empirical_transition_operator,
)
from scripts.paper_figures import (  # noqa: E402
    plot_topic5_figure6_multiscale_scaffold_v0_5 as base,
)


DEFAULT_RESPONSE = (
    ROOT / "results/topic5_latent_propagation_landscape_v0_2"
    / "spatial_control_field/patch_operator"
)
DEFAULT_OUTPUT = (
    ROOT / "results/paper-ready-figure"
    / "fig6_interictal_crossstate_response_r5_candidate"
    / "panel_i_standalone/figures"
)
STEM = "topic5_figure6_panel_i_response_vs_heldout_transitions"
BLUE = "#397bb7"
GREY = "#a7adaf"
DARK = "#293236"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contact_names() -> list[str]:
    provenance = json.loads(
        (
            ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3/cache"
            / base.FIT_ID / "provenance.json"
        ).read_text()
    )
    return [str(value) for value in provenance.get("joint_contacts", provenance["contacts"])]


def build_case_matrices(response_root: Path) -> tuple[np.ndarray, np.ndarray]:
    consensus = np.nanmedian(
        np.stack([
            r5.load_fit_operator(response_root, base.FIT_ID, arm)
            for arm in r5.REAL_ARMS
        ]),
        axis=0,
    )
    response = contact_space_operator(consensus, base.FIT_ID, "L3", 0)
    transition, _ = empirical_transition_operator(base.FIT_ID)
    return response, transition


def relative_pattern(matrix: np.ndarray) -> np.ndarray:
    """Map off-diagonal matrix entries to centered within-matrix ranks.

    The cohort endpoint is an off-diagonal Spearman correlation, so this
    display transform preserves exactly the ordering used by that endpoint.
    It also lets matrices with different physical units share one colorbar.
    """
    source = np.asarray(matrix, dtype=float)
    display = np.full(source.shape, np.nan, dtype=float)
    mask = np.isfinite(source) & ~np.eye(source.shape[0], dtype=bool)
    values = source[mask]
    if not len(values):
        return display
    if np.nanmax(values) == np.nanmin(values):
        display[mask] = 0.0
        return display
    ranks = rankdata(values, method="average")
    display[mask] = 2.0 * (ranks - 1.0) / (len(values) - 1.0) - 1.0
    return display


def draw_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    names: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    norm: mpl.colors.Normalize,
) -> mpl.image.AxesImage:
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f0f2f3")
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )
    positions = np.arange(len(names))
    ax.set_xticks(positions, names, rotation=55, ha="right", rotation_mode="anchor")
    ax.set_yticks(positions, names)
    ax.tick_params(axis="both", length=2.2, width=.6, labelsize=7.8, pad=1.5)
    ax.set_xlabel(xlabel, fontsize=9.4, labelpad=5, color=DARK)
    ax.set_ylabel(ylabel, fontsize=9.4, labelpad=6, color=DARK)
    ax.set_title(title, fontsize=10.2, fontweight="normal", pad=7, color=DARK)
    for spine in ax.spines.values():
        spine.set_linewidth(.75)
        spine.set_color("#737b7f")
    return image


def draw_colorbar(
    fig: plt.Figure,
    cax: plt.Axes,
    image: mpl.image.AxesImage,
) -> None:
    colorbar = fig.colorbar(image, cax=cax, orientation="vertical")
    colorbar.set_ticks([-1.0, 0.0, 1.0])
    colorbar.ax.set_yticklabels(["Lower", "Median", "Higher"])
    colorbar.ax.tick_params(labelsize=7.4, length=2.0, width=.6, pad=2)
    colorbar.ax.set_title("Relative\npattern", fontsize=7.8, pad=4, color=DARK)
    colorbar.outline.set_linewidth(.6)
    colorbar.outline.set_edgecolor("#737b7f")


def draw_violin(ax: plt.Axes, observed: np.ndarray, null: np.ndarray) -> dict[str, float | int]:
    positions = [0.0, 1.0]
    values = [observed, null]
    colors = [BLUE, GREY]
    parts = ax.violinplot(
        values, positions=positions, widths=.62,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(.24)

    rng = np.random.default_rng(6417)
    for values_i, position, color in zip(values, positions, colors):
        jitter = rng.uniform(-.13, .13, len(values_i))
        ax.scatter(
            position + jitter, values_i,
            s=30, color=color, alpha=.68,
            edgecolor="white", linewidth=.4, zorder=3,
        )
        q1, median, q3 = np.nanpercentile(values_i, [25, 50, 75])
        ax.plot(
            [position, position], [q1, q3], color=color,
            linewidth=4.0, solid_capstyle="round", zorder=4,
        )
        ax.plot(
            [position - .13, position + .13], [median, median],
            color=DARK, linewidth=2.2, zorder=5,
        )

    difference = observed - null
    p_value = float(wilcoxon(difference, alternative="greater").pvalue)
    ax.axhline(0, color="#8a9093", linewidth=.9, linestyle="--", zorder=0)
    ax.plot([0, 0, 1, 1], [.635, .66, .66, .635], color=DARK, linewidth=1.0)
    ax.text(.5, .675, f"P = {p_value:.4f}", ha="center", va="bottom",
            fontsize=11.0, fontweight="bold", color=DARK)
    ax.set_xlim(-.48, 1.48)
    ax.set_ylim(-.32, .76)
    ax.set_xticks(
        positions,
        ["RNN response\nvs held-out data", "Within-electrode\ncontact-shuffle null"],
    )
    ax.tick_params(axis="x", labelsize=11.0, pad=6, colors=DARK)
    ax.tick_params(axis="y", labelsize=10.2, colors=DARK)
    ax.set_ylabel("Spearman correlation with\nheld-out contact transitions",
                  fontsize=12.0, labelpad=9, color=DARK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#596267")
    return {
        "n_patients": int(len(observed)),
        "median_observed": float(np.nanmedian(observed)),
        "median_within_shaft_null": float(np.nanmedian(null)),
        "median_paired_difference": float(np.nanmedian(difference)),
        "patients_above_null": int(np.sum(difference > 0)),
        "p_one_sided": p_value,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-root", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    response_root = args.response_root.resolve()
    output = args.output_dir.resolve()

    names = contact_names()
    response, transition = build_case_matrices(response_root)
    if response.shape != transition.shape or response.shape[0] != len(names):
        raise RuntimeError(
            f"matrix/contact mismatch: response={response.shape}, "
            f"transition={transition.shape}, contacts={len(names)}"
        )
    response_display = relative_pattern(response)
    transition_display = relative_pattern(transition)
    shared_norm = mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)

    source = pd.read_csv(response_root / "OPERATOR_DATA_ALIGNMENT.csv")
    patient = source.groupby("patient", as_index=False).median(numeric_only=True)
    observed = patient["consensus_alignment"].to_numpy(float)
    null = patient["within_shaft_null_median"].to_numpy(float)
    finite = np.isfinite(observed) & np.isfinite(null)
    observed, null = observed[finite], null[finite]
    patient = patient.loc[finite].copy()

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(8.4, 6.8), facecolor="white")
    grid = fig.add_gridspec(
        2, 4,
        height_ratios=(1.02, .90),
        width_ratios=(1.0, .105, 1.0, .045),
        left=.115, right=.955, bottom=.105, top=.935,
        hspace=.49, wspace=.20,
    )
    left = fig.add_subplot(grid[0, 0])
    arrow = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2])
    shared_colorbar = fig.add_subplot(grid[0, 3])
    lower = fig.add_subplot(grid[1, :])

    left_image = draw_matrix(
        left, response_display, names,
        title="RNN perturbation response",
        xlabel="Perturbed contact",
        ylabel="Future model-output contact",
        norm=shared_norm,
    )
    draw_matrix(
        right, transition_display, names,
        title="Held-out contact transitions",
        xlabel="Earlier contact in held-out event",
        ylabel="Contact appearing 1–3 steps later",
        norm=shared_norm,
    )
    draw_colorbar(fig, shared_colorbar, left_image)
    arrow.axis("off")
    arrow.annotate(
        "", xy=(.98, .5), xytext=(.02, .5), xycoords="axes fraction",
        arrowprops={"arrowstyle": "<->", "linewidth": 1.35, "color": DARK},
    )
    stats = draw_violin(lower, observed, null)
    fig.text(.025, .958, "I", fontsize=15.5, fontweight="bold", ha="left", va="top", color=DARK)

    output.mkdir(parents=True, exist_ok=True)
    source_dir = output / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(response, index=names, columns=names).to_csv(
        source_dir / "e1146_rnn_perturbation_response.csv"
    )
    pd.DataFrame(transition, index=names, columns=names).to_csv(
        source_dir / "e1146_heldout_contact_transitions.csv"
    )
    pd.DataFrame(response_display, index=names, columns=names).to_csv(
        source_dir / "e1146_rnn_perturbation_response_relative_pattern.csv"
    )
    pd.DataFrame(transition_display, index=names, columns=names).to_csv(
        source_dir / "e1146_heldout_contact_transitions_relative_pattern.csv"
    )
    patient.to_csv(source_dir / "patient_level_response_vs_null.csv", index=False)

    stem = output / STEM
    assets = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
        assets.append(path)
    plt.close(fig)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract": "figure6_panel_i_standalone_shared_pattern_scale_v0.2",
        "representative_fit": base.FIT_ID,
        "contact_order": names,
        "upper_left": {
            "signed_response_definition": "(logits[h+delta] - logits[h-delta]) / (2 * |delta|)",
            "dose": "0.5 local hidden-state SD",
            "future_steps_averaged": [1, 2, 3],
            "event_phases_averaged": [0.25, 0.5, 0.75],
            "real_order_network_designs": list(r5.REAL_ARMS),
            "seeds_per_design": 3,
        },
        "upper_right": {
            "split": "held-out test events",
            "transition_lags": [1, 2, 3],
            "within_event_rank_shuffle_expectation_subtracted": True,
        },
        "upper_display": {
            "transform": "off-diagonal centered within-matrix ranks",
            "range": [-1.0, 1.0],
            "shared_colorbar": True,
            "reason": "preserves the ordering tested by off-diagonal Spearman while allowing matrices with different physical units to share a scale",
            "blue_semantics": "lower within the same matrix, not necessarily a negative raw response",
            "response_raw_range": [float(np.nanmin(response)), float(np.nanmax(response))],
            "transition_raw_range": [float(np.nanmin(transition)), float(np.nanmax(transition))],
        },
        "lower": stats,
        "assets_sha256": {path.name: sha256(path) for path in assets},
    }
    (output / "FIGURE_METADATA.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "README.md").write_text(
        "### topic5_figure6_panel_i_response_vs_heldout_transitions.png / .pdf / .svg\n\n"
        "上排以 E1146 为例：左图为四种真实顺序 RNN 在早、中、晚参考状态下，分别对局部组织施加等幅正、负扰动后，以两支未来输出之差除以两倍扰动幅度得到的带符号响应。带符号响应先在参考状态、未来1–3步和阶段内取均值，再在3个种子及4种网络设计之间取中位数。右图为模型未见留出事件中，较晚触点在较早触点后1–3个 rank 出现的超额频率。原始矩阵单位不同，因此画面将各自非对角元素转换为矩阵内相对排序，并共用一个色标；蓝色表示该矩阵内相对较低，不等于原始响应必为负。\n\n"
        "下排每个点是一位患者，蓝色为扰动响应矩阵与留出触点转移矩阵的非对角 Spearman 相关，灰色为在同一电极杆内交换触点身份512次所得零模型中位数。42个 fit 先折叠为28位患者；该面板检验响应形状与真实转移是否对齐，不检验扰动前后预测准确率或连接必要性。\n\n"
        "**关注点**：患者内配对差中位数为 +0.0676，21/28 位患者高于杆内打乱零模型；图中报告单侧配对检验 P=0.0017。\n"
    )


if __name__ == "__main__":
    main()
