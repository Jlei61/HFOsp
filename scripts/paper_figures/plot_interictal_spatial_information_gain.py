#!/usr/bin/env python3
"""Render the paper-ready held-out interictal spatial-information gain figure."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_interictal_template_direction_rose import (  # noqa: E402
    TA_COLOR,
    TB_COLOR,
)
from scripts.run_interictal_spatial_information_gain import (  # noqa: E402
    DEFAULT_MAX_EVENTS,
    DEFAULT_MIN_CLUSTER_EVENTS,
    DEFAULT_SEED,
    load_subject_analysis_inputs,
)
from scripts.run_interictal_spatial_information_gain_all_events import (  # noqa: E402
    load_all_event_inputs,
)
from src.interictal_spatial_information_gain import (  # noqa: E402
    METHOD_HYBRID,
    METHOD_TEMPORAL,
    fit_evaluate_all_event_crossfit_fold,
    fit_evaluate_crossfit_fold,
)
from src.topic5_interictal_direction_rose import (  # noqa: E402
    axis_pair_display_basis,
    project_directions_to_angles,
)
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    bootstrap_median_ci,
)


DEFAULT_ANALYSIS_ROOT = (
    ROOT / "results/interictal_propagation_masked/spatial_information_gain"
)
DEFAULT_PAPER_ROOT = (
    ROOT / "results/paper-ready-figure/fig2b_spatial_information_gain"
)
FIGURE_STEM = "interictal_spatial_information_gain"
PAPER_STEM = "fig2b-spatial-information-gain"
VIOLIN_FIGURE_STEM = "interictal_spatial_information_gain_paired_violin"
VIOLIN_PAPER_STEM = "fig2b-spatial-information-gain-paired-violin"
DEFAULT_EXAMPLES = ("epilepsiae_1146", "epilepsiae_548")
DEFAULT_EXAMPLE_LABELS = ("E1146", "E548")
ROSE_FOLD_INDEX = 0
ROSE_BINS = 18

TEMPORAL_COLOR = "#5B7894"
HYBRID_COLOR = "#C75D3A"
EPILEPSIAE_COLOR = "#3E6F8E"
YUQUAN_COLOR = "#B95F47"
NEUTRAL = "#777777"
LIGHT_NEUTRAL = "#D2D2D2"
TEXT = "#202020"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.relative_to(ROOT)) if value.is_relative_to(ROOT) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                **row,
                "timing_only_score": float(row["timing_only_score"]),
                "timing_plus_space_score": float(row["timing_plus_space_score"]),
                "spatial_information_gain": float(row["spatial_information_gain"]),
            })
    return rows


def build_heldout_rose_payload(
    subject_id: str,
    *,
    all_events: bool = False,
) -> dict[str, Any]:
    """Build fold-0 roses with one event set and one display basis per patient."""
    if all_events:
        inputs = load_all_event_inputs(
            subject_id,
            max_events=0,
            seed=DEFAULT_SEED,
        )
        fold = fit_evaluate_all_event_crossfit_fold(
            np.asarray(inputs["ranks"], float),
            np.asarray(inputs["bools"], bool),
            np.asarray(inputs["directions"], float),
            np.asarray(inputs["blocks"], int),
            np.asarray(inputs["coords"], float),
            fold_index=ROSE_FOLD_INDEX,
            min_cluster_events=DEFAULT_MIN_CLUSTER_EVENTS,
        )
    else:
        inputs = load_subject_analysis_inputs(
            subject_id,
            max_events=DEFAULT_MAX_EVENTS,
            seed=DEFAULT_SEED,
        )
        qc_indices = np.asarray(inputs["qc_indices"], int)
        fold = fit_evaluate_crossfit_fold(
            np.asarray(inputs["ranks"], float)[:, qc_indices],
            np.asarray(inputs["bools"], bool)[:, qc_indices],
            np.asarray(inputs["directions"], float)[qc_indices],
            np.asarray(inputs["blocks"], int)[qc_indices],
            np.asarray(inputs["coords"], float),
            fold_index=ROSE_FOLD_INDEX,
            min_cluster_events=DEFAULT_MIN_CLUSTER_EVENTS,
        )
    hybrid_axes = np.asarray(fold["models"][METHOD_HYBRID]["axes"], float)
    common_basis = axis_pair_display_basis(hybrid_axes[0], hybrid_axes[1])
    if all_events:
        common_directions = np.asarray(
            fold["evaluations"]["common_test_directions"], float
        )
    else:
        common_directions = np.asarray(
            fold["evaluations"][METHOD_TEMPORAL]["test_directions"], float
        )
    common_projection = project_directions_to_angles(
        common_directions,
        common_basis["axis_a"],
        common_basis["transverse"],
    )
    all_event_angles = np.asarray(common_projection["angles"], float)
    all_event_angles = all_event_angles[np.isfinite(all_event_angles)]

    methods: dict[str, Any] = {}
    for method in (METHOD_TEMPORAL, METHOD_HYBRID):
        model = fold["models"][method]
        evaluation = fold["evaluations"][method]
        axes = np.asarray(model["axes"], float)
        axis_projection = project_directions_to_angles(
            axes,
            common_basis["axis_a"],
            common_basis["transverse"],
        )
        methods[method] = {
            "direction_score": float(evaluation["direction_score"]),
            "cluster_scores": np.asarray(evaluation["cluster_scores"], float),
            "test_cluster_counts": np.asarray(
                evaluation[
                    "score_cluster_counts" if all_events else "test_cluster_counts"
                ],
                int,
            ),
            "train_cluster_counts": np.asarray(
                model["train_cluster_counts"], int
            ),
            "axes": axes,
            "axis_angles_rad": np.asarray(axis_projection["angles"], float),
            "axis_projection_norm": np.asarray(
                axis_projection["projection_norm"], float
            ),
            "spatial_scale": float(model["spatial_scale"]),
            "assignments": np.asarray(
                evaluation["score_assignments" if all_events else "assignments"],
                int,
            ),
        }
    temporal_assignment = methods[METHOD_TEMPORAL]["assignments"]
    hybrid_assignment = methods[METHOD_HYBRID]["assignments"]
    return {
        "subject_id": subject_id,
        "fold": int(fold["fold"]),
        "n_train_events": int(len(fold["train_indices"])),
        "n_test_events": int(len(fold["test_indices"])),
        "train_blocks": np.asarray(fold["train_blocks"], int),
        "test_blocks": np.asarray(fold["test_blocks"], int),
        "train_label_ami": float(fold["train_label_ami"]),
        "train_label_overlap": float(fold["train_label_overlap"]),
        "hybrid_label_swap_to_temporal": bool(
            fold["hybrid_label_swap_to_temporal"]
        ),
        "common_display_basis": common_basis,
        "all_event_angles": all_event_angles,
        "n_common_rose_events": int(all_event_angles.size),
        "test_label_overlap": float(np.mean(
            temporal_assignment == hybrid_assignment
        )),
        "test_reassigned_fraction": float(np.mean(
            temporal_assignment != hybrid_assignment
        )),
        "methods": methods,
    }


def _histogram_proportions(values: Sequence[float], edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, float), bins=edges)
    total = int(counts.sum())
    return counts.astype(float) / total if total else np.zeros_like(counts, float)


def _rose_max(payloads: Sequence[Mapping[str, Any]], edges: np.ndarray) -> float:
    maximum = 0.0
    for payload in payloads:
        proportions = _histogram_proportions(payload["all_event_angles"], edges)
        if proportions.size:
            maximum = max(maximum, float(np.max(proportions)))
    return 0.10 if maximum <= 0 else float(np.ceil(maximum / 0.05) * 0.05)


def draw_heldout_probability_rose(
    ax: plt.Axes,
    payload: Mapping[str, Any],
    *,
    method: str,
    edges: np.ndarray,
    rmax: float,
    row_label: str | None,
    show_radial_labels: bool,
) -> None:
    """Draw the same held-out directions with method-specific fitted axes."""
    method_payload = payload["methods"][method]
    centers = edges[:-1] + 0.5 * np.diff(edges)
    width = float(np.diff(edges)[0] * 0.92)
    proportions = _histogram_proportions(payload["all_event_angles"], edges)
    ax.bar(
        centers,
        proportions,
        width=width,
        facecolor=matplotlib.colors.to_rgba("#8B8B8B", 0.24),
        edgecolor="#6F6F6F",
        linewidth=0.68,
        zorder=2,
    )

    line_top = rmax * 1.02
    for theta, color in zip(
        np.asarray(method_payload["axis_angles_rad"], float),
        (TA_COLOR, TB_COLOR),
    ):
        ax.plot([theta, theta], [0.0, line_top], color=color, lw=2.15, zorder=5)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=5.8)
    ax.tick_params(axis="x", pad=-1.5)
    ax.set_ylim(0.0, rmax * 1.04)
    tick_step = 0.10 if rmax >= 0.20 else 0.05
    radial_ticks = np.arange(tick_step, rmax + 1e-9, tick_step)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels(
        [f"{100 * value:.0f}%" for value in radial_ticks]
        if show_radial_labels else []
    )
    if show_radial_labels:
        for label in ax.get_yticklabels():
            label.set_fontsize(5.6)
    ax.set_rlabel_position(140)
    ax.grid(color="#D7D7D7", linewidth=0.48, alpha=0.95)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.62)
    if row_label is not None:
        ax.text(
            -0.16,
            1.04,
            row_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=TEXT,
            clip_on=False,
        )
        ax.text(
            -0.16,
            0.93,
            f"same n={payload['n_common_rose_events']:,}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color="#555555",
            clip_on=False,
        )
    score_text = f"score={method_payload['direction_score']:.2f}"
    if method == METHOD_HYBRID:
        score_text += f"   {100 * payload['test_reassigned_fraction']:.0f}% reassigned"
    ax.text(
        0.50,
        -0.14,
        score_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color="#444444",
        clip_on=False,
    )


def draw_combined_probability_rose(
    ax: plt.Axes,
    payload: Mapping[str, Any],
    *,
    edges: np.ndarray,
    rmax: float,
    row_label: str,
) -> None:
    """Draw one shared held-out rose with old dashed and new solid axes."""
    centers = edges[:-1] + 0.5 * np.diff(edges)
    width = float(np.diff(edges)[0] * 0.92)
    proportions = _histogram_proportions(payload["all_event_angles"], edges)
    ax.bar(
        centers,
        proportions,
        width=width,
        facecolor=matplotlib.colors.to_rgba("#8B8B8B", 0.24),
        edgecolor="#6F6F6F",
        linewidth=0.68,
        zorder=2,
    )

    line_top = rmax * 1.02
    hybrid_angles = np.asarray(
        payload["methods"][METHOD_HYBRID]["axis_angles_rad"], float
    )
    timing_angles = np.asarray(
        payload["methods"][METHOD_TEMPORAL]["axis_angles_rad"], float
    )
    for theta, color in zip(hybrid_angles, (TA_COLOR, TB_COLOR)):
        ax.plot([theta, theta], [0.0, line_top], color=color, lw=2.35, zorder=5)
    for theta, color in zip(timing_angles, (TA_COLOR, TB_COLOR)):
        ax.plot(
            [theta, theta],
            [0.0, line_top],
            color=color,
            lw=1.45,
            ls=(0, (3.0, 2.0)),
            zorder=6,
        )

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=6.2)
    ax.tick_params(axis="x", pad=-1.5)
    ax.set_ylim(0.0, rmax * 1.04)
    tick_step = 0.10 if rmax >= 0.20 else 0.05
    radial_ticks = np.arange(tick_step, rmax + 1e-9, tick_step)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{100 * value:.0f}%" for value in radial_ticks], fontsize=5.8)
    ax.set_rlabel_position(140)
    ax.grid(color="#D7D7D7", linewidth=0.48, alpha=0.95)
    ax.spines["polar"].set_color("#777777")
    ax.spines["polar"].set_linewidth(0.62)
    ax.text(
        -0.18,
        1.00,
        row_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def _p_stars(p_value: float) -> str:
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(labelsize=8.0, length=3, color="#777777")
    ax.grid(False)


def _violin_box(
    ax: plt.Axes,
    values: np.ndarray,
    x: float,
    *,
    color: str,
) -> None:
    if values.size >= 2 and not np.allclose(values, values[0]):
        violin = ax.violinplot(
            [values],
            positions=[x],
            widths=0.54,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )["bodies"][0]
        violin.set_facecolor(color)
        violin.set_edgecolor("none")
        violin.set_alpha(0.22)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    low, high = float(values.min()), float(values.max())
    ax.plot([x, x], [low, high], color=color, lw=0.8, alpha=0.9, zorder=3)
    ax.add_patch(plt.Rectangle(
        (x - 0.10, q1),
        0.20,
        q3 - q1,
        facecolor="white",
        edgecolor=color,
        lw=0.9,
        zorder=4,
    ))
    ax.plot([x - 0.10, x + 0.10], [median, median], color=TEXT, lw=1.4, zorder=5)


def draw_paired_scores(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    title: str | None = "Paired held-out scores",
    ylabel: str = (
        "Direction score on held-out blocks\n(mean signed cosine)"
    ),
) -> None:
    timing = np.asarray([row["timing_only_score"] for row in rows], float)
    hybrid = np.asarray([row["timing_plus_space_score"] for row in rows], float)
    x_timing, x_hybrid = 0.0, 1.0
    order = np.argsort(hybrid - timing)
    for index in order:
        improved = hybrid[index] > timing[index]
        ax.plot(
            [x_timing, x_hybrid],
            [timing[index], hybrid[index]],
            color="#8D8D8D" if improved else "#B14F4F",
            lw=0.60,
            alpha=0.52,
            zorder=1,
        )
    _violin_box(ax, timing, x_timing, color=TEMPORAL_COLOR)
    _violin_box(ax, hybrid, x_hybrid, color=HYBRID_COLOR)
    ax.scatter(
        np.full(len(timing), x_timing),
        timing,
        s=15,
        facecolors=TEMPORAL_COLOR,
        edgecolors="white",
        linewidths=0.45,
        zorder=6,
    )
    ax.scatter(
        np.full(len(hybrid), x_hybrid),
        hybrid,
        s=15,
        facecolors=HYBRID_COLOR,
        edgecolors="white",
        linewidths=0.45,
        zorder=6,
    )

    top = max(float(timing.max()), float(hybrid.max())) + 0.055
    bracket_height = 0.018
    ax.plot(
        [x_timing, x_timing, x_hybrid, x_hybrid],
        [top, top + bracket_height, top + bracket_height, top],
        color=TEXT,
        lw=0.85,
        clip_on=False,
    )
    ax.text(
        0.5,
        top + bracket_height + 0.006,
        _p_stars(float(summary["paired_wilcoxon_greater_p"])),
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=TEXT,
    )
    ax.axhline(0.0, color="#B8B8B8", lw=0.65, ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(-0.48, 1.48)
    ax.set_ylim(-0.055, max(1.04, top + 0.075))
    ax.set_xticks([x_timing, x_hybrid])
    ax.set_xticklabels(
        ["Timing", "Timing + space"],
        fontsize=8.5,
    )
    ax.set_ylabel(ylabel, fontsize=9.2, labelpad=7)
    if title:
        ax.set_title(title, fontsize=9.2, fontweight="bold", pad=7)
    _style_axis(ax)


def draw_gain_and_null(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    cohort_null: np.ndarray,
    *,
    example_labels: Mapping[str, str],
) -> None:
    ordered = sorted(rows, key=lambda row: float(row["spatial_information_gain"]))
    gains = np.asarray([row["spatial_information_gain"] for row in ordered], float)
    y = np.arange(len(ordered), dtype=float)
    for index, row in enumerate(ordered):
        gain = float(row["spatial_information_gain"])
        color = EPILEPSIAE_COLOR if row["dataset"] == "epilepsiae" else YUQUAN_COLOR
        ax.plot([0.0, gain], [y[index], y[index]], color=LIGHT_NEUTRAL, lw=0.75, zorder=1)
        ax.scatter(
            gain,
            y[index],
            s=18,
            marker="o" if row["dataset"] == "epilepsiae" else "s",
            facecolor=color,
            edgecolor="white",
            linewidth=0.45,
            zorder=4,
        )
        subject_id = str(row["subject_id"])
        if subject_id in example_labels:
            ax.scatter(
                gain,
                y[index],
                s=43,
                marker="o" if row["dataset"] == "epilepsiae" else "s",
                facecolors="none",
                edgecolors=TEXT,
                linewidths=0.75,
                zorder=5,
            )
            ax.annotate(
                example_labels[subject_id],
                xy=(gain, y[index]),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=6.7,
                color=TEXT,
                zorder=6,
            )

    observed_median = float(summary["median_gain"])
    ci_lo, ci_hi = map(float, summary["median_gain_bootstrap_ci95"])
    summary_y = len(ordered) + 0.75
    ax.errorbar(
        observed_median,
        summary_y,
        xerr=np.array([[observed_median - ci_lo], [ci_hi - observed_median]]),
        fmt="D",
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=TEXT,
        markeredgewidth=0.9,
        ecolor=TEXT,
        elinewidth=1.05,
        capsize=2.2,
        zorder=5,
    )
    ax.text(
        ci_hi + 0.018,
        summary_y,
        "Median ± 95% CI",
        ha="left",
        va="center",
        fontsize=6.9,
        color="#555555",
    )

    null = np.asarray(cohort_null, float)
    q_lo, q_hi = np.percentile(null, [2.5, 97.5])
    null_median = float(np.median(null))
    null_y = -1.35
    ax.errorbar(
        null_median,
        null_y,
        xerr=np.array([[null_median - q_lo], [q_hi - null_median]]),
        fmt="D",
        markersize=3.7,
        markerfacecolor="white",
        markeredgecolor=NEUTRAL,
        markeredgewidth=0.9,
        ecolor=NEUTRAL,
        elinewidth=1.0,
        capsize=2.0,
        zorder=4,
    )
    ax.text(
        q_hi + 0.018,
        null_y,
        "Block-shuffle null (95%)",
        ha="left",
        va="center",
        fontsize=6.9,
        color="#5A5A5A",
    )
    ax.axvline(0.0, color="#777777", lw=0.75, ls=(0, (3, 2)), zorder=0)
    ax.axvline(observed_median, color=HYBRID_COLOR, lw=0.9, ls=(0, (3, 2)), zorder=0)
    ax.set_yticks([])
    ax.set_ylim(-2.25, len(ordered) + 2.0)
    pad = 0.05 * max(0.2, float(gains.max() - gains.min()))
    ax.set_xlim(min(-0.20, float(gains.min()) - pad), float(gains.max()) + 0.11)
    ax.set_xlabel(
        "Spatial-information gain (Δ signed cosine)",
        fontsize=9.2,
        labelpad=6,
    )
    ax.set_ylabel("Patients (ordered)", fontsize=9.2, labelpad=7)
    ax.set_title(
        "Two-way held-out cohort comparison",
        fontsize=9.5,
        fontweight="bold",
        pad=7,
    )
    ax.text(
        0.985,
        0.97,
        f"{summary['n_positive_gain']}/{summary['n']} improve",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color="#555555",
    )
    _style_axis(ax)


def draw_absolute_scores_and_null(
    ax: plt.Axes,
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    hybrid_null: np.ndarray,
    *,
    example_labels: Mapping[str, str],
) -> None:
    """Show absolute scores, the spatial null and the paired method test."""
    ordered = sorted(rows, key=lambda row: float(row["timing_plus_space_score"]))
    timing = np.asarray([row["timing_only_score"] for row in ordered], float)
    hybrid = np.asarray([row["timing_plus_space_score"] for row in ordered], float)
    y = np.arange(3, len(ordered) + 3, dtype=float)
    null = np.asarray(hybrid_null, float)
    null = null[np.isfinite(null)]
    null_lo, null_hi = np.percentile(null, [2.5, 97.5])
    null_median = float(np.median(null))

    ax.axvline(0.0, color="#8A8A8A", lw=0.80, ls=(0, (3, 2)), zorder=0)
    ax.axvspan(null_lo, null_hi, color="#BDBDBD", alpha=0.16, zorder=0)
    for index, row in enumerate(ordered):
        improved = hybrid[index] > timing[index]
        ax.plot(
            [timing[index], hybrid[index]],
            [y[index], y[index]],
            color="#C9C9C9" if improved else "#C4867A",
            lw=0.85,
            alpha=0.90,
            zorder=1,
        )
    ax.scatter(
        timing,
        y,
        s=18,
        marker="o",
        facecolor="white",
        edgecolor=TEMPORAL_COLOR,
        linewidth=0.80,
        zorder=3,
    )
    ax.scatter(
        hybrid,
        y,
        s=22,
        marker="o",
        facecolor=HYBRID_COLOR,
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
    )
    for index, row in enumerate(ordered):
        subject_id = str(row["subject_id"])
        if subject_id not in example_labels:
            continue
        ax.scatter(
            hybrid[index],
            y[index],
            s=50,
            facecolor="none",
            edgecolor=TEXT,
            linewidth=1.0,
            zorder=5,
        )
        ax.annotate(
            example_labels[subject_id],
            xy=(hybrid[index], y[index]),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.8,
            fontweight="bold",
            color=TEXT,
            zorder=6,
        )

    absolute = summary.get("absolute_direction_scores", {})
    timing_median = float(absolute.get("timing_only_median", np.median(timing)))
    timing_ci = absolute.get(
        "timing_only_bootstrap_ci95",
        bootstrap_median_ci(timing, n_boot=5000, seed=DEFAULT_SEED + 1),
    )
    hybrid_median = float(
        absolute.get("timing_plus_space_median", np.median(hybrid))
    )
    hybrid_ci = absolute.get(
        "timing_plus_space_bootstrap_ci95",
        bootstrap_median_ci(hybrid, n_boot=5000, seed=DEFAULT_SEED + 2),
    )

    def bootstrap_medians(values: np.ndarray, *, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        medians = np.empty(10_000, float)
        for start in range(0, len(medians), 1_000):
            stop = min(start + 1_000, len(medians))
            samples = values[
                rng.integers(0, len(values), size=(stop - start, len(values)))
            ]
            medians[start:stop] = np.median(samples, axis=1)
        return medians

    timing_bootstrap = bootstrap_medians(timing, seed=DEFAULT_SEED + 2)
    hybrid_bootstrap = bootstrap_medians(hybrid, seed=DEFAULT_SEED + 1)
    # The ridge is a visual bootstrap density.  The error bars use the persisted
    # cohort CI, which remains authoritative when row ordering changes between
    # the all-event and QC-clean result tables.

    def draw_score_ridge(
        values: np.ndarray,
        baseline_y: float,
        color: str,
        *,
        height: float,
        alpha: float,
    ) -> None:
        ridge_x = np.linspace(float(values.min()), float(values.max()), 300)
        ridge_density = gaussian_kde(values)(ridge_x)
        ridge_density = height * ridge_density / float(np.max(ridge_density))
        ax.fill_between(
            ridge_x,
            baseline_y,
            baseline_y + ridge_density,
            color=color,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )
        ax.plot(
            ridge_x,
            baseline_y + ridge_density,
            color=color,
            lw=0.70,
            alpha=0.90,
            zorder=3,
        )

    ridge_y = -1.20
    for ridge_values, color, height, alpha in (
        (null, "#7C7C7C", 1.55, 0.42),
        (timing_bootstrap, TEMPORAL_COLOR, 1.55, 0.20),
        (hybrid_bootstrap, HYBRID_COLOR, 1.55, 0.20),
    ):
        draw_score_ridge(
            ridge_values,
            ridge_y,
            color,
            height=height,
            alpha=alpha,
        )

    for median, ci, color, y_offset in (
        (null_median, (null_lo, null_hi), "#666666", 0.00),
        (timing_median, timing_ci, TEMPORAL_COLOR, -0.045),
        (hybrid_median, hybrid_ci, HYBRID_COLOR, 0.045),
    ):
        ci_lo, ci_hi = map(float, ci)
        ax.errorbar(
            median,
            ridge_y + 0.04 + y_offset,
            xerr=np.asarray([[median - ci_lo], [ci_hi - median]]),
            fmt="D",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.95,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.0,
            zorder=5,
        )

    label_y = ridge_y - 0.30
    ax.text(
        null_median,
        label_y,
        "Null",
        ha="center",
        va="top",
        fontsize=6.4,
        color=TEXT,
    )
    ax.text(
        timing_median - 0.012,
        label_y,
        "Timing",
        ha="right",
        va="top",
        fontsize=6.5,
        color=TEMPORAL_COLOR,
    )
    ax.text(
        hybrid_median + 0.012,
        label_y,
        "+Space",
        ha="left",
        va="top",
        fontsize=6.5,
        color=HYBRID_COLOR,
    )

    paired_bracket_y = ridge_y + 1.78
    ax.plot(
        [timing_median, timing_median, hybrid_median, hybrid_median],
        [
            paired_bracket_y - 0.12,
            paired_bracket_y,
            paired_bracket_y,
            paired_bracket_y - 0.12,
        ],
        color=TEXT,
        lw=0.85,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        0.5 * (timing_median + hybrid_median),
        paired_bracket_y + 0.04,
        _p_stars(float(summary["paired_wilcoxon_greater_p"])),
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color=TEXT,
        zorder=6,
    )
    null_p = float(absolute.get(
        "hybrid_observed_vs_null_empirical_p",
        (1 + np.sum(null >= hybrid_median - 1e-15)) / (len(null) + 1),
    ))
    # Keep both inferential comparisons next to the summary distributions.
    # The longer +Space-vs-null bracket sits above the shorter paired bracket,
    # but remains below the first patient row.
    bracket_y = ridge_y + 2.70
    ax.plot(
        [null_median, null_median, hybrid_median, hybrid_median],
        [bracket_y - 0.28, bracket_y, bracket_y, bracket_y - 0.28],
        color=TEXT,
        lw=0.85,
        clip_on=False,
    )
    ax.text(
        0.5 * (null_median + hybrid_median),
        bracket_y + 0.05,
        _p_stars(null_p),
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
        color=TEXT,
    )

    ax.text(
        0.02,
        0.80,
        f"{summary['n_positive_gain']}/{summary['n']} improve",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=TEXT,
        linespacing=1.25,
    )
    ax.set_xlim(
        min(-0.10, float(null_lo) - 0.02, float(timing.min()) - 0.03),
        max(0.98, float(hybrid.max()) + 0.08),
    )
    ax.set_ylim(-2.05, len(ordered) + 4.55)
    ax.set_yticks([])
    ax.set_xlabel("Held-out direction score", fontsize=9.2, labelpad=6)
    ax.set_ylabel("Patients (ordered)", fontsize=9.2, labelpad=6)
    _style_axis(ax)


def _write_readme(
    figures: Path,
    filename: str,
    violin_filename: str,
    summary: Mapping[str, Any],
    sensitivity: Mapping[str, Any],
) -> None:
    ci_lo, ci_hi = summary["median_gain_bootstrap_ci95"]
    null = summary["direction_shuffle_null"]
    absolute = summary["absolute_direction_scores"]
    lines = [
        f"### {filename}.png / .pdf",
        "",
        (
            "左侧固定复用原 Fig. 2B 的 E1146 与 E548，每位患者只画一张 fold-0 held-out 全事件 rose；同色虚线是 Timing 训练模板轴，同色实线是 Timing + space 训练模板轴，两种方法严格共享事件集合和二维显示基底，并将空间模型的 Mode 1 红色实线固定为 0°。"
        ),
        (
            f"右侧恢复原 Fig. 2B 的绝对 direction-score/零假设语法：底部同一行叠加蓝色 Timing、橙色 +Space 的 10,000 次患者 bootstrap cohort-median 分布，以及冻结 Timing + space 模型后在 held-out recording block 内打乱事件方向得到的灰色 cohort-median null；空间模型真实中位分数为 {absolute['timing_plus_space_median']:.3f}，对 null 的经验 p={absolute['hybrid_observed_vs_null_empirical_p']:.4g}。"
        ),
        (
            f"每条患者内连线从 Timing 指向 Timing + space，{summary['n']} 位可评估患者中 {summary['n_positive_gain']} 位提高，增益中位数为 {summary['median_gain']:.3f}（95% bootstrap CI {ci_lo:.3f}–{ci_hi:.3f}），单侧配对 Wilcoxon p={summary['paired_wilcoxon_greater_p']:.4g}；增益对 block-shuffle null 的 p={null['empirical_p_observed_median_greater']:.4g}。"
        ),
        "该结果只支持空间信息能提高可估计事件子集的跨 block 方向一致性，不代表未见患者泛化、真实组织轨迹、传播速度或因果机制。",
        "",
        "**关注点**：先看单张 rose 内同色虚线和实线的方向差异，再看右侧绝对分数高于零假设且多数患者连线向右。底部分布区的长横括号检验 +Space 相对方向置换零模型，短横括号检验 +Space 相对 Timing；图内 p 值只显示星号，精确值保留在本说明和 metadata。",
        "",
        f"### {violin_filename}.png / .pdf",
        "",
        (
            f"该补充图把同一 {summary['n']} 位患者的 Timing 与 Timing + space held-out direction score 画成配对连线、violin、IQR 和中位数，显著性使用患者级单侧配对 Wilcoxon 检验。"
        ),
        (
            f"Timing 中位数为 {summary['timing_only_median']:.3f}，Timing + space 中位数为 {summary['timing_plus_space_median']:.3f}，{summary['n_positive_gain']}/{summary['n']} 位提高，p={summary['paired_wilcoxon_greater_p']:.4g}。"
        ),
        "它用于显示配对统计的完整分布，不替代主图中的绝对零假设比较。",
        "",
        "**关注点**：看每条患者内连线的方向和整体中位数移动，不要只比较两个 violin 的边际形状。",
        "",
    ]
    (figures / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_figure(
    *,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    paper_root: Path = DEFAULT_PAPER_ROOT,
    examples: Sequence[str] = DEFAULT_EXAMPLES,
    example_labels: Sequence[str] = DEFAULT_EXAMPLE_LABELS,
) -> Mapping[str, Path]:
    if len(examples) != 2 or len(example_labels) != 2:
        raise ValueError("Fig. 2B comparison requires exactly two rose examples")
    subject_csv = analysis_root / "subject_spatial_information_gain.csv"
    summary_json = analysis_root / "spatial_information_gain_summary.json"
    null_npz = analysis_root / "cohort_direction_shuffle_null.npz"
    sensitivity_json = analysis_root / "sampling_seed_sensitivity.json"
    rows = _load_rows(subject_csv)
    summary_payload = json.loads(summary_json.read_text())
    summary = summary_payload["cohort_statistics"]
    sensitivity = (
        json.loads(sensitivity_json.read_text())
        if sensitivity_json.exists()
        else {"status": "not_applicable_all_event_primary"}
    )
    all_events = summary_payload.get("contract", {}).get("hard_event_qc_used") is False
    with np.load(null_npz, allow_pickle=False) as cached:
        cohort_null_gain = np.asarray(cached["cohort_median_null_gain"], float)
        if "cohort_median_null_timing_plus_space_score" in cached.files:
            cohort_null_hybrid = np.asarray(
                cached["cohort_median_null_timing_plus_space_score"], float
            )
        elif "patient_null_timing_plus_space_score" in cached.files:
            cohort_null_hybrid = np.median(
                np.asarray(cached["patient_null_timing_plus_space_score"], float),
                axis=0,
            )
        else:
            raise RuntimeError(
                "absolute direction-score null missing; rerun the spatial-information analysis"
            )
        cached_subject_ids = [str(value) for value in cached["subject_ids"].tolist()]
    if cached_subject_ids != [str(row["subject_id"]) for row in rows]:
        raise RuntimeError("subject CSV and cohort null use different patient order")
    if int(summary["n"]) != len(rows):
        raise RuntimeError("summary and subject CSV denominators disagree")
    row_ids = {str(row["subject_id"]) for row in rows}
    missing_examples = [subject_id for subject_id in examples if subject_id not in row_ids]
    if missing_examples:
        raise RuntimeError(f"rose examples absent from held-out cohort: {missing_examples}")
    rose_payloads = [
        build_heldout_rose_payload(subject_id, all_events=all_events)
        for subject_id in examples
    ]
    edges = np.linspace(0.0, 2.0 * np.pi, ROSE_BINS + 1)
    rmax = _rose_max(rose_payloads, edges)
    example_label_map = dict(zip(examples, example_labels))

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.15, 4.55), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.05, 1.35),
        height_ratios=(1, 1),
        left=0.028,
        right=0.992,
        top=0.945,
        bottom=0.145,
        wspace=0.18,
        hspace=0.26,
    )
    rose_axes: list[plt.Axes] = []
    for row_index, (payload, label) in enumerate(zip(rose_payloads, example_labels)):
        ax = fig.add_subplot(grid[row_index, 0], projection="polar")
        draw_combined_probability_rose(
            ax,
            payload,
            edges=edges,
            rmax=rmax,
            row_label=label,
        )
        rose_axes.append(ax)

    gain_ax = fig.add_subplot(grid[:, 1])
    draw_absolute_scores_and_null(
        gain_ax,
        rows,
        summary,
        cohort_null_hybrid,
        example_labels=example_label_map,
    )
    rose_legend = [
        Patch(
            facecolor=matplotlib.colors.to_rgba("#8B8B8B", 0.24),
            edgecolor="#6F6F6F",
            linewidth=0.8,
            label="Events",
        ),
        (
            Line2D([0], [0], color=TA_COLOR, lw=2.0),
            Line2D([0], [0], color=TB_COLOR, lw=2.0),
        ),
        (
            Line2D([0], [0], color=TEXT, lw=1.4, ls=(0, (3, 2))),
            Line2D([0], [0], color=TEXT, lw=2.2),
        ),
    ]
    fig.legend(
        handles=rose_legend,
        labels=["Events", "Modes 1 / 2", "Timing dashed / +Space solid"],
        loc="lower left",
        bbox_to_anchor=(0.028, 0.012),
        ncol=3,
        frameon=False,
        fontsize=6.8,
        handlelength=1.7,
        columnspacing=0.95,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    )

    analysis_figures = analysis_root / "figures"
    paper_figures = paper_root / "figures"
    analysis_figures.mkdir(parents=True, exist_ok=True)
    paper_figures.mkdir(parents=True, exist_ok=True)
    outputs = {
        "analysis_png": analysis_figures / f"{FIGURE_STEM}.png",
        "analysis_pdf": analysis_figures / f"{FIGURE_STEM}.pdf",
        "paper_png": paper_figures / f"{PAPER_STEM}.png",
        "paper_pdf": paper_figures / f"{PAPER_STEM}.pdf",
        "analysis_violin_png": analysis_figures / f"{VIOLIN_FIGURE_STEM}.png",
        "analysis_violin_pdf": analysis_figures / f"{VIOLIN_FIGURE_STEM}.pdf",
        "paper_violin_png": paper_figures / f"{VIOLIN_PAPER_STEM}.png",
        "paper_violin_pdf": paper_figures / f"{VIOLIN_PAPER_STEM}.pdf",
        "metadata": paper_figures / f"{PAPER_STEM}_metadata.json",
    }
    fig.savefig(outputs["analysis_png"], dpi=600, facecolor="white")
    fig.savefig(outputs["analysis_pdf"], facecolor="white")
    plt.close(fig)
    shutil.copy2(outputs["analysis_png"], outputs["paper_png"])
    shutil.copy2(outputs["analysis_pdf"], outputs["paper_pdf"])

    violin_fig, violin_ax = plt.subplots(figsize=(3.45, 3.20), facecolor="white")
    violin_fig.subplots_adjust(left=0.24, right=0.98, top=0.89, bottom=0.18)
    draw_paired_scores(violin_ax, rows, summary)
    violin_fig.savefig(outputs["analysis_violin_png"], dpi=600, facecolor="white")
    violin_fig.savefig(outputs["analysis_violin_pdf"], facecolor="white")
    plt.close(violin_fig)
    shutil.copy2(outputs["analysis_violin_png"], outputs["paper_violin_png"])
    shutil.copy2(outputs["analysis_violin_pdf"], outputs["paper_violin_pdf"])

    rose_metadata = []
    for payload, label in zip(rose_payloads, example_labels):
        rose_metadata.append({
            "subject_id": payload["subject_id"],
            "display_label": label,
            "visual_fold": payload["fold"],
            "n_train_events": payload["n_train_events"],
            "n_test_events": payload["n_test_events"],
            "train_blocks": payload["train_blocks"],
            "test_blocks": payload["test_blocks"],
            "train_label_ami": payload["train_label_ami"],
            "train_label_overlap_after_symmetry_match": payload[
                "train_label_overlap"
            ],
            "test_label_overlap": payload["test_label_overlap"],
            "test_reassigned_fraction": payload["test_reassigned_fraction"],
            "n_common_rose_events": payload["n_common_rose_events"],
            "common_display_basis": payload["common_display_basis"],
            "hybrid_label_swap_to_temporal": payload[
                "hybrid_label_swap_to_temporal"
            ],
            "methods": {
                method: {
                    "direction_score": payload["methods"][method][
                        "direction_score"
                    ],
                    "cluster_scores": payload["methods"][method][
                        "cluster_scores"
                    ],
                    "train_cluster_counts": payload["methods"][method][
                        "train_cluster_counts"
                    ],
                    "test_cluster_counts": payload["methods"][method][
                        "test_cluster_counts"
                    ],
                    "template_axes_xyz": payload["methods"][method]["axes"],
                    "axis_angle_in_common_display_basis_rad": payload[
                        "methods"
                    ][method]["axis_angles_rad"],
                    "axis_projection_norm": payload["methods"][method][
                        "axis_projection_norm"
                    ],
                    "spatial_scale": payload["methods"][method]["spatial_scale"],
                }
                for method in (METHOD_TEMPORAL, METHOD_HYBRID)
            },
        })

    metadata = {
        "figure_role": "Figure 2B replacement candidate: held-out spatial-information gain",
        "scientific_question": summary_payload["contract"]["scientific_question"],
        "cohort_flow": summary_payload["cohort_flow"],
        "cohort_statistics": summary,
        "sampling_seed_sensitivity": sensitivity,
        "analysis_contract": summary_payload["contract"],
        "rose_examples": rose_metadata,
        "visual_contract": {
            "left": "one shared held-out all-event rose per locked E1146/E548 example; timing axes are dashed and timing-plus-space axes are solid in matched mode colors",
            "right": "original Figure 2B absolute-score/null grammar with paired timing-to-space patient endpoints",
            "rose_fold_role": "visual explanation only; cohort inference uses both cross-fit directions",
            "rose_event_identity": (
                "within each patient, both methods show exactly the same held-out all-event score denominator as one neutral distribution"
                if all_events
                else "within each patient, both methods show exactly the same held-out QC-clean events as one neutral all-event distribution"
            ),
            "rose_assignment": "rank-template distance only; held-out directions are opened after assignment",
            "rose_display_basis": "one shared timing-plus-space axis plane per patient for both methods; timing-plus-space Mode 1 red solid axis is rotated to 0 degrees",
            "cluster_label_matching": "hybrid IDs are matched to temporal IDs by maximum train-event overlap; score is unchanged",
            "patient_labels_displayed": list(example_labels),
            "score_encoding": "timing is an open blue circle and blue summary ridge; timing-plus-space is a filled terracotta circle and terracotta summary ridge; connectors preserve patient pairing",
            "absolute_null": "cohort-median held-out timing-plus-space scores after within-block direction shuffle on frozen models",
            "summary_ridges": "aligned cohort-median densities: 10000 patient-bootstrap medians for Timing and +Space, and the frozen-model within-block direction-shuffle null",
            "zero_reference": "signed cosine equals zero",
            "p_value_display": "stars in figures; exact values in metadata and README",
            "paired_test_display": "short horizontal bracket between the Timing and +Space medians above their shared summary ridge row",
            "null_test_display": "long horizontal bracket between Null and +Space in the lower summary-distribution region",
            "supplement": "paired violin with patient connectors, IQR, median, and exact paired Wilcoxon p value",
            "panel_letters": False,
            "dpi": 600,
            "font": "Arial with DejaVu Sans fallback",
        },
        "source": {
            "subject_csv": subject_csv,
            "subject_csv_sha256": _sha256(subject_csv),
            "summary_json": summary_json,
            "summary_json_sha256": _sha256(summary_json),
            "cohort_null_npz": null_npz,
            "cohort_null_npz_sha256": _sha256(null_npz),
            "sampling_seed_sensitivity_json": (
                sensitivity_json if sensitivity_json.exists() else None
            ),
            "sampling_seed_sensitivity_json_sha256": (
                _sha256(sensitivity_json) if sensitivity_json.exists() else None
            ),
            "rose_subject_jsons": {
                subject_id: {
                    "path": analysis_root / "per_subject" / f"{subject_id}.json",
                    "sha256": _sha256(
                        analysis_root / "per_subject" / f"{subject_id}.json"
                    ),
                }
                for subject_id in examples
            },
            "producer": "scripts/paper_figures/plot_interictal_spatial_information_gain.py",
        },
        "claim_boundary": summary_payload["contract"]["claim_boundary"],
        "outputs": outputs,
        "output_sha256": {
            "analysis_png": _sha256(outputs["analysis_png"]),
            "analysis_pdf": _sha256(outputs["analysis_pdf"]),
            "paper_png": _sha256(outputs["paper_png"]),
            "paper_pdf": _sha256(outputs["paper_pdf"]),
            "analysis_violin_png": _sha256(outputs["analysis_violin_png"]),
            "analysis_violin_pdf": _sha256(outputs["analysis_violin_pdf"]),
            "paper_violin_png": _sha256(outputs["paper_violin_png"]),
            "paper_violin_pdf": _sha256(outputs["paper_violin_pdf"]),
        },
    }
    outputs["metadata"].write_text(
        json.dumps(_jsonable(metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(
        analysis_figures,
        FIGURE_STEM,
        VIOLIN_FIGURE_STEM,
        summary,
        sensitivity,
    )
    _write_readme(
        paper_figures,
        PAPER_STEM,
        VIOLIN_PAPER_STEM,
        summary,
        sensitivity,
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--paper-root", type=Path, default=DEFAULT_PAPER_ROOT)
    parser.add_argument("--examples", nargs=2, default=DEFAULT_EXAMPLES)
    parser.add_argument("--example-labels", nargs=2, default=DEFAULT_EXAMPLE_LABELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_figure(
        analysis_root=args.analysis_root,
        paper_root=args.paper_root,
        examples=args.examples,
        example_labels=args.example_labels,
    )
    print(outputs["paper_png"])


if __name__ == "__main__":
    main()
