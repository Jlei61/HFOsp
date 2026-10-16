"""Paper producer functions copied locally; input/output adapters are separate.
Source: scripts/paper_figures/plot_interictal_spatial_information_gain.py
"""
from __future__ import annotations
from typing import Any, List, Mapping, Sequence
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
TA_COLOR='#B2182B'
TB_COLOR='#2166AC'
METHOD_TEMPORAL='timing_only'
METHOD_HYBRID='timing_plus_space'
TEXT='#202020'


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
    ax.set_rlabel_position(112)
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
