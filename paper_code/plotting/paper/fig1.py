"""Paper producer functions copied locally; input/output adapters are separate.
Source: results/paper-ready-figure/fig1/source/code_snapshot/scripts/plot_interictal_propagation.py
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


def _viridis_with_lightgray_bad():
    """Return a viridis cmap copy that renders NaN cells as lightgray.

    Used so non-participating channels (Topic 0 §3.1 phantom-rank mask) are
    visually distinguished from real ranks. Without this, pcolormesh would
    color the phantom int rank from the legacy `argsort(argsort(x))` producer.
    """
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    return cmap

def _mask_phantom_cells(display_ranks: np.ndarray, display_bools: np.ndarray | None) -> np.ndarray:
    """Return a float copy of `display_ranks` with non-participating cells NaN'd.

    `display_bools` has the same shape as `display_ranks`; True = participating
    event-channel, False = non-participating (legacy producer wrote a phantom
    int rank there). If `display_bools` is None, returns the input unchanged.
    Shape mismatch is a programmer error and is asserted.
    """
    if display_bools is None:
        return np.asarray(display_ranks, dtype=float)
    out = np.asarray(display_ranks, dtype=float).copy()
    mask = np.asarray(display_bools, dtype=bool)
    if mask.shape != out.shape:
        raise ValueError(
            f"display_bools shape {mask.shape} != display_ranks shape {out.shape}"
        )
    out[~mask] = np.nan
    return out

def _plot_rank_heatmap(
    ax: plt.Axes,
    display_ranks: np.ndarray,
    channel_names: List[str],
    title: str,
    show_ylabels: bool = True,
    display_bools: np.ndarray | None = None,
    ytick_fontsize: float | None = None,
    title_fontsize: float = 18,
    xtick_fontsize: float = 14,
) -> Any:
    n_ch = display_ranks.shape[0]
    masked = _mask_phantom_cells(display_ranks, display_bools)
    im = ax.pcolormesh(masked, rasterized=True, cmap=_viridis_with_lightgray_bad())
    if ytick_fontsize is None:
        ytick_fontsize = 14 if n_ch > 24 else 16
    ax.set_yticks(np.arange(n_ch) + 0.5)
    ax.set_yticklabels(
        channel_names if show_ylabels else [],
        fontsize=ytick_fontsize,
    )
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    return im

def _plot_cluster_rank_fig4(
    ax: plt.Axes,
    ranks: np.ndarray,
    bools: np.ndarray,
    valid_events: np.ndarray,
    labels: np.ndarray,
    channel_order: np.ndarray,
    channel_names: List[str],
    title: str,
    *,
    ytick_fontsize: float | None = None,
    label_fontsize: float = 16,
    title_fontsize: float = 18,
    xtick_fontsize: float = 14,
    legend_fontsize: float = 14,
    show_legend: bool = True,
    invert_yaxis: bool = True,
    show_ylabels: bool = True,
    marker_size: float = 6.0,
    line_colors: List[str] | None = None,
    label_names: List[str] | None = None,
) -> None:
    """Per-cluster mean rank line + shaded mean +/- std band on fixed channel order."""
    n_ch = len(channel_order)
    ordered_names = [channel_names[idx] for idx in channel_order]
    if ytick_fontsize is None:
        ytick_fontsize = 14 if n_ch > 24 else 16
    unique_k = np.unique(labels)
    n_k = len(unique_k)
    _base_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    if line_colors is None:
        line_colors = [_base_colors[i % len(_base_colors)] for i in range(n_k)]
    elif len(line_colors) < n_k:
        raise ValueError("line_colors must cover every displayed cluster")
    y_pos = np.arange(n_ch, dtype=float)

    for ki, cid in enumerate(unique_k):
        mask_cluster = labels == cid
        eidx = valid_events[mask_cluster]
        means = np.full(n_ch, np.nan)
        stds = np.full(n_ch, np.nan)
        for ci_plot, ci_raw in enumerate(channel_order):
            vals = np.asarray(ranks[ci_raw, eidx], dtype=float)
            bmask = np.asarray(bools[ci_raw, eidx], dtype=bool)
            vals = vals[bmask & np.isfinite(vals)]
            if vals.size > 0:
                means[ci_plot] = np.mean(vals)
                stds[ci_plot] = np.std(vals)

        valid = np.isfinite(means)
        ax.fill_betweenx(
            y_pos[valid],
            (means - stds)[valid], (means + stds)[valid],
            color=line_colors[ki], alpha=0.15, linewidth=0,
        )
        ax.plot(
            means[valid], y_pos[valid],
            "-o", color=line_colors[ki], lw=2.5, ms=marker_size, zorder=10,
            label=(
                f"{label_names[ki]} (n={int(mask_cluster.sum())})"
                if label_names is not None and ki < len(label_names)
                else f"C{int(cid)} (n={int(mask_cluster.sum())})"
            ),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_names if show_ylabels else [], fontsize=ytick_fontsize)
    if not show_ylabels:
        ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_ylim(-0.5, n_ch - 0.5)
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_xlabel("Rank", fontsize=label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim(-0.5, n_ch - 0.5)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    if show_legend:
        ax.legend(fontsize=legend_fontsize, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=n_k)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
