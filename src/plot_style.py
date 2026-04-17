"""Shared publication-quality plotting style for all topics.

Provides unified semantic colors, panel styling, and reusable plot helpers
following Nature/Science conventions.  Every plotting script in this repo
should ``from src.plot_style import ...`` instead of defining its own
style constants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# =========================================================================
# Semantic color palette — Morandi-inspired (muted, low-saturation tones)
# =========================================================================

# Datasets
COL_YUQUAN = "#6F8FA8"        # Morandi blue
COL_EPILEPSIAE = "#B07A6E"    # Morandi terracotta

# Significance
COL_SIG = "#A35E48"           # Morandi rust
COL_NONSIG = "#9B9B9B"        # neutral gray

# SOZ vs non-SOZ
COL_SOZ = "#C49B92"           # Morandi rose
COL_NONSOZ = "#8C9AA3"        # Morandi slate

# Empirical / analytic / surrogate triplet
COL_EMPIRICAL = "#6F8FA8"     # blue (matches Yuquan)
COL_ANALYTIC = "#A35E48"      # rust
COL_SURROGATE = "#9DAA90"     # Morandi sage

COL_DETRENDED = "#C9A86A"     # Morandi mustard

# Day / night
COL_DAY = "#D8C7A8"           # Morandi cream
COL_NIGHT = "#7E6E84"         # Morandi plum

# Misc
COL_NEUTRAL = "#A89B8A"       # Morandi dust
COL_OSCILLATOR = "#A67C5A"    # warm brown (toy oscillator)
COL_REFRACTORY = "#6F8FA8"    # blue (toy refractory)

# Convenience alias kept for backward compat with Topic 1 scripts
COL_YQ = COL_YUQUAN
COL_EPI = COL_EPILEPSIAE

# =========================================================================
# Font-size presets (PPT-friendly)
# =========================================================================

FS_TICK = 14
FS_LABEL = 14
FS_TITLE = 16
FS_PANEL_LETTER = 24
FS_SUPTITLE = 18

DPI_PUB = 300

# =========================================================================
# Panel styling
# =========================================================================


def style_panel(
    ax: plt.Axes,
    label: str = "",
    label_x: float = -0.18,
    label_y: float = 1.12,
) -> None:
    """Apply Nature/Science publication conventions to a single axis.

    - Remove top and right spines
    - Thicken left and bottom spines (1.4 pt)
    - Enlarge tick labels
    - Optionally place a bold panel letter (a, b, c ...) at top-left.
      `label_x`/`label_y` allow per-panel tuning when ticks or annotations
      collide with the default position.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(labelsize=FS_TICK, width=1.4, length=6)
    if label:
        ax.text(
            label_x, label_y, label,
            transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER, fontweight="bold",
            va="bottom", ha="right", fontfamily="sans-serif",
        )


# =========================================================================
# Significance brackets
# =========================================================================


def add_significance_bracket(
    ax: plt.Axes,
    x0: float, x1: float, y: float,
    p: float, dy: float = 0.02,
    fontsize: int = 18,
) -> None:
    """Draw a bracket with significance stars between two x positions."""
    if p < 0.001:
        txt = "***"
    elif p < 0.01:
        txt = "**"
    elif p < 0.05:
        txt = "*"
    else:
        txt = "n.s."
    ax.plot([x0, x0, x1, x1], [y, y + dy, y + dy, y], lw=1.5, color="black")
    ax.text(
        (x0 + x1) / 2, y + dy * 1.3, txt,
        ha="center", va="bottom", fontsize=fontsize, fontweight="bold",
    )


# =========================================================================
# Violin + scatter distribution plot
# =========================================================================


def violin_with_scatter(
    ax: plt.Axes, vals: np.ndarray, pos: float,
    color: str, width: float = 0.45, scatter_size: float = 55,
    alpha_body: float = 0.2, rng_seed: int = 42,
) -> None:
    """Violin + boxplot + jittered scatter at a given position."""
    if vals.size == 0:
        return
    vp = ax.violinplot(
        vals, positions=[pos], widths=width,
        bw_method="silverman",
        showmeans=False, showmedians=False, showextrema=False,
    )
    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(alpha_body)
        pc.set_edgecolor(color)
        pc.set_linewidth(1.2)
    bp = ax.boxplot(
        vals, positions=[pos], widths=width * 0.5,
        patch_artist=True, showfliers=False, showcaps=False,
        medianprops=dict(linewidth=2.5, color="black"),
        whiskerprops=dict(linewidth=1.2, color="black"),
        zorder=2,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.0)
    rng = np.random.default_rng(rng_seed)
    jit = np.full(vals.size, pos) + rng.normal(0, 0.04, vals.size)
    ax.scatter(
        jit, vals, s=scatter_size, c=color,
        edgecolors="white", linewidths=0.8, zorder=3, alpha=0.85,
    )


# =========================================================================
# Savefig wrapper
# =========================================================================


def savefig_pub(fig: plt.Figure, path, dpi: int = DPI_PUB) -> Path:
    """Save figure with publication defaults (white bg, tight bbox, high dpi)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# =========================================================================
# Utility helpers
# =========================================================================


def dataset_color(dataset: str) -> str:
    """Return the semantic color for a dataset name."""
    if "yuquan" in dataset.lower():
        return COL_YUQUAN
    return COL_EPILEPSIAE


def format_subject_label(dataset: str, subject: str, max_len: int = 12) -> str:
    """Short label for axis tick marks."""
    short = subject[:max_len] if len(subject) > max_len else subject
    prefix = "Y" if "yuquan" in dataset.lower() else "E"
    return f"{prefix}:{short}"


def new_figure(nrows: int = 1, ncols: int = 1, *,
               figsize: Optional[tuple] = None,
               **kwargs) -> tuple:
    """Create a figure with white background and sensible defaults."""
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    fig.patch.set_facecolor("white")
    return fig, axes
