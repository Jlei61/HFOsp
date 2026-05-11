"""Shared publication-quality plotting style for all topics.

Provides unified semantic colors, panel styling, and reusable plot helpers
following Nature/Science conventions.  Every plotting script in this repo
should ``from src.plot_style import ...`` instead of defining its own
style constants.

----------------------------------------------------------------------------
Convention: cluster-rank / endpoint-anchoring small multiples
----------------------------------------------------------------------------

When rendering per-subject small-multiples that show **per-cluster mean
rank profiles with subset-channel highlighting** (e.g. PR-6 swap-node
figures, propagation cluster_rank_fig4 style), follow these rules. They
were settled iteratively during the PR-6 swap-cohort supplementary work
(2026-05-11); ``scripts/plot_pr6_swap_cluster_rank_multiples.py`` is the
reference implementation.

1.  **No figure-level suptitle.** Prose narrative lives in the figure
    ``README.md`` / caption, not in the figure canvas. Per-panel titles
    only.

2.  **Per-panel title.** Two-line maximum, ``pad=10`` so the title lifts
    clearly above the axes top spine. Format:
      line 1 = ``"{D}:{subject}"`` (D = first letter of dataset uppercased)
      line 2 = compact stats only — ``"p=X.XXX   n_swap=N"`` style;
               **never** include redundant labels like ``swap_score=X``,
               ``k=X``, ``cluster_id=X`` that aren't needed by an
               out-of-context reader.

3.  **Tight axes** for naturally-bounded rank/index variables:
      ``ax.set_xlim(0, n_ch - 1)``
      ``ax.set_ylim(0, n_ch - 1); ax.invert_yaxis()``
    First channel flush with the top edge, last channel flush with the
    x-axis, x starts exactly at 0. No decorative whitespace.

4.  **Highlight pattern for "important" channels** (swap nodes, source/
    sink endpoints, etc.):
      marker:  large filled star (``marker="*", s=180``) + black edge,
               full saturation, ``zorder=12``, ``clip_on=False`` so
               markers at axis boundary aren't clipped.
      y-tick:  bold, orange ``COL_SWAP_LABEL = "#D2691E"`` (Morandi rust).
      Non-highlight channels: small faded circle (``s=28``, alpha≈0.30),
      muted gray y-tick (``#888888``).

5.  **Two-cluster T0/T1 conventions:**
      T0 (forward / cluster_id_a) = blue ``"#1f77b4"``
      T1 (reverse / cluster_id_b) = red  ``"#d62728"``
      Shaded mean ± 1 SD band (``alpha=0.12``) + solid line (``lw=1.8``).

6.  **Legend in a dedicated right column.** No bottom-row legend that
    cramps panel layout. Reserve 3.0–3.8 inches on the figure right
    (more for figures with many or wider panels); place ``fig.legend``
    with ``loc="center right"``, ``bbox_to_anchor=(0.995, 0.5)``,
    ``ncol=1``. Single shared legend, never per-panel.

7.  **Layout sizing:** ``gridspec`` ``right = panel_block_w / total_w``
    (panel area ends before the reserved legend column);
    ``top ≈ 0.91``, ``hspace ≈ 0.45–0.50`` to give the lifted titles
    breathing room.

8.  **Save both PNG (``dpi=200``) and PDF** from the same script call so
    downstream consumers always have a vector copy.

These rules are *additive* to the umbrella paper-grade-self-contained
contract (no internal §X / PR-N / cluster_id terminology in legends or
axis labels; render → eyeball → fix → re-render before commit).
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

# Highlight color for "subset-of-interest" y-tick labels in small-multiples
# (e.g. swap nodes, endpoint channels). Pairs with the COL_SIG marker family.
COL_SWAP_LABEL = "#D2691E"    # vivid Morandi rust, bold-tick highlight

# Two-cluster T0/T1 line colors used in cluster_rank_fig4-style panels
COL_CLUSTER_T0 = "#1f77b4"    # forward / cluster_id_a — blue
COL_CLUSTER_T1 = "#d62728"    # reverse / cluster_id_b — red

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
