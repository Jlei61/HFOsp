"""Shared typography for complete-width supplementary figures.

The main-figure typography contract distinguishes semantic hierarchy from
panel density.  Supplementary figures are exported as complete layouts, so a
single producer-side scale is used here to prevent figure-to-figure drift.
"""
from __future__ import annotations

import matplotlib as mpl


PANEL_LETTER_SIZE = 11.0
AXIS_LABEL_SIZE = 8.2
TICK_LABEL_SIZE = 7.3
LEGEND_SIZE = 7.3
ANNOTATION_SIZE = 7.0
SIGNIFICANCE_SIZE = 9.0
IDENTITY_SIZE = 8.0


def apply_supplementary_rcparams() -> None:
    """Apply the common complete-layout typography and line contract."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": TICK_LABEL_SIZE,
            "font.weight": "normal",
            "axes.titlesize": IDENTITY_SIZE,
            "axes.titleweight": "normal",
            "axes.labelsize": AXIS_LABEL_SIZE,
            "axes.labelweight": "normal",
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "legend.title_fontsize": LEGEND_SIZE,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def normalize_axis_text(axis: mpl.axes.Axes, *, scale: float = 1.0) -> None:
    """Normalize axis labels, ticks and legends without changing semantics.

    ``scale`` is reserved for raster compositors whose source panels are
    reduced again inside a complete-layout canvas.
    """
    axis_size = AXIS_LABEL_SIZE * scale
    tick_size = TICK_LABEL_SIZE * scale
    legend_size = LEGEND_SIZE * scale
    axis.xaxis.label.set_fontsize(axis_size)
    axis.yaxis.label.set_fontsize(axis_size)
    axis.xaxis.label.set_fontweight("normal")
    axis.yaxis.label.set_fontweight("normal")
    axis.tick_params(axis="both", labelsize=tick_size)
    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
        tick.set_fontweight("normal")
    legend = axis.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(legend_size)
            text.set_fontweight("normal")
        title = legend.get_title()
        if title is not None:
            title.set_fontsize(legend_size)
            title.set_fontweight("normal")


__all__ = [
    "ANNOTATION_SIZE",
    "AXIS_LABEL_SIZE",
    "IDENTITY_SIZE",
    "LEGEND_SIZE",
    "PANEL_LETTER_SIZE",
    "SIGNIFICANCE_SIZE",
    "TICK_LABEL_SIZE",
    "apply_supplementary_rcparams",
    "normalize_axis_text",
]
