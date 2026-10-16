"""Original Fig3C field painter reused by the Fig5 onset-field panel."""
from __future__ import annotations
from typing import Mapping,Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from analysis.display_field import _smooth_rank_field_mm
AXIS_LABEL_FONTSIZE=14.5
TICK_LABEL_FONTSIZE=11.5
FIELD_TITLE_FONTSIZE=14.

def _event_field(fz, values, support):
    """Call the established Topic 3/5 physical-mm field smoother without reimplementation."""
    pts = np.asarray(fz["points_mm"], float)
    return _smooth_rank_field_mm(
        pts[:, 0], pts[:, 1], np.clip(np.asarray(values, float), 0.0, None),
        np.asarray(support, float), fz["display_xlim_mm"], fz["display_ylim_mm"],
        fz["display_sigma_mm"],
    )

def _normalize_minmax(values: Sequence[float]) -> np.ndarray:
    """Continuous min-max display transform; never rank or sign-flip values."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("cannot normalize an all-missing vector")
    lo = float(np.nanmin(values[finite]))
    hi = float(np.nanmax(values[finite]))
    if not hi > lo:
        raise ValueError("cannot normalize a constant vector")
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out

def _draw_field(
    ax: plt.Axes,
    fz: Mapping[str, object],
    values: np.ndarray,
    support: np.ndarray,
    *,
    cmap: str,
    colorbar_values: np.ndarray,
    title: str,
    title_color: str,
    show_y: bool,
) -> ScalarMappable:
    x_grid, y_grid, field, _, _ = _event_field(fz, values, support)
    points = np.asarray(fz["points_mm"], dtype=float)
    ax.imshow(
        field,
        origin="lower",
        extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
        aspect="equal",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    finite = np.isfinite(values)
    ax.scatter(
        points[finite, 0],
        points[finite, 1],
        c=values[finite],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=38,
        edgecolors="white",
        linewidths=0.85,
        zorder=3,
    )
    ax.set_xlim(float(x_grid.min()), float(x_grid.max()))
    ax.set_ylim(float(y_grid.min()), float(y_grid.max()))
    ax.set_title(
        title,
        fontsize=FIELD_TITLE_FONTSIZE,
        pad=7,
        color=title_color,
        fontweight="bold",
    )
    ax.set_xlabel("shared TA axis (mm)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
    ax.tick_params(
        axis="both", labelsize=TICK_LABEL_FONTSIZE, length=2.8, pad=2.5
    )
    if show_y:
        ax.set_ylabel("Y (mm)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=5)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    colorbar_values = np.asarray(colorbar_values, dtype=float)
    finite_colorbar = colorbar_values[np.isfinite(colorbar_values)]
    if finite_colorbar.size == 0:
        raise ValueError("colorbar values are all missing")
    return ScalarMappable(
        Normalize(float(finite_colorbar.min()), float(finite_colorbar.max())),
        cmap=cmap,
    )

