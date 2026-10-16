"""Paper producer functions copied locally; input/output adapters are separate.
Source: scripts/plot_topic5_interictal_event_envelope_field.py
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
CMAP=LinearSegmentedColormap.from_list('fig2c_soft_teal_navy', ['#f7f9f8','#dce9e4','#b8d2ca','#78aaa1','#3f747d','#314766'])
FIELD_DISPLAY_NORM=PowerNorm(gamma=.5,vmin=0,vmax=1)
FRAME_TITLE_SIZE=14
FIELD_TICK_LABELSIZE=11
TICK_LENGTH=2.5
AXIS_LABELSIZE=16


def _static_frame_relative_values(raw_values, participant):
    """Normalize one static frame by its three strongest participating contacts.

    Static small multiples are intended to show where the envelope is concentrated.  A single
    complete-window q99 makes low-amplitude but direction-qualified late frames nearly white after
    the fixed 6-mm smoother.  The top-three mean is a robust frame-local reference: it does not let
    one contact set the scale, and the full-field visibility gate prevents quiet noise frames from
    being amplified into the paper panel.
    """
    raw = np.clip(np.asarray(raw_values, float), 0.0, None)
    part = np.asarray(participant, bool)
    if raw.ndim != 1 or part.shape != raw.shape:
        raise ValueError("static frame values and participant mask must be aligned vectors")
    if int(part.sum()) < 3:
        raise ValueError("static frame normalization requires at least three participants")
    scale = float(np.mean(np.sort(raw[part])[-3:]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("static frame top-three participant scale must be positive and finite")
    return np.clip(raw / scale, 0.0, 1.0), scale

def _panel(ax, grid, T, pts, cvals, part, vmax, title, *, show_y=False, show_x=False):
    """Established Topic 3/5 physical-mm field grammar with a shared A/B frame."""
    X, Y = grid
    ax.imshow(T, origin="lower",
              extent=[X.min(), X.max(), Y.min(), Y.max()], aspect="equal", cmap=CMAP,
              norm=FIELD_DISPLAY_NORM, interpolation="bilinear")
    ax.scatter(pts[part, 0], pts[part, 1], c=np.clip(cvals[part], 0, vmax), cmap=CMAP,
               norm=FIELD_DISPLAY_NORM, s=34, edgecolors="#34434a", linewidths=0.9, zorder=3)
    ax.scatter(pts[~part, 0], pts[~part, 1], facecolors="none", edgecolors="0.62", s=22,
               linewidths=0.7, zorder=3)
    ax.set_title(title, fontsize=FRAME_TITLE_SIZE)
    ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
    ax.tick_params(axis="both", labelsize=FIELD_TICK_LABELSIZE, length=TICK_LENGTH)
    if not show_y:
        ax.set_yticklabels([])
    if show_x:
        ax.set_xlabel("shared TA axis (mm)", fontsize=AXIS_LABELSIZE)

from analysis.event_spectrogram import full_extent_edges
FIG1A_CMAP="coolwarm"
CENTROID_FACE='#ffb000'
TEMPLATE_COLORS={'TA':'#B2182B','TB':'#2166AC'}
READOUT_BOX_ASPECT=1.18
CONTACT_TICK_LABELSIZE=9.5
READOUT_TICK_LABELSIZE=11
TEMPLATE_LABEL_SIZE=14

def _readout(
    ax, e, fz, order, xlim, _stats, title=None, *, template,
    show_xlabel=True, row_label=None,
):
    """Exact Fig1a stacked spectrogram + dominant-enhancement centroid trajectory."""
    sel = [c for c in order if e["usable"][c]]
    if not sel:
        raise ValueError("readout has no participating contact with a finite Fig1a centroid")
    n_freq = len(e["spec_freq_hz"])
    shown = np.concatenate([e["spec"][c] for c in sel], axis=0)
    t_edges = full_extent_edges(
        np.asarray(e["spec_t_ms"], float), float(e["tile_lo_ms"]), float(e["tile_hi_ms"]),
    )
    y_edges = np.arange(len(sel) * n_freq + 1, dtype=float)
    im = ax.pcolormesh(t_edges, y_edges, shown, cmap=FIG1A_CMAP, vmin=0.0, vmax=1.0,
                       shading="flat", rasterized=True)
    for k in range(1, len(sel)):
        ax.axhline(k * n_freq, color="#c7c7c7", linewidth=0.45, linestyle="--")
    xs = np.asarray([e["centroid_ms"][c] for c in sel], float)
    ys = np.asarray(
        [k * n_freq + e["centroid_freq_index"][c] + 0.5 for k, c in enumerate(sel)], float,
    )
    semantic = TEMPLATE_COLORS[template]
    ax.plot(xs, ys, "-", color=semantic, lw=1.0, alpha=0.95, zorder=4)
    ax.scatter(xs, ys, s=13, facecolors=CENTROID_FACE, edgecolors=semantic,
               linewidth=0.35, zorder=5)
    ax.set_xlim(*xlim)
    ax.set_ylim(len(sel) * n_freq, 0.0)
    ax.set_box_aspect(READOUT_BOX_ASPECT)
    ax.margins(x=0.0)
    ax.set_yticks((np.arange(len(sel)) + 0.5) * n_freq)
    ax.set_yticklabels([fz["names"][c] for c in sel], fontsize=CONTACT_TICK_LABELSIZE)
    if show_xlabel:
        ax.set_xlabel("time (ms)", fontsize=AXIS_LABELSIZE)
    if title:
        ax.set_title(
            title, fontsize=TEMPLATE_LABEL_SIZE, color=semantic, fontweight="bold",
            x=0.96, ha="right", pad=3,
        )
    if row_label:
        ax.set_ylabel(
            row_label, fontsize=TEMPLATE_LABEL_SIZE,
            color=semantic, fontweight="bold",
        )
    ax.tick_params(axis="x", labelsize=READOUT_TICK_LABELSIZE, length=TICK_LENGTH)
    ax.tick_params(axis="y", labelsize=CONTACT_TICK_LABELSIZE, length=0)
    return im, sel, xs, ys
