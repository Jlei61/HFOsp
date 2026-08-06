#!/usr/bin/env python3
"""Paper-ready E1146 MZ early-field bridge.

This renderer deliberately reuses the accepted legacy Figure-5 grammar:

* one continuous virtual-SEEG strip from the native z+m trajectory, containing
  a representative returning interictal-like burst and a single early-onset
  marker at 120 ms before the operational t120 reference;
* one compact z+m slow-state path beside the readout; the lower row contains the
  exact event-order field, early-onset energy field, and two frozen-q rate-field
  leading-mode context panels (baseline and early-onset -120 ms).

The displayed event is selected without consulting the target energy field.  It
is the last eligible native event in the majority slow-off direction before
``t_recruit``.  Both fields are smooth contact-readout projections on the fixed
E1146 plane, with the complete virtual montage overlaid.  Grey grain is the fixed
E-neuron substrate geometry only; no local recruitment is implied because the
local-participation audit is incomplete.

Plotting/readout only.  No SNN simulation is rerun.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_mz_early_field_bridge import (  # noqa: E402
    associate,
    burst_envelope,
    event_contact_timing,
)


# V2 (z+m, tau_adp=500): read the V2 output dir; NEVER the V1 z-only dir (task §12).
BRIDGE = ROOT / "results/topic4_sef_hfo/mz_early_field_bridge_v2_zm_tau500"
SEED_ROOT = BRIDGE / "per_seed/seed1"
FIGDIR = ROOT / "results/paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures"
SUBSTRATE = (
    ROOT
    / "results/topic4_sef_hfo/mz_slowvars/readout_ready/"
    / "readout_zA_q75_tz10000_seed1.npz"        # E-neuron geometry only (candidate-independent, seed1)
)
STEM = "fig_mz_early_bridge_v2_zm_tau500"
DISPLAY_DPI = 300
FIELD_GRID_N = 220
FIELD_SIGMA_MM = 3.0
TRACE_BAND_HZ = (30.0, 80.0)
TRACE_OFF = 1.48
TRACE_GAIN = 0.68
TRACE_SCALE_PERCENTILE = 95.0
INTERICTAL_SHADE = "#6F9FD8"
TA_COL, TB_COL = "#B2182B", "#2166AC"
SHAFT_COLS = ["#e8743b", "#1f9e9e", "#7b5cb8", "#3b7a3b"]
PRIMARY_WK = "early_0_50_ms"
EARLY_ONSET_OFFSET_MS = 120.0
SLOW_STATE_PANEL = SEED_ROOT / "slow_state_panel_seed1.json"
SLOW_STATE_COLOR = "#334E73"
MODE_CONTEXT = BRIDGE / "mode_context/frozen_q_rate_field_mode_pair.json"
MODE_CMAP = "magma"


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _shaft(name):
    match = re.match(r"[A-Za-z]+", str(name))
    return match.group(0) if match else str(name)


def _shaft_color(name, shafts):
    return SHAFT_COLS[shafts.index(_shaft(name)) % len(SHAFT_COLS)]


def _project(coordinates, axis, center):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    transverse = np.array([-axis[1], axis[0]])
    centered = np.asarray(coordinates, float) - np.asarray(center, float)[None, :]
    return np.column_stack((centered @ axis, centered @ transverse))


def _normalize_minmax(values):
    values = np.asarray(values, float)
    out = np.full(values.shape, np.nan, float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        raise ValueError("field requires at least two finite values")
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if not hi > lo:
        raise ValueError("field values are constant")
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def _close_short_gaps(mask, max_gap):
    out = np.asarray(mask, bool).copy()
    i, n = 0, out.size
    while i < n:
        if out[i]:
            i += 1
            continue
        j = i
        while j < n and not out[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= int(max_gap):
            out[i:j] = True
        i = j
    return out


def _components(mask):
    mask = np.asarray(mask, bool)
    out = []
    i, n = 0, mask.size
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def _select_display_event(slowoff, native, native_json):
    """Select a representative native event without looking at target energy."""
    times = np.asarray(native["times"], float)
    r20 = np.asarray(native["r20"], float)
    onset = native_json["onset"]
    t_recruit = float(onset["t_recruit_ms"])
    theta = float(onset["theta_recruit"])

    supra = _close_short_gaps(r20 > theta, max_gap=5)
    candidates = [(float(a), float(b)) for a, b in _components(supra) if float(b) < t_recruit]
    if not candidates:
        raise RuntimeError("no complete pre-t_recruit native event candidates")

    event_dirs = [str(x) for x in slowoff["event_dir"]]
    counts = {direction: event_dirs.count(direction) for direction in ("A_to_B", "B_to_A")}
    majority_direction = max(counts, key=counts.get)

    env = burst_envelope(native["lfp_trace"], times)
    scored = []
    for index, (t_on, t_off) in enumerate(candidates):
        next_on = candidates[index + 1][0] if index + 1 < len(candidates) else t_recruit
        timing = event_contact_timing(
            env,
            times,
            {"t_on": t_on, "t_off": t_off},
            next_event_t_on=next_on,
            record_end_ms=float(times[-1]),
            quiet_med=slowoff["qmed"],
            quiet_mad=slowoff["qmad"],
            contact_axis=slowoff["contact_axis"],
        )
        if timing.eligible and timing.direction == majority_direction:
            scored.append((t_on, t_off, next_on, timing))
    if not scored:
        raise RuntimeError(f"no eligible native event in majority direction {majority_direction}")

    t_on, t_off, next_on, timing = scored[-1]
    readout_end = min(t_off + 40.0, next_on, float(times[-1]))
    return {
        "t_on_ms": t_on,
        "t_off_ms": t_off,
        "readout_end_ms": readout_end,
        "direction": majority_direction,
        "timing": timing,
        "theta_recruit": theta,
        "selection_rule": (
            "last eligible native event in the majority slow-off direction before t_recruit; "
            "selection does not use target early energy"
        ),
    }


def _signed_burst(lfp, times, scale_mask):
    times = np.asarray(times, float)
    dt_ms = float(np.median(np.diff(times)))
    sos = butter(4, TRACE_BAND_HZ, btype="bandpass", fs=1000.0 / dt_ms, output="sos")
    burst = sosfiltfilt(sos, np.asarray(lfp, float), axis=0)
    scale = np.percentile(np.abs(burst[np.asarray(scale_mask, bool)]), TRACE_SCALE_PERCENTILE, axis=0)
    finite_positive = scale[np.isfinite(scale) & (scale > 1e-12)]
    if finite_positive.size == 0:
        raise ValueError("pre-recruitment burst trace is constant")
    scale = np.maximum(scale, 0.15 * float(np.median(finite_positive)))
    return TRACE_GAIN * burst / scale[None, :]


def _plot_continuous_trace(ax, native, names, display_event, t_recruit, t120):
    times_abs = np.asarray(native["times"], float)
    event_on = float(display_event["t_on_ms"])
    display_start = float(np.floor((event_on - 450.0) / 50.0) * 50.0)
    display_end = min(float(times_abs[-1]), float(t120 + 110.0))
    keep = (times_abs >= display_start) & (times_abs <= display_end)
    times = times_abs[keep] - display_start
    trace = _signed_burst(native["lfp_trace"], times_abs, times_abs < float(t_recruit))[keep]
    shafts = sorted(set(_shaft(name) for name in names))
    y = np.arange(len(names), dtype=float) * TRACE_OFF

    event_start = float(display_event["t_on_ms"] - display_start)
    event_end = float(display_event["readout_end_ms"] - display_start)
    energy_window_start = float(t_recruit - display_start)
    energy_window_end = float(t_recruit + 50.0 - display_start)
    early_onset_absolute = float(t120 - EARLY_ONSET_OFFSET_MS)
    onset_x = float(early_onset_absolute - display_start)
    ax.axvspan(event_start, event_end, color=INTERICTAL_SHADE, alpha=0.10, lw=0, zorder=0)
    ax.axvline(onset_x, color="crimson", lw=1.6, ls="--", alpha=0.95, zorder=8)
    for ci, name in enumerate(names):
        ax.plot(
            times,
            trace[:, ci] + y[ci],
            color=_shaft_color(name, shafts),
            lw=0.95,
            alpha=0.94,
            zorder=3,
            clip_on=True,
        )
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_ylim(-0.55, y[-1] + 1.75)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.8)
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(_shaft_color(name, shafts))
    ax.set_xlabel("Simulation time (ms)", fontsize=11.0)
    ax.set_ylabel("Virtual-SEEG (30–80 Hz)", fontsize=11.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.2, length=3)
    ax.tick_params(axis="y", length=2.5)
    return {
        "display_start_absolute_ms": display_start,
        "display_end_absolute_ms": display_end,
        "display_event_start_ms": event_start,
        "display_event_end_ms": event_end,
        "energy_window_start_ms": energy_window_start,
        "energy_window_end_ms": energy_window_end,
        "early_onset_absolute_ms": early_onset_absolute,
        "early_onset_display_ms": onset_x,
        "operational_t120_absolute_ms": float(t120),
        "operational_t120_display_ms": float(t120 - display_start),
    }


def _smooth_contact_field(points, values, xlim, ylim, sigma_mm):
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("contact field requires at least two finite values")
    x_grid = np.linspace(float(xlim[0]), float(xlim[1]), FIELD_GRID_N)
    y_grid = np.linspace(float(ylim[0]), float(ylim[1]), FIELD_GRID_N)
    X, Y = np.meshgrid(x_grid, y_grid)
    d2 = (X[..., None] - points[valid, 0]) ** 2 + (Y[..., None] - points[valid, 1]) ** 2
    weights = np.exp(-0.5 * d2 / max(float(sigma_mm), 1e-6) ** 2)
    weight_sum = np.sum(weights, axis=-1)
    field = np.sum(weights * values[valid], axis=-1) / np.maximum(weight_sum, 1e-12)
    confidence = 1.0 - np.exp(-weight_sum)
    confidence /= max(float(np.max(confidence)), 1e-12)
    return X, Y, np.clip(field, 0.0, 1.0), np.clip(confidence, 0.0, 1.0)


def _draw_field(
    ax,
    points,
    values_display,
    values_colorbar,
    xlim,
    ylim,
    *,
    cmap,
    title,
    title_color,
    show_y,
    substrate_points,
):
    X, Y, field, confidence = _smooth_contact_field(
        points, values_display, xlim, ylim, FIELD_SIGMA_MM
    )
    rgba = plt.get_cmap(cmap)(field)
    rgba[..., 3] = 0.72 * confidence
    ax.imshow(
        rgba,
        origin="lower",
        extent=[X.min(), X.max(), Y.min(), Y.max()],
        aspect="equal",
        interpolation="bilinear",
        zorder=1,
    )
    substrate_points = np.asarray(substrate_points, float)
    ax.scatter(
        substrate_points[:, 0],
        substrate_points[:, 1],
        s=1.0,
        c="0.70",
        alpha=0.28,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    values_display = np.asarray(values_display, float)
    valid = np.isfinite(values_display)
    ax.scatter(
        points[~valid, 0],
        points[~valid, 1],
        s=50,
        facecolors="white",
        edgecolors="black",
        linewidths=0.95,
        zorder=5,
    )
    ax.scatter(
        points[valid, 0],
        points[valid, 1],
        c=values_display[valid],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=50,
        edgecolors="black",
        linewidths=0.95,
        zorder=6,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=13.0, pad=7, color=title_color, fontweight="bold")
    ax.set_xlabel("TA shared axis (mm)", fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.2, length=2.5)
    if show_y:
        ax.set_ylabel("y (mm)", fontsize=11.0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    raw = np.asarray(values_colorbar, float)
    finite_raw = raw[np.isfinite(raw)]
    return ScalarMappable(Normalize(float(finite_raw.min()), float(finite_raw.max())), cmap=cmap)


def _draw_mode_field(ax, field, xlim, ylim, *, title, vmax, show_y=False):
    """Render the accepted frozen-q rate-field loading on the registered plane."""
    image = ax.imshow(
        # Producer stores [long-axis, transverse]; imshow expects [row=y, column=x].
        np.asarray(field, float).T,
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=MODE_CMAP,
        vmin=0.0,
        vmax=float(vmax),
        interpolation="nearest",
        aspect="equal",
        rasterized=True,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=11.4, pad=6, fontweight="bold")
    ax.set_xlabel("TA shared axis (mm)", fontsize=10.0, labelpad=4)
    ax.set_xticks([-10, 0, 10])
    ax.set_yticks([-10, 0, 10])
    ax.tick_params(axis="both", labelsize=7.8, length=2.2)
    if show_y:
        ax.set_ylabel("y (mm)", fontsize=9.2, labelpad=3)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return image


def _draw_spatial_probe_schematic(ax):
    """Add a compact, explicitly schematic local E-rate perturbation glyph.

    The displayed mode is obtained from the frozen Jacobian, not by applying
    this particular kick.  The inset therefore communicates the perturbation
    concept only and must not be interpreted as an additional simulation.
    """
    inset = ax.inset_axes([0.055, 0.64, 0.245, 0.245], zorder=9)
    coord = np.linspace(-1.0, 1.0, 81)
    xx, yy = np.meshgrid(coord, coord)
    probe = np.exp(-0.5 * (xx**2 + yy**2) / 0.23**2)
    inset.imshow(
        probe,
        origin="lower",
        cmap="Greys",
        vmin=0.0,
        vmax=1.0,
        interpolation="bilinear",
    )
    inset.contour(probe, levels=[0.25, 0.55], colors="white", linewidths=0.45, alpha=0.9)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_title("spatial probe", fontsize=7.1, color="white", pad=2.0, fontweight="bold")
    for spine in inset.spines.values():
        spine.set_color("white")
        spine.set_linewidth(0.8)
    ax.annotate(
        "",
        xy=(0.42, 0.762),
        xytext=(0.315, 0.762),
        xycoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="white",
            lw=1.25,
            mutation_scale=9,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=10,
    )


def _legend_handles():
    return [
        Line2D([0], [0], color="crimson", lw=1.6, ls="--", label="early onset"),
        Patch(
            facecolor=INTERICTAL_SHADE,
            alpha=0.12,
            edgecolor="none",
            label="TB sample event",
        ),
    ]


def _add_path_arrow(ax, x, y, fraction):
    index = int(np.clip(round(fraction * (len(x) - 1)), 1, len(x) - 2))
    half_width = max(3, len(x) // 110)
    left = max(0, index - half_width)
    right = min(len(x) - 1, index + half_width)
    ax.annotate(
        "",
        xy=(x[right], y[right]),
        xytext=(x[left], y[left]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=SLOW_STATE_COLOR,
            lw=1.6,
            mutation_scale=10,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=7,
    )


def _plot_slow_state_path(ax, slow_state):
    """Compact paper panel: one natural z+m path crossing its operational boundary."""
    times = np.asarray(slow_state["t_ms"], float)
    disinhibition = np.asarray(slow_state["D"], float)
    adaptation = np.asarray(slow_state["a"], float)
    crossing_ms = float(slow_state["crossing_ms"])
    if not np.isfinite(crossing_ms):
        raise ValueError("registered z+m representative has no operational-runaway crossing")

    adaptation = 1e4 * adaptation
    d_boundary = float(slow_state["display_boundary_D"])
    a_boundary = 1e4 * float(slow_state["display_boundary_a"])

    x_max = max(0.101, float(np.nanmax(disinhibition)) + 0.004)
    y_max = max(8.0, float(np.nanmax(adaptation)) + 0.45)
    ax.axvspan(d_boundary, x_max, color="crimson", alpha=0.055, lw=0, zorder=0)
    ax.axvline(d_boundary, color="crimson", lw=1.35, ls=(0, (3.0, 2.4)), zorder=2)
    ax.plot(
        disinhibition,
        adaptation,
        color=SLOW_STATE_COLOR,
        lw=1.65,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=5,
    )
    _add_path_arrow(ax, disinhibition, adaptation, 0.70)
    _add_path_arrow(ax, disinhibition, adaptation, 0.965)
    ax.scatter(
        disinhibition[0], adaptation[0], s=31,
        facecolor="white", edgecolor=SLOW_STATE_COLOR, linewidth=1.35, zorder=8,
    )
    ax.scatter(
        d_boundary, a_boundary, s=31,
        facecolor="crimson", edgecolor="white", linewidth=0.8, zorder=9,
    )
    ax.text(
        d_boundary - 0.0020, y_max - 0.28, r"$\mathcal{S}$",
        color="crimson", fontsize=13.0, ha="right", va="top",
    )

    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks([0.00, 0.04, 0.08])
    ax.set_yticks([0, 3, 6])
    ax.set_xlabel(r"Disinhibition  $D=1-\bar z$", fontsize=12.4, labelpad=6)
    ax.set_ylabel(r"Adaptation  $a$  ($\times10^{-4}$)", fontsize=12.4, labelpad=7)
    ax.tick_params(axis="both", labelsize=9.5, width=0.9, length=3.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.9)
    ax.set_box_aspect(0.82)
    return {
        "input": str(SLOW_STATE_PANEL),
        "source_artifact": slow_state["source_artifact"],
        "source_sha256": slow_state["source_sha256"],
        "display_transform": slow_state["display_transform"],
        "condition": str(slow_state["condition"]),
        "seed": int(slow_state["seed"]),
        "crossing_ms": crossing_ms,
        "display_boundary_D": d_boundary,
        "boundary_semantics": (
            "operational first-crossing boundary drawn as a schematic separatrix; "
            "not an analytically fitted basin boundary"
        ),
    }


def main():
    slowoff = _load_npz(SEED_ROOT / "slowoff.npz")
    native = _load_npz(SEED_ROOT / "native.npz")
    slow_state = _load_json(SLOW_STATE_PANEL)
    mode_context = _load_json(MODE_CONTEXT)
    native_json = _load_json(SEED_ROOT / "native.json")
    bridge_json = _load_json(SEED_ROOT / "bridge_metrics.json")
    templates = _load_npz(SEED_ROOT / "templates.npz")
    substrate = _load_npz(SUBSTRATE)

    names = [str(x) for x in slowoff["names"]]
    if names != [str(x) for x in native["names"]] or names != [str(x) for x in substrate["names"]]:
        raise ValueError("contact order differs across bridge/substrate artifacts")
    if not np.allclose(native["contacts"], substrate["contacts"], atol=1e-6):
        raise ValueError("substrate geometry does not match bridge contacts")

    mapping = _load_json(SEED_ROOT / "templates.json")["direction_axis_mapping"]
    contacts = _project(native["contacts"], mapping["axis_unit"], mapping["center"])
    neurons = _project(substrate["posE"], mapping["axis_unit"], mapping["center"])
    xlim = ylim = (-10.0, 10.0)

    event = _select_display_event(slowoff, native, native_json)
    timing = event["timing"]
    contact_rank = np.asarray(timing.rank, float)
    rank_display = _normalize_minmax(contact_rank)
    energy_raw = np.asarray(native[f"contact_energy__{PRIMARY_WK}"], float)
    energy_display = _normalize_minmax(energy_raw)
    t_recruit = float(native_json["onset"]["t_recruit_ms"])
    t120 = float(native_json["t120_ms"])

    template_key = "contact_B" if event["direction"] == "B_to_A" else "contact_A"
    template = np.asarray(templates[template_key], float)
    shared = np.isfinite(contact_rank) & np.isfinite(template)
    template_similarity = float(spearmanr(contact_rank[shared], template[shared]).statistic)
    event_energy = associate(contact_rank, energy_raw)
    direction_label = "TB" if event["direction"] == "B_to_A" else "TA"
    direction_color = TB_COL if direction_label == "TB" else TA_COL

    mode_baseline = np.asarray(mode_context["fields"]["baseline"], float)
    mode_pre_onset = np.asarray(mode_context["fields"]["pre_onset_120ms"], float)
    if mode_baseline.shape != mode_pre_onset.shape or mode_baseline.ndim != 2:
        raise ValueError("baseline/pre-onset mode fields must be matching 2D arrays")
    if not np.isfinite(mode_baseline).all() or not np.isfinite(mode_pre_onset).all():
        raise ValueError("mode fields contain non-finite values")
    mode_vmax = float(mode_context["display"]["vmax"])

    fig = plt.figure(figsize=(14.6, 7.45), facecolor="white")
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[0.78, 1.22],
        width_ratios=[3.18, 1.0],
        left=0.060,
        right=0.975,
        bottom=0.075,
        top=0.900,
        hspace=0.38,
        wspace=0.12,
    )
    trace_ax = fig.add_subplot(gs[0, 0])
    trace_meta = _plot_continuous_trace(trace_ax, native, names, event, t_recruit, t120)
    trace_ax.legend(
        handles=_legend_handles(),
        frameon=False,
        fontsize=8.2,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.075),
        ncol=2,
        handlelength=1.7,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    slow_state_meta = _plot_slow_state_path(fig.add_subplot(gs[0, 1]), slow_state)

    lower = gs[1, :].subgridspec(
        1,
        7,
        width_ratios=[1.0, 0.050, 1.0, 0.050, 1.0, 1.0, 0.050],
        wspace=0.24,
    )

    ax_left = fig.add_subplot(lower[0, 0])
    map_left = _draw_field(
        ax_left,
        contacts,
        rank_display,
        contact_rank,
        xlim,
        ylim,
        cmap="viridis",
        title=f"{direction_label} event order",
        title_color=direction_color,
        show_y=True,
        substrate_points=neurons,
    )
    cb_left = fig.colorbar(map_left, cax=fig.add_subplot(lower[0, 1]))
    rank_lo = float(np.nanmin(contact_rank))
    rank_hi = float(np.nanmax(contact_rank))
    rank_mid = 0.5 * (rank_lo + rank_hi)
    cb_left.set_ticks([rank_lo, rank_mid, rank_hi])
    cb_left.set_ticklabels([f"{rank_lo:g}", f"{rank_mid:.1f}", f"{rank_hi:g}"])
    cb_left.ax.yaxis.set_ticks_position("right")
    cb_left.ax.set_title("contact\nrank", fontsize=9.0, pad=6)
    cb_left.ax.tick_params(labelsize=8.5, length=2.2)

    ax_right = fig.add_subplot(lower[0, 2], sharey=ax_left)
    map_right = _draw_field(
        ax_right,
        contacts,
        energy_display,
        energy_raw,
        xlim,
        ylim,
        cmap="Blues",
        title="Early-onset energy",
        title_color="black",
        show_y=False,
        substrate_points=neurons,
    )
    cb_right = fig.colorbar(map_right, cax=fig.add_subplot(lower[0, 3]))
    energy_ticks = np.linspace(float(np.nanmin(energy_raw)), float(np.nanmax(energy_raw)), 4)
    cb_right.set_ticks(energy_ticks)
    cb_right.set_ticklabels([f"{value / 1000.0:.1f}" for value in energy_ticks])
    cb_right.ax.yaxis.set_ticks_position("right")
    cb_right.ax.set_title("energy\n(×10³ a.u.)", fontsize=9.0, pad=6)
    cb_right.ax.tick_params(labelsize=8.5, length=2.2)

    ax_mode_baseline = fig.add_subplot(lower[0, 4])
    _draw_mode_field(
        ax_mode_baseline,
        mode_baseline,
        xlim,
        ylim,
        title="Baseline mode",
        vmax=mode_vmax,
    )
    _draw_spatial_probe_schematic(ax_mode_baseline)
    ax_mode_pre = fig.add_subplot(lower[0, 5], sharey=ax_mode_baseline)
    mode_map = _draw_mode_field(
        ax_mode_pre,
        mode_pre_onset,
        xlim,
        ylim,
        title="Early-onset mode −120 ms",
        vmax=mode_vmax,
    )
    cb_mode = fig.colorbar(mode_map, cax=fig.add_subplot(lower[0, 6]))
    cb_mode.ax.set_title("mode\namplitude", fontsize=8.2, pad=5)
    cb_mode.ax.tick_params(labelsize=7.6, length=2.0)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    png = FIGDIR / f"{STEM}.png"
    pdf = FIGDIR / f"{STEM}.pdf"
    fig.savefig(png, dpi=DISPLAY_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    formal = (
        bridge_json["by_window"][PRIMARY_WK]["contact"]["all_support"]
    )
    metadata = {
        "schema_id": "topic4_mz_continuous_early_field_figure_v3_slow_state_modes",
        "status": (
            "LOCKED 2026-07-22 as the Figure 5 upper-half layout; V2 z+m tau_adp=500; "
            "continuous native MZ trajectory; observation-layer model bridge with slow-state "
            "path and frozen-q rate-field mode context"
        ),
        "figure_assignment": "Figure 5 upper half",
        "layout_status": "LOCKED 2026-07-22; lower-half composition remains separate",
        "candidate": "zA_q75_tz5000__mA0p001_tau500 (use_z+use_m; I_th_EI=95.19851312666987, tau_z=5000, "
                     "tau_adp=500, eta_m=0.007451594355587098)",
        "paper_role": "Figure 5 upper half — same scaffold, state-dependent readout (z+m)",
        "canonical_producer": "scripts/paper_figures/plot_fig_mz_early_bridge_v2.py",
        "input_artifacts": {
            "slowoff": str(SEED_ROOT / "slowoff.npz"),
            "native": str(SEED_ROOT / "native.npz"),
            "bridge_metrics": str(SEED_ROOT / "bridge_metrics.json"),
            "substrate_geometry_only": str(SUBSTRATE),
            "mode_context": str(MODE_CONTEXT),
            "bridge_provenance_fingerprint": bridge_json.get("provenance_fingerprint"),
        },
        "continuous_trace": trace_meta,
        "slow_state_panel": slow_state_meta,
        "rate_field_mode_context": {
            "contract": mode_context["mode_contract"],
            "model_contract": mode_context["model_contract"],
            "candidate": mode_context["candidate"],
            "seeds": mode_context["seeds"],
            "aggregation": mode_context["aggregation"],
            "states": mode_context["states"],
            "display": mode_context["display"],
            "provenance": mode_context["provenance"],
            "claim_boundary": mode_context["claim_boundary"],
            "perturbation_schematic": {
                "display": "localized positive E-rate pulse inset on the baseline-mode panel",
                "role": "schematic spatial-probe glyph only",
                "claim_boundary": (
                    "the leading-mode fields come from the frozen Jacobian; they are not responses "
                    "to this particular pulse, and no additional simulation is implied"
                ),
            },
        },
        "display_event": {
            "selection_rule": event["selection_rule"],
            "direction": event["direction"],
            "t_on_absolute_ms": event["t_on_ms"],
            "t_off_absolute_ms": event["t_off_ms"],
            "readout_end_absolute_ms": event["readout_end_ms"],
            "n_readable_contacts": int(timing.n_readable),
            "axis_spearman": float(timing.axis_spearman),
            "similarity_to_slowoff_direction_template": template_similarity,
            "descriptive_earliness_energy_spearman": event_energy["earliness_energy_spearman"],
        },
        "early_field": {
            "window": PRIMARY_WK,
            "t_recruit_absolute_ms": t_recruit,
            "early_onset_absolute_ms": t120 - EARLY_ONSET_OFFSET_MS,
            "energy_window_relative_to_early_onset_ms": [
                t_recruit - (t120 - EARLY_ONSET_OFFSET_MS),
                t_recruit + 50.0 - (t120 - EARLY_ONSET_OFFSET_MS),
            ],
            "t120_absolute_ms": t120,
            "contact_n": int(np.isfinite(energy_raw).sum()),
        },
        "formal_multievent_bridge": {
            "rho_maxab": formal["maxab"]["rho_maxab"],
            "within_shaft_p": formal["within_shaft_null"]["p_one_sided"],
            "tier": "formal statistic remains the held-out-validated slowoff-template bridge",
        },
        "spatial_rendering": {
            "method": "3-mm Gaussian contact-readout field with smooth confidence fade on the fixed E1146 plane",
            "substrate_grain": "fixed E-neuron positions only; no local recruitment values encoded",
            "claim_boundary": "continuous readout field, not direct local-tissue recruitment",
        },
        "framing": (
            "operational runaway is a model proxy, not clinical seizure; virtual-LFP 30-80-Hz "
            "early energy is not clinical broadband power"
        ),
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    (FIGDIR / f"{STEM}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"wrote {png}\nwrote {pdf}\nwrote {FIGDIR / f'{STEM}_metadata.json'}")


if __name__ == "__main__":
    main()
