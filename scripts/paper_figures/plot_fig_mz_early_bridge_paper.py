#!/usr/bin/env python3
"""Paper-ready E1146 MZ early-field bridge.

This renderer deliberately reuses the accepted legacy Figure-5 grammar:

* one continuous virtual-SEEG strip from the native z-only trajectory, containing
  both a representative returning interictal-like burst and the pre-t120 early
  recruitment window;
* two lower fields only: the exact displayed event's contact recruitment order
  and the pre-t120 contact-energy field.

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


BRIDGE = ROOT / "results/topic4_sef_hfo/mz_early_field_bridge"
SEED_ROOT = BRIDGE / "per_seed/seed1"
FIGDIR = ROOT / "results/paper-ready-figure/fig_mz_early_bridge/figures"
SUBSTRATE = (
    ROOT
    / "results/topic4_sef_hfo/mz_slowvars/readout_ready/"
    / "readout_zA_q75_tz10000_seed1.npz"
)
STEM = "fig_mz_early_bridge"
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
    early_start = float(t_recruit - display_start)
    early_end = float(t_recruit + 50.0 - display_start)
    onset_x = float(t120 - display_start)
    ax.axvspan(event_start, event_end, color=INTERICTAL_SHADE, alpha=0.10, lw=0, zorder=0)
    ax.axvspan(early_start, early_end, color="crimson", alpha=0.11, lw=0, zorder=0)
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
    ax.text(
        onset_x + 10.0,
        y[-1] + 1.05,
        f"runaway\n{onset_x:.0f} ms",
        color="crimson",
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_ylim(-0.55, y[-1] + 1.75)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.8)
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(_shaft_color(name, shafts))
    ax.set_xlabel("time in displayed continuous trajectory (ms)", fontsize=11.0)
    ax.set_ylabel("contacts", fontsize=11.0)
    ax.set_title("Virtual-SEEG", fontsize=13.0, fontweight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.2, length=3)
    ax.tick_params(axis="y", length=2.5)
    return {
        "display_start_absolute_ms": display_start,
        "display_end_absolute_ms": display_end,
        "display_event_start_ms": event_start,
        "display_event_end_ms": event_end,
        "early_window_start_ms": early_start,
        "early_window_end_ms": early_end,
        "t120_display_ms": onset_x,
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
    ax.set_xlabel("E1146 long axis (mm)", fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.2, length=2.5)
    if show_y:
        ax.set_ylabel("transverse (mm)", fontsize=11.0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    raw = np.asarray(values_colorbar, float)
    finite_raw = raw[np.isfinite(raw)]
    return ScalarMappable(Normalize(float(finite_raw.min()), float(finite_raw.max())), cmap=cmap)


def _legend_handles(names, direction):
    shafts = sorted(set(_shaft(name) for name in names))
    label = "TB" if direction == "B_to_A" else "TA"
    return [
        Line2D([0], [0], color=_shaft_color("ICL1", shafts), lw=1.6, label="ICL"),
        Line2D([0], [0], color=_shaft_color("SCL6", shafts), lw=1.6, label="SCL"),
        Line2D([0], [0], color="crimson", lw=1.6, ls="--", label="runaway onset"),
        Patch(facecolor=INTERICTAL_SHADE, alpha=0.12, edgecolor="none", label=f"{label} event used below"),
        Patch(facecolor="crimson", alpha=0.12, edgecolor="none", label="pre-t120 early window"),
    ]


def main():
    slowoff = _load_npz(SEED_ROOT / "slowoff.npz")
    native = _load_npz(SEED_ROOT / "native.npz")
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

    fig = plt.figure(figsize=(10.8, 8.25), facecolor="white")
    gs = gridspec.GridSpec(
        3,
        1,
        height_ratios=[0.92, 0.09, 1.24],
        left=0.075,
        right=0.965,
        bottom=0.075,
        top=0.925,
        hspace=0.30,
    )
    trace_meta = _plot_continuous_trace(
        fig.add_subplot(gs[0, 0]), native, names, event, t_recruit, t120
    )

    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_legend_handles(names, event["direction"]),
        frameon=False,
        fontsize=9.3,
        loc="center",
        ncol=5,
        handlelength=2.0,
        columnspacing=1.6,
    )

    lower = gs[2, 0].subgridspec(1, 2, wspace=0.19)
    left_pair = lower[0, 0].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.035)
    right_pair = lower[0, 1].subgridspec(1, 2, width_ratios=[1.0, 0.045], wspace=0.035)

    ax_left = fig.add_subplot(left_pair[0, 0])
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
    cb_left = fig.colorbar(map_left, cax=fig.add_subplot(left_pair[0, 1]))
    rank_lo = float(np.nanmin(contact_rank))
    rank_hi = float(np.nanmax(contact_rank))
    rank_mid = 0.5 * (rank_lo + rank_hi)
    cb_left.set_ticks([rank_lo, rank_mid, rank_hi])
    cb_left.set_ticklabels([f"{rank_lo:g} (early)", f"{rank_mid:.1f}", f"{rank_hi:g} (late)"])
    cb_left.ax.set_title("contact\nrank", fontsize=9.0, pad=6)
    cb_left.ax.tick_params(labelsize=8.5, length=2.2)

    ax_right = fig.add_subplot(right_pair[0, 0], sharey=ax_left)
    map_right = _draw_field(
        ax_right,
        contacts,
        energy_display,
        energy_raw,
        xlim,
        ylim,
        cmap="Blues",
        title="Pre-t120 early energy",
        title_color="black",
        show_y=False,
        substrate_points=neurons,
    )
    cb_right = fig.colorbar(map_right, cax=fig.add_subplot(right_pair[0, 1]))
    energy_ticks = np.linspace(float(np.nanmin(energy_raw)), float(np.nanmax(energy_raw)), 4)
    cb_right.set_ticks(energy_ticks)
    cb_right.set_ticklabels([f"{value / 1000.0:.1f}" for value in energy_ticks])
    cb_right.ax.set_title("energy\n(×10³ a.u.)", fontsize=9.0, pad=6)
    cb_right.ax.tick_params(labelsize=8.5, length=2.2)

    fig.suptitle("E1146", x=0.075, y=0.985, ha="left", fontsize=16.0, fontweight="bold")
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
        "schema_id": "topic4_mz_continuous_early_field_figure_v2",
        "status": "Figure 5 candidate; continuous native MZ trajectory; observation-layer model bridge",
        "paper_role": "same scaffold, state-dependent readout",
        "canonical_producer": "scripts/paper_figures/plot_fig_mz_early_bridge_paper.py",
        "input_artifacts": {
            "slowoff": str(SEED_ROOT / "slowoff.npz"),
            "native": str(SEED_ROOT / "native.npz"),
            "bridge_metrics": str(SEED_ROOT / "bridge_metrics.json"),
            "substrate_geometry_only": str(SUBSTRATE),
            "bridge_provenance_fingerprint": bridge_json.get("provenance_fingerprint"),
        },
        "continuous_trace": trace_meta,
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
