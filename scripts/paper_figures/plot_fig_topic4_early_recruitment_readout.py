#!/usr/bin/env python3
"""Continuous E1146 M3 readout with single-event order and early-runaway energy.

The whole figure consumes one accepted M3A-v2.1 q_I build-up-to-runaway
trajectory exported by ``run_topic4_m3_runaway_readout.py``. The upper trace is
continuous and marks the operational runaway onset. The two lower fields reuse
the paper Fig. 3B visual grammar on the transition-side template plane: raw
single-event contact recruitment rank on the left and onset-locked contact
energy on the right.

Plotting-only. No simulation is rerun and no GIF is produced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig_subject_snn import (  # noqa: E402
    _shaft,
    _shaft_color,
)


DEFAULT_RUN = ROOT / "results/topic4_sef_hfo/early_recruitment_readout"
DEFAULT_FIGURE_OUT = ROOT / "results/paper-ready-figure/fig5_snn_state_readout/figures"
DEFAULT_STEM = "fig5_candidate_E1146_snn_state_readout"
DISPLAY_DPI = 300
FIELD_GRID_N = 220
FIELD_SIGMA_MM = 3.0
LATENCY_CMAP = "viridis"
ENERGY_CMAP = "Blues"
MODE_CMAP = "magma"
TEMPLATE_COLORS = {"TA": "#B2182B", "TB": "#2166AC"}
INTERICTAL_EVENT_SHADE = "#6F9FD8"
TRACE_OFF = 1.48
TRACE_GAIN = 0.68
TRACE_BAND_HZ = (30.0, 80.0)
TRACE_SCALE_PERCENTILE = 95.0


def _load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=True)


def _sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_mode_pair(arrays, summary):
    """Three-seed mean leading-mode amplitude at baseline and pre-onset -100 ms.

    These are frozen-q rate-field Jacobian loadings, not empirical full-SNN
    perturbation modes.  All seeds must be resolved at both registered states;
    otherwise the figure fails closed instead of silently changing the cohort.
    """
    labels = [str(x) for x in summary["labels"]]
    wanted = {"baseline": "baseline_1000ms", "pre_onset": "pre_onset_100ms"}
    idx = {name: labels.index(label) for name, label in wanted.items()}
    seeds = [int(s) for s in summary["seeds"]]
    fields = {name: [] for name in wanted}
    records = {name: [] for name in wanted}
    for seed in seeds:
        resolved = np.asarray(arrays[f"{seed}__resolved"], bool)
        cube = np.asarray(arrays[f"{seed}__fields"], float)
        by_label = {
            str(rec["label"]): rec for rec in summary["per_seed"][str(seed)]["records"]
        }
        for name, label in wanted.items():
            ii = idx[name]
            rec = by_label[label]
            if not resolved[ii] or rec.get("op_status") != "resolved":
                raise ValueError(f"mode panel requires resolved {label} for seed {seed}")
            field = np.asarray(cube[ii], float)
            if field.ndim != 2 or not np.isfinite(field).all() or np.linalg.norm(field) <= 0:
                raise ValueError(f"invalid mode field for seed {seed}, {label}")
            fields[name].append(field)
            records[name].append(rec)
    mean_fields = {name: np.mean(np.stack(vals), axis=0) for name, vals in fields.items()}
    metrics = {
        name: {
            "axis_score_mean": float(np.mean([r["axis_score"] for r in records[name]])),
            "globality_mean": float(np.mean([r["globality"] for r in records[name]])),
            "time_to_runoff_ms": [float(r["time_to_runoff_ms"]) for r in records[name]],
        }
        for name in wanted
    }
    return mean_fields, metrics, seeds


def _geometry(path: Path):
    fd = _load_npz(path)
    names = [str(x) for x in fd["names"]]
    return fd, names


def _pre_runaway_burst_trace(arrays, onset_ms):
    """Expose the signed 30--80 Hz burst component without runaway rescaling."""
    times = np.asarray(arrays["times_ms"], float)
    lfp = np.asarray(arrays["lfp_trace"], float)
    dt_ms = float(np.median(np.diff(times)))
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError("virtual-SEEG time axis must have positive regular spacing")
    sampling_hz = 1000.0 / dt_ms
    sos = butter(
        4,
        TRACE_BAND_HZ,
        btype="bandpass",
        fs=sampling_hz,
        output="sos",
    )
    burst = sosfiltfilt(sos, lfp, axis=0)
    pre = times < float(onset_ms)
    if pre.sum() < 2:
        raise ValueError("fewer than two pre-runaway samples for trace scaling")
    scale = np.percentile(np.abs(burst[pre]), TRACE_SCALE_PERCENTILE, axis=0)
    finite_positive = scale[np.isfinite(scale) & (scale > 1e-12)]
    if finite_positive.size == 0:
        raise ValueError("pre-runaway burst trace is constant")
    scale_floor = 0.15 * float(np.median(finite_positive))
    scale = np.maximum(scale, max(scale_floor, 1e-12))
    return TRACE_GAIN * burst / scale[None, :]


def _plot_continuous_readout(ax, arrays, template_label):
    times = np.asarray(arrays["times_ms"], float)
    names = [str(x) for x in arrays["contact_names"]]
    shafts = sorted(set(_shaft(name) for name in names))
    y = np.arange(len(names), dtype=float) * TRACE_OFF
    onset = float(arrays["runaway_start_ms"])
    energy_start = float(arrays["runaway_energy_start_ms"])
    energy_end = float(arrays["runaway_energy_end_ms"])
    event_start = float(arrays["interictal_display_event_t0_ms"])
    event_end = float(arrays["interictal_display_event_t1_ms"])
    burst_trace = _pre_runaway_burst_trace(arrays, onset)

    # One pre-runaway event is shaded because it is exactly the event displayed
    # in the lower-left recruitment-order field. No peak connectors are drawn.
    ax.axvspan(
        event_start,
        event_end,
        color=INTERICTAL_EVENT_SHADE,
        alpha=0.10,
        lw=0,
        zorder=0,
    )
    ax.axvspan(energy_start, energy_end, color="crimson", alpha=0.11, lw=0, zorder=0)
    ax.axvline(onset, color="crimson", lw=1.6, ls="--", alpha=0.95, zorder=8)
    for ci, name in enumerate(names):
        ax.plot(
            times,
            burst_trace[:, ci] + y[ci],
            color=_shaft_color(name, shafts),
            lw=0.95,
            alpha=0.94,
            zorder=3,
            clip_on=True,
        )

    y_top = y[-1] + 1.75
    ax.text(
        onset + 10.0,
        y[-1] + 1.05,
        f"runaway\n{onset:.0f} ms",
        color="crimson",
        fontsize=10.2,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    ax.set_xlim(float(times[0]), float(times[-1]))
    ax.set_ylim(-0.55, y_top)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.8)
    for tick, name in zip(ax.get_yticklabels(), names):
        tick.set_color(_shaft_color(name, shafts))
    ax.set_xlabel("time (ms)", fontsize=11.0)
    ax.set_ylabel("contacts", fontsize=11.0)
    ax.set_title("Virtual-SEEG", fontsize=13.0, fontweight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.2, length=3)
    ax.tick_params(axis="y", length=2.5)
    return {
        "runaway_start_ms": onset,
        "energy_start_ms": energy_start,
        "energy_end_ms": energy_end,
        "interictal_event_start_ms": event_start,
        "interictal_event_end_ms": event_end,
        "trace_display_band_hz": list(TRACE_BAND_HZ),
        "trace_display_scaling": (
            f"per-contact signed {TRACE_BAND_HZ[0]:g}-{TRACE_BAND_HZ[1]:g}-Hz component "
            f"divided by its pre-runaway {TRACE_SCALE_PERCENTILE:g}th absolute percentile; "
            "runaway excluded from scaling and intentionally allowed to clip"
        ),
    }


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


def _project_to_registered_plane(coordinates, axis, transverse, center):
    centered = np.asarray(coordinates, float) - np.asarray(center, float)[None, :]
    return np.column_stack((centered @ axis, centered @ transverse))


def _shared_template_plane(geometry_fd, source):
    """Use the accepted E1146 registered plane without template-dependent mirroring."""
    del source
    contacts = np.asarray(geometry_fd["contacts"], float)
    reg = geometry_fd["reg"].item()
    axis = np.asarray(reg["axis_unit"], float)
    axis /= np.linalg.norm(axis)
    transverse = np.asarray([-axis[1], axis[0]], float)
    center = np.asarray(reg["center"], float)
    points = _project_to_registered_plane(contacts, axis, transverse, center)
    half = 0.5 * float(geometry_fd["L"])
    return points, (-half, half), (-half, half), axis, transverse, center


def _smooth_contact_field(points, values, xlim, ylim, sigma_mm):
    points = np.asarray(points, float)
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    if valid.sum() < 2:
        raise ValueError("contact field requires at least two finite values")
    x_grid = np.linspace(float(xlim[0]), float(xlim[1]), FIELD_GRID_N)
    y_grid = np.linspace(float(ylim[0]), float(ylim[1]), FIELD_GRID_N)
    X, Y = np.meshgrid(x_grid, y_grid)
    d2 = (
        (X[..., None] - points[valid, 0]) ** 2
        + (Y[..., None] - points[valid, 1]) ** 2
    )
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
    axis_label,
    sigma_mm,
    show_y,
    neuron_points,
    neuron_values_display,
    neuron_active,
):
    X, Y, field, confidence = _smooth_contact_field(
        points, values_display, xlim, ylim, sigma_mm
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
    neuron_points = np.asarray(neuron_points, float)
    neuron_values_display = np.asarray(neuron_values_display, float)
    neuron_active = np.asarray(neuron_active, bool)
    if neuron_points.shape != (neuron_values_display.size, 2):
        raise ValueError("neuron points and display values do not align")
    ax.scatter(
        neuron_points[:, 0],
        neuron_points[:, 1],
        s=1.05,
        c="0.70",
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    active = neuron_active & np.isfinite(neuron_values_display)
    ax.scatter(
        neuron_points[active, 0],
        neuron_points[active, 1],
        c=neuron_values_display[active],
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        s=2.0,
        alpha=0.62,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )
    # The montage is fixed independently of readout support. Unsupported
    # latency contacts remain hollow, but every contact has the same black rim.
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
    ax.set_xlabel(axis_label, fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.2, length=2.5)
    if show_y:
        ax.set_ylabel("transverse (mm)", fontsize=11.0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    raw = np.asarray(values_colorbar, float)
    finite_raw = raw[np.isfinite(raw)]
    return ScalarMappable(
        Normalize(float(finite_raw.min()), float(finite_raw.max())), cmap=cmap
    )


def _draw_mode_field(ax, field, xlim, ylim, *, title, vmax, show_y):
    image = ax.imshow(
        # The accepted rate-field producer stores [long-axis, transverse];
        # imshow expects [row=y, column=x], hence the explicit transpose.
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
    ax.set_title(title, fontsize=13.0, pad=7, fontweight="bold")
    ax.set_xlabel("E1146 long axis (mm)", fontsize=11.0)
    ax.tick_params(axis="both", labelsize=9.2, length=2.5)
    if show_y:
        ax.set_ylabel("transverse (mm)", fontsize=11.0)
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)
    return image


def _legend_handles(names, template_label):
    shafts = sorted(set(_shaft(name) for name in names))
    return [
        Line2D([0], [0], color=_shaft_color("ICL1", shafts), lw=1.6, label="ICL"),
        Line2D([0], [0], color=_shaft_color("SCL6", shafts), lw=1.6, label="SCL"),
        Line2D([0], [0], color="crimson", lw=1.6, ls="--", label="runaway onset"),
        Patch(
            facecolor=INTERICTAL_EVENT_SHADE,
            alpha=0.12,
            edgecolor="none",
            label=f"{template_label} event used below",
        ),
        Patch(
            facecolor="crimson",
            alpha=0.12,
            edgecolor="none",
            label="early runaway window",
        ),
    ]


def _build_figure(
    arrays,
    summary,
    geometry_fd,
    geometry_names,
    sigma_mm,
    *,
    mode_arrays=None,
    mode_summary=None,
):
    del summary
    array_names = [str(x) for x in arrays["contact_names"]]
    if array_names != geometry_names:
        raise ValueError(f"contact order mismatch: arrays={array_names}, geometry={geometry_names}")
    artifact_contacts = np.asarray(arrays["contacts_reference_mm"], float)
    geometry_contacts = np.asarray(geometry_fd["contacts"], float)
    if not np.allclose(artifact_contacts, geometry_contacts, atol=1e-8):
        raise ValueError("M3 exported contacts do not match accepted E1146 geometry")

    latency = np.asarray(arrays["interictal_reference_latency_ms"], float)
    contact_rank = np.asarray(arrays["interictal_event_contact_rank"], float)
    support = np.isfinite(contact_rank)
    rank_display = _normalize_minmax(contact_rank)

    energy_raw = np.asarray(arrays["runaway_energy"], float)
    energy_display = _normalize_minmax(energy_raw)
    source = str(arrays["interictal_reference_source"].item())
    template_label = "TB" if source == "tempB" else "TA"
    points, xlim, ylim, axis, transverse, center = _shared_template_plane(
        geometry_fd, source
    )
    neuron_positions = np.asarray(arrays["neuron_positions_reference_mm"], float)
    neuron_points = _project_to_registered_plane(
        neuron_positions, axis, transverse, center
    )
    neuron_latency = np.asarray(
        arrays["neuron_interictal_reference_latency_ms"], float
    )
    neuron_latency_relative = np.full(neuron_latency.shape, np.nan, float)
    neuron_latency_active = np.isfinite(neuron_latency)
    neuron_latency_relative[neuron_latency_active] = (
        neuron_latency[neuron_latency_active]
        - float(np.min(neuron_latency[neuron_latency_active]))
    )
    neuron_latency_display = _normalize_minmax(neuron_latency_relative)
    neuron_runaway_rate = np.asarray(arrays["neuron_runaway_rate_hz"], float)
    neuron_runaway_active = neuron_runaway_rate > 0.0
    neuron_rate_display = np.zeros(neuron_runaway_rate.shape, float)
    neuron_rate_cap = float(np.percentile(neuron_runaway_rate[neuron_runaway_active], 99.0))
    neuron_rate_display[neuron_runaway_active] = np.clip(
        neuron_runaway_rate[neuron_runaway_active] / max(neuron_rate_cap, 1e-12),
        0.0,
        1.0,
    )
    locally_recruited = np.asarray(
        arrays["interictal_event_locally_recruited"], bool
    )
    scl_mask = np.asarray([name.startswith("SCL") for name in array_names], bool)

    has_modes = mode_arrays is not None and mode_summary is not None
    fig = plt.figure(figsize=((16.4 if has_modes else 10.8), 8.25), facecolor="white")
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
    top_stats = _plot_continuous_readout(
        fig.add_subplot(gs[0, 0]), arrays, template_label
    )

    legend_ax = fig.add_subplot(gs[1, 0])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_legend_handles(array_names, template_label),
        frameon=False,
        fontsize=9.3,
        loc="center",
        ncol=5,
        handlelength=2.0,
        columnspacing=1.6,
    )

    lower = gs[2, 0].subgridspec(
        1,
        (3 if has_modes else 2),
        width_ratios=([1.0, 1.0, 1.78] if has_modes else [1.0, 1.0]),
        wspace=(0.15 if has_modes else 0.19),
    )
    left_pair = lower[0, 0].subgridspec(
        1, 2, width_ratios=[1.0, 0.045], wspace=0.035
    )
    right_pair = lower[0, 1].subgridspec(
        1, 2, width_ratios=[1.0, 0.045], wspace=0.035
    )

    axis_label = "E1146 long axis (mm)"
    ax_l = fig.add_subplot(left_pair[0, 0])
    map_l = _draw_field(
        ax_l,
        points,
        rank_display,
        contact_rank,
        xlim,
        ylim,
        cmap=LATENCY_CMAP,
        title=f"{template_label} event order",
        title_color=TEMPLATE_COLORS[template_label],
        axis_label=axis_label,
        sigma_mm=sigma_mm,
        show_y=True,
        neuron_points=neuron_points,
        neuron_values_display=neuron_latency_display,
        neuron_active=neuron_latency_active,
    )
    cb_l = fig.colorbar(map_l, cax=fig.add_subplot(left_pair[0, 1]))
    rank_lo = float(np.nanmin(contact_rank))
    rank_hi = float(np.nanmax(contact_rank))
    rank_mid = 0.5 * (rank_lo + rank_hi)
    cb_l.set_ticks([rank_lo, rank_mid, rank_hi])
    cb_l.set_ticklabels(
        [f"{rank_lo:g} (early)", f"{rank_mid:.1f}", f"{rank_hi:g} (late)"]
    )
    cb_l.ax.set_title("contact\nrank", fontsize=9.0, pad=6)
    cb_l.ax.tick_params(labelsize=8.5, length=2.2)

    # Keep raw model units visible while shortening the colorbar labels.
    energy_million = energy_raw / 1e6
    ax_r = fig.add_subplot(right_pair[0, 0], sharey=ax_l)
    map_r = _draw_field(
        ax_r,
        points,
        energy_display,
        energy_million,
        xlim,
        ylim,
        cmap=ENERGY_CMAP,
        title="Early runaway energy",
        title_color="black",
        axis_label=axis_label,
        sigma_mm=sigma_mm,
        show_y=False,
        neuron_points=neuron_points,
        neuron_values_display=neuron_rate_display,
        neuron_active=neuron_runaway_active,
    )
    cb_r = fig.colorbar(map_r, cax=fig.add_subplot(right_pair[0, 1]))
    energy_lo = float(np.nanmin(energy_million))
    energy_hi = float(np.nanmax(energy_million))
    energy_ticks = np.linspace(energy_lo, energy_hi, 4)
    cb_r.set_ticks(energy_ticks)
    cb_r.set_ticklabels([f"{value:.2f}" for value in energy_ticks])
    cb_r.ax.set_title("energy\n(×10⁶)", fontsize=9.0, pad=6)
    cb_r.ax.tick_params(labelsize=8.5, length=2.2)

    mode_display = None
    if has_modes:
        mean_modes, mode_metrics, mode_seeds = _load_mode_pair(mode_arrays, mode_summary)
        mode_group = lower[0, 2].subgridspec(
            1, 3, width_ratios=[1.0, 1.0, 0.045], wspace=0.06
        )
        mode_vmax = float(max(np.max(mean_modes["baseline"]), np.max(mean_modes["pre_onset"])))
        ax_mb = fig.add_subplot(mode_group[0, 0])
        _draw_mode_field(
            ax_mb,
            mean_modes["baseline"],
            xlim,
            ylim,
            title="Baseline mode",
            vmax=mode_vmax,
            show_y=False,
        )
        ax_mp = fig.add_subplot(mode_group[0, 1], sharey=ax_mb)
        mode_map = _draw_mode_field(
            ax_mp,
            mean_modes["pre_onset"],
            xlim,
            ylim,
            title="Pre-onset mode  −100 ms",
            vmax=mode_vmax,
            show_y=False,
        )
        cb_m = fig.colorbar(mode_map, cax=fig.add_subplot(mode_group[0, 2]))
        cb_m.ax.set_title("mode\namplitude", fontsize=9.0, pad=6)
        cb_m.ax.tick_params(labelsize=8.5, length=2.2)
        mode_display = {
            "contract": mode_summary["mode_contract"],
            "model_contract": mode_summary["model_contract"],
            "candidate": mode_summary["candidate"],
            "seeds": mode_seeds,
            "aggregation": "mean raw non-negative loading across three resolved seeds; shared color scale",
            "states": mode_metrics,
            "colormap": MODE_CMAP,
            "display_vmax": mode_vmax,
        }

    fig.suptitle("E1146", x=0.075, y=0.985, ha="left", fontsize=16.0, fontweight="bold")
    common = support & np.isfinite(energy_raw)
    relation = {
        "n": int(common.sum()),
        "earliness_energy_spearman": (
            float(spearmanr(-contact_rank[common], energy_raw[common]).statistic)
            if common.sum() >= 3
            else None
        ),
    }
    display = {
        "template_label": template_label,
        "common_support_n": int(support.sum()),
        "montage_contact_n": int(points.shape[0]),
        "left_contact_value_n": int(np.isfinite(contact_rank).sum()),
        "right_contact_value_n": int(np.isfinite(energy_raw).sum()),
        "neuron_n": int(neuron_points.shape[0]),
        "interictal_display_neuron_n": int(neuron_latency_active.sum()),
        "SCL_readout_contact_n": int(np.sum(support & scl_mask)),
        "SCL_locally_recruited_contact_n": int(
            np.sum(locally_recruited & scl_mask)
        ),
        "runaway_active_neuron_n": int(neuron_runaway_active.sum()),
        "shared_axis_unit": axis.tolist(),
        "transverse_unit": transverse.tolist(),
        "plane_center_mm": center.tolist(),
        "display_xlim_mm": list(map(float, xlim)),
        "display_ylim_mm": list(map(float, ylim)),
        "contact_rank_limits": [rank_lo, rank_hi],
        "energy_limits_raw": [float(np.nanmin(energy_raw)), float(np.nanmax(energy_raw))],
        "neuron_runaway_rate_display_cap_hz": neuron_rate_cap,
    }
    return fig, top_stats, source, relation, display, mode_display


def _write_sidecars(
    outdir,
    stem,
    png,
    pdf,
    artifact_npz,
    artifact_json,
    summary,
    source,
    top_stats,
    relation,
    display,
    sigma_mm,
    mode_artifact_npz=None,
    mode_artifact_json=None,
    mode_display=None,
):
    template_label = display["template_label"]
    metadata = {
        "schema_id": "topic4_m3_continuous_runaway_field_figure_v10",
        "status": "Figure 5 candidate; single-trajectory model readout; no seizure/recovery proof",
        "paper_role": "Figure 5 candidate: same scaffold, state-dependent readout",
        "central_argument": (
            "the spatial order expressed by one interictal-like group event on a fixed "
            "E1146 model scaffold aligns descriptively with the early operational-runaway "
            "energy gradient on the same continuous trajectory"
        ),
        "canonical_producer": "scripts/paper_figures/plot_fig_topic4_early_recruitment_readout.py",
        "computation_artifact": {"npz": str(artifact_npz), "json": str(artifact_json)},
        "layout": (
            "top continuous virtual-SEEG with one matched TB event; separate legend; "
            "bottom single-event contact rank, onset-locked runaway energy, and two "
            "rate-field leading-mode context panels"
            if mode_display is not None
            else "top continuous virtual-SEEG with one matched TB event; separate legend; bottom single-event contact rank and onset-locked runaway energy"
        ),
        "runaway": {
            **top_stats,
            "operational_definition": summary["runaway_onset"]["operational_definition"],
            "separatrix_boundary": summary["runaway_onset"]["separatrix_boundary"],
        },
        "interictal_field": {
            **summary["interictal_reference"],
            "display": (
                f"1..N contact recruitment rank from the displayed event's 30-80-Hz "
                "burst-envelope peak latency on the fixed E1146 registered plane; "
                "continuous min-max interpolation; viridis dark=early; all 15 contacts "
                "shown and unsupported contacts retained as hollow markers"
            ),
        },
        "runaway_energy_field": {
            **summary["runaway_energy_window"],
            "display": (
                "raw contact mean excess virtual-LFP energy on all 15 virtual electrodes; "
                "continuous min-max interpolation; Blues high=dark"
            ),
        },
        "descriptive_relation": {
            **relation,
            "tier": "single-trajectory mechanism visualization; not cohort inference",
        },
        "spatial_rendering": {
            **display,
            "sigma_mm": float(sigma_mm),
            "method": (
                "Fig3B-style paired fields on the identical fixed E1146 montage, registered "
                "plane, and extent; panel-specific finite contact values use Gaussian projection "
                "with smooth confidence fade; grain is direct per-neuron spiking from the same run"
            ),
            "grain_contract": {
                "background": "all simulated E-neuron positions",
                "interictal": "first-spike latency for neurons active in the identical displayed TB event",
                "runaway": "per-neuron firing rate in the onset-locked early-runaway window",
            },
            "left_colormap": LATENCY_CMAP,
            "right_colormap": ENERGY_CMAP,
        },
        "rate_field_mode_context": (
            {
                **mode_display,
                "artifact_npz": str(mode_artifact_npz),
                "artifact_json": str(mode_artifact_json),
                "artifact_npz_sha256": _sha256(mode_artifact_npz),
                "artifact_json_sha256": _sha256(mode_artifact_json),
                "claim_boundary": (
                    "frozen-q rate-field Jacobian loading; not an empirical full-SNN response mode, "
                    "not propagation direction, and not a mode computed from the displayed M3 run"
                ),
            }
            if mode_display is not None
            else None
        ),
        "claim_boundary": summary["claim_boundary"] + ([
            "the two added leading-mode panels are a frozen-q rate-field context layer, not empirical SNN eigenmodes from the displayed trajectory"
        ] if mode_display is not None else []),
        "outputs": {"png": str(png), "pdf": str(pdf)},
    }
    meta = outdir / f"{stem}_metadata.json"
    meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    readme = f"""# Figure 5 candidate — E1146 SNN state-dependent readout

### {stem}.png / .pdf

上方是 M3A-v2.1 `q_I build-up → runaway` 的同一条连续 0–1500 ms virtual-SEEG。蓝色只标出左下图实际使用的单次 `{template_label}` 群体事件（{top_stats['interictal_event_start_ms']:.1f}–{top_stats['interictal_event_end_ms']:.1f} ms）；浅红色标出右下能量场实际平均的 {top_stats['energy_start_ms']:.1f}–{top_stats['energy_end_ms']:.1f} ms，红虚线是 operational runaway onset（{top_stats['runaway_start_ms']:.1f} ms）。不画 peak 点或传播连线。上图画 signed 30–80 Hz component；每个触点按自身 runaway 前 95% absolute amplitude 定标，runaway 不参与定标，超出纵轴的部分允许裁切。

下方前两图复用正式 Fig3-B 的成对 field 语法，并保留完整的 E1146 15-contact montage 和原注册平面。左图不是多事件模板，也不是 variance：它把蓝色窗内每个 virtual contact 的 30–80 Hz burst-envelope peak latency 排成 `1…N` recruitment rank，再投影到 field；`viridis` 深色更早。该事件有 {display['left_contact_value_n']} 个触点达到 readout 阈值，其余触点以空心电极显示。右图使用完整 {display['right_contact_value_n']} 个触点的 onset-locked mean-squared positive excess virtual-LFP energy，使用 `Blues`，深蓝更高。投影 kernel 为 {sigma_mm:.1f} mm。

{('下方最右两图是三 seed 的 frozen-q rate-field leading-mode loading：baseline 近全局/各向同性，pre-onset −100 ms 沿 E1146 长轴集中；两图使用同一色标。它们是线性化 rate-field 的机制背景，不是从上方这条 M3 SNN trace 直接辨识出的 empirical SNN eigenmode，也不编码传播方向。' if mode_display is not None else '')}

平滑 wash 表示 virtual-SEEG contact readout；颗粒层直接来自同一仿真的 {display['neuron_n']} 个 E neurons。左侧彩色颗粒是同一个蓝色窗内实际发放的 {display['interictal_display_neuron_n']} 个神经元，颜色为各自 first-spike latency 的相对 early-to-late order；右侧彩色颗粒是 early-runaway window 内实际发放的 {display['runaway_active_neuron_n']} 个神经元，颜色来自逐神经元 firing rate。灰色颗粒是同一 run 的完整模拟 E-neuron 位置；两图完整 15-contact montage 均使用统一黑色外边框。

四个 SCL 均有该单次事件的 virtual-contact burst peak（{display['SCL_readout_contact_n']}/4），但按“contact 周围 1.5 mm 内至少 5% E neurons 发放”的局部组织门，SCL 通过数为 {display['SCL_locally_recruited_contact_n']}/4。因此图只支持 upper contacts participate in the group readout，不支持 SCL 下方局部组织已被直接招募。

**关注点**：这是 Figure 5 的候选模型 panel，用来检查“同一固定 scaffold 上，runaway 刚开始时的能量增强位置是否沿用此前间期单事件的传播次序”。它是单模型、单 seed、单连续轨迹的 observation-layer bridge。runaway onset 是 sustained-rate 操作定义，不是独立求出的解析 separatrix `q_I*`；virtual-LFP excess energy 也不是临床 broadband SEEG power，当前也没有发作终止或恢复。
"""
    (outdir / "README.md").write_text(readme, encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact-npz", type=Path, default=DEFAULT_RUN / "m3_runaway_readout.npz")
    ap.add_argument("--artifact-json", type=Path, default=DEFAULT_RUN / "m3_runaway_readout.json")
    ap.add_argument("--geometry-npz", type=Path, default=None)
    ap.add_argument("--field-sigma-mm", type=float, default=FIELD_SIGMA_MM)
    ap.add_argument("--mode-artifact-npz", type=Path, default=None)
    ap.add_argument("--mode-artifact-json", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=DEFAULT_FIGURE_OUT)
    ap.add_argument("--stem", default=DEFAULT_STEM)
    args = ap.parse_args()

    summary = _load_json(args.artifact_json)
    arrays = _load_npz(args.artifact_npz)
    if (args.mode_artifact_npz is None) != (args.mode_artifact_json is None):
        raise ValueError("mode artifact NPZ and JSON must be provided together")
    mode_arrays = _load_npz(args.mode_artifact_npz) if args.mode_artifact_npz else None
    mode_summary = _load_json(args.mode_artifact_json) if args.mode_artifact_json else None
    geometry_path = args.geometry_npz or Path(summary["geometry_npz"])
    geometry_fd, names = _geometry(geometry_path)
    fig, top_stats, source, relation, display, mode_display = _build_figure(
        arrays,
        summary,
        geometry_fd,
        names,
        args.field_sigma_mm,
        mode_arrays=mode_arrays,
        mode_summary=mode_summary,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    png = args.outdir / f"{args.stem}.png"
    pdf = args.outdir / f"{args.stem}.pdf"
    fig.savefig(png, dpi=DISPLAY_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    meta = _write_sidecars(
        args.outdir,
        args.stem,
        png,
        pdf,
        args.artifact_npz,
        args.artifact_json,
        summary,
        source,
        top_stats,
        relation,
        display,
        args.field_sigma_mm,
        mode_artifact_npz=args.mode_artifact_npz,
        mode_artifact_json=args.mode_artifact_json,
        mode_display=mode_display,
    )
    print(f"wrote {png}\nwrote {pdf}\nwrote {meta}\nwrote {args.outdir / 'README.md'}")


if __name__ == "__main__":
    main()
