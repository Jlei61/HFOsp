#!/usr/bin/env python3
"""Paper-facing Figure 5 in the accepted A-D transition layout.

A  one continuous signed 30-80 Hz virtual-contact current-proxy readout
B  h-weighted Z/M trajectory
C  one rule-selected pre-transition event and early-transition activity energy
D  response to the same frozen source-site probe at low activity and -500 ms

The diagnostic three-panel GIF remains a supplement. This producer never
re-simulates the SNN and refuses to render Panel D without both perturbation
states from at least two paired network seeds.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "svg.fonttype": "none"})

ICL = "#F1783A"
SCL = "#29A6B5"
ONSET = "#D62745"
TRAJ = "#304D73"
EVENT_SHADE = "#DCEAF6"
STATE_SHADE = "#F7E9ED"
REFERENCE_FIGDATA = (
    "results/topic4_sef_hfo/field_swap_subject_snn/"
    "figdata_epilepsiae_1146_tsrc_cr1p5_s4_T20000_20260706.npz"
)


def _registered_xy(xy, axis_unit, origin):
    xy = np.asarray(xy, float)
    u = np.asarray(axis_unit, float)
    u = u / np.linalg.norm(u)
    p = np.asarray([-u[1], u[0]])
    centred = xy - np.asarray(origin, float)
    return np.column_stack([centred @ u, centred @ p])


def _fit_accepted_display(current_xy, current_names, reference_xy,
                          reference_names):
    """Rigidly register the current sheet to the accepted E1146 display plane."""
    current_xy = np.asarray(current_xy, float)
    reference_xy = np.asarray(reference_xy, float)
    current_names = np.asarray(current_names).astype(str)
    reference_names = np.asarray(reference_names).astype(str)
    if set(current_names) != set(reference_names):
        raise ValueError("current and accepted display montages do not share all contacts")
    source = np.asarray([
        current_xy[np.flatnonzero(current_names == name)[0]]
        for name in reference_names
    ])
    target = reference_xy - np.mean(reference_xy, axis=0)
    source_center = np.mean(source, axis=0)
    source_zero = source - source_center
    target_zero = target - np.mean(target, axis=0)
    u, singular, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(np.sum(singular) / np.sum(source_zero * source_zero))
    offset = -scale * source_center @ rotation
    fitted = scale * source @ rotation + offset
    residual = fitted - target
    rmse = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    maximum = float(np.max(np.linalg.norm(residual, axis=1)))
    if rmse > 0.01 or maximum > 0.02:
        raise RuntimeError(
            f"accepted E1146 display registration drifted: rmse={rmse:.4g}, "
            f"max={maximum:.4g} mm")
    return {
        "rotation": rotation,
        "scale": scale,
        "offset": offset,
        "fit_rmse_mm": rmse,
        "fit_max_error_mm": maximum,
        "reference_center_mm": np.mean(reference_xy, axis=0),
    }


def _accepted_display_xy(xy, display):
    xy = np.asarray(xy, float)
    return (float(display["scale"]) * xy @
            np.asarray(display["rotation"], float)
            + np.asarray(display["offset"], float))


def _load_accepted_display(replay, path):
    with np.load(path, allow_pickle=True) as handle:
        display = _fit_accepted_display(
            replay["contact_xy_mm"], replay["contact_names"],
            np.asarray(handle["contacts"], float),
            np.asarray(handle["names"]).astype(str))
    display["reference_figdata"] = str(path)
    display["contract"] = "accepted E1146 registered-plane montage"
    return display


def _grid_mean(xy, values, extent, n=80, *, positive=False):
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    if positive:
        values = np.clip(values, 0.0, None)
    lo, hi = float(extent[0]), float(extent[1])
    edges = np.linspace(lo, hi, int(n) + 1)
    total, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges, edges),
                                 weights=values)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges, edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = total / count
    return gaussian_filter(np.nan_to_num(grid), sigma=1.15)


def _signed_bandpass(raw, dt_ms, band=(30.0, 80.0)):
    raw = np.asarray(raw, float)
    fs_hz = 1000.0 / float(dt_ms)
    if fs_hz <= 2.0 * float(band[1]):
        raise ValueError("virtual-contact sampling rate is below the 30-80 Hz Nyquist limit")
    sos = butter(4, band, btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, raw, axis=0)


def _runaway_ema(rate_hz, dt_ms, tau_ms=20.0):
    """Match the causal rate EMA used by ``kick_probe`` runaway detection."""
    rate_hz = np.asarray(rate_hz, float)
    alpha = 1.0 - np.exp(-float(dt_ms) / float(tau_ms))
    output = np.empty_like(rate_hz)
    state = 0.0
    for index, value in enumerate(rate_hz):
        state += alpha * (float(value) - state)
        output[index] = state
    return output


def _require_sustained_runaway(replay_meta, *, allow_exploratory_workpoint=False):
    """Reject rate-threshold crossings that lack the frozen morphology."""
    morphology = replay_meta.get("runaway_morphology")
    if not isinstance(morphology, dict):
        raise RuntimeError(
            "Figure 5A requires a runaway morphology audit; an operational "
            "rate-threshold crossing is not sufficient")
    classification = morphology.get("classification")
    if not isinstance(classification, dict):
        raise RuntimeError("runaway morphology audit has no classification")
    checks = classification.get("checks", {})
    failed = sorted(name for name, passed in checks.items() if not bool(passed))
    if not bool(classification.get("all_checks_pass")) or failed:
        recruitment = morphology.get("full_field_recruitment", {})
        allowed_failures = {
            "majority_E_active_for_95pct_windows",
            "majority_sheet_recruited_for_95pct_windows",
        }
        majority_duty = min(
            float(recruitment.get("fraction_windows_majority_E_active", 0.0)),
            float(recruitment.get("fraction_windows_majority_sheet_recruited", 0.0)),
        )
        exploratory_ok = (
            allow_exploratory_workpoint
            and set(failed).issubset(allowed_failures)
            and majority_duty >= 0.90
        )
        if exploratory_ok:
            morphology["figure_workpoint_status"] = (
                "AUTHOR_SELECTED_GLOBAL_HIGH_FREQUENCY_WORKPOINT_WITH_BRIEF_DROPOUTS")
            return morphology
        detail = ", ".join(failed) if failed else "unspecified morphology check"
        raise RuntimeError(
            "Figure 5A refuses a non-ictal threshold crossing; failed: " + detail)
    return morphology


def _contact_order(names):
    def number(name):
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else -1
    # bottom -> top: SCL6..9, then ICL1..11, matching the accepted reference.
    return np.asarray(sorted(range(len(names)), key=lambda i: (
        0 if str(names[i]).startswith("SCL") else 1, number(names[i]))), int)


def _load_npz(path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def _seed_from_path(path):
    match = re.search(r"seed_(\d+)", Path(path).name)
    if not match:
        raise ValueError(f"cannot parse seed from {path}")
    return int(match.group(1))


def _aggregate_probe_fields(paths, replay, output_root, extent, display,
                            site_id="source",
                            field_key="excess_per_neuron_early",
                            require_e1=False):
    grids, used, excluded, ignited = [], [], [], []
    probe_xy = None
    for path in sorted(paths):
        block = _load_npz(path)
        ids = np.asarray(block["site_id"]).astype(str)
        where = np.flatnonzero(ids == site_id)
        if where.size != 1:
            raise RuntimeError(f"{path} has no unique frozen {site_id!r} site")
        row = int(where[0])
        seed = _seed_from_path(path)
        is_ignited = bool(block["probe_attributable_event_200ms"][row]
                          or block["reached_model_ictal_200ms"][row])
        if is_ignited:
            ignited.append(seed)
        if require_e1 and not bool(block["e1_evaluable"][row]):
            excluded.append(seed)
            continue
        worker = _load_npz(output_root / "workers"
                           / f"joint_04_control_seed_{seed}.npz")
        xy = _accepted_display_xy(worker["positions_E"], display)
        if field_key not in block:
            raise RuntimeError(f"{path} predates the {field_key!r} map contract")
        grids.append(_grid_mean(xy, block[field_key][row], extent,
                                positive=True))
        current_probe = _accepted_display_xy(
            block["site_xy_mm"][[row]], display)[0]
        if probe_xy is None:
            probe_xy = current_probe
        elif not np.allclose(probe_xy, current_probe, atol=1e-8, rtol=0.0):
            raise RuntimeError("the frozen source probe moved across network seeds")
        used.append(seed)
    if len(grids) < 2:
        raise RuntimeError(
            f"Panel D needs >=2 evaluable network seeds; got {used}, excluded={excluded}")
    return np.mean(grids, axis=0), probe_xy, used, excluded, ignited


def _style_spatial(ax, extent, show_ylabel=False):
    ax.set_xlim(extent); ax.set_ylim(extent); ax.set_aspect("equal")
    ax.set_xlabel("TA shared axis (mm)", fontsize=8.2)
    if show_ylabel:
        ax.set_ylabel("registered transverse axis (mm)", fontsize=8.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7.2, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_label(ax, label, x=-0.18, y=1.11):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
            ha="left", va="top")


def _plot_readout(ax, replay, onset_ms, morphology):
    names = np.asarray(replay["contact_names"]).astype(str)
    shafts = np.asarray(replay["shaft_ids"]).astype(str)
    order = _contact_order(names)
    signed = _signed_bandpass(replay["lfp_trace"], float(replay["lfp_dt_ms"]))
    times = np.arange(signed.shape[0]) * float(replay["lfp_dt_ms"])
    event_on = float(replay["sample_event_t_on_ms"])
    event_off = float(replay["sample_event_t_off_ms"])
    start = max(0.0, min(event_on - 100.0, onset_ms - 1200.0))
    stop = min(float(times[-1]), onset_ms + 2000.0)
    mask = (times >= start) & (times <= stop)
    trace = signed[mask][:, order]
    ts = times[mask] - start
    pre_runaway = ts < (onset_ms - start)
    if not np.any(pre_runaway):
        raise RuntimeError("readout window has no pre-runaway scale interval")
    scale = float(np.percentile(np.abs(trace[pre_runaway]), 99.0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise RuntimeError("signed 30-80 Hz current proxy is constant")
    trace = 0.72 * trace / scale
    y = np.arange(len(order), dtype=float) * 1.18
    recruitment_time = np.asarray(replay["full_field_time_ms"], float)
    recruitment_mask = (recruitment_time >= start) & (recruitment_time <= stop)
    recruitment_ts = recruitment_time[recruitment_mask] - start
    neuron_fraction = np.asarray(
        replay["active_neuron_fraction_20ms"], float)[recruitment_mask]
    spatial_fraction = np.asarray(
        replay["recruited_spatial_fraction_1mm"], float)[recruitment_mask]

    original = ax.get_position()
    gap = 0.025 * original.height
    rate_height = 0.205 * original.height
    trace_height = original.height - rate_height - gap
    ax.set_position([original.x0, original.y0, original.width, trace_height])
    rate_ax = ax.figure.add_axes([
        original.x0, original.y0 + trace_height + gap,
        original.width, rate_height,
    ], sharex=ax)

    ax.axvspan(event_on - start, event_off - start, color=EVENT_SHADE, alpha=0.60,
               lw=0, zorder=0)
    ax.axvspan(onset_ms - start, stop - start, color=STATE_SHADE, alpha=0.92,
               lw=0, zorder=0)
    for row, ci in enumerate(order):
        color = ICL if shafts[ci] == "ICL" else SCL
        ax.plot(ts, trace[:, row] + y[row], color=color, lw=0.78, alpha=0.96)
    ax.axvline(onset_ms - start, color=ONSET, lw=1.0, ls="--")
    ax.set_xlim(0, float(ts[-1])); ax.set_ylim(-0.8, y[-1] + 1.0)
    ax.set_yticks(y); ax.set_yticklabels(names[order], fontsize=7.5)
    for tick, ci in zip(ax.get_yticklabels(), order):
        tick.set_color(ICL if shafts[ci] == "ICL" else SCL)
    ax.set_xlabel("Time in displayed continuous window (ms)", fontsize=9.0)
    ax.set_ylabel("Virtual-SEEG proxy (30-80 Hz)", fontsize=9.0)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    rate_ax.axvspan(event_on - start, event_off - start, color=EVENT_SHADE,
                    alpha=0.60, lw=0, zorder=0)
    rate_ax.axvspan(onset_ms - start, stop - start, color=STATE_SHADE,
                    alpha=0.92, lw=0, zorder=0)
    criterion_start = onset_ms - start - 100.0
    rate_ax.axvspan(criterion_start, onset_ms - start, color=ONSET,
                    alpha=0.08, lw=0, zorder=1)
    rate_ax.plot(recruitment_ts, 100.0 * neuron_fraction, color="0.18", lw=0.9,
                 label="active E neurons")
    rate_ax.plot(recruitment_ts, 100.0 * spatial_fraction, color="#6D7F91",
                 lw=0.9, label="recruited sheet")
    rate_ax.axhline(50.0, color=ONSET, lw=0.85, ls=":")
    rate_ax.axvline(onset_ms - start, color=ONSET, lw=1.0, ls="--")
    rate_ax.set_xlim(0, float(ts[-1])); rate_ax.set_ylim(0.0, 102.0)
    rate_ax.set_yticks([0.0, 50.0, 100.0])
    rate_ax.set_ylabel("global\nrecruitment (%)", fontsize=7.2, labelpad=2)
    rate_ax.tick_params(axis="y", labelsize=6.6, length=2.2)
    rate_ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    rate_ax.spines[["top", "right"]].set_visible(False)
    rate_ax.text(onset_ms - start + 10, 96.0, "global high-frequency state",
                 fontsize=7.0, ha="left", va="top", color=ONSET,
                 fontweight="bold")
    rate_ax.text(float(ts[-1]) - 12, 52.0, "majority threshold",
                 fontsize=6.4, ha="right", va="bottom", color=ONSET)
    rate_ax.legend(handles=[
        Line2D([0], [0], color=ONSET, ls="--", lw=1.1, label="runaway onset"),
        Patch(facecolor=EVENT_SHADE, edgecolor="none", label="sample interictal event"),
    ], frameon=False, fontsize=7.5, loc="upper right", ncol=2,
       bbox_to_anchor=(1.0, 1.42), borderaxespad=0.0)
    ax.text(float(ts[-1]) - 12, y[-1] + 0.62, "shared pre-runaway scale",
            fontsize=6.8, ha="right", va="top", color="0.35")
    frequency = morphology["population_rate_frequency"]
    return {"window_start_ms": start, "window_stop_ms": stop,
            "sample_event_t_on_ms": event_on, "sample_event_t_off_ms": event_off,
            "scale_interval": "all displayed samples before runaway onset",
            "global_recruitment_strip": {
                "measure": ("20-ms active-E fraction and fraction of 1-mm sheet "
                            "bins with >=50% local recruitment"),
                "majority_threshold": 0.5,
            },
            "population_frequency_pre_hz": frequency["spectral_centroid_pre_hz"],
            "population_frequency_post_hz": frequency["spectral_centroid_post_hz"],
            "morphology_status": morphology["classification"]["status"]}, rate_ax


def _plot_trajectory(ax, replay, onset_ms, eta_m):
    time = np.asarray(replay["zm_h_weighted_time_ms"], float)
    d = 1.0 - np.asarray(replay["zm_h_weighted_z"], float)
    a = float(eta_m) * np.asarray(replay["zm_h_weighted_m"], float)
    keep = time <= onset_ms + 1e-9
    ax.plot(d[keep], a[keep], color=TRAJ, lw=1.35)
    event_mid = 0.5 * (float(replay["sample_event_t_on_ms"])
                       + float(replay["sample_event_t_off_ms"]))
    event_i = int(np.argmin(np.abs(time - event_mid)))
    pre_i = int(np.argmin(np.abs(time - (onset_ms - 500.0))))
    onset_i = int(np.argmin(np.abs(time - onset_ms)))
    ax.scatter(d[event_i], a[event_i], marker="^", s=28, color=TRAJ, zorder=4)
    ax.scatter(d[pre_i], a[pre_i], s=22, fc="white", ec=TRAJ, lw=0.9, zorder=4)
    ax.scatter(d[onset_i], a[onset_i], s=30, color=ONSET, ec="white", lw=0.6, zorder=5)
    ax.axvline(d[onset_i], color=ONSET, ls="--", lw=0.9)
    ax.text(d[onset_i] - 0.002, a[onset_i], r"$\mathcal{S}$", color=ONSET,
            fontsize=13, ha="right", va="bottom")
    ax.set_xlabel(r"Disinhibition $D = 1-z$", fontsize=9.0)
    ax.set_ylabel(r"Adaptation $A = \eta_m m$", fontsize=9.0)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    return {"event_time_ms": event_mid, "pre_transition_time_ms": onset_ms - 500.0}


def _plot_event_order(ax, replay, registered_pos, registered_contacts, extent):
    first = np.asarray(replay["sample_first_spike_ms"], float)
    valid = np.isfinite(first)
    ranks = np.asarray(replay["sample_contact_ranks"], float)
    max_rank = max(1.0, float(np.nanmax(ranks)))
    ax.scatter(registered_pos[~valid, 0], registered_pos[~valid, 1], s=0.16,
               color="0.82", alpha=0.22, rasterized=True)
    if np.any(valid):
        relative = first[valid] - np.nanmin(first[valid])
        denom = max(float(np.nanmax(relative)), 1e-9)
        neuron_rank = relative / denom * max_rank
        ax.scatter(registered_pos[valid, 0], registered_pos[valid, 1], s=0.42,
                   c=neuron_rank, cmap="viridis", norm=Normalize(0, max_rank),
                   alpha=0.22, linewidths=0, rasterized=True)
    contact_valid = np.isfinite(ranks)
    mappable = ax.scatter(registered_contacts[contact_valid, 0],
                          registered_contacts[contact_valid, 1], s=43,
                          c=ranks[contact_valid], cmap="viridis",
                          norm=Normalize(0, max_rank), ec="black", lw=0.75, zorder=5)
    ax.set_title("Sample event order", fontsize=10.0, fontweight="bold")
    _style_spatial(ax, extent, show_ylabel=True)
    return mappable


def _sample_contact_field(positions, values, contacts, sigma_mm=0.75):
    positions = np.asarray(positions, float)
    values = np.asarray(values, float)
    out = np.empty(len(contacts), float)
    for index, contact in enumerate(np.asarray(contacts, float)):
        distance2 = np.sum(np.square(positions - contact), axis=1)
        weights = np.exp(-0.5 * distance2 / float(sigma_mm) ** 2)
        out[index] = float(np.dot(weights, values) / max(weights.sum(), 1e-12))
    return out


def _plot_energy(ax, replay, native_pos, native_contacts, registered_pos,
                 registered_contacts, extent):
    values = np.asarray(replay["early_activity_energy"], float) / 1e3
    grid = _grid_mean(registered_pos, values, extent, n=100, positive=True)
    positive_grid = grid[grid > 0]
    high = (float(np.percentile(positive_grid, 99.5))
            if positive_grid.size else 1.0)
    mappable = ax.imshow(
        grid.T, origin="lower", extent=(*extent, *extent), cmap="Blues",
        vmin=0.0, vmax=high, interpolation="bilinear", alpha=0.92)
    ax.scatter(registered_pos[:, 0], registered_pos[:, 1], s=0.10,
               color="0.72", alpha=0.10, linewidths=0, rasterized=True)
    active = values > 0
    ax.scatter(registered_pos[active, 0], registered_pos[active, 1], s=0.16,
               c=np.clip(values[active], 0, high), cmap="Blues", vmin=0, vmax=high,
               alpha=0.12, linewidths=0, rasterized=True)
    contact_energy = _sample_contact_field(
        native_pos, values, native_contacts)
    ax.scatter(registered_contacts[:, 0], registered_contacts[:, 1], s=43,
               c=contact_energy, cmap="Blues", vmin=0, vmax=high,
               ec="black", lw=0.75, zorder=5)
    ax.set_title("Early high-state activity", fontsize=10.0, fontweight="bold")
    _style_spatial(ax, extent, show_ylabel=False)
    return mappable


def _probe_inset(ax):
    inset = ax.inset_axes([0.05, 0.66, 0.27, 0.27])
    xx = np.linspace(-1, 1, 35)
    x, y = np.meshgrid(xx, xx)
    g = np.exp(-(x * x + y * y) / 0.12)
    inset.imshow(g, origin="lower", cmap="Greys", extent=(-1, 1, -1, 1))
    inset.set_xticks([]); inset.set_yticks([])
    for side in inset.spines.values(): side.set_visible(False)
    inset.set_title("spatial probe", fontsize=6.8, loc="left", pad=1.5)


def _plot_response(ax, grid, probe_xy, extent, title, vmax, show_ylabel=False,
                   show_probe=False):
    image = ax.imshow(grid.T, origin="lower", extent=(*extent, *extent), cmap="magma",
                      vmin=0.0, vmax=vmax, interpolation="bilinear")
    ax.scatter([probe_xy[0]], [probe_xy[1]], s=36, fc="none", ec="white",
               lw=1.0, zorder=5)
    ax.set_title(title, fontsize=10.0, fontweight="bold")
    _style_spatial(ax, extent, show_ylabel=show_ylabel)
    if show_probe:
        _probe_inset(ax)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_ictal_transition_v1.json")
    parser.add_argument("--reference-figdata", default=REFERENCE_FIGDATA)
    parser.add_argument("--replay", default=(
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition/fig5_replay/"
        "joint_04_control_seed_1801_frames.npz"))
    parser.add_argument("--output-root", default=(
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition"))
    parser.add_argument("--out-dir", default="results/paper-ready-figure/fig5/figures")
    parser.add_argument("--extent-mm", type=float, default=12.0)
    parser.add_argument("--minimum-perturbation-seeds", type=int, default=2)
    parser.add_argument("--allow-exploratory-workpoint", action="store_true")
    parser.add_argument("--stem", default="fig5-data-driven-zm-main")
    args = parser.parse_args()

    replay_path = ROOT / args.replay
    replay = _load_npz(replay_path)
    replay_meta = json.loads(replay_path.with_suffix(".json").read_text())
    morphology = _require_sustained_runaway(
        replay_meta,
        allow_exploratory_workpoint=bool(args.allow_exploratory_workpoint),
    )
    verification = replay_meta.get(
        "verification_against_reference_run",
        replay_meta.get("verification_against_archived_run"),
    )
    if not isinstance(verification, dict) or not verification.get("all_match"):
        raise RuntimeError("the Figure 5 replay does not match its frozen reference")
    required = {"lfp_trace", "sample_first_spike_ms", "early_activity_energy"}
    missing = sorted(required.difference(replay))
    if missing:
        raise RuntimeError(f"replay predates the main-figure contract; missing {missing}")

    config = json.loads((ROOT / args.config).read_text())
    output_root = ROOT / args.output_root
    low_paths = sorted((output_root / "perturbation").glob(
        "joint_04_control_seed_*_low_activity_representative.npz"))
    pre_paths = sorted((output_root / "perturbation").glob(
        "joint_04_control_seed_*_pre_ictal_representative.npz"))
    if not low_paths or not pre_paths:
        raise RuntimeError(
            "Panel D requires low-activity and pre-transition representative-site artifacts")

    onset_ms = float(replay_meta.get(
        "morphology_onset_ms", replay_meta["model_ictal_onset_ms"]))
    extent = (-float(args.extent_mm), float(args.extent_mm))
    display = _load_accepted_display(replay, ROOT / args.reference_figdata)
    positions = _accepted_display_xy(replay["positions_E"], display)
    contacts = _accepted_display_xy(replay["contact_xy_mm"], display)
    low_grid, low_probe, low_used, low_excluded, low_ignited = _aggregate_probe_fields(
        low_paths, replay, output_root, extent, display)
    pre_grid, pre_probe, pre_used, pre_excluded, pre_ignited = _aggregate_probe_fields(
        pre_paths, replay, output_root, extent, display)
    if not np.allclose(low_probe, pre_probe, atol=1e-8, rtol=0.0):
        raise RuntimeError("low-activity and pre-transition probes differ")
    paired = sorted(set(low_used).intersection(pre_used))
    minimum_seeds = int(args.minimum_perturbation_seeds)
    if minimum_seeds < 1:
        raise ValueError("minimum perturbation seeds must be positive")
    if len(paired) < minimum_seeds:
        raise RuntimeError(
            f"Panel D needs >={minimum_seeds} paired seeds, got {paired}")

    fig = plt.figure(figsize=(15.4, 7.4), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[0.78, 1.10],
                             left=0.055, right=0.985, bottom=0.09, top=0.96,
                             hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=[3.65, 1.0], wspace=0.16)
    bottom = outer[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.0],
                                  wspace=0.22)
    ax_a = fig.add_subplot(top[0, 0]); ax_b = fig.add_subplot(top[0, 1])
    ax_c1 = fig.add_subplot(bottom[0, 0]); ax_c2 = fig.add_subplot(bottom[0, 1])
    ax_d1 = fig.add_subplot(bottom[0, 2]); ax_d2 = fig.add_subplot(bottom[0, 3])

    readout_meta, rate_ax = _plot_readout(ax_a, replay, onset_ms, morphology)
    trajectory_meta = _plot_trajectory(ax_b, replay, onset_ms,
                                       config["zm"]["eta_m"])
    rank_map = _plot_event_order(ax_c1, replay, positions, contacts, extent)
    energy_map = _plot_energy(
        ax_c2, replay, replay["positions_E"], replay["contact_xy_mm"],
        positions, contacts, extent)
    vmax = max(float(np.percentile(low_grid, 99)), float(np.percentile(pre_grid, 99)), 1e-9)
    response_map = _plot_response(ax_d1, low_grid, low_probe, extent,
                                  "Baseline probe response", vmax,
                                  show_ylabel=False, show_probe=True)
    _plot_response(ax_d2, pre_grid, pre_probe, extent,
                   "Pre-transition response (-500 ms)", vmax,
                   show_ylabel=False, show_probe=False)

    _panel_label(rate_ax, "A", x=-0.075, y=1.35)
    _panel_label(ax_b, "B", x=-0.18, y=1.16)
    _panel_label(ax_c1, "C", x=-0.25, y=1.16)
    _panel_label(ax_d1, "D", x=-0.18, y=1.16)

    cb_rank = fig.colorbar(rank_map, ax=ax_c1, fraction=0.045, pad=0.025)
    cb_rank.set_label("contact order\n(0 = first)", fontsize=7.2)
    cb_rank.ax.tick_params(labelsize=6.8)
    cb_energy = fig.colorbar(energy_map, ax=ax_c2, fraction=0.045, pad=0.025)
    cb_energy.set_label(r"activity energy\n($\times 10^3$ Hz$^2$)", fontsize=7.2)
    cb_energy.ax.tick_params(labelsize=6.8)
    cb_resp = fig.colorbar(response_map, ax=[ax_d1, ax_d2], fraction=0.026, pad=0.018)
    cb_resp.set_label("immediate descendant response\n(0-50 ms excess spikes per local E cell)",
                      fontsize=7.2)
    cb_resp.ax.tick_params(labelsize=6.8)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.stem)
    outputs = []
    for suffix, kwargs in (("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", **kwargs)
        outputs.append(str(path.relative_to(ROOT)))
    plt.close(fig)

    metadata = {
        "figure": stem,
        "layout_contract": "reference A-D transition layout supplied by the author",
        "substrate": "frozen data-driven Joint (Node + E->E + E->I), Z/M active",
        "workpoint_parameters": replay_meta.get("workpoint_parameters"),
        "exploratory_workpoint_override": bool(args.allow_exploratory_workpoint),
        "seed": int(replay_meta["seed"]),
        "selection": replay_meta["sample_event_selection"],
        "registered_display": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in display.items()
        },
        "panel_A": {**readout_meta,
                    "readout": "current-based LFPRecorder proxy, signed 30-80 Hz",
                    "scaling": "one shared p99 frozen from all pre-runaway contacts",
                    "morphology_contract": morphology["classification"]},
        "panel_B": trajectory_meta,
        "panel_C": {"event_order": ("per-neuron first spike plus frozen contact order; "
                                      "no patient-mode label is assigned"),
                    "energy": ("100 ms post-transition spike-rate-squared field in 1e3 Hz^2; "
                               "contact colours sample that same field with a fixed 0.75 mm Gaussian kernel")},
        "panel_D": {"probe_site": "source (geometry-frozen; not response-selected)",
                    "dose_cells": 64,
                    "endpoint": ("exploratory 0-50 ms immediate descendant response; "
                                 "the preregistered 200 ms finite-response gate remains failed"),
                    "paired_seeds": paired,
                    "low_activity_used": low_used, "pre_transition_used": pre_used,
                    "low_activity_excluded": low_excluded,
                    "pre_transition_excluded": pre_excluded,
                    "baseline_source_ignited_seeds": low_ignited,
                    "pre_transition_source_ignited_seeds": pre_ignited,
                    "response": "positive descendant excess over paired sham, 0-50 ms"},
        "claim_boundary": ("single data-driven SNN trajectory plus three-network perturbation "
                           "canary; operational model transition, not clinical seizure onset; "
                           "NO_SUBEVENT_PROBE_REGIME remains the formal 200 ms E1 result"),
        "outputs": outputs,
    }
    atomic_write_json(metadata, str(out_dir / f"{stem}-metadata.json"))
    print(json.dumps({"figure": stem, "outputs": outputs, "paired_seeds": paired}))


if __name__ == "__main__":
    main()
