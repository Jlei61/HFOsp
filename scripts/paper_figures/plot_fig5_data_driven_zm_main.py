#!/usr/bin/env python3
"""Paper-facing Figure 5 in the accepted A-D transition layout.

A  one continuous signed 30-80 Hz virtual-contact current-proxy readout
B  h-weighted Z/M trajectory
C  one rule-selected interictal event and early-runaway recruitment
D  signed response to one frozen probe at low activity and runaway onset

The diagnostic three-panel GIF remains a supplement. This producer never
re-simulates the SNN.  Panel C uses the same raw 0-20 mm sheet coordinates as
Figure 4; Panel D consumes an exact-resume state contrast from this trajectory.
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
            "population_frequency_increased",
        }
        majority_duty = min(
            float(recruitment.get("fraction_windows_majority_E_active", 0.0)),
            float(recruitment.get("fraction_windows_majority_sheet_recruited", 0.0)),
        )
        population = morphology.get("population_rate_frequency", {})
        population_frequency_override = (
            "population_frequency_increased" not in failed
            or (
                bool(checks.get("contact_frequency_increased"))
                and bool(checks.get("population_rate_increased"))
                and float(population.get("spectral_centroid_shift_hz", -np.inf)) >= 5.0
            )
        )
        exploratory_ok = (
            allow_exploratory_workpoint
            and set(failed).issubset(allowed_failures)
            and majority_duty >= 0.90
            and population_frequency_override
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


def _style_spatial(ax, extent, show_ylabel=False):
    ax.set_xlim(extent); ax.set_ylim(extent); ax.set_aspect("equal")
    ax.set_xlabel("sheet x (mm)", fontsize=8.2)
    if show_ylabel:
        ax.set_ylabel("sheet y (mm)", fontsize=8.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7.2, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_label(ax, label, x=-0.18, y=1.11):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18, fontweight="bold",
            ha="left", va="top")


def _plot_readout(ax, replay, onset_ms, morphology, *, window_start_ms=None):
    names = np.asarray(replay["contact_names"]).astype(str)
    shafts = np.asarray(replay["shaft_ids"]).astype(str)
    order = _contact_order(names)
    signed = _signed_bandpass(replay["lfp_trace"], float(replay["lfp_dt_ms"]))
    times = np.arange(signed.shape[0]) * float(replay["lfp_dt_ms"])
    event_on = float(replay["sample_event_t_on_ms"])
    event_off = float(replay["sample_event_t_off_ms"])
    if window_start_ms is None:
        start = max(0.0, min(event_on - 100.0, onset_ms - 1200.0))
    else:
        start = max(0.0, float(window_start_ms))
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
        Line2D([0], [0], color=ONSET, ls="--", lw=1.1,
               label="high-state onset"),
        Patch(facecolor=EVENT_SHADE, edgecolor="none", label="sample interictal event"),
    ], frameon=False, fontsize=7.5, loc="upper right", ncol=2,
       bbox_to_anchor=(1.0, 1.42), borderaxespad=0.0)
    ax.text(float(ts[-1]) - 12, y[-1] + 0.62, "shared pre-runaway scale",
            fontsize=6.8, ha="right", va="top", color="0.35")
    frequency = morphology["population_rate_frequency"]
    return {"window_start_ms": start, "window_stop_ms": stop,
            "sample_event_t_on_ms": event_on, "sample_event_t_off_ms": event_off,
            "scale_interval": "all displayed samples before high-state onset",
            "global_recruitment_strip": {
                "measure": ("20-ms active-E fraction and fraction of 1-mm sheet "
                            "bins with >=50% local recruitment"),
                "majority_threshold": 0.5,
            },
            "population_frequency_pre_hz": frequency["spectral_centroid_pre_hz"],
            "population_frequency_post_hz": frequency["spectral_centroid_post_hz"],
            "figure_workpoint_status": morphology.get(
                "figure_workpoint_status", morphology["classification"]["status"]),
            "formal_morphology_status": morphology["classification"]["status"]}, rate_ax


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


def _event_order_grid(positions, first_spike_ms, extent, n=64):
    positions = np.asarray(positions, float)
    first = np.asarray(first_spike_ms, float)
    lo, hi = float(extent[0]), float(extent[1])
    edges = np.linspace(lo, hi, int(n) + 1)
    occupancy, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1], bins=(edges, edges))
    valid = np.isfinite(first)
    participating, _, _ = np.histogram2d(
        positions[valid, 0], positions[valid, 1], bins=(edges, edges))
    relative = first[valid] - float(np.nanmin(first))
    weighted, _, _ = np.histogram2d(
        positions[valid, 0], positions[valid, 1], bins=(edges, edges),
        weights=relative)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_onset = weighted / participating
        participation = participating / occupancy
    mean_onset[participating == 0] = np.nan
    participation = np.nan_to_num(participation)
    return mean_onset, participation


def _plot_event_order(ax, replay, positions, contacts, extent):
    first = np.asarray(replay["sample_first_spike_ms"], float)
    valid = np.isfinite(first)
    ranks = np.asarray(replay["sample_contact_ranks"], float)
    max_rank = max(1.0, float(np.nanmax(ranks)))
    mean_onset, participation = _event_order_grid(
        positions, first, extent, n=64)
    duration = max(float(np.nanmax(first[valid]) - np.nanmin(first[valid])), 1e-9)
    order_grid = mean_onset / duration * max_rank
    alpha = np.clip((participation - 0.05) / 0.75, 0.0, 1.0)
    image = ax.imshow(
        order_grid.T, origin="lower", extent=(*extent, *extent),
        cmap="viridis", norm=Normalize(0, max_rank), interpolation="nearest",
        alpha=alpha.T)
    axis = np.linspace(extent[0], extent[1], participation.shape[0])
    if np.nanmax(participation) >= 0.5:
        ax.contour(axis, axis, participation.T, levels=[0.5], colors=["0.2"],
                   linewidths=0.65, alpha=0.8)
    quantiles = np.quantile(first[valid], [0.15, 0.85])
    early = np.mean(positions[valid & (first <= quantiles[0])], axis=0)
    late = np.mean(positions[valid & (first >= quantiles[1])], axis=0)
    ax.annotate("", xy=late, xytext=early,
                arrowprops={"arrowstyle": "-|>", "color": "white",
                            "lw": 3.0, "mutation_scale": 12}, zorder=6)
    ax.annotate("", xy=late, xytext=early,
                arrowprops={"arrowstyle": "-|>", "color": "0.15",
                            "lw": 1.25, "mutation_scale": 11}, zorder=7)
    contact_valid = np.isfinite(ranks)
    mappable = ax.scatter(contacts[contact_valid, 0], contacts[contact_valid, 1], s=43,
                          c=ranks[contact_valid], cmap="viridis",
                          norm=Normalize(0, max_rank), ec="black", lw=0.75, zorder=5)
    # Keep the image and contacts on the identical normalization contract.
    image.set_norm(mappable.norm)
    ax.set_title("Interictal event propagation", fontsize=10.0, fontweight="bold")
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


def _runaway_activity_grid(replay, onset_ms, activity_window_ms,
                           *, start_offset_ms=0.0, duration_ms=100.0):
    time = np.asarray(replay["frame_time_ms"], float)
    start = float(onset_ms) + float(start_offset_ms)
    stop = start + float(duration_ms)
    keep = (time >= start) & (time < stop)
    if not np.any(keep):
        raise RuntimeError("replay has no activity frames in the requested runaway window")
    counts = np.asarray(replay["activity_spike_counts"], float)[keep]
    occupancy = np.asarray(replay["activity_cell_occupancy"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.mean(counts, axis=0) / occupancy / (float(activity_window_ms) * 1e-3)
    return np.nan_to_num(rate), start, stop


def _plot_runaway_activity(ax, replay, contacts, extent, onset_ms,
                           activity_window_ms, start_offset_ms=0.0):
    grid, start, stop = _runaway_activity_grid(
        replay, onset_ms, activity_window_ms,
        start_offset_ms=start_offset_ms, duration_ms=100.0)
    positive = grid[grid > 0]
    high = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    mappable = ax.imshow(
        grid.T, origin="lower", extent=(*extent, *extent), cmap="magma",
        vmin=0.0, vmax=high, interpolation="nearest")
    ax.scatter(contacts[:, 0], contacts[:, 1], s=43,
               fc="white", ec="black", lw=0.75, zorder=5)
    ax.set_title("Runaway recruitment (0 to +100 ms)",
                 fontsize=10.0, fontweight="bold")
    _style_spatial(ax, extent, show_ylabel=False)
    return mappable, {"window_start_ms": start, "window_stop_ms": stop,
                      "measure": "mean 10-ms local E-neuron firing rate"}


def _probe_inset(ax):
    inset = ax.inset_axes([0.05, 0.66, 0.27, 0.27])
    xx = np.linspace(-1, 1, 35)
    x, y = np.meshgrid(xx, xx)
    g = np.exp(-(x * x + y * y) / 0.12)
    inset.imshow(g, origin="lower", cmap="Greys", extent=(-1, 1, -1, 1))
    inset.set_xticks([]); inset.set_yticks([])
    for side in inset.spines.values(): side.set_visible(False)
    inset.set_title("spatial probe", fontsize=6.8, loc="left", pad=1.5)


def _plot_response(ax, contacts, response_grid, probe_xy, extent, title, vmax,
                   show_ylabel=False, show_probe=False):
    image = ax.imshow(response_grid.T, origin="lower", extent=(*extent, *extent),
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="bilinear")
    ax.scatter(contacts[:, 0], contacts[:, 1], s=16, fc="white", ec="0.2",
               lw=0.45, alpha=0.85, zorder=4)
    ax.scatter([probe_xy[0]], [probe_xy[1]], s=36, fc="none", ec="white",
               lw=1.0, zorder=5)
    ax.set_title(title, fontsize=10.0, fontweight="bold")
    _style_spatial(ax, extent, show_ylabel=show_ylabel)
    if show_probe:
        _probe_inset(ax)
    return image


def _state_contrast_payload(path, replay_meta, *, baseline_early_ceiling):
    block = _load_npz(path)
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    if not bool(meta.get("continuation_rate_exact")):
        raise RuntimeError("Panel D continuation does not match the verified replay")
    if int(meta["seed"]) != int(replay_meta["seed"]):
        raise RuntimeError("Panel D state contrast and replay use different seeds")
    if meta["workpoint_parameters"] != replay_meta["workpoint_parameters"]:
        raise RuntimeError("Panel D state contrast and replay use different work points")
    doses = np.asarray(block["dose_cells"], int)
    eligible = [
        row for row in meta["low_probe_scan"]
        if bool(row["e1_evaluable"])
        and abs(float(row["excess_spikes_early"])) <= float(baseline_early_ceiling)
    ]
    if not eligible:
        raise RuntimeError("Panel D has no near-zero low-activity probe dose")
    selected = max(int(row["dose_cells"]) for row in eligible)
    where = np.flatnonzero(doses == selected)
    if where.size != 1:
        raise RuntimeError("Panel D selected probe dose is not unique")
    low_rows = {int(row["dose_cells"]): row for row in meta["low_probe_scan"]}
    if not bool(low_rows[selected]["e1_evaluable"]):
        raise RuntimeError("Panel D selected dose ignites the low-activity network")
    return block, meta, int(where[0]), selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_ictal_transition_v1.json")
    parser.add_argument("--replay", default=(
        "results/topic4_sef_hfo/data_driven_zm_ictal_transition/fig5_replay/"
        "joint_04_control_seed_1801_frames.npz"))
    parser.add_argument("--state-contrast", required=True)
    parser.add_argument("--out-dir", default="results/paper-ready-figure/fig5/figures")
    parser.add_argument("--sheet-size-mm", type=float, default=20.0)
    parser.add_argument("--baseline-early-response-ceiling", type=float, default=50.0)
    parser.add_argument("--display-onset-offset-ms", type=float, default=300.0)
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
    required = {"lfp_trace", "sample_first_spike_ms", "frame_time_ms",
                "activity_spike_counts", "activity_cell_occupancy"}
    missing = sorted(required.difference(replay))
    if missing:
        raise RuntimeError(f"replay predates the main-figure contract; missing {missing}")

    config = json.loads((ROOT / args.config).read_text())
    morphology_onset_ms = float(replay_meta.get(
        "morphology_onset_ms", replay_meta["model_ictal_onset_ms"]))
    display_onset_ms = morphology_onset_ms + float(args.display_onset_offset_ms)
    display_window_start_ms = max(0.0, morphology_onset_ms - 1200.0)
    extent = (0.0, float(args.sheet_size_mm))
    positions = np.asarray(replay["positions_E"], float)
    contacts = np.asarray(replay["contact_xy_mm"], float)
    if (np.min(positions) < extent[0] - 1e-9
            or np.max(positions) > extent[1] + 1e-9):
        raise RuntimeError("Panel C positions fall outside the Figure 4 sheet coordinates")
    contrast, contrast_meta, dose_index, selected_dose = _state_contrast_payload(
        ROOT / args.state_contrast, replay_meta,
        baseline_early_ceiling=float(args.baseline_early_response_ceiling))
    if not np.array_equal(positions.astype(np.float32), contrast["positions_E"]):
        raise RuntimeError("Panel D state contrast uses a different neuron sheet")
    probe_xy = np.asarray(contrast["site_xy_mm"], float)
    low_response = np.asarray(contrast["low_response_early"][dose_index], float)
    post_response = np.asarray(contrast["post_response_early"][dose_index], float)
    low_response_grid = _grid_mean(
        positions, low_response, extent, n=100, positive=False)
    post_response_grid = _grid_mean(
        positions, post_response, extent, n=100, positive=False)
    grid_values = np.abs(np.concatenate([
        low_response_grid.ravel(), post_response_grid.ravel()]))
    grid_values = grid_values[grid_values > 1e-12]
    if grid_values.size == 0:
        raise RuntimeError("Panel D has no nonzero signed response in either state")
    vmax = max(float(np.percentile(grid_values, 99.5)), 1e-6)
    expected_post_time = display_onset_ms
    if not np.isclose(float(contrast_meta["post_time_ms"]), expected_post_time,
                      atol=1e-6):
        raise RuntimeError(
            "Panel D post state does not match the figure high-state onset")

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

    readout_meta, rate_ax = _plot_readout(
        ax_a, replay, display_onset_ms, morphology,
        window_start_ms=display_window_start_ms)
    eta_m = replay_meta.get("workpoint_parameters", config["zm"])["eta_m"]
    trajectory_meta = _plot_trajectory(ax_b, replay, display_onset_ms, eta_m)
    rank_map = _plot_event_order(ax_c1, replay, positions, contacts, extent)
    activity_map, activity_meta = _plot_runaway_activity(
        ax_c2, replay, contacts, extent, display_onset_ms,
        float(replay_meta["activity_window_ms"]), start_offset_ms=0.0)
    response_map = _plot_response(
        ax_d1, contacts, low_response_grid, probe_xy, extent,
        "Low-activity response (1.0 s)", vmax,
        show_ylabel=False, show_probe=True)
    _plot_response(
        ax_d2, contacts, post_response_grid, probe_xy, extent,
        "Runaway-onset response", vmax,
        show_ylabel=False, show_probe=False)

    _panel_label(rate_ax, "A", x=-0.075, y=1.35)
    _panel_label(ax_b, "B", x=-0.18, y=1.16)
    _panel_label(ax_c1, "C", x=-0.25, y=1.16)
    _panel_label(ax_d1, "D", x=-0.18, y=1.16)

    cb_rank = fig.colorbar(rank_map, ax=ax_c1, fraction=0.045, pad=0.025)
    cb_rank.set_label("contact order\n(0 = first)", fontsize=7.2)
    cb_rank.ax.tick_params(labelsize=6.8)
    cb_activity = fig.colorbar(activity_map, ax=ax_c2, fraction=0.045, pad=0.025)
    cb_activity.set_label("local E-neuron rate (Hz)", fontsize=7.2)
    cb_activity.ax.tick_params(labelsize=6.8)
    cb_resp = fig.colorbar(response_map, ax=[ax_d1, ax_d2], fraction=0.026, pad=0.018)
    cb_resp.set_label("signed probe effect\n(0-50 ms excess spikes per local E cell)",
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
        "coordinate_contract": ("raw 0-20 mm sheet x/y, identical to Figure 4; "
                                "no display rotation, reflection or similarity transform"),
        "panel_A": {**readout_meta,
                    "readout": "current-based LFPRecorder proxy, signed 30-80 Hz",
                    "scaling": "one shared p99 frozen from all pre-runaway contacts",
                    "morphology_contract": morphology["classification"]},
        "panel_B": trajectory_meta,
        "time_contract": {
            "formal_morphology_onset_ms": morphology_onset_ms,
            "figure_high_state_onset_ms": display_onset_ms,
            "figure_offset_from_formal_onset_ms": float(
                args.display_onset_offset_ms),
            "reason": ("author-selected onset of visibly global sustained high "
                       "activity; the formal detector onset remains unchanged"),
        },
        "panel_C": {"event_order": ("per-neuron first spike plus frozen contact order; "
                                      "64x64 unsmoothed onset bins, 50% recruitment contour, "
                                      "15%-to-85% onset-centroid arrow; no patient-mode label"),
                    "runaway_activity": activity_meta},
        "panel_D": {
            "state_contrast": str((ROOT / args.state_contrast).relative_to(ROOT)),
            "probe_site": "source (geometry-frozen; not response-selected)",
            "dose_cells": selected_dose,
            "dose_selection": ("largest low-activity non-igniting dose with "
                               "absolute 0-50 ms excess <= "
                               f"{float(args.baseline_early_response_ceiling):g} spikes; "
                               "post-onset response was not used for selection"),
            "low_time_ms": contrast_meta["low_time_ms"],
            "post_time_ms": contrast_meta["post_time_ms"],
            "post_offset_from_scientific_onset_ms": contrast_meta[
                "post_offset_from_scientific_onset_ms"],
            "response": "signed descendant probe-minus-sham effect, 0-50 ms",
            "rendering": "response only; no slow-state or substrate overlay",
            "not_rendered_slow_state_context": contrast_meta["h_weighted_state"],
            "continuation_rate_exact": contrast_meta["continuation_rate_exact"],
        },
        "claim_boundary": ("single selected data-driven SNN trajectory and its exact-resume "
                           "paired perturbation contrast; operational model transition, not "
                           "clinical seizure onset or multi-seed confirmation"),
        "outputs": outputs,
    }
    atomic_write_json(metadata, str(out_dir / f"{stem}-metadata.json"))
    print(json.dumps({"figure": stem, "outputs": outputs,
                      "selected_dose_cells": selected_dose}))


if __name__ == "__main__":
    main()
