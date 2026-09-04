#!/usr/bin/env python3
"""Render the paper-ready A-D Fig. 5 layout for spatial Z/M + OU tonic runaway.

Panel semantics deliberately match the author-supplied reference:
A continuous virtual-contact readout plus global recruitment;
B Z/M state trajectory;
C low-state event order plus early-runaway activity energy;
D paired low-state versus early-runaway perturbation response.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter, uniform_filter1d


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
ARCHIVE = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou"
)
SOURCE_NPZ = ARCHIVE / "tonic_confirmation_v2/tonic_b0_v2_s1842.npz"
SOURCE_JSON = SOURCE_NPZ.with_suffix(".json")
AGGREGATE_JSON = ARCHIVE / "tonic_runaway_aggregate.json"
STATIC_NPZ = ARCHIVE / "paper_ready_fig5_static/seed1842_static_panels.npz"
STATIC_JSON = STATIC_NPZ.with_suffix(".json")
DEFAULT_OUT = (
    CANONICAL_ROOT / "results/paper-ready-figure/"
    "fig5_spatial_zm_ou_tonic/figures"
)

ICL = "#F1783A"
SCL = "#29A6B5"
ONSET = "#D62745"
TRAJ = "#304D73"
EVENT_SHADE = "#DCEAF6"
STATE_SHADE = "#F7E9ED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def grid_mean(xy, values, extent, n=100):
    xy = np.asarray(xy, float)
    values = np.asarray(values, float)
    edges = np.linspace(float(extent[0]), float(extent[1]), int(n) + 1)
    total, _, _ = np.histogram2d(
        xy[:, 0], xy[:, 1], bins=(edges, edges), weights=values)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(edges, edges))
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = total / count
    return gaussian_filter(np.nan_to_num(grid), sigma=1.15)


def contact_order(names):
    def number(name):
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else -1

    return np.asarray(sorted(range(len(names)), key=lambda index: (
        0 if str(names[index]).startswith("SCL") else 1,
        number(names[index]),
    )), int)


def normalised_contact_plateau(raw, *, dt_ms, onset_ms):
    raw = np.asarray(raw, float)
    smooth = uniform_filter1d(
        raw, size=max(1, int(round(5.0 / dt_ms))), axis=0, mode="nearest")
    time = np.arange(len(smooth), dtype=float) * float(dt_ms)
    pre = (time >= max(0.0, onset_ms - 500.0)) & (time < onset_ms)
    post = (time >= onset_ms + 300.0) & (time < onset_ms + 1300.0)
    baseline = np.median(smooth[pre], axis=0)
    plateau = np.median(smooth[post], axis=0)
    scale = plateau - baseline
    if not np.all(scale > 1e-9):
        raise RuntimeError("a virtual contact lacks a positive tonic step")
    return (smooth - baseline[None, :]) / scale[None, :]


def style_spatial(ax, extent, show_ylabel=False):
    ax.set_xlim(extent)
    ax.set_ylim(extent)
    ax.set_aspect("equal")
    ax.set_xlabel("sheet x (mm)", fontsize=8.2)
    if show_ylabel:
        ax.set_ylabel("sheet y (mm)", fontsize=8.2)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=7.2, length=2.5)
    ax.spines[["top", "right"]].set_visible(False)


def panel_label(ax, label, x=-0.18, y=1.11):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=18,
            fontweight="bold", ha="left", va="top")


def plot_readout(ax, source, arrays, static):
    onset_ms = float(source["scientific_onset_ms"])
    dt_ms = float(arrays["lfp_dt_ms"])
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    order = contact_order(names)
    trace = normalised_contact_plateau(
        arrays["lfp_trace"], dt_ms=dt_ms, onset_ms=onset_ms)[:, order]
    times = np.arange(len(trace), dtype=float) * dt_ms
    event_on = float(static["sample_event_start_ms"])
    event_off = float(static["sample_event_stop_ms"])
    offsets = np.arange(len(order), dtype=float) * 1.18

    original = ax.get_position()
    gap = 0.025 * original.height
    recruitment_height = 0.205 * original.height
    trace_height = original.height - recruitment_height - gap
    ax.set_position([original.x0, original.y0, original.width, trace_height])
    recruitment_ax = ax.figure.add_axes([
        original.x0, original.y0 + trace_height + gap,
        original.width, recruitment_height,
    ], sharex=ax)

    for axis in (ax, recruitment_ax):
        axis.axvspan(event_on, event_off, color=EVENT_SHADE, alpha=0.62,
                    lw=0, zorder=0)
        axis.axvspan(onset_ms, times[-1], color=STATE_SHADE, alpha=0.95,
                    lw=0, zorder=0)
        axis.axvline(onset_ms, color=ONSET, lw=1.0, ls="--")
        axis.set_xlim(0.0, float(times[-1]))

    for row, contact_index in enumerate(order):
        colour = ICL if shafts[contact_index] == "ICL" else SCL
        ax.plot(times, np.clip(trace[:, row], -0.25, 1.25) + offsets[row],
                color=colour, lw=0.78, alpha=0.97)
    ax.set_ylim(-0.8, offsets[-1] + 1.0)
    ax.set_yticks(offsets)
    ax.set_yticklabels(names[order], fontsize=7.5)
    for tick, contact_index in zip(ax.get_yticklabels(), order):
        tick.set_color(ICL if shafts[contact_index] == "ICL" else SCL)
    ax.set_xlabel("Time in continuous trajectory (ms)", fontsize=9.0)
    ax.set_ylabel("Virtual-SEEG current proxy\n(normalized tonic level)",
                  fontsize=9.0)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(times[-1] - 12.0, offsets[-1] + 0.62,
            "each contact: pre median = 0, plateau median = 1",
            fontsize=6.5, ha="right", va="top", color="0.35")

    recruitment_time = np.asarray(arrays["full_field_time_ms"], float)
    recruitment_ax.plot(
        recruitment_time,
        100.0 * np.asarray(arrays["active_neuron_fraction_20ms"], float),
        color="0.18", lw=0.9, label="active E neurons")
    recruitment_ax.plot(
        recruitment_time,
        100.0 * np.asarray(arrays["recruited_spatial_fraction_1mm"], float),
        color="#6D7F91", lw=0.9, label="recruited sheet")
    recruitment_ax.axhline(50.0, color=ONSET, lw=0.8, ls=":")
    recruitment_ax.set_ylim(0.0, 102.0)
    recruitment_ax.set_yticks((0.0, 50.0, 100.0))
    recruitment_ax.set_ylabel("Global\nrecruitment (%)", fontsize=7.2,
                              labelpad=2.0)
    recruitment_ax.tick_params(axis="y", labelsize=6.6, length=2.2)
    recruitment_ax.tick_params(axis="x", which="both", bottom=False,
                               labelbottom=False)
    recruitment_ax.spines[["top", "right"]].set_visible(False)
    recruitment_ax.text(onset_ms + 10.0, 96.0, "tonic global runaway",
                        fontsize=7.0, ha="left", va="top", color=ONSET,
                        fontweight="bold")
    recruitment_ax.text(times[-1] - 12.0, 52.0, "majority threshold",
                        fontsize=6.4, ha="right", va="bottom", color=ONSET)
    recruitment_ax.legend(handles=[
        Line2D([0], [0], color=ONSET, ls="--", lw=1.1,
               label="runaway onset"),
        Patch(facecolor=EVENT_SHADE, edgecolor="none",
              label="sample low-state event"),
    ], frameon=False, fontsize=7.5, loc="upper right", ncol=2,
       bbox_to_anchor=(1.0, 1.42), borderaxespad=0.0)
    return recruitment_ax, {
        "measure": (
            "20-ms active-E fraction and fraction of 1-mm sheet bins with "
            ">=50% local recruitment"),
        "readout": (
            "current-based LFPRecorder proxy; 5-ms smoothing; each contact "
            "mapped so pre-onset median=0 and onset+300..1300-ms median=1; "
            "no detrending or band-pass filtering"),
        "sample_event_window_ms": [event_on, event_off],
    }


def plot_trajectory(ax, source, arrays, static):
    time = np.asarray(arrays["slow_time_ms"], float)
    d = 1.0 - np.asarray(arrays["slow_q_core_mean"], float)
    a = np.asarray(arrays["slow_adaptation_current_mean"], float)
    onset_ms = float(source["scientific_onset_ms"])
    keep = time <= onset_ms + 1e-9
    ax.plot(d[keep], a[keep], color=TRAJ, lw=1.35)
    event_ms = float(static["sample_event_peak_ms"])
    low_ms = 200.0
    event_i = int(np.argmin(np.abs(time - event_ms)))
    low_i = int(np.argmin(np.abs(time - low_ms)))
    onset_i = int(np.argmin(np.abs(time - onset_ms)))
    ax.scatter(d[event_i], a[event_i], marker="^", s=28,
               color=TRAJ, zorder=4)
    ax.scatter(d[low_i], a[low_i], s=22, fc="white", ec=TRAJ,
               lw=0.9, zorder=4)
    ax.scatter(d[onset_i], a[onset_i], s=30, color=ONSET, ec="white",
               lw=0.6, zorder=5)
    ax.axvline(d[onset_i], color=ONSET, ls="--", lw=0.9)
    ax.text(d[onset_i] - 0.002, a[onset_i], r"$\mathcal{S}$",
            color=ONSET, fontsize=13, ha="right", va="bottom")
    ax.set_xlabel(r"Disinhibition $D = 1-q_{core}$", fontsize=9.0)
    ax.set_ylabel(r"Adaptation $A = \langle I_M\rangle$", fontsize=9.0)
    ax.tick_params(labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    return {
        "trajectory_until_ms": onset_ms,
        "event_marker_ms": event_ms,
        "open_marker_ms": low_ms,
        "onset_marker_ms": onset_ms,
        "D": "1 - archived high-h/core mean q",
        "A": "archived population-mean M/gK adaptation current",
    }


def plot_event_order(ax, static, positions, contacts, extent):
    first = np.asarray(static["sample_first_spike_ms"], float)
    contact_first = np.asarray(static["sample_contact_first_spike_ms"], float)
    event_start = float(static["sample_event_start_ms"])
    valid = np.isfinite(first)
    relative = first[valid] - event_start
    contact_relative = contact_first - event_start
    high = max(float(np.nanmax(relative)), 1.0)
    norm = Normalize(0.0, high)
    ax.scatter(positions[~valid, 0], positions[~valid, 1], s=0.20,
               color="0.80", alpha=0.18, linewidths=0, rasterized=True)
    ax.scatter(positions[valid, 0], positions[valid, 1], s=1.45,
               c=relative, cmap="viridis", norm=norm, alpha=0.82,
               linewidths=0, rasterized=True)
    contact_valid = np.isfinite(contact_relative)
    mappable = ax.scatter(
        contacts[contact_valid, 0], contacts[contact_valid, 1], s=43,
        c=contact_relative[contact_valid], cmap="viridis", norm=norm,
        ec="black", lw=0.75, zorder=5)
    ax.set_title("Low-state event order", fontsize=10.0, fontweight="bold")
    style_spatial(ax, extent, show_ylabel=True)
    return mappable, {
        "measure": "exact per-neuron first-spike time in the fixed event window",
        "event_window_ms": [event_start, float(static["sample_event_stop_ms"])],
        "contact_measure": "median first-spike time among local E neurons",
        "spatial_smoothing": "none",
    }


def plot_activity_energy(ax, static, positions, contacts, extent):
    energy = np.asarray(static["early_activity_energy"], float) / 1e3
    grid = grid_mean(positions, energy, extent, n=100)
    positive = grid[grid > 0]
    high = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    image = ax.imshow(
        grid.T, origin="lower", extent=(*extent, *extent), cmap="Blues",
        vmin=0.0, vmax=high, interpolation="bilinear")
    contact_energy = np.empty(len(contacts), float)
    for index, contact in enumerate(contacts):
        distance2 = np.sum(np.square(positions - contact), axis=1)
        weights = np.exp(-0.5 * distance2 / 0.75 ** 2)
        contact_energy[index] = np.dot(weights, energy) / max(weights.sum(), 1e-12)
    ax.scatter(contacts[:, 0], contacts[:, 1], s=43, c=contact_energy,
               cmap="Blues", vmin=0.0, vmax=high, ec="black", lw=0.75,
               zorder=5)
    ax.set_title("Early-runaway activity energy", fontsize=10.0,
                 fontweight="bold")
    style_spatial(ax, extent, show_ylabel=False)
    return image, {
        "measure": "square of per-neuron firing rate, displayed in 1e3 Hz^2",
        "window_ms": [float(static["early_activity_energy_start_ms"]),
                      float(static["early_activity_energy_stop_ms"])],
        "selection": "frozen onset-aligned 100-ms window; image pixels unused",
    }


def plot_response(ax, contacts, response_grid, extent, title, vmax,
                  show_ylabel=False):
    image = ax.imshow(
        response_grid.T, origin="lower", extent=(*extent, *extent),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="bilinear")
    ax.scatter(contacts[:, 0], contacts[:, 1], s=16, fc="white", ec="0.2",
               lw=0.45, alpha=0.85, zorder=4)
    ax.set_title(title, fontsize=10.0, fontweight="bold")
    style_spatial(ax, extent, show_ylabel=show_ylabel)
    return image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stem", default="fig5-spatial-zm-ou-tonic-main-v1")
    args = parser.parse_args()

    source = json.loads(SOURCE_JSON.read_text())
    aggregate = json.loads(AGGREGATE_JSON.read_text())
    static_meta = json.loads(STATIC_JSON.read_text())
    if static_meta.get("status") != "FIG5_SPATIAL_ZM_OU_TONIC_STATIC_ASSETS_COMPLETE":
        raise RuntimeError("static A-D assets are not complete")
    if not (
        source["tonic_global_runaway"]["all_checks_pass"]
        and int(source["seed"]) == 1842
        and source["hybrid_config"]["use_SG"] is False
    ):
        raise RuntimeError("source is not the accepted seed-1842 tonic Z/M run")
    family = aggregate["primary_confirmation_family"]
    if not (
        family["eligible_multi_seed_family"]
        and family["single_frozen_config"]
        and int(family["n_passed_seeds"]) == int(family["n_unique_seeds"]) == 3
    ):
        raise RuntimeError("the three-seed frozen confirmation family is incomplete")

    arrays = load_npz(SOURCE_NPZ)
    static = load_npz(STATIC_NPZ)
    positions = np.asarray(static["positions_E"], float)
    contacts = np.asarray(static["contact_xy_mm"], float)
    if not (
        np.array_equal(static["positions_E"], arrays["positions_E"])
        and np.array_equal(static["contact_xy_mm"], arrays["contact_xy_mm"])
    ):
        raise RuntimeError("static panels and continuous trajectory use different geometry")
    extent = (0.0, 20.0)
    if np.min(positions) < 0.0 or np.max(positions) > 20.0:
        raise RuntimeError("raw neuron coordinates fall outside the frozen sheet")

    low_response = np.asarray(static["low_response_early_mean"], float)
    high_response = np.asarray(static["early_runaway_response_early_mean"], float)
    low_grid = grid_mean(positions, low_response, extent, n=100)
    high_grid = grid_mean(positions, high_response, extent, n=100)
    nonzero = np.abs(np.concatenate((low_grid.ravel(), high_grid.ravel())))
    nonzero = nonzero[nonzero > 1e-12]
    if nonzero.size == 0:
        raise RuntimeError("paired perturbation maps are identically zero")
    response_vmax = max(float(np.percentile(nonzero, 99.5)), 1e-6)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(15.4, 7.4), facecolor="white")
    outer = fig.add_gridspec(
        2, 1, height_ratios=(0.78, 1.10),
        left=0.055, right=0.985, bottom=0.09, top=0.96, hspace=0.42)
    top = outer[0].subgridspec(1, 2, width_ratios=(3.65, 1.0), wspace=0.16)
    bottom = outer[1].subgridspec(
        1, 4, width_ratios=(1.0, 1.0, 1.0, 1.0), wspace=0.22)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    ax_c1 = fig.add_subplot(bottom[0, 0])
    ax_c2 = fig.add_subplot(bottom[0, 1])
    ax_d1 = fig.add_subplot(bottom[0, 2])
    ax_d2 = fig.add_subplot(bottom[0, 3])

    recruitment_ax, readout_meta = plot_readout(ax_a, source, arrays, static)
    trajectory_meta = plot_trajectory(ax_b, source, arrays, static)
    order_map, event_meta = plot_event_order(
        ax_c1, static, positions, contacts, extent)
    energy_map, energy_meta = plot_activity_energy(
        ax_c2, static, positions, contacts, extent)
    response_map = plot_response(
        ax_d1, contacts, low_grid, extent, "Low-activity mean response",
        response_vmax, show_ylabel=False)
    plot_response(
        ax_d2, contacts, high_grid, extent, "Early-runaway mean response",
        response_vmax, show_ylabel=False)

    panel_label(recruitment_ax, "A", x=-0.075, y=1.35)
    panel_label(ax_b, "B", x=-0.18, y=1.16)
    panel_label(ax_c1, "C", x=-0.25, y=1.16)
    panel_label(ax_d1, "D", x=-0.18, y=1.16)

    order_cb = fig.colorbar(order_map, ax=ax_c1, fraction=0.045, pad=0.025)
    order_cb.set_label("event time\n(ms after window start)", fontsize=7.2)
    order_cb.ax.tick_params(labelsize=6.8)
    energy_cb = fig.colorbar(energy_map, ax=ax_c2, fraction=0.045, pad=0.025)
    energy_cb.set_label("activity energy\n" + r"($\times 10^3$ Hz$^2$)",
                        fontsize=7.2)
    energy_cb.ax.tick_params(labelsize=6.8)
    response_cb = fig.colorbar(
        response_map, ax=[ax_d1, ax_d2], fraction=0.026, pad=0.018)
    response_cb.set_label(
        "mean signed probe effect\n(0–50 ms excess spikes per local E cell)",
        fontsize=7.2)
    response_cb.ax.tick_params(labelsize=6.8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / args.stem
    outputs = {}
    for suffix, save_args in (
        ("png", {"dpi": 240}), ("pdf", {}), ("svg", {})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                    facecolor="white", **save_args)
        outputs[suffix] = path
    plt.close(fig)

    metadata = {
        "figure": args.stem,
        "status": "FIG5_SPATIAL_ZM_OU_TONIC_PAPER_READY_CANDIDATE",
        "registry_status": "CANDIDATE_PENDING_AUTHOR_VISUAL_ACCEPTANCE",
        "layout_contract": "author-supplied paper-ready Fig5 A-D transition layout",
        "panel_semantics": {
            "A": {
                "question": "Does one continuous trajectory enter a sustained globally recruited state that is visible at all virtual contacts?",
                **readout_meta,
            },
            "B": {
                "question": "Where does the trajectory move in the Z/M slow-state plane before runaway?",
                **trajectory_meta,
            },
            "C_left": {
                "question": "What is the spatial first-spike order of one rule-selected low-state event?",
                **event_meta,
            },
            "C_right": {
                "question": "Where is SNN firing energy expressed in the fixed first 100 ms of runaway?",
                **energy_meta,
            },
            "D": {
                "question": "How does the same weak perturbation propagate from the low and early-runaway states?",
                **static_meta["panel_D"],
                "rendering": "response only; one shared signed scale; no slow-state or substrate overlay",
                "interpretation_boundary": (
                    "the early-runaway state may be non-evaluable as susceptibility because the sham is already tonic; the signed probe-minus-sham field remains a descriptive saturation response"),
            },
        },
        "representative_seed": 1842,
        "representative_selection": "seed nearest the median onset in the frozen all-pass confirmation family",
        "confirmation_family": {
            "parameter_set_id": family["parameter_set_id"],
            "seeds": family["seeds"],
            "n_passed": family["n_passed_seeds"],
            "n_total": family["n_unique_seeds"],
            "single_frozen_config": family["single_frozen_config"],
            "parameter_contract_sha256": family["parameter_contract_sha256"],
        },
        "coordinate_contract": "raw 0-20 mm sheet x/y; no rotation, reflection, or similarity transform",
        "model_contract": {
            "slow_variables": "spatial Z/q inhibition resource and per-neuron M/gK adaptation",
            "use_SG": source["hybrid_config"]["use_SG"],
            "stationary_spatial_ou": source["applied_spatial_ou"],
            "full_edges": source["full_edge_contract"],
            "timed_pulse_train": False,
        },
        "source_files": {
            "trajectory_npz": {"path": str(SOURCE_NPZ), "sha256": sha256(SOURCE_NPZ)},
            "trajectory_json": {"path": str(SOURCE_JSON), "sha256": sha256(SOURCE_JSON)},
            "aggregate_json": {"path": str(AGGREGATE_JSON), "sha256": sha256(AGGREGATE_JSON)},
            "static_npz": {"path": str(STATIC_NPZ), "sha256": sha256(STATIC_NPZ)},
            "static_json": {"path": str(STATIC_JSON), "sha256": sha256(STATIC_JSON)},
            "producer": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256(Path(__file__).resolve())},
        },
        "outputs": {
            suffix: {"path": str(path), "sha256": sha256(path)}
            for suffix, path in outputs.items()
        },
        "claim_boundary": (
            "synthetic tonic model-state transition on one frozen patient-derived scaffold; not clinical seizure reproduction, a recoverable seizure cycle, waveform fitting, or patient-mechanism identification"),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    metadata_path = stem.with_name(stem.name + "-metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    readme = (
        f"### {args.stem}.png / .pdf / .svg\n\n"
        "严格按作者提供的 paper-ready Figure 5 A–D 布局生成的静态候选图。"
        "A 是同一条连续轨迹的全局招募与 15 个虚拟触点 tonic-level readout；"
        "B 是转变前的 Z/q–M/gK 状态轨迹；C 左是按群体率规则选出的低态事件"
        "逐神经元首放电次序，C 右是固定 onset 后 100 ms 的活动能量；D 是同一"
        "seed、同一 16-cell 弱 probe、同一组 16 个分层随机位点在 200 ms 低态"
        "和 600 ms early-runaway 状态的 probe-minus-sham 平均响应。\n\n"
        "**关注点**：A–D 没有混入旧 seed 1801 或其他工作点，C/D 来自 seed "
        "1842 的 bit-identical replay 与 exact-resume checkpoint。代表轨迹属于"
        "固定参数 3/3 confirmation family；三种子统计只写入 metadata，不替换"
        "原 D 的状态微扰语义。该图展示模型中的 tonic global runaway，不要求"
        "30–80 Hz 深调制，也不能表述为临床发作或患者机制证明。\n")
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({
        "status": metadata["status"],
        "png": str(outputs["png"]),
        "pdf": str(outputs["pdf"]),
        "svg": str(outputs["svg"]),
        "metadata": str(metadata_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
