#!/usr/bin/env python3
"""Render the candidate full Fig. 5 for spatial Z/M + stationary OU.

The producer is read-only with respect to the simulation: it combines the
locked seed-1842 replay, its bit-identical SNN frame capture, and the frozen
three-seed confirmation family.  It never re-simulates or selects frames from
image appearance.
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
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label, uniform_filter1d


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
ARCHIVE = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou"
)
AGGREGATE = ARCHIVE / "tonic_runaway_aggregate.json"
SOURCE_NPZ = ARCHIVE / "tonic_confirmation_v2/tonic_b0_v2_s1842.npz"
SOURCE_JSON = SOURCE_NPZ.with_suffix(".json")
CAPTURE_NPZ = ARCHIVE / "snn_gif_capture/tonic_b0_v2_s1842_snn_frames.npz"
CAPTURE_JSON = ARCHIVE / "snn_gif_capture/tonic_b0_v2_s1842_snn_frames_metadata.json"
DEFAULT_OUT = (
    CANONICAL_ROOT / "results/paper-ready-figure/"
    "fig5_spatial_zm_ou_tonic_candidate/figures"
)

INK = "#252525"
SHEET = "#6D7F91"
ICL = "#F1783A"
SCL = "#29A6B5"
QCOL = "#7B4D6D"
QSUR = "#B78FA6"
MCOL = "#6F7E3C"
ONSET = "#D62745"
STATE_SHADE = "#F7E9ED"
SEED_COLORS = ("#355C7D", "#C06C84", "#6C9A8B")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as handle:
        return {key: handle[key] for key in handle.files}


def contact_order(names: np.ndarray) -> np.ndarray:
    def number(name: str) -> int:
        match = re.search(r"(\d+)$", str(name))
        return int(match.group(1)) if match else -1

    return np.asarray(sorted(
        range(len(names)),
        key=lambda index: (
            0 if str(names[index]).startswith("SCL") else 1,
            number(str(names[index])),
        ),
    ), dtype=int)


def grid_mean(values, positions, *, length: float, n_grid: int):
    positions = np.asarray(positions, float)
    ix = np.clip((positions[:, 0] / length * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((positions[:, 1] / length * n_grid).astype(int), 0, n_grid - 1)
    flat = iy * n_grid + ix
    total = np.bincount(flat, weights=np.asarray(values, float),
                        minlength=n_grid * n_grid)
    count = np.bincount(flat, minlength=n_grid * n_grid)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return mean.reshape(n_grid, n_grid), flat, count.reshape(n_grid, n_grid)


def normalised_contact_plateau(raw, *, dt_ms: float, onset_ms: float):
    raw = np.asarray(raw, float)
    smooth = uniform_filter1d(
        raw, size=max(1, int(round(5.0 / dt_ms))), axis=0, mode="nearest")
    time = np.arange(len(smooth), dtype=float) * dt_ms
    pre = (time >= max(0.0, onset_ms - 500.0)) & (time < onset_ms)
    post = (time >= onset_ms + 300.0) & (time < onset_ms + 1300.0)
    baseline = np.median(smooth[pre], axis=0)
    plateau = np.median(smooth[post], axis=0)
    scale = plateau - baseline
    if not np.all(scale > 1e-9):
        raise RuntimeError("a virtual contact lacks a positive tonic plateau")
    return (smooth - baseline[None, :]) / scale[None, :]


def draw_contacts(ax, xy, shafts, *, labels=None, size=10.0):
    for shaft in np.unique(shafts):
        selected = np.flatnonzero(shafts == shaft)
        colour = SCL if shaft == "SCL" else ICL
        marker = "s" if shaft == "SCL" else "o"
        ax.plot(xy[selected, 0], xy[selected, 1], color=colour,
                lw=0.55, alpha=0.7, zorder=6)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=size, marker=marker,
                   facecolor="white", edgecolor=colour, linewidth=0.55,
                   zorder=7)
        if labels is not None:
            for index in selected:
                ax.text(
                    xy[index, 0], xy[index, 1], labels[index], fontsize=3.2,
                    color=colour, ha="center", va="center", zorder=8,
                    path_effects=[pe.withStroke(linewidth=1.0,
                                                foreground="white")],
                )


def spatial_style(ax, length: float, *, show_y=True):
    ax.set_xlim(0.0, length)
    ax.set_ylim(0.0, length)
    ax.set_aspect("equal")
    ax.set_xticks((0, 10, 20))
    ax.set_yticks((0, 10, 20))
    ax.set_xlabel("x (mm)", fontsize=5.7, labelpad=1.0)
    if show_y:
        ax.set_ylabel("y (mm)", fontsize=5.7, labelpad=1.0)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=5.0, length=1.8, pad=1.1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def panel_label(ax, letter: str, *, x=-0.20, y=1.13):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12.5,
            fontweight="bold", ha="left", va="top")


def validate_confirmation(aggregate: dict, source: dict, capture_meta: dict):
    family = aggregate.get("primary_confirmation_family")
    representative = aggregate.get("primary_confirmation_candidate")
    if not isinstance(family, dict) or not isinstance(representative, dict):
        raise RuntimeError("the aggregate lacks a formal confirmation family")
    if not (
        family.get("eligible_multi_seed_family")
        and family.get("single_frozen_config")
        and int(family.get("n_unique_seeds", 0)) >= 3
        and int(family.get("n_passed_seeds", 0))
        == int(family.get("n_unique_seeds", -1))
    ):
        raise RuntimeError("the tonic family is not a frozen all-pass multi-seed family")
    if not (
        int(representative.get("seed", -1)) == 1842
        and representative.get("all_checks_pass")
        and int(source.get("seed", -1)) == 1842
        and source.get("tonic_global_runaway", {}).get("all_checks_pass")
        and source.get("hybrid_config", {}).get("use_SG") is False
    ):
        raise RuntimeError("seed 1842 violates the locked Z/M representative contract")
    if capture_meta.get("status") != "LOCKED_TRAJECTORY_REPLAY_BIT_IDENTICAL":
        raise RuntimeError("SNN frames are not from a bit-identical locked replay")
    return family, representative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    aggregate = load_json(AGGREGATE)
    source = load_json(SOURCE_JSON)
    capture_meta = load_json(CAPTURE_JSON)
    family, representative = validate_confirmation(
        aggregate, source, capture_meta)
    arrays = load_npz(SOURCE_NPZ)
    movie = load_npz(CAPTURE_NPZ)

    frame_time = np.asarray(movie["frame_time_ms"], float)
    n_e = int(movie["n_E"])
    active = np.unpackbits(movie["active_E_packbits"], axis=1)[:, :n_e].astype(bool)
    positions = np.asarray(arrays["positions_E"], float)
    length = 20.0
    n_grid = int(movie["q_grid"].shape[1])
    h_grid, flat, occupancy = grid_mean(
        arrays["h_E"], positions, length=length, n_grid=n_grid)
    if len(frame_time) != len(active):
        raise RuntimeError("captured SNN frame times and spike masks are misaligned")

    activity_grid = np.empty((len(active), n_grid, n_grid), np.float32)
    count_flat = occupancy.ravel()
    for index, mask in enumerate(active):
        fired = np.bincount(flat[mask], minlength=n_grid * n_grid)
        activity_grid[index] = np.divide(
            fired, count_flat, out=np.zeros(n_grid * n_grid, float),
            where=count_flat > 0,
        ).reshape(n_grid, n_grid)

    threshold = 0.5
    component_labels, n_components = label(h_grid >= threshold)
    if int(n_components) != 1:
        raise RuntimeError(
            f"expected one h>=0.5 grid component, found {n_components}")
    neuron_core = np.asarray(arrays["h_E"], float) >= threshold
    n_core_neurons = int(np.sum(neuron_core))
    weights = np.asarray(arrays["h_E"], float)[neuron_core]
    core_centroid = np.average(positions[neuron_core], axis=0, weights=weights)

    onset_ms = float(source["scientific_onset_ms"])
    frame_targets = (100.0, onset_ms, 900.0)
    frame_indices = [int(np.argmin(np.abs(frame_time - target)))
                     for target in frame_targets]
    if len(set(frame_indices)) != len(frame_indices):
        raise RuntimeError("preselected display times collapsed onto one frame")

    dt_ms = float(arrays["lfp_dt_ms"])
    time = np.arange(len(arrays["rate_E_hz"]), dtype=float) * dt_ms
    duration_ms = float(time[-1])
    slow_time = np.asarray(arrays["slow_time_ms"], float)
    recruit_time = np.asarray(arrays["full_field_time_ms"], float)
    smooth_rate = uniform_filter1d(
        np.asarray(arrays["rate_E_hz"], float),
        size=max(1, int(round(20.0 / dt_ms))), mode="nearest")
    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    contact_xy = np.asarray(arrays["contact_xy_mm"], float)
    order = contact_order(names)
    contact = normalised_contact_plateau(
        arrays["lfp_trace"], dt_ms=dt_ms, onset_ms=onset_ms)[:, order]

    records = list(family["records"])
    if [int(record["seed"]) for record in records] != [1841, 1842, 1843]:
        raise RuntimeError("unexpected confirmation seed order")
    pre_rate = np.asarray([record["median_rate_pre_hz"] for record in records], float)
    post_rate = np.asarray([record["median_rate_post_hz"] for record in records], float)
    active_post = 100.0 * np.asarray([
        record["median_active_neuron_fraction_20ms"] for record in records], float)
    sheet_post = 100.0 * np.asarray([
        record["median_recruited_spatial_fraction_1mm"] for record in records], float)
    duty = 100.0 * np.asarray([
        record["joint_global_recruitment_duty"] for record in records], float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6.9,
        "axes.linewidth": 0.65,
        "xtick.major.width": 0.55,
        "ytick.major.width": 0.55,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(7.25, 7.75), facecolor="white")
    outer = fig.add_gridspec(
        3, 1, height_ratios=(1.12, 0.94, 1.60), hspace=0.43,
        left=0.075, right=0.975, bottom=0.07, top=0.895,
    )

    # A-B: the frozen continuous substrate and exact SNN spatial frames.
    top = outer[0].subgridspec(1, 4, wspace=0.31)
    ax_h = fig.add_subplot(top[0])
    frame_axes = [fig.add_subplot(top[index]) for index in (1, 2, 3)]
    extent = (0.0, length, 0.0, length)
    h_image = ax_h.imshow(
        h_grid, origin="lower", extent=extent, cmap="magma",
        vmin=0.0, vmax=1.0, interpolation="nearest")
    coords = (np.arange(n_grid) + 0.5) * length / n_grid
    ax_h.contour(coords, coords, h_grid, levels=[threshold], colors="white",
                 linewidths=0.75, linestyles="--")
    draw_contacts(ax_h, contact_xy, shafts, labels=None, size=8.0)
    spatial_style(ax_h, length, show_y=True)
    ax_h.set_title("data-driven substrate", fontsize=7.2,
                   fontweight="bold", pad=3.0)
    ax_h.text(
        0.03, 0.04, r"one $h\geq0.5$ component", transform=ax_h.transAxes,
        fontsize=5.0, color="white", ha="left", va="bottom",
        path_effects=[pe.withStroke(linewidth=1.3, foreground="black")])
    panel_label(ax_h, "A")
    h_cb = fig.colorbar(h_image, ax=ax_h, orientation="horizontal",
                        fraction=0.05, pad=0.15, aspect=18)
    h_cb.set_label("excitability field h", fontsize=5.2, labelpad=1.2)
    h_cb.ax.tick_params(labelsize=4.8, length=1.5)

    frame_titles = ("low state", "transition", "tonic plateau")
    activity_image = None
    for column, (axis, index, title) in enumerate(zip(
            frame_axes, frame_indices, frame_titles)):
        activity_image = axis.imshow(
            activity_grid[index], origin="lower", extent=extent,
            cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        axis.contour(coords, coords, h_grid, levels=[threshold], colors="white",
                     linewidths=0.55, linestyles="--", alpha=0.9)
        draw_contacts(axis, contact_xy, shafts, labels=None, size=6.0)
        spatial_style(axis, length, show_y=False)
        axis.set_title(
            f"{title}\n{frame_time[index]:.0f} ms · {100*np.mean(active[index]):.0f}% active",
            fontsize=6.6, fontweight="bold" if column == 1 else "normal",
            color=ONSET if column == 1 else INK, pad=2.5)
    panel_label(frame_axes[0], "B")
    a_cb = fig.colorbar(activity_image, ax=frame_axes, orientation="horizontal",
                        fraction=0.05, pad=0.15, aspect=48)
    a_cb.set_label("E neurons active in previous 10 ms (fraction)",
                   fontsize=5.2, labelpad=1.2)
    a_cb.ax.tick_params(labelsize=4.8, length=1.5)

    # C: slow variables, rate, recruitment, and the same virtual-contact readout.
    middle = outer[1].subgridspec(3, 1, height_ratios=(1.0, 0.82, 0.82), hspace=0.08)
    ax_slow = fig.add_subplot(middle[0])
    ax_rate = fig.add_subplot(middle[1], sharex=ax_slow)
    ax_recruit = fig.add_subplot(middle[2], sharex=ax_slow)
    for axis in (ax_slow, ax_rate, ax_recruit):
        axis.axvspan(onset_ms, duration_ms, color=STATE_SHADE, lw=0, zorder=0)
        axis.axvline(onset_ms, color=ONSET, lw=0.85, ls="--", zorder=5)
        axis.set_xlim(0.0, duration_ms)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=5.2, length=1.8, pad=1.4)
    q_core = 100.0 * (1.0 - np.asarray(arrays["slow_q_core_mean"], float))
    q_surround = 100.0 * (1.0 - np.asarray(arrays["slow_q_surround_mean"], float))
    line_core, = ax_slow.plot(slow_time, q_core, color=QCOL, lw=0.8,
                              label="Z/q permissivity, high-h")
    line_sur, = ax_slow.plot(slow_time, q_surround, color=QSUR, lw=0.75,
                             ls="--", label="surround")
    ax_slow.set_ylim(0.0, 25.0)
    ax_slow.set_ylabel("1−Z/q (%)", fontsize=5.8, color=QCOL, labelpad=3.0)
    ax_slow.tick_params(axis="y", colors=QCOL)
    ax_slow.tick_params(axis="x", labelbottom=False)
    m_axis = ax_slow.twinx()
    line_m, = m_axis.plot(
        slow_time, np.asarray(arrays["slow_adaptation_current_mean"], float),
        color=MCOL, lw=0.75, label="M/gK current")
    m_axis.set_ylim(0.0, max(0.12, 1.05 * float(np.max(line_m.get_ydata()))))
    m_axis.set_ylabel("M/gK (a.u.)", fontsize=5.8, color=MCOL, labelpad=3.0)
    m_axis.tick_params(axis="y", colors=MCOL, labelsize=5.2, length=1.8)
    m_axis.spines["top"].set_visible(False)
    ax_slow.legend(
        [line_core, line_sur, line_m],
        [line_core.get_label(), line_sur.get_label(), line_m.get_label()],
        loc="lower left", bbox_to_anchor=(0.0, 1.015), frameon=False,
        ncol=3, fontsize=5.0, handlelength=1.3, columnspacing=0.75,
        borderaxespad=0.0)
    ax_rate.plot(time, smooth_rate, color=INK, lw=0.75)
    ax_rate.axhline(120.0, color=ONSET, lw=0.55, ls=":")
    ax_rate.set_ylim(0.0, 500.0)
    ax_rate.set_ylabel("E rate\n(Hz)", fontsize=5.8, labelpad=3.0)
    ax_rate.tick_params(axis="x", labelbottom=False)
    ax_recruit.plot(
        recruit_time,
        100.0 * np.asarray(arrays["active_neuron_fraction_20ms"], float),
        color=INK, lw=0.75, label="active E")
    ax_recruit.plot(
        recruit_time,
        100.0 * np.asarray(arrays["recruited_spatial_fraction_1mm"], float),
        color=SHEET, lw=0.75, label="recruited sheet")
    ax_recruit.set_ylim(0.0, 103.0)
    ax_recruit.set_ylabel("global\nrecruitment (%)", fontsize=5.8, labelpad=3.0)
    ax_recruit.set_xlabel("time in continuous trajectory (ms)", fontsize=6.0,
                          labelpad=1.5)
    ax_recruit.legend(loc="lower right", frameon=False, ncol=2, fontsize=5.0,
                      handlelength=1.3, columnspacing=0.8)
    panel_label(ax_slow, "C", x=-0.078, y=1.12)
    ax_rate.text(
        onset_ms + 16.0, 472.0, "tonic transition", fontsize=5.2,
        color=ONSET, ha="left", va="top")

    bottom = outer[2].subgridspec(1, 2, width_ratios=(2.65, 1.0), wspace=0.30)
    ax_contacts = fig.add_subplot(bottom[0])
    ax_contacts.axvspan(onset_ms, duration_ms, color=STATE_SHADE, lw=0, zorder=0)
    ax_contacts.axvline(onset_ms, color=ONSET, lw=0.85, ls="--", zorder=5)
    offsets = np.arange(len(order), dtype=float) * 0.92
    for row, contact_index in enumerate(order):
        colour = SCL if shafts[contact_index] == "SCL" else ICL
        ax_contacts.plot(
            time, np.clip(contact[:, row], -0.25, 1.25) + offsets[row],
            color=colour, lw=0.48)
    ax_contacts.set_xlim(0.0, duration_ms)
    ax_contacts.set_ylim(-0.5, offsets[-1] + 1.25)
    ax_contacts.set_yticks(offsets)
    ax_contacts.set_yticklabels(names[order], fontsize=4.8)
    for tick, contact_index in zip(ax_contacts.get_yticklabels(), order):
        tick.set_color(SCL if shafts[contact_index] == "SCL" else ICL)
    ax_contacts.set_xlabel("time in continuous trajectory (ms)", fontsize=6.0)
    ax_contacts.set_ylabel("virtual-contact current proxy\n(normalized tonic level)",
                           fontsize=5.8)
    ax_contacts.tick_params(axis="x", labelsize=5.2, length=1.8)
    ax_contacts.tick_params(axis="y", length=1.4, pad=1.0)
    ax_contacts.spines[["top", "right"]].set_visible(False)
    ax_contacts.set_title("same-state virtual-SEEG readout", fontsize=6.6,
                          loc="left", pad=2.0)

    # D: all three seeds, no image-based selection.
    dgrid = bottom[1].subgridspec(2, 1, height_ratios=(1.0, 1.0), hspace=0.43)
    ax_d_rate = fig.add_subplot(dgrid[0])
    ax_d_global = fig.add_subplot(dgrid[1])
    for seed_index, (before, after, colour) in enumerate(zip(
            pre_rate, post_rate, SEED_COLORS)):
        horizontal_offset = (seed_index - 1) * 0.035
        ax_d_rate.plot((horizontal_offset, 1 + horizontal_offset),
                       (before, after), color=colour, lw=0.75,
                       alpha=0.75)
        ax_d_rate.scatter((horizontal_offset, 1 + horizontal_offset),
                          (before, after), s=15, color=colour,
                          edgecolor="white", linewidth=0.35, zorder=3)
    ax_d_rate.axhline(300.0, color=ONSET, lw=0.55, ls=":")
    ax_d_rate.set_xticks((0, 1), ("pre", "post"))
    ax_d_rate.set_ylim(0.0, 430.0)
    ax_d_rate.set_ylabel("median E rate (Hz)", fontsize=5.5)
    ax_d_rate.tick_params(labelsize=5.0, length=1.7, pad=1.1)
    ax_d_rate.spines[["top", "right"]].set_visible(False)
    ax_d_rate.set_title("three-seed confirmation", fontsize=6.7,
                        fontweight="bold", pad=2.5)
    panel_label(ax_d_rate, "D", x=-0.34, y=1.18)
    ax_d_rate.text(
        0.98, 0.04, "3/3 pass · each 15/15 contacts",
        transform=ax_d_rate.transAxes, fontsize=4.7, color="0.28",
        ha="right", va="bottom")

    metrics = (active_post, sheet_post, duty)
    x = np.arange(3)
    for seed_index, colour in enumerate(SEED_COLORS):
        ax_d_global.scatter(
            x + (seed_index - 1) * 0.08,
            [metric[seed_index] for metric in metrics],
            s=15, color=colour, edgecolor="white", linewidth=0.35, zorder=3)
    thresholds = (85.0, 85.0, 80.0)
    for xpos, threshold_value in zip(x, thresholds):
        ax_d_global.plot((xpos - 0.22, xpos + 0.22),
                         (threshold_value, threshold_value),
                         color=ONSET, lw=0.6, ls=":")
    ax_d_global.set_xticks(x, ("active E", "sheet", "global\nduty"))
    ax_d_global.set_ylim(70.0, 103.0)
    ax_d_global.set_ylabel("post-state (%)", fontsize=5.5)
    ax_d_global.tick_params(labelsize=4.8, length=1.7, pad=1.1)
    ax_d_global.spines[["top", "right"]].set_visible(False)
    onset_values = [float(record["scientific_onset_ms"]) for record in records]
    post_dwell_s = np.asarray(
        [record["observed_post_transition_ms"] for record in records], float) / 1000.0
    ax_d_global.text(
        0.98, 0.05,
        (f"median onset {np.median(onset_values):.0f} ms\n"
         f"post-state ≥{np.min(post_dwell_s):.2f} s"),
        transform=ax_d_global.transAxes, fontsize=4.6, color="0.28",
        ha="right", va="bottom")

    fig.text(
        0.50, 0.977,
        "Spatial Z/M dynamics under stationary OU input enter a global tonic runaway",
        fontsize=10.2, fontweight="bold", ha="center", va="top")
    fig.text(
        0.50, 0.945,
        ("frozen E1146 scaffold · full learned E→E/E→I · one suprathreshold "
         "h component · 3/3 confirmation seeds"),
        fontsize=6.2, color="0.30", ha="center", va="top")
    fig.text(
        0.50, 0.012,
        ("continuous stochastic drive; no pulse train · Z/q and M/gK only "
         "(use_SG=false) · model-state morphology, not clinical seizure reproduction"),
        fontsize=5.4, color="0.32", ha="center", va="bottom")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / "fig5-spatial-zm-ou-tonic-global-runaway-candidate"
    outputs = {}
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=300, facecolor="white")
        outputs[suffix] = path
    plt.close(fig)

    metadata = {
        "status": "FIG5_TONIC_GLOBAL_RUNAWAY_CANDIDATE_RENDERED",
        "registry_status": "CANDIDATE_PENDING_AUTHOR_VISUAL_ACCEPTANCE",
        "scientific_claim": (
            "On one frozen patient-derived spatial scaffold, Z/q and M/gK "
            "dynamics under stationary spatial OU input enter a persistent "
            "near-saturated globally recruited tonic state."),
        "claim_boundary": (
            "This is a synthetic model-state morphology. It is not a clinical "
            "seizure waveform fit, a recoverable seizure cycle, or patient-"
            "mechanism identification."),
        "core_contract": {
            "field": "continuous data-driven h_E",
            "operational_threshold": threshold,
            "connected_components_on_64x64_display_grid": int(n_components),
            "neurons_at_or_above_threshold": n_core_neurons,
            "h_weighted_centroid_mm": core_centroid.tolist(),
            "interpretation": (
                "one suprathreshold high-h component; subthreshold field bumps "
                "and the two electrode shafts are not additional cores"),
        },
        "model_contract": {
            "slow_variables": "Z/q inhibition resource plus M/gK adaptation",
            "use_SG": source["hybrid_config"]["use_SG"],
            "spatial_ou": source["applied_spatial_ou"],
            "edges": source["full_edge_contract"],
            "no_timed_stimulus": True,
        },
        "confirmation_family": family,
        "representative": representative,
        "panel_contract": {
            "A": "continuous h field, h>=0.5 contour, and frozen contact geometry",
            "B": {
                "measure": (
                    "fraction of E neurons firing at least once in the preceding "
                    "10 ms in each frozen 64x64 spatial bin"),
                "fixed_scale": [0.0, 1.0],
                "preselected_times_ms": [float(frame_time[i]) for i in frame_indices],
                "frame_selection": "fixed low/onset/post times; no image-pixel selection",
            },
            "C": {
                "population_rate": "20-ms uniform smoothing of archived raw E rate",
                "recruitment": "archived 20-ms-window neuron and 1-mm sheet metrics",
                "virtual_contacts": (
                    "5-ms smoothing; pre-onset median mapped to 0 and "
                    "onset+300..1300-ms median mapped to 1; no detrending or band-pass"),
            },
            "D": "all three seeds from one frozen all-pass confirmation family",
        },
        "source_files": {
            "aggregate": {"path": str(AGGREGATE), "sha256": sha256(AGGREGATE)},
            "candidate_json": {"path": str(SOURCE_JSON), "sha256": sha256(SOURCE_JSON)},
            "candidate_npz": {"path": str(SOURCE_NPZ), "sha256": sha256(SOURCE_NPZ)},
            "capture_npz": {"path": str(CAPTURE_NPZ), "sha256": sha256(CAPTURE_NPZ)},
            "capture_metadata": {"path": str(CAPTURE_JSON), "sha256": sha256(CAPTURE_JSON)},
            "producer": {"path": str(Path(__file__).resolve()),
                         "sha256": sha256(Path(__file__).resolve())},
        },
        "outputs": {
            suffix: {"path": str(path), "sha256": sha256(path)}
            for suffix, path in outputs.items()
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    metadata_path = stem.with_name(stem.name + "-metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    readme = (
        "### fig5-spatial-zm-ou-tonic-global-runaway-candidate.png / .pdf / .svg\n\n"
        "新版 Figure 5 候选图。A 展示冻结的连续 data-driven `h` 场、唯一的 "
        "`h≥0.5` 连通区和原始电极摆位；B 用固定色标展示同一 seed 1842 "
        "轨迹的低态、转变时刻和 tonic plateau 三个真实 SNN 放电帧。C 将同一"
        "轨迹的 Z/q、M/gK、群体率、全局招募和 15 个虚拟触点同步画出；D 展示"
        "同一冻结参数下 1841–1843 三个 confirmation seeds 的全部结果。\n\n"
        "**关注点**：正式 core 定义下只有一个阈值以上的主区；两个弱的场峰和"
        "两条电极 shaft 不是两个 core。3/3 seeds 均进入持续、近饱和、全片招募"
        "的 tonic runaway。该图不要求 30–80 Hz 深调制，也不能解读为临床发作"
        "波形、可恢复发作周期或患者机制证明。\n")
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({
        "status": metadata["status"],
        "core_components": int(n_components),
        "confirmation": f"{family['n_passed_seeds']}/{family['n_unique_seeds']}",
        "png": str(outputs["png"]),
        "pdf": str(outputs["pdf"]),
        "svg": str(outputs["svg"]),
        "metadata": str(metadata_path),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
