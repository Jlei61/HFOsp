#!/usr/bin/env python3
"""Render the locked seed-1842 spatial Z/M + OU SNN activity GIF."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter1d


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou"
)
SOURCE_NPZ = ARCHIVE / "tonic_confirmation_v2/tonic_b0_v2_s1842.npz"
SOURCE_JSON = SOURCE_NPZ.with_suffix(".json")
CAPTURE_NPZ = ARCHIVE / "snn_gif_capture/tonic_b0_v2_s1842_snn_frames.npz"
CAPTURE_JSON = ARCHIVE / "snn_gif_capture/tonic_b0_v2_s1842_snn_frames_metadata.json"
DEFAULT_OUT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/"
    "fig5a_spatial_zm_ou_tonic/figures"
)

INK = "#252525"
SHEET = "#6D7F91"
ICL = "#F1783A"
SCL = "#29A6B5"
QCOL = "#7B4D6D"
MCOL = "#6F7E3C"
ONSET = "#D62745"
STATE_SHADE = "#F7E9ED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


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


def normalised_contact_plateau(raw, *, dt_ms, onset_ms):
    raw = np.asarray(raw, float)
    smooth_steps = max(1, int(round(5.0 / float(dt_ms))))
    smooth = uniform_filter1d(raw, size=smooth_steps, axis=0, mode="nearest")
    time = np.arange(len(smooth), dtype=float) * float(dt_ms)
    pre = (time >= max(0.0, float(onset_ms) - 500.0)) & (time < onset_ms)
    post = ((time >= onset_ms + 300.0) & (time < onset_ms + 1300.0))
    baseline = np.median(smooth[pre], axis=0)
    plateau = np.median(smooth[post], axis=0)
    scale = plateau - baseline
    if not np.all(scale > 1e-9):
        raise RuntimeError("contact plateau normalization is not positive")
    return (smooth - baseline[None, :]) / scale[None, :]


def grid_mean(values, positions, *, length, n_grid):
    ix = np.clip((positions[:, 0] / length * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((positions[:, 1] / length * n_grid).astype(int), 0, n_grid - 1)
    flat = iy * n_grid + ix
    total = np.bincount(flat, weights=np.asarray(values, float),
                        minlength=n_grid * n_grid)
    count = np.bincount(flat, minlength=n_grid * n_grid)
    out = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return out.reshape(n_grid, n_grid), flat, count.reshape(n_grid, n_grid)


def draw_contacts(ax, xy, names, shafts):
    for shaft in np.unique(shafts):
        selected = np.flatnonzero(shafts == shaft)
        colour = SCL if shaft == "SCL" else ICL
        marker = "s" if shaft == "SCL" else "o"
        ax.plot(xy[selected, 0], xy[selected, 1], color=colour,
                lw=0.75, alpha=0.65, zorder=6)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=17, marker=marker,
                   facecolor="white", edgecolor=colour, linewidth=0.7,
                   zorder=7)
        for index in selected:
            ax.text(
                xy[index, 0], xy[index, 1], names[index], fontsize=4.3,
                color=colour, ha="center", va="center", zorder=8,
                path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            )


def spatial_style(ax, length):
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)", fontsize=6.4)
    ax.set_ylabel("y (mm)", fontsize=6.4)
    ax.tick_params(labelsize=5.7, length=2.0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frame-duration-ms", type=int, default=90)
    parser.add_argument("--end-hold-frames", type=int, default=12)
    args = parser.parse_args()

    source = json.loads(SOURCE_JSON.read_text())
    capture_meta = json.loads(CAPTURE_JSON.read_text())
    if capture_meta.get("status") != "LOCKED_TRAJECTORY_REPLAY_BIT_IDENTICAL":
        raise RuntimeError("SNN capture is not a verified locked-trajectory replay")
    with np.load(SOURCE_NPZ, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    with np.load(CAPTURE_NPZ, allow_pickle=False) as handle:
        movie = {key: handle[key] for key in handle.files}

    frame_time = np.asarray(movie["frame_time_ms"], float)
    n_e = int(movie["n_E"])
    active = np.unpackbits(movie["active_E_packbits"], axis=1)[:, :n_e].astype(bool)
    q_grid = np.asarray(movie["q_grid"], float)
    m_grid = np.asarray(movie["m_grid"], float)
    if not (len(frame_time) == len(active) == len(q_grid) == len(m_grid)):
        raise RuntimeError("captured SNN and slow-state frames are misaligned")

    dt_ms = float(arrays["lfp_dt_ms"])
    time = np.arange(len(arrays["rate_E_hz"]), dtype=float) * dt_ms
    duration_ms = float(time[-1])
    onset_ms = float(source["scientific_onset_ms"])
    positions = np.asarray(arrays["positions_E"], float)
    length = float(np.ceil(np.max(positions)))
    n_grid = int(q_grid.shape[1])
    h_grid, flat, cell_count = grid_mean(
        arrays["h_E"], positions, length=length, n_grid=n_grid)

    activity_grid = np.empty((len(frame_time), n_grid, n_grid), np.float32)
    flat_size = n_grid * n_grid
    count_flat = cell_count.ravel()
    for index, mask in enumerate(active):
        fired = np.bincount(flat[mask], minlength=flat_size)
        activity_grid[index] = np.divide(
            fired, count_flat, out=np.zeros(flat_size, float),
            where=count_flat > 0,
        ).reshape(n_grid, n_grid)

    names = np.asarray(arrays["contact_names"]).astype(str)
    shafts = np.asarray(arrays["shaft_ids"]).astype(str)
    contacts = np.asarray(arrays["contact_xy_mm"], float)
    order = contact_order(names)
    contact = normalised_contact_plateau(
        arrays["lfp_trace"], dt_ms=dt_ms, onset_ms=onset_ms)[:, order]
    offsets = np.arange(len(order), dtype=float) * 0.96
    smooth_rate = uniform_filter1d(
        np.asarray(arrays["rate_E_hz"], float),
        size=max(1, int(round(20.0 / dt_ms))), mode="nearest",
    )
    recruit_time = np.asarray(arrays["full_field_time_ms"], float)
    active_fraction = 100.0 * np.asarray(
        arrays["active_neuron_fraction_20ms"], float)
    sheet_fraction = 100.0 * np.asarray(
        arrays["recruited_spatial_fraction_1mm"], float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(14.6, 5.8), facecolor="white")
    outer = fig.add_gridspec(
        1, 3, width_ratios=(0.92, 1.28, 2.35),
        left=0.045, right=0.985, bottom=0.125, top=0.83, wspace=0.15,
    )
    slow_grid = outer[0, 0].subgridspec(2, 1, hspace=0.30)
    ax_q = fig.add_subplot(slow_grid[0])
    ax_m = fig.add_subplot(slow_grid[1])
    ax_activity = fig.add_subplot(outer[0, 1])
    right = outer[0, 2].subgridspec(2, 1, height_ratios=(0.82, 2.3), hspace=0.13)
    ax_rate = fig.add_subplot(right[0])
    ax_contact = fig.add_subplot(right[1], sharex=ax_rate)

    q_max = max(0.225, float(np.max(1.0 - q_grid)))
    m_max = max(1e-6, float(np.max(m_grid)))
    extent = [0, length, 0, length]
    q_image = ax_q.imshow(
        1.0 - q_grid[0], origin="lower", extent=extent, cmap="magma",
        vmin=0.0, vmax=q_max, interpolation="nearest",
    )
    m_image = ax_m.imshow(
        m_grid[0], origin="lower", extent=extent, cmap="YlGn",
        vmin=0.0, vmax=m_max, interpolation="nearest",
    )
    activity_image = ax_activity.imshow(
        activity_grid[0], origin="lower", extent=extent, cmap="viridis",
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    for axis in (ax_q, ax_m, ax_activity):
        if float(np.min(h_grid)) < 0.5 < float(np.max(h_grid)):
            coord = (np.arange(n_grid) + 0.5) * length / n_grid
            axis.contour(coord, coord, h_grid, levels=[0.5], colors="white",
                         linewidths=0.7, linestyles="--", alpha=0.85)
        spatial_style(axis, length)
    draw_contacts(ax_q, contacts, names, shafts)
    draw_contacts(ax_m, contacts, names, shafts)
    draw_contacts(ax_activity, contacts, names, shafts)
    q_cb = fig.colorbar(q_image, ax=ax_q, fraction=0.046, pad=0.025)
    m_cb = fig.colorbar(m_image, ax=ax_m, fraction=0.046, pad=0.025)
    a_cb = fig.colorbar(activity_image, ax=ax_activity, fraction=0.046, pad=0.025)
    q_cb.set_label("1 − Z/q", fontsize=5.8)
    m_cb.set_label("M state (a.u.)", fontsize=5.8)
    a_cb.set_label("active E fraction", fontsize=5.8)
    for colorbar in (q_cb, m_cb, a_cb):
        colorbar.ax.tick_params(labelsize=5.4, length=1.8)

    q_title = ax_q.set_title("Z/q permissivity", fontsize=8.0,
                             fontweight="bold", pad=3)
    m_title = ax_m.set_title("M adaptation", fontsize=8.0,
                             fontweight="bold", pad=3)
    activity_title = ax_activity.set_title(
        "2D SNN E spiking | previous 10 ms", fontsize=8.5,
        fontweight="bold", pad=4,
    )

    ax_rate.axvspan(onset_ms, duration_ms, color=STATE_SHADE, lw=0, zorder=0)
    ax_rate.plot(time, smooth_rate, color="0.78", lw=0.8, zorder=1)
    rate_past, = ax_rate.plot([], [], color=INK, lw=0.95, zorder=3,
                             label="E population rate")
    ax_rate.axhline(120.0, color=ONSET, lw=0.65, ls=":", zorder=2)
    ax_rate.axvline(onset_ms, color=ONSET, lw=0.9, ls="--", zorder=4)
    rate_cursor = ax_rate.axvline(0.0, color="black", lw=1.1, zorder=5)
    ax_rate.set_xlim(0.0, duration_ms)
    ax_rate.set_ylim(0.0, 500.0)
    ax_rate.set_ylabel("E rate (Hz)", fontsize=6.5)
    ax_rate.tick_params(axis="both", labelsize=5.7, length=2.0)
    ax_rate.tick_params(axis="x", labelbottom=False)
    ax_rate.spines[["top", "right"]].set_visible(False)
    rec_ax = ax_rate.twinx()
    rec_ax.plot(recruit_time, active_fraction, color=SHEET, lw=0.65,
                alpha=0.55, label="active E")
    rec_ax.plot(recruit_time, sheet_fraction, color=SCL, lw=0.65,
                alpha=0.55, ls="--", label="recruited sheet")
    rec_ax.set_ylim(0.0, 103.0)
    rec_ax.set_ylabel("recruitment (%)", fontsize=6.3, color=SHEET)
    rec_ax.tick_params(axis="y", colors=SHEET, labelsize=5.5, length=2.0)
    rec_ax.spines["top"].set_visible(False)

    ax_contact.axvspan(onset_ms, duration_ms, color=STATE_SHADE, lw=0, zorder=0)
    contact_lines = []
    for row, contact_index in enumerate(order):
        colour = SCL if shafts[contact_index] == "SCL" else ICL
        full = np.clip(contact[:, row], -0.25, 1.25) + offsets[row]
        ax_contact.plot(time, full, color="0.82", lw=0.46, zorder=1)
        line, = ax_contact.plot([], [], color=colour, lw=0.68, zorder=3)
        contact_lines.append((line, full))
    ax_contact.axvline(onset_ms, color=ONSET, lw=0.9, ls="--", zorder=4)
    contact_cursor = ax_contact.axvline(0.0, color="black", lw=1.1, zorder=5)
    ax_contact.set_ylim(-0.55, offsets[-1] + 1.35)
    ax_contact.set_yticks(offsets)
    ax_contact.set_yticklabels(names[order], fontsize=5.7)
    for tick, contact_index in zip(ax_contact.get_yticklabels(), order):
        tick.set_color(SCL if shafts[contact_index] == "SCL" else ICL)
    ax_contact.set_xlabel("time in continuous trajectory (ms)", fontsize=7.0)
    ax_contact.set_title(
        "15 virtual-contact current proxies (normalized tonic level)",
        fontsize=6.5, loc="left", pad=2.5,
    )
    ax_contact.tick_params(axis="x", labelsize=5.8, length=2.0)
    ax_contact.tick_params(axis="y", length=1.8, pad=1.5)
    ax_contact.spines[["top", "right"]].set_visible(False)

    fig.text(0.018, 0.955, "A", fontsize=16.0, fontweight="bold",
             ha="left", va="top")
    main_title = fig.text(
        0.50, 0.954,
        "Spatial Z/M + stationary OU | exact SNN replay | seed 1842",
        fontsize=11.2, fontweight="bold", ha="center", va="top",
    )
    clock = fig.text(0.50, 0.902, "", fontsize=8.0, color="0.25",
                     ha="center", va="top")
    fig.text(
        0.50, 0.045,
        "full learned E→E/E→I · continuous stochastic drive · no pulse train · "
        "tonic global runaway is a model-state morphology, not clinical SEEG",
        fontsize=6.7, color="0.30", ha="center", va="bottom",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = args.out_dir / "fig5a-spatial-zm-ou-tonic-snn-activity.gif"
    preview_path = args.out_dir / "fig5a-spatial-zm-ou-tonic-snn-activity-final.png"
    metadata_path = args.out_dir / "fig5a-spatial-zm-ou-tonic-snn-activity-metadata.json"
    total_frames = len(frame_time) + max(0, int(args.end_hold_frames))
    with imageio.get_writer(
        gif_path, mode="I", duration=int(args.frame_duration_ms), loop=0,
        palettesize=256, subrectangles=False,
    ) as writer:
        for output_index in range(total_frames):
            index = min(output_index, len(frame_time) - 1)
            now = float(frame_time[index])
            q_image.set_data(1.0 - q_grid[index])
            m_image.set_data(m_grid[index])
            activity_image.set_data(activity_grid[index])
            q_title.set_text(
                f"Z/q permissivity | mean {np.mean(1.0-q_grid[index]):.3f}")
            m_title.set_text(
                f"M adaptation | mean {np.mean(m_grid[index]):.2f}")
            activity_title.set_text(
                "2D SNN E spiking | previous 10 ms | "
                f"active {100.0*np.mean(active[index]):.0f}%")
            stop = min(len(time), int(np.searchsorted(time, now, side="right")))
            rate_past.set_data(time[:stop], smooth_rate[:stop])
            rate_cursor.set_xdata([now, now])
            contact_cursor.set_xdata([now, now])
            for line, full in contact_lines:
                line.set_data(time[:stop], full[:stop])
            state = "low state" if now < onset_ms else "tonic global runaway"
            clock.set_text(
                f"t = {now:.0f} ms  |  onset = {onset_ms:.0f} ms  |  {state}")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            writer.append_data(frame)
            if output_index == len(frame_time) - 1:
                fig.savefig(preview_path, dpi=180, facecolor="white")
    plt.close(fig)

    with Image.open(gif_path) as encoded:
        encoded_frames = int(encoded.n_frames)
        encoded_loop = encoded.info.get("loop")
        encoded_durations_ms = []
        for index in range(encoded_frames):
            encoded.seek(index)
            encoded_durations_ms.append(int(encoded.info.get("duration", 0)))

    metadata = {
        "status": "FIG5A_TONIC_SNN_GIF_RENDERED_FROM_BIT_IDENTICAL_REPLAY",
        "representative_seed": 1842,
        "scientific_onset_ms": onset_ms,
        "trajectory_duration_ms": source["trajectory_duration_ms"],
        "n_biological_frames": int(len(frame_time)),
        "n_appended_frames": int(total_frames),
        "n_encoded_frames": encoded_frames,
        "frame_dt_ms": float(np.median(np.diff(frame_time))),
        "frame_duration_ms": int(args.frame_duration_ms),
        "encoded_total_duration_ms": int(sum(encoded_durations_ms)),
        "loop": encoded_loop,
        "activity_window_ms": float(movie["activity_window_ms"]),
        "display_contract": {
            "SNN_activity": (
                "fraction of E neurons that fired at least once in the previous "
                "10 ms, binned on the frozen 64x64 sheet; fixed 0..1 scale; no "
                "frame-wise normalization"
            ),
            "Z_q": "captured q grid shown as 1-q; fixed 0..0.225 scale",
            "M": "mean per-neuron M state in each spatial bin; one fixed movie-wide scale",
            "population_rate": "20-ms uniform smoothing of archived raw E rate",
            "virtual_contacts": (
                "archived current proxy with 5-ms smoothing; each contact maps "
                "pre-onset median to 0 and onset+300..1300-ms median to 1; no "
                "detrending or band-pass filtering"
            ),
            "future_context": "future traces are light grey; elapsed traces are coloured",
        },
        "scientific_boundary": source["boundary"],
        "full_edge_contract": source["full_edge_contract"],
        "applied_spatial_ou": source["applied_spatial_ou"],
        "source_files": {
            "candidate_json": str(SOURCE_JSON),
            "candidate_json_sha256": sha256(SOURCE_JSON),
            "candidate_npz": str(SOURCE_NPZ),
            "candidate_npz_sha256": sha256(SOURCE_NPZ),
            "SNN_capture_npz": str(CAPTURE_NPZ),
            "SNN_capture_npz_sha256": sha256(CAPTURE_NPZ),
            "SNN_capture_metadata": str(CAPTURE_JSON),
            "SNN_capture_metadata_sha256": sha256(CAPTURE_JSON),
            "producer": str(Path(__file__).resolve()),
            "producer_sha256": sha256(Path(__file__).resolve()),
        },
        "outputs": {
            "gif": {"path": str(gif_path), "sha256": sha256(gif_path)},
            "final_png": {"path": str(preview_path), "sha256": sha256(preview_path)},
        },
    }
    atomic_json(metadata_path, metadata)
    print(json.dumps({
        "status": metadata["status"],
        "gif": str(gif_path),
        "gif_sha256": metadata["outputs"]["gif"]["sha256"],
        "frames": encoded_frames,
        "total_duration_ms": metadata["encoded_total_duration_ms"],
        "metadata": str(metadata_path),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
