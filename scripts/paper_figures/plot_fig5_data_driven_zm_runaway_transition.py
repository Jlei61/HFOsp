#!/usr/bin/env python3
"""Base Figure 5: data-driven Joint substrate, Z/M on, low activity to runaway.

Strict reproduction of the qI/gK runaway-transition syntax, with the substrate
swapped for the frozen data-driven Node + E->E + E->I one:

    dynamic Z/M net slow field | same-instant 2-D E activity | continuous readout

One trajectory, one continuous 15-contact trace with no splice, a red dashed
line at the operational onset. Seed 1801 is used because its onset (4115 ms) is
the median of the three canary seeds (3877 / 4115 / 4541), not because it looks
better.

Nothing here is a statistic. No KMeans, no perturbation, no spatial null.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402

TRACE_OFF = 1.55
SHAFT_COLORS = {"ICL": "#E8871A", "SCL": "#1F9FBF"}   # axis shaft orange, cross shaft cyan
ICTAL_GREY = "#3A3A3A"                                # model ictal state: its own dark grey


def _field_grid(values, positions, grid_n, sheet_l):
    ix = np.clip((positions[:, 0] / sheet_l * grid_n).astype(int), 0, grid_n - 1)
    iy = np.clip((positions[:, 1] / sheet_l * grid_n).astype(int), 0, grid_n - 1)
    flat = ix * grid_n + iy
    counts = np.bincount(flat, minlength=grid_n * grid_n).astype(float)
    total = np.bincount(flat, weights=values, minlength=grid_n * grid_n)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid = (total / counts).reshape(grid_n, grid_n)
    return grid


def _style_spatial(ax, sheet_l):
    ax.set_xlim(0, sheet_l); ax.set_ylim(0, sheet_l)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.6)


def _draw_contacts(ax, contact_xy, shaft_ids):
    for shaft in np.unique(shaft_ids):
        mask = shaft_ids == shaft
        ax.scatter(contact_xy[mask, 0], contact_xy[mask, 1], s=26, marker="o",
                   fc="white", ec=SHAFT_COLORS.get(str(shaft), "black"),
                   lw=0.9, zorder=6)


def _normalise_readout(envelope, envelope_dt, onset_ms):
    """Normalise mainly from PRE-onset activity, but keep the post-onset trace."""
    times = np.arange(envelope.shape[1]) * envelope_dt
    pre = times <= (onset_ms if onset_ms else times[-1])
    base = np.median(envelope[:, pre], axis=1, keepdims=True)
    pre_scale = np.maximum(
        np.percentile(envelope[:, pre], 99, axis=1, keepdims=True) - base, 1e-9)
    full_scale = np.maximum(
        np.percentile(envelope, 99, axis=1, keepdims=True) - base, 1e-9)
    scale = np.maximum(pre_scale, 0.35 * full_scale)
    return times, (envelope - base) / scale


def _render(data, meta, out_dir, gif_stride):
    sheet_l = 20.0
    positions = np.asarray(data["positions_E"], float)
    frame_ms = np.asarray(data["frame_time_ms"], float)
    field = np.asarray(data["net_slow_field"], float)
    # counts -> per-neuron rate (Hz), then a light spatial smooth. The measure
    # is per-cell spike COUNT because a 64x64 occupancy fraction saturates at
    # ~7.8 neurons per cell and renders binary.
    counts = np.asarray(data["activity_spike_counts"], float)
    occupancy = np.asarray(data["activity_cell_occupancy"], float)
    occupancy_safe = np.where(occupancy > 0, occupancy, np.nan)
    window_s = float(meta["activity_window_ms"]) * 1e-3
    activity = counts / occupancy_safe / window_s
    activity = np.stack([gaussian_filter(np.nan_to_num(frame), sigma=1.0)
                         for frame in activity])
    contact_xy_raw = np.asarray(data["contact_xy_mm"], float)
    names_raw = [str(n) for n in data["contact_names"]]
    shafts_raw = np.asarray(data["shaft_ids"]).astype(str)
    # group by shaft, as the original two-shaft montage did: the interleaved
    # contract order hides which shaft a trace belongs to. The spatial panels
    # keep the UNPERMUTED coordinates.
    order = np.lexsort((np.arange(len(names_raw)), shafts_raw))
    names = [names_raw[i] for i in order]
    shaft_ids = shafts_raw[order]
    onset_ms = float(meta["model_ictal_onset_ms"])
    envelope = np.abs(np.asarray(data["contact_envelope"], float))[order]
    envelope_dt = float(data["contact_envelope_dt_ms"])
    times, ztrace = _normalise_readout(envelope, envelope_dt, onset_ms)
    trace_y = np.arange(len(names)) * TRACE_OFF
    readout_hi = float(times[-1])

    field_grids = [_field_grid(field[i], positions, 64, sheet_l)
                   for i in range(len(frame_ms))]
    finite = np.concatenate([g[np.isfinite(g)] for g in field_grids])
    f_lo, f_hi = np.percentile(finite, [1, 99])
    # scale from PRE-onset activity so the interictal events stay visible; the
    # ictal frames are allowed to saturate -- that saturation is the finding.
    pre_mask = frame_ms < (onset_ms if onset_ms else frame_ms[-1])
    pre_live = activity[pre_mask]
    pre_live = pre_live[np.isfinite(pre_live) & (pre_live > 0)]
    a_hi = float(np.percentile(pre_live, 99)) if pre_live.size else 1.0

    axis_src = np.asarray(data["axis_source_xy"], float)
    axis_snk = np.asarray(data["axis_sink_xy"], float)

    def _frame(index):
        tm = float(frame_ms[index])
        after = onset_ms is not None and tm >= onset_ms
        fig = plt.figure(figsize=(13.6, 4.8), facecolor="white")
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 2.15], left=0.045,
                              right=0.985, bottom=0.14, top=0.82, wspace=0.30)

        ax0 = fig.add_subplot(gs[0, 0])
        im0 = ax0.imshow(field_grids[index].T, origin="lower",
                         extent=[0, sheet_l, 0, sheet_l], cmap="plasma",
                         vmin=f_lo, vmax=f_hi)
        ax0.plot([axis_src[0], axis_snk[0]], [axis_src[1], axis_snk[1]],
                 color="white", lw=1.1, alpha=0.85, zorder=5)
        _draw_contacts(ax0, contact_xy_raw, shafts_raw)
        _style_spatial(ax0, sheet_l)
        ax0.set_title("Z/M net slow current", fontsize=9.0, fontweight="bold", pad=4)
        cb0 = fig.colorbar(im0, ax=ax0, fraction=0.040, pad=0.015)
        cb0.ax.tick_params(labelsize=6.0)
        cb0.set_label("D − A", fontsize=6.5, labelpad=1)

        ax1 = fig.add_subplot(gs[0, 1])
        im1 = ax1.imshow(activity[index].T, origin="lower",
                         extent=[0, sheet_l, 0, sheet_l], cmap="viridis",
                         vmin=0.0, vmax=a_hi)
        ax1.plot([axis_src[0], axis_snk[0]], [axis_src[1], axis_snk[1]],
                 color="white", lw=1.1, alpha=0.9, zorder=5)
        _draw_contacts(ax1, contact_xy_raw, shafts_raw)
        _style_spatial(ax1, sheet_l)
        ax1.set_title("2D SNN activity", fontsize=9.0, fontweight="bold", pad=4)
        cb1 = fig.colorbar(im1, ax=ax1, fraction=0.040, pad=0.015)
        cb1.ax.tick_params(labelsize=6.0)
        cb1.set_label("E rate (Hz)", fontsize=6.5, labelpad=1)

        ax2 = fig.add_subplot(gs[0, 2])
        if onset_ms is not None:
            ax2.axvspan(onset_ms, readout_hi, color=ICTAL_GREY, alpha=0.10, lw=0, zorder=0)
        for i, name in enumerate(names):
            ax2.plot(times, ztrace[i] + trace_y[i],
                     color=SHAFT_COLORS.get(str(shaft_ids[i]), "black"),
                     lw=0.72, alpha=0.9, zorder=3)
        ax2.axvline(tm, color="black", lw=1.2, alpha=0.9, zorder=7)
        if onset_ms is not None:
            ax2.axvline(onset_ms, color="crimson", lw=1.0, ls="--", alpha=0.9, zorder=6)
        ax2.set_xlim(0.0, readout_hi)
        ax2.set_yticks(trace_y); ax2.set_yticklabels(names, fontsize=6.8)
        for tick, name in zip(ax2.get_yticklabels(), names):
            tick.set_color(SHAFT_COLORS.get(str(shaft_ids[list(names).index(name)]), "black"))
        ax2.set_ylim(trace_y[0] - TRACE_OFF, trace_y[-1] + TRACE_OFF)
        ax2.set_xlabel("time (ms)", fontsize=8)
        ax2.set_ylabel("virtual contact activity (firing-density envelope)",
                       fontsize=7.5, labelpad=2)
        ax2.tick_params(axis="x", labelsize=7)
        ax2.set_title("continuous 15-contact readout", fontsize=9.0,
                      fontweight="bold", pad=4)
        for side in ("top", "right"):
            ax2.spines[side].set_visible(False)

        fig.suptitle(f"data-driven Node + E→E + E→I substrate, Z/M active   "
                     f"|   seed {meta['seed']}   |   t = {tm:.0f} ms"
                     + ("   (model ictal state)" if after else ""),
                     fontsize=9.5, fontweight="bold", y=0.955,
                     path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
        return fig

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "fig5-data-driven-zm-runaway-transition"
    images = []
    for index in range(0, len(frame_ms), gif_stride):
        fig = _frame(index)
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    final = _frame(len(frame_ms) - 1)
    final.savefig(out_dir / f"{stem}-final.png", dpi=190)
    with PdfPages(out_dir / f"{stem}-final.pdf") as pdf:
        pdf.savefig(final)
    plt.close(final)

    import imageio.v2 as imageio
    imageio.mimsave(out_dir / f"{stem}.gif", images, duration=0.05, loop=0)
    return stem, len(images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--out-dir",
                        default="results/paper-ready-figure/fig5/figures")
    parser.add_argument("--gif-stride", type=int, default=4)
    args = parser.parse_args()

    replay = Path(args.replay)
    meta = json.loads(replay.with_suffix(".json").read_text())
    if not meta["verification_against_archived_run"]["all_match"]:
        raise SystemExit("replay does not match the archived run; refusing to plot")
    with np.load(replay, allow_pickle=False) as handle:
        data = {key: handle[key] for key in handle.files}

    out_dir = ROOT / args.out_dir
    stem, n_frames = _render(data, meta, out_dir, int(args.gif_stride))
    atomic_write_json({
        "figure": "fig5_data_driven_zm_runaway_transition",
        "status": ("visual diagnostic, single continuous trajectory; no recovery, "
                   "termination or clinical-seizure claim"),
        "substrate": "frozen data-driven Node + E->E + E->I (joint_04_control)",
        "seed": meta["seed"],
        "seed_selection": ("median onset of the three canary seeds "
                           "(3877 / 4115 / 4541 ms), not chosen for appearance"),
        "zm": "active, use_z and use_m both on",
        "model_ictal_onset_ms": meta["model_ictal_onset_ms"],
        "onset_is_operational": ("20 ms EMA of the population E rate >= 120 Hz for "
                                 ">= 100 ms; not a clinical seizure onset"),
        "frame_dt_ms": meta["frame_dt_ms"], "n_gif_frames": n_frames,
        "replay_verified_against_archived_run":
            meta["verification_against_archived_run"],
        "readout": ("firing-density envelope at 15 virtual contacts, NOT a "
                    "synaptic-current LFP and never an SEEG voltage"),
        "not_included": ["KMeans", "perturbation", "spatial null", "12-seed statistics"],
        "outputs": {"gif": f"{args.out_dir}/{stem}.gif",
                    "final_png": f"{args.out_dir}/{stem}-final.png",
                    "final_pdf": f"{args.out_dir}/{stem}-final.pdf"},
    }, str(out_dir / f"{stem}-metadata.json"))
    print(json.dumps({"stem": stem, "frames": n_frames,
                      "onset_ms": meta["model_ictal_onset_ms"]}))


if __name__ == "__main__":
    main()
