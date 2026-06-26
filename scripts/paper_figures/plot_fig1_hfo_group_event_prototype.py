#!/usr/bin/env python3
"""Build a compact paper-ready demo of interictal HFO group events.

The figure reuses current pipeline artifacts instead of the legacy hard-coded
paper scripts:
  - envelope cache with 80-250 Hz bandpassed traces
  - packedTimes group-event windows
  - groupAnalysis event participation and centroid times
  - groupTF tile cache for normalized HFO power
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_RUN_SUMMARY = Path(
    "/mnt/yuquan_data/yuquan_24h_edf/chengshuai/temp/run_summary.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/paper-ready-figure/fig1_hfo_group_event_demo/figures"
)


def _as_names(values: Iterable[object]) -> list[str]:
    return [str(v) for v in list(values)]


def _load_run_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = [
        "env_cache_path",
        "packed_times_path",
        "group_analysis_path",
        "group_tf_tile_cache_path",
    ]
    missing = [k for k in required if not data.get(k)]
    if missing:
        raise ValueError(f"run_summary is missing required paths: {missing}")
    return data


def _event_spread(centroid_time: np.ndarray, events_bool: np.ndarray) -> np.ndarray:
    spread = np.zeros(centroid_time.shape[1], dtype=np.float64)
    for ei in range(centroid_time.shape[1]):
        vals = centroid_time[:, ei]
        ok = events_bool[:, ei] & np.isfinite(vals)
        if int(ok.sum()) > 1:
            spread[ei] = float(np.nanmax(vals[ok]) - np.nanmin(vals[ok]))
    return spread


def _select_events(
    events_bool: np.ndarray,
    centroid_time: np.ndarray,
    n_events: int,
    min_active_channels: int,
) -> list[int]:
    active = events_bool.sum(axis=0).astype(np.float64)
    spread = _event_spread(centroid_time, events_bool)
    valid = active >= float(min_active_channels)
    if not np.any(valid):
        valid = active > 0
    score = spread * np.sqrt(np.maximum(active, 1.0))
    score = np.where(valid, score, -np.inf)
    picks = np.argsort(-score)[: int(n_events)]
    return sorted(int(i) for i in picks if np.isfinite(score[i]))


def _select_channels(
    ch_names: list[str],
    events_bool: np.ndarray,
    centroid_time: np.ndarray,
    event_indices: list[int],
    n_channels: int,
    min_fraction: float,
) -> list[str]:
    if not event_indices:
        raise ValueError("No event indices available for channel selection.")

    sub_bool = events_bool[:, event_indices]
    participation = sub_bool.sum(axis=1)
    min_hits = max(1, int(np.ceil(float(min_fraction) * len(event_indices))))
    candidates = np.flatnonzero(participation >= min_hits)
    if candidates.size < min(n_channels, events_bool.shape[0]):
        candidates = np.argsort(-participation)[: int(n_channels)]
    else:
        candidates = candidates[np.argsort(-participation[candidates])[: int(n_channels)]]

    med_times = {}
    for ci in candidates:
        vals = centroid_time[int(ci), event_indices]
        ok = sub_bool[int(ci)] & np.isfinite(vals)
        med_times[int(ci)] = float(np.nanmedian(vals[ok])) if np.any(ok) else np.inf

    ordered = sorted([int(c) for c in candidates], key=lambda ci: (med_times[ci], -participation[ci]))
    return [ch_names[i] for i in ordered]


def _crop_bounds(start: float, end: float, plot_window_sec: float) -> tuple[float, float]:
    center = 0.5 * (float(start) + float(end))
    half = 0.5 * float(plot_window_sec)
    return center - half, center + half


def _robust_norm_rows(mat: np.ndarray, lo_pct: float = 5.0, hi_pct: float = 99.0) -> np.ndarray:
    out = np.zeros_like(mat, dtype=np.float64)
    for i in range(mat.shape[0]):
        row = np.asarray(mat[i], dtype=np.float64)
        finite = np.isfinite(row)
        if not finite.any():
            continue
        lo, hi = np.nanpercentile(row[finite], [lo_pct, hi_pct])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.nanmin(row[finite]))
            hi = float(np.nanmax(row[finite]))
        if hi <= lo:
            continue
        normed = np.clip((row - lo) / (hi - lo), 0.0, 1.0)
        normed[~finite] = 0.0
        out[i] = normed
    return out


def _build_concat_data(
    *,
    env_cache: np.lib.npyio.NpzFile,
    tf_cache: np.lib.npyio.NpzFile,
    group_analysis: np.lib.npyio.NpzFile,
    packed: np.ndarray,
    channels: list[str],
    event_indices: list[int],
    plot_window_sec: float,
    tf_freq_percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    env_names = _as_names(env_cache["ch_names"].tolist())
    tf_names = _as_names(tf_cache["channel_names"].tolist())
    tf_event_labels = [int(x) for x in np.asarray(tf_cache["event_indices"]).tolist()]
    tf_event_map = {ev: pos for pos, ev in enumerate(tf_event_labels)}

    env_idx = {name: idx for idx, name in enumerate(env_names)}
    tf_idx = {name: idx for idx, name in enumerate(tf_names)}

    sfreq = float(np.asarray(env_cache["sfreq"]).ravel()[0])
    x_band = np.asarray(env_cache["x_band"], dtype=np.float64)
    power_db = np.asarray(tf_cache["power_db"], dtype=np.float64)
    time_axis = np.asarray(tf_cache["time_axis"], dtype=np.float64)
    window_sec = float(np.asarray(tf_cache["window_sec"]).ravel()[0])
    crop_start_rel = 0.5 * window_sec - 0.5 * float(plot_window_sec)
    crop_end_rel = crop_start_rel + float(plot_window_sec)
    tf_mask = (time_axis >= crop_start_rel) & (time_axis < crop_end_rel)
    if not np.any(tf_mask):
        raise ValueError("plot_window_sec does not overlap TF tile time_axis.")

    trace_rows: list[np.ndarray] = []
    heat_rows: list[np.ndarray] = []
    segment_lengths: list[float] = []
    target_samples = int(round(float(plot_window_sec) * sfreq))

    for ch in channels:
        if ch not in env_idx:
            raise ValueError(f"Channel {ch} not found in env cache.")
        if ch not in tf_idx:
            raise ValueError(f"Channel {ch} not found in TF cache.")
        trace_segments = []
        heat_segments = []
        for ev in event_indices:
            if ev not in tf_event_map:
                raise ValueError(f"Event {ev} not found in TF cache.")
            s, e = packed[int(ev)]
            crop_s, crop_e = _crop_bounds(float(s), float(e), float(plot_window_sec))
            i0 = max(0, int(round(crop_s * sfreq)))
            i1 = min(x_band.shape[1], int(round(crop_e * sfreq)))
            seg = x_band[env_idx[ch], i0:i1]
            if seg.shape[0] < target_samples:
                seg = np.pad(seg, (0, target_samples - seg.shape[0]), constant_values=np.nan)
            elif seg.shape[0] > target_samples:
                seg = seg[:target_samples]
            trace_segments.append(seg)

            tile = power_db[tf_idx[ch], tf_event_map[int(ev)]]
            tile_crop = tile[:, tf_mask]
            if np.isfinite(tile_crop).any():
                heat = np.nanpercentile(tile_crop, float(tf_freq_percentile), axis=0)
                heat = np.where(np.isfinite(heat), heat, np.nanmin(heat[np.isfinite(heat)]))
            else:
                heat = np.zeros(int(tf_mask.sum()), dtype=np.float64)
            if heat.shape[0] < target_samples:
                heat = np.interp(
                    np.linspace(0.0, 1.0, target_samples),
                    np.linspace(0.0, 1.0, heat.shape[0]),
                    heat,
                )
            elif heat.shape[0] > target_samples:
                heat = heat[:target_samples]
            heat_segments.append(heat)

        trace_rows.append(np.concatenate(trace_segments))
        heat_rows.append(np.concatenate(heat_segments))

    for _ in event_indices:
        segment_lengths.append(float(plot_window_sec))

    traces = np.vstack(trace_rows)
    heat = _robust_norm_rows(np.vstack(heat_rows))
    time_axis_concat = np.arange(traces.shape[1], dtype=np.float64) / sfreq
    return traces, heat, time_axis_concat, segment_lengths


def _plot_figure(
    *,
    traces: np.ndarray,
    heat: np.ndarray,
    time_axis: np.ndarray,
    segment_lengths: list[float],
    channels: list[str],
    event_indices: list[int],
    group_analysis: np.lib.npyio.NpzFile,
    plot_window_sec: float,
    title: str,
    output_png: Path,
    output_pdf: Path,
) -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    n_ch = len(channels)
    total_time = float(sum(segment_lengths))
    cmap = LinearSegmentedColormap.from_list(
        "hfo_power_blue_red",
        ["#173b74", "#f2f4f7", "#b2182b"],
        N=256,
    )

    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.18)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    spacing = 2.8
    centered = traces - np.nanmedian(traces, axis=1, keepdims=True)
    scale = np.nanpercentile(np.abs(centered), 99, axis=1, keepdims=True)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    y = centered / scale
    for ci, ch in enumerate(channels):
        offset = float(ci) * spacing
        ax_trace.plot(time_axis, y[ci] + offset, color="black", linewidth=0.45)
    ax_trace.set_yticks([i * spacing for i in range(n_ch)])
    ax_trace.set_yticklabels(channels, fontsize=8)
    ax_trace.invert_yaxis()
    ax_trace.set_xlim(0.0, total_time)
    ax_trace.set_xlabel("Time (s; concatenated events)", fontsize=9)
    ax_trace.set_title("80-250 Hz filtered SEEG", fontsize=10, fontweight="bold")
    ax_trace.tick_params(axis="both", labelsize=8, length=2)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)
    ax_trace.spines["left"].set_visible(False)
    ax_trace.tick_params(axis="y", length=0)

    im = ax_heat.imshow(
        heat,
        aspect="auto",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
        extent=(0.0, total_time, n_ch, 0.0),
    )
    ax_heat.set_yticks([i + 0.5 for i in range(n_ch)])
    ax_heat.set_yticklabels(channels, fontsize=8)
    ax_heat.set_xlabel("Time (s; concatenated events)", fontsize=9)
    ax_heat.set_title("Normalized HFO power", fontsize=10, fontweight="bold")
    ax_heat.tick_params(axis="both", labelsize=8, length=2)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)

    ga_names = _as_names(group_analysis["ch_names"].tolist())
    ga_idx = {name: idx for idx, name in enumerate(ga_names)}
    events_bool = np.asarray(group_analysis["events_bool"], dtype=bool)
    centroid_time = np.asarray(group_analysis["centroid_time"], dtype=np.float64)
    window_sec = float(np.asarray(group_analysis["window_sec"]).ravel()[0])
    crop_start_rel = 0.5 * window_sec - 0.5 * float(plot_window_sec)

    event_starts = np.cumsum([0.0] + segment_lengths[:-1])
    for pos, ev in enumerate(event_indices):
        xs = []
        ys = []
        for ci, ch in enumerate(channels):
            gi = ga_idx.get(ch)
            if gi is None or not bool(events_bool[gi, int(ev)]):
                continue
            tc = float(centroid_time[gi, int(ev)])
            if not np.isfinite(tc):
                continue
            rel = tc - crop_start_rel
            if rel < 0.0 or rel > float(plot_window_sec):
                continue
            xs.append(float(event_starts[pos]) + rel)
            ys.append(float(ci) + 0.5)
        if xs:
            order = np.argsort(xs)
            xs_arr = np.asarray(xs)[order]
            ys_arr = np.asarray(ys)[order]
            if xs_arr.size > 1:
                ax_heat.plot(
                    xs_arr,
                    ys_arr,
                    color="#d95f02",
                    linewidth=0.85,
                    alpha=0.85,
                    zorder=4,
                )
            ax_heat.scatter(
                xs_arr,
                ys_arr,
                s=16,
                facecolors="white",
                edgecolors="#d95f02",
                linewidth=0.9,
                zorder=5,
            )

    cur = 0.0
    for seg_len in segment_lengths:
        ax_trace.axvline(cur, color="#9aa0a6", linewidth=0.45, alpha=0.7)
        ax_heat.axvline(cur, color="white", linewidth=0.8, alpha=0.8)
        cur += seg_len
    ax_trace.axvline(cur, color="#9aa0a6", linewidth=0.45, alpha=0.7)
    ax_heat.axvline(cur, color="white", linewidth=0.8, alpha=0.8)

    tick_step = 0.5 if total_time <= 3.5 else 1.0
    xticks = np.arange(0.0, total_time + 1e-9, tick_step)
    for ax in (ax_trace, ax_heat):
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:g}" for x in xticks], fontsize=8)
        ax.margins(x=0.0)

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.034, pad=0.02)
    cbar.set_label("Normalized power", fontsize=8, rotation=270, labelpad=11)
    cbar.ax.tick_params(labelsize=8, length=2)

    fig.text(0.012, 0.96, "a", fontsize=13, fontweight="bold")
    fig.text(0.47, 0.96, "b", fontsize=13, fontweight="bold")
    fig.suptitle(title, fontsize=9, fontweight="bold", y=1.01)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-summary", type=Path, default=DEFAULT_RUN_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="yuquan_y1_hfo_group_event_demo")
    parser.add_argument("--figure-title", default="Yuquan Y1")
    parser.add_argument("--n-events", type=int, default=8)
    parser.add_argument("--n-channels", type=int, default=7)
    parser.add_argument("--min-active-channels", type=int, default=6)
    parser.add_argument("--channel-min-fraction", type=float, default=0.5)
    parser.add_argument("--plot-window-sec", type=float, default=0.32)
    parser.add_argument("--tf-freq-percentile", type=float, default=90.0)
    parser.add_argument(
        "--event-indices",
        type=str,
        default="",
        help="Comma-separated packed event indices. Empty means automatic selection.",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="",
        help="Comma-separated channel names. Empty means automatic selection.",
    )
    args = parser.parse_args()

    run_summary = _load_run_summary(args.run_summary)
    env_path = Path(run_summary["env_cache_path"])
    packed_path = Path(run_summary["packed_times_path"])
    group_path = Path(run_summary["group_analysis_path"])
    tf_path = Path(run_summary["group_tf_tile_cache_path"])

    env_cache = np.load(env_path, allow_pickle=True)
    packed = np.load(packed_path, allow_pickle=True)
    group_analysis = np.load(group_path, allow_pickle=True)
    tf_cache = np.load(tf_path, allow_pickle=True)

    ch_names = _as_names(group_analysis["ch_names"].tolist())
    events_bool = np.asarray(group_analysis["events_bool"], dtype=bool)
    centroid_time = np.asarray(group_analysis["centroid_time"], dtype=np.float64)

    if args.event_indices.strip():
        event_indices = [int(x.strip()) for x in args.event_indices.split(",") if x.strip()]
    else:
        event_indices = _select_events(
            events_bool,
            centroid_time,
            n_events=int(args.n_events),
            min_active_channels=int(args.min_active_channels),
        )
    if not event_indices:
        raise ValueError("No events selected.")

    if args.channels.strip():
        channels = [x.strip() for x in args.channels.split(",") if x.strip()]
    else:
        channels = _select_channels(
            ch_names,
            events_bool,
            centroid_time,
            event_indices=event_indices,
            n_channels=int(args.n_channels),
            min_fraction=float(args.channel_min_fraction),
        )
    if not channels:
        raise ValueError("No channels selected.")

    traces, heat, concat_t, segment_lengths = _build_concat_data(
        env_cache=env_cache,
        tf_cache=tf_cache,
        group_analysis=group_analysis,
        packed=packed,
        channels=channels,
        event_indices=event_indices,
        plot_window_sec=float(args.plot_window_sec),
        tf_freq_percentile=float(args.tf_freq_percentile),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf"
    metadata_path = args.output_dir / f"{args.output_stem}_metadata.json"

    _plot_figure(
        traces=traces,
        heat=heat,
        time_axis=concat_t,
        segment_lengths=segment_lengths,
        channels=channels,
        event_indices=event_indices,
        group_analysis=group_analysis,
        plot_window_sec=float(args.plot_window_sec),
        title=str(args.figure_title),
        output_png=output_png,
        output_pdf=output_pdf,
    )

    active_counts = events_bool.sum(axis=0)
    channel_hits = events_bool[:, event_indices].sum(axis=1)
    channel_hits_by_name = {
        ch: int(channel_hits[ch_names.index(ch)]) for ch in channels if ch in ch_names
    }
    metadata = {
        "run_summary": str(args.run_summary),
        "source_paths": {
            "env_cache": str(env_path),
            "packed_times": str(packed_path),
            "group_analysis": str(group_path),
            "group_tf_tile_cache": str(tf_path),
        },
        "selection": {
            "event_indices": event_indices,
            "event_active_channel_counts": [int(active_counts[i]) for i in event_indices],
            "channels": channels,
            "channel_selected_event_hits": channel_hits_by_name,
            "event_selection_rule": "score = centroid_time_spread * sqrt(active_channel_count), filtered by min_active_channels",
            "channel_selection_rule": "top participating channels, ordered by median centroid_time",
        },
        "plot": {
            "plot_window_sec": float(args.plot_window_sec),
            "band_hz": [80, 250],
            "tf_freq_percentile": float(args.tf_freq_percentile),
            "outputs": [str(output_png), str(output_pdf)],
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {output_png}")
    print(f"[OK] wrote {output_pdf}")
    print(f"[OK] wrote {metadata_path}")
    print(f"[INFO] events: {event_indices}")
    print(f"[INFO] channels: {channels}")


if __name__ == "__main__":
    main()
