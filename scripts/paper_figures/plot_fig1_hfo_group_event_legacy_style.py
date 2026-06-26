#!/usr/bin/env python3
"""Legacy-style paper panel for interictal HFO group events.

This script intentionally follows the old ReplayIED spectrogram layout:
event windows are cut from EDF, bipolar-referenced, 80-250 Hz filtered,
concatenated, and then a fresh spectrogram is computed over the concatenated
signal. That gives the old "Normalized Spectrogram" look with one frequency
block per channel and a mass-center trajectory per group event.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ["HOME"] = "/tmp/hfo-paper-home"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, spectrogram


DEFAULT_SUBJECT_DIR = Path("/mnt/yuquan_data/yuquan_24h_edf/chengshuai")
DEFAULT_RECORD = "FC10477Q"
DEFAULT_OUTPUT_DIR = Path("results/paper-ready-figure/fig1_hfo_group_event_demo/figures")
DEFAULT_CHANNELS = (
    "K3,K4,K5,K6,K7,K8,K9,K11,K12,K13,E8,E10,E11,E12,E13"
)
DEFAULT_EVENT_INDICES = "22,237,1458"


def _standard_channel_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"^(EEG|POL)\s+", "", name)
    name = re.sub(r"-(Ref|REF)$", "", name)
    name = name.replace(" ", "")
    return name


def _display_label(bipolar_name: str) -> str:
    return str(bipolar_name).split("-")[0]


def _bandpass(data: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    nyq = 0.5 * float(fs)
    b, a = butter(3, [float(low) / nyq, float(high) / nyq], btype="bandpass")
    return filtfilt(b, a, data, axis=-1)


def _notch(data: np.ndarray, fs: float, freqs: list[float]) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64)
    for freq in freqs:
        if freq >= 0.5 * fs:
            continue
        b, a = iirnotch(w0=float(freq), Q=30.0, fs=float(fs))
        out = filtfilt(b, a, out, axis=-1)
    return out


def _resolve_channels(requested_labels: list[str], gpu_names: list[str]) -> list[str]:
    by_label = {_display_label(name): name for name in gpu_names}
    out = []
    missing = []
    for label in requested_labels:
        if label in by_label:
            out.append(by_label[label])
        else:
            missing.append(label)
    if missing:
        print(f"[WARN] requested channels not found in GPU bipolar list: {missing}")
    if not out:
        raise ValueError("No requested channels were found in GPU bipolar list.")
    return out


def _event_centers_from_gpu(
    *,
    gpu_dets: np.ndarray,
    channel_indices: list[int],
    packed_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    centers = np.full((len(channel_indices), packed_times.shape[0]), np.nan, dtype=np.float64)
    active = np.zeros_like(centers, dtype=bool)
    for row, ch_idx in enumerate(channel_indices):
        dets = np.asarray(gpu_dets[ch_idx], dtype=np.float64)
        if dets.size == 0:
            continue
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        det_centers = 0.5 * (dets[:, 0] + dets[:, 1])
        for ei, (start, end) in enumerate(packed_times):
            overlap = (dets[:, 0] < float(end)) & (dets[:, 1] > float(start))
            if not np.any(overlap):
                continue
            mid = 0.5 * (float(start) + float(end))
            candidates = det_centers[overlap]
            centers[row, ei] = float(candidates[np.argmin(np.abs(candidates - mid))])
            active[row, ei] = True
    return centers, active


def _select_events(
    centers: np.ndarray,
    active: np.ndarray,
    packed_times: np.ndarray,
    n_events: int,
    min_active_channels: int,
) -> list[int]:
    active_count = active.sum(axis=0)
    spread = np.zeros(active.shape[1], dtype=np.float64)
    for ei in range(active.shape[1]):
        vals = centers[:, ei]
        ok = active[:, ei] & np.isfinite(vals)
        if int(ok.sum()) > 1:
            spread[ei] = np.nanmax(vals[ok]) - np.nanmin(vals[ok])
    valid = active_count >= int(min_active_channels)
    if not np.any(valid):
        valid = active_count > 0
    score = np.where(valid, active_count * (1.0 + spread / 0.08), -np.inf)
    picks = np.argsort(-score)[: int(n_events)]
    return sorted(int(i) for i in picks if np.isfinite(score[i]))


def _select_and_sort_channels(
    bipolar_names: list[str],
    centers: np.ndarray,
    active: np.ndarray,
    packed_times: np.ndarray,
    event_indices: list[int],
    n_channels: int,
) -> tuple[list[int], list[str]]:
    hits = active[:, event_indices].sum(axis=1)
    top = np.argsort(-hits)[: int(n_channels)]
    med = {}
    rel_starts = packed_times[event_indices, 0]
    for row in top:
        vals = centers[int(row), event_indices]
        ok = active[int(row), event_indices] & np.isfinite(vals)
        if np.any(ok):
            med[int(row)] = float(np.nanmedian(vals[ok] - rel_starts[ok]))
        else:
            med[int(row)] = np.inf
    ordered = sorted([int(x) for x in top], key=lambda row: (med[row], -hits[row]))
    return ordered, [bipolar_names[i] for i in ordered]


def _load_bipolar_event_snippets(
    *,
    edf_path: Path,
    bipolar_pairs: list[tuple[str, str]],
    event_windows: np.ndarray,
    window_sec: float,
    fs_out: float,
    band_hz: tuple[float, float],
) -> tuple[np.ndarray, float]:
    raw = mne.io.read_raw_edf(
        str(edf_path),
        preload=False,
        encoding="latin1",
        verbose="ERROR",
    )
    fs_in = float(raw.info["sfreq"])
    raw_name_map = {_standard_channel_name(name): idx for idx, name in enumerate(raw.ch_names)}
    target_samples = int(round(float(window_sec) * float(fs_out)))
    event_segments = []

    for start, end in event_windows:
        center = 0.5 * (float(start) + float(end))
        crop_start = max(0.0, center - 0.5 * float(window_sec))
        crop_end = crop_start + float(window_sec)
        start_idx = int(round(crop_start * fs_in))
        stop_idx = int(round(crop_end * fs_in))

        needed_names = []
        for left, right in bipolar_pairs:
            needed_names.extend([left, right])
        missing = [name for name in needed_names if name not in raw_name_map]
        if missing:
            raise ValueError(f"EDF missing channels needed for bipolar pairs: {sorted(set(missing))[:8]}")
        picks = sorted({raw_name_map[name] for name in needed_names})
        data = raw.get_data(picks=picks, start=start_idx, stop=stop_idx)
        pick_to_row = {pick: row for row, pick in enumerate(picks)}

        bipolar = []
        for left, right in bipolar_pairs:
            left_row = pick_to_row[raw_name_map[left]]
            right_row = pick_to_row[raw_name_map[right]]
            bipolar.append(data[left_row] - data[right_row])
        bipolar = np.asarray(bipolar, dtype=np.float64)
        if fs_out != fs_in:
            up = int(round(fs_out))
            down = int(round(fs_in))
            bipolar = resample_poly(bipolar, up, down, axis=-1)
        bipolar = _notch(bipolar, fs_out, [50, 100, 150, 200, 250])
        bipolar = _bandpass(bipolar, fs_out, band_hz[0], band_hz[1])
        if bipolar.shape[1] < target_samples:
            bipolar = np.pad(
                bipolar,
                ((0, 0), (0, target_samples - bipolar.shape[1])),
                mode="constant",
                constant_values=np.nan,
            )
        elif bipolar.shape[1] > target_samples:
            bipolar = bipolar[:, :target_samples]
        event_segments.append(bipolar)

    return np.concatenate(event_segments, axis=1), float(fs_out)


def _norm_specs_to_event_max(
    all_specs_cat: np.ndarray,
    n_freq: int,
    spec_times: np.ndarray,
    split_borders: np.ndarray,
) -> np.ndarray:
    normed = np.zeros_like(all_specs_cat, dtype=np.float64)
    split_edges = np.asarray([0.0] + split_borders.tolist(), dtype=np.float64)
    windows = np.vstack([split_edges[:-1], split_edges[1:]]).T
    n_channels = all_specs_cat.shape[0] // int(n_freq)
    for start, end in windows:
        tmask = (spec_times > start) & (spec_times < end)
        if not np.any(tmask):
            continue
        for ci in range(n_channels):
            row_slice = slice(ci * n_freq, (ci + 1) * n_freq)
            block = all_specs_cat[row_slice, :][:, tmask]
            denom = float(np.nanmax(block)) if np.isfinite(block).any() else 0.0
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            normed[row_slice, :][:, tmask] = np.clip(block / denom, 0.0, 1.0)
    return normed


def _spec_centers(
    channel_spec: np.ndarray,
    spec_times: np.ndarray,
    split_borders: np.ndarray,
) -> list[tuple[float, float]]:
    split_edges = np.asarray([0.0] + split_borders.tolist(), dtype=np.float64)
    centers = []
    for start, end in np.vstack([split_edges[:-1], split_edges[1:]]).T:
        tmask = (spec_times > start) & (spec_times < end)
        win = np.asarray(channel_spec[:, tmask], dtype=np.float64)
        if win.size == 0 or not np.isfinite(win).any() or np.nanmax(win) <= 0:
            centers.append((np.nan, np.nan))
            continue
        win = np.nan_to_num(win, nan=0.0) ** 3
        denom = float(np.sum(win))
        if denom <= 0:
            centers.append((np.nan, np.nan))
            continue
        weight = win / denom
        tvals = spec_times[tmask]
        time_grid = np.tile(tvals, (channel_spec.shape[0], 1))
        freq_grid = np.tile(np.arange(channel_spec.shape[0]), (len(tvals), 1)).T
        centers.append((float(np.sum(weight * time_grid)), float(np.sum(weight * freq_grid))))
    return centers


def _compute_legacy_specs(
    split_conti_high: np.ndarray,
    fs: float,
    split_borders: np.ndarray,
    spec_win_sec: float,
    spec_freq_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_specs = []
    spec_times = None
    spec_freqs = None
    for row in split_conti_high:
        freqs, times, spec = spectrogram(
            row,
            fs,
            window="hamming",
            nperseg=int(round(float(spec_win_sec) * fs)),
            noverlap=int(round(0.8 * float(spec_win_sec) * fs)),
            nfft=int(round(float(spec_win_sec) * fs)),
            mode="magnitude",
        )
        spec = gaussian_filter(spec, sigma=1.5)
        fmask = (freqs > float(spec_freq_range[0])) & (freqs < float(spec_freq_range[1]))
        all_specs.append(spec[fmask])
        spec_times = times
        spec_freqs = freqs[fmask]
    assert spec_times is not None and spec_freqs is not None
    all_specs_cat = np.concatenate(all_specs, axis=0)
    normed = _norm_specs_to_event_max(all_specs_cat, len(spec_freqs), spec_times, split_borders)
    centers = np.asarray([_spec_centers(spec, spec_times, split_borders) for spec in all_specs])
    return normed, spec_times, spec_freqs, centers


def _plot(
    *,
    split_conti_high: np.ndarray,
    fs: float,
    normed_specs: np.ndarray,
    spec_times: np.ndarray,
    spec_freqs: np.ndarray,
    centers: np.ndarray,
    split_borders: np.ndarray,
    channels: list[str],
    output_png: Path,
    output_pdf: Path,
) -> None:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False

    labels = [_display_label(ch) for ch in channels]
    n_channels = len(labels)
    n_freq = len(spec_freqs)
    total_t = split_conti_high.shape[1] / float(fs)

    fig = plt.figure(figsize=(6.5, 4.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.52, 1.45], wspace=0.055)
    ax_trace = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])

    gap = np.nanstd(split_conti_high) * 8.0
    x = np.arange(split_conti_high.shape[1], dtype=np.float64) / float(fs)
    for ci in range(n_channels):
        ax_trace.plot(x, split_conti_high[ci] + ci * gap, color="black", linewidth=0.42)
    for border in split_borders:
        ax_trace.axvline(float(border), color="#b9b9b9", linestyle="--", linewidth=0.55, alpha=0.9)
    ax_trace.set_yticks(np.arange(n_channels) * gap)
    ax_trace.set_yticklabels(labels, fontsize=8)
    ax_trace.set_xlim(0.0, total_t)
    ax_trace.set_ylim(-0.8 * gap, (n_channels - 0.2) * gap)
    ax_trace.invert_yaxis()
    ax_trace.set_title("80-250Hz", fontsize=10)
    ax_trace.set_xlabel("Time (s)", fontsize=9)
    ax_trace.tick_params(axis="x", labelsize=8, length=2)
    ax_trace.tick_params(axis="y", length=0)
    ax_trace.spines["top"].set_visible(False)
    ax_trace.spines["right"].set_visible(False)

    im = ax_spec.imshow(
        normed_specs,
        origin="upper",
        aspect="auto",
        interpolation="nearest",
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        extent=(0.0, total_t, n_channels * n_freq, 0.0),
    )
    for border in split_borders:
        ax_spec.axvline(float(border), color="white", linestyle="-", linewidth=0.9, alpha=0.95)
    for ci in range(1, n_channels):
        ax_spec.axhline(ci * n_freq, color="#c7c7c7", linewidth=0.45, linestyle="--")
    for ev in range(centers.shape[1]):
        xy = []
        for ci in range(n_channels):
            t_center, f_center = centers[ci, ev]
            if not np.isfinite(t_center) or not np.isfinite(f_center):
                continue
            xy.append((t_center, ci * n_freq + f_center))
        if xy:
            arr = np.asarray(xy, dtype=np.float64)
            ax_spec.plot(arr[:, 0], arr[:, 1], color="red", linewidth=0.8, alpha=0.9, zorder=4)
            ax_spec.scatter(
                arr[:, 0],
                arr[:, 1],
                s=10,
                facecolors="#ffb000",
                edgecolors="red",
                linewidth=0.35,
                zorder=5,
            )
    ax_spec.set_yticks([])
    ax_spec.set_yticklabels([])
    ax_spec.set_xlim(0.0, total_t)
    ax_spec.set_title("Normalized Spectrogram", fontsize=10)
    ax_spec.set_xlabel("Time (s)", fontsize=9)
    ax_spec.tick_params(axis="x", labelsize=8, length=2)
    ax_spec.tick_params(axis="y", length=0)
    ax_spec.spines["top"].set_visible(False)
    ax_spec.spines["right"].set_visible(False)

    cbar = fig.colorbar(im, ax=ax_spec, fraction=0.018, pad=0.015)
    cbar.set_ticks([0, 1])
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.outline.set_visible(False)

    fig.suptitle("Yuquan Y1", x=0.11, y=0.98, ha="left", fontsize=11, fontweight="bold")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-dir", type=Path, default=DEFAULT_SUBJECT_DIR)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="yuquan_y1_hfo_group_event_demo")
    parser.add_argument("--channels", default=DEFAULT_CHANNELS)
    parser.add_argument("--n-events", type=int, default=3)
    parser.add_argument("--n-channels", type=int, default=10)
    parser.add_argument("--min-active-channels", type=int, default=6)
    parser.add_argument("--window-sec", type=float, default=0.32)
    parser.add_argument("--fs-out", type=float, default=1000.0)
    parser.add_argument(
        "--event-indices",
        default=DEFAULT_EVENT_INDICES,
        help="Comma-separated packed event indices. Use an empty string for automatic selection.",
    )
    args = parser.parse_args()

    subject_dir = args.subject_dir
    record = str(args.record)
    edf_path = subject_dir / f"{record}.edf"
    gpu_path = subject_dir / f"{record}_gpu.npz"
    packed_path = subject_dir / f"{record}_packedTimes.npy"

    gpu = np.load(gpu_path, allow_pickle=True)
    packed = np.load(packed_path, allow_pickle=True)
    gpu_names = [str(x) for x in gpu["chns_names"].tolist()]
    gpu_index = {name: idx for idx, name in enumerate(gpu_names)}
    requested = [x.strip() for x in str(args.channels).split(",") if x.strip()]
    candidate_channels = _resolve_channels(requested, gpu_names)
    candidate_indices = [gpu_index[ch] for ch in candidate_channels]
    centers, active = _event_centers_from_gpu(
        gpu_dets=gpu["whole_dets"],
        channel_indices=candidate_indices,
        packed_times=packed,
    )

    fixed_event_indices = bool(str(args.event_indices).strip())
    if fixed_event_indices:
        event_indices = [int(x.strip()) for x in str(args.event_indices).split(",") if x.strip()]
    else:
        event_indices = _select_events(
            centers,
            active,
            packed,
            n_events=int(args.n_events),
            min_active_channels=int(args.min_active_channels),
        )
    if not event_indices:
        raise ValueError("No group events selected.")

    selected_rows, selected_channels = _select_and_sort_channels(
        candidate_channels,
        centers,
        active,
        packed_times=packed,
        event_indices=event_indices,
        n_channels=int(args.n_channels),
    )
    selected_candidate_indices = [candidate_indices[row] for row in selected_rows]
    selected_pairs = [tuple(x) for x in gpu["bipolar_pairs"][selected_candidate_indices]]

    split_high, fs = _load_bipolar_event_snippets(
        edf_path=edf_path,
        bipolar_pairs=selected_pairs,
        event_windows=packed[event_indices],
        window_sec=float(args.window_sec),
        fs_out=float(args.fs_out),
        band_hz=(80.0, 250.0),
    )
    split_borders = np.arange(1, len(event_indices) + 1, dtype=np.float64) * float(args.window_sec)
    normed_specs, spec_times, spec_freqs, spec_centers = _compute_legacy_specs(
        split_high,
        fs=fs,
        split_borders=split_borders,
        spec_win_sec=0.05,
        spec_freq_range=(50.0, 300.0),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_dir / f"{args.output_stem}.png"
    output_pdf = args.output_dir / f"{args.output_stem}.pdf"
    metadata_path = args.output_dir / f"{args.output_stem}_metadata.json"
    _plot(
        split_conti_high=split_high,
        fs=fs,
        normed_specs=normed_specs,
        spec_times=spec_times,
        spec_freqs=spec_freqs,
        centers=spec_centers,
        split_borders=split_borders,
        channels=selected_channels,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    metadata = {
        "legacy_code_reference": [
            "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/for523_p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool.py",
            "ReplayIED/inter_events/yuquan_24h_perPatientAnalysis_dropRef/p16_packGroupEvents_per2h_showSpecs_bipolar_refine_bool_withFreqCenter.py",
        ],
        "source_paths": {
            "edf": str(edf_path),
            "gpu_npz": str(gpu_path),
            "packed_times": str(packed_path),
        },
        "selection": {
            "figure_label": "Fig1-A",
            "candidate_channels": candidate_channels,
            "selected_channels": selected_channels,
            "display_labels": [_display_label(ch) for ch in selected_channels],
            "event_indices": event_indices,
            "event_indices_mode": "fixed_clean_examples" if fixed_event_indices else "automatic",
            "event_windows_sec": packed[event_indices].tolist(),
            "event_active_channel_counts_in_candidate_set": active[:, event_indices].sum(axis=0).astype(int).tolist(),
            "event_selection_rule": (
                "fixed clean examples selected after visual screening"
                if fixed_event_indices
                else "top active candidate-channel events weighted by within-event detection-center spread"
            ),
            "channel_selection_rule": "top participating candidate channels, sorted by median detection center over selected events",
        },
        "plot": {
            "window_sec": float(args.window_sec),
            "fs_out": float(args.fs_out),
            "band_hz": [80, 250],
            "spectrogram": {
                "window": "hamming",
                "nperseg_sec": 0.05,
                "overlap_fraction": 0.8,
                "freq_range_hz": [50, 300],
                "gaussian_sigma": 1.5,
                "normalization": "per channel per event window max, matching old ReplayIED norm_theSpec_toMaxOne",
            },
            "outputs": [str(output_png), str(output_pdf)],
            "layout": {
                "figure_size_in": [6.5, 4.25],
                "panel_width_ratios": [0.52, 1.45],
                "x_extent_sec": [0.0, float(split_high.shape[1] / float(fs))],
            },
        },
    }
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {output_png}")
    print(f"[OK] wrote {output_pdf}")
    print(f"[OK] wrote {metadata_path}")
    print(f"[INFO] events: {event_indices}")
    print(f"[INFO] channels: {selected_channels}")


if __name__ == "__main__":
    main()
