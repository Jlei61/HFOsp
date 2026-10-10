"""Shared spectrogram kernel for Figure 1 Panels a1 and a2."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, generate_binary_structure, label
from scipy.signal import spectrogram


ALGORITHM_ID = "fig1_gaussian_smoothed_magnitude_v1"


def compute_smoothed_magnitude_spectrogram(
    signal: np.ndarray,
    fs: float,
    *,
    window: str = "hann",
    window_sec: float = 0.18,
    overlap_sec: float = 0.16,
    freq_range_hz: tuple[float, float] = (0.0, 240.0),
    gaussian_sigma: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the common Figure 1 magnitude/Gaussian spectrogram quantity."""
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"signal must be 1D, got {values.shape}")
    nperseg = int(round(float(window_sec) * float(fs)))
    noverlap = int(round(float(overlap_sec) * float(fs)))
    if nperseg < 2 or noverlap < 0 or noverlap >= nperseg:
        raise ValueError(
            f"invalid spectrogram window/overlap: nperseg={nperseg}, noverlap={noverlap}"
        )
    freqs, times, magnitude = spectrogram(
        values,
        fs=float(fs),
        window=str(window),
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nperseg,
        mode="magnitude",
    )
    low, high = map(float, freq_range_hz)
    fmask = (freqs >= low) & (freqs <= high)
    if not np.any(fmask):
        raise ValueError(f"no spectrogram bins in requested range {freq_range_hz}")
    smoothed = gaussian_filter(magnitude[fmask], sigma=float(gaussian_sigma))
    return freqs[fmask], times, np.maximum(smoothed, 0.0)


def normalise_event_spectrograms(
    stacked_magnitude: np.ndarray,
    n_freq: int,
    spec_times: np.ndarray,
    split_borders: np.ndarray,
) -> np.ndarray:
    """Normalize the displayed magnitude within each channel x event block."""
    normed = np.zeros_like(stacked_magnitude, dtype=np.float64)
    split_edges = np.asarray([0.0] + np.asarray(split_borders, float).tolist(), dtype=np.float64)
    windows = np.vstack([split_edges[:-1], split_edges[1:]]).T
    n_channels = stacked_magnitude.shape[0] // int(n_freq)
    for start, end in windows:
        tmask = (spec_times > start) & (spec_times < end)
        if not np.any(tmask):
            continue
        for ci in range(n_channels):
            row_slice = slice(ci * n_freq, (ci + 1) * n_freq)
            block = stacked_magnitude[row_slice, :][:, tmask]
            denom = float(np.nanmax(block)) if np.isfinite(block).any() else 0.0
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            normed[row_slice, :][:, tmask] = np.clip(block / denom, 0.0, 1.0)
    return normed


def dominant_enhancement_centroids(
    channel_magnitude: np.ndarray,
    spec_times: np.ndarray,
    split_borders: np.ndarray,
    edge_guard_sec: float,
    enhancement_threshold: float,
) -> list[tuple[float, float]]:
    """Centroid the dominant displayed HFO enhancement in every event tile.

    The support is the 8-connected component containing the within-event maximum
    at ``enhancement_threshold`` times that maximum.  This is the exact Figure 1a
    reader-facing marker contract: the marker is computed from the same smoothed
    magnitude that is displayed and cannot land in a valley between two bursts.
    """
    if not 0.0 < float(enhancement_threshold) <= 1.0:
        raise ValueError("enhancement_threshold must be in (0, 1]")
    split_edges = np.asarray([0.0] + np.asarray(split_borders, float).tolist(), dtype=np.float64)
    centers = []
    for start, end in np.vstack([split_edges[:-1], split_edges[1:]]).T:
        tmask = (spec_times > start + float(edge_guard_sec)) & (
            spec_times < end - float(edge_guard_sec)
        )
        win = np.asarray(channel_magnitude[:, tmask], dtype=np.float64)
        if win.size == 0 or not np.isfinite(win).any() or np.nanmax(win) <= 0:
            centers.append((np.nan, np.nan))
            continue
        win = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        peak_index = np.unravel_index(int(np.argmax(win)), win.shape)
        high_energy = win >= float(enhancement_threshold) * float(win[peak_index])
        components, _ = label(
            high_energy,
            structure=generate_binary_structure(rank=2, connectivity=2),
        )
        peak_component = int(components[peak_index])
        if peak_component <= 0:
            centers.append((np.nan, np.nan))
            continue
        win = np.where(components == peak_component, win, 0.0)
        denom = float(np.sum(win))
        if denom <= 0:
            centers.append((np.nan, np.nan))
            continue
        weight = win / denom
        tvals = spec_times[tmask]
        time_grid = np.tile(tvals, (channel_magnitude.shape[0], 1))
        freq_grid = np.tile(np.arange(channel_magnitude.shape[0]), (len(tvals), 1)).T
        centers.append((float(np.sum(weight * time_grid)), float(np.sum(weight * freq_grid))))
    return centers


def compute_group_event_spectrogram_stack(
    split_conti_high: np.ndarray,
    fs: float,
    split_borders: np.ndarray,
    *,
    spec_window: str = "hamming",
    spec_win_sec: float = 0.05,
    spec_overlap_sec: float = 0.04,
    spec_freq_range: tuple[float, float] = (50.0, 300.0),
    gaussian_sigma: float = 1.5,
    enhancement_threshold: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the complete Figure 1a stacked-spectrogram/centroid payload."""
    all_specs = []
    spec_times = None
    spec_freqs = None
    for row in np.asarray(split_conti_high, float):
        freqs, times, magnitude = compute_smoothed_magnitude_spectrogram(
            row,
            fs,
            window=str(spec_window),
            window_sec=float(spec_win_sec),
            overlap_sec=float(spec_overlap_sec),
            freq_range_hz=spec_freq_range,
            gaussian_sigma=float(gaussian_sigma),
        )
        all_specs.append(magnitude)
        spec_times = times
        spec_freqs = freqs
    if spec_times is None or spec_freqs is None:
        raise ValueError("split_conti_high contains no channel")
    stacked = np.concatenate(all_specs, axis=0)
    normed = normalise_event_spectrograms(stacked, len(spec_freqs), spec_times, split_borders)
    centers = np.asarray(
        [
            dominant_enhancement_centroids(
                magnitude,
                spec_times,
                split_borders,
                edge_guard_sec=float(spec_win_sec),
                enhancement_threshold=float(enhancement_threshold),
            )
            for magnitude in all_specs
        ]
    )
    return normed, spec_times, spec_freqs, centers


def full_extent_edges(centers: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Build pcolormesh edges with exact outer bounds and true interior timing."""
    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.diff(values) > 0):
        raise ValueError("centers must be a strictly increasing 1D array with >=2 values")
    if not lower < values[0] or not values[-1] < upper:
        raise ValueError(f"outer bounds {lower, upper} must contain center range")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[0] = float(lower)
    edges[-1] = float(upper)
    edges[1:-1] = 0.5 * (values[:-1] + values[1:])
    if not np.all(np.diff(edges) > 0):
        raise ValueError("derived pcolormesh edges are not increasing")
    return edges


def centroid_alignment_audit(
    normed_specs: np.ndarray,
    spec_times: np.ndarray,
    spec_freqs: np.ndarray,
    centers: np.ndarray,
    window_sec: float,
    acceptance_threshold: float,
) -> dict:
    """Verify that every Figure 1a centroid lands on displayed enhancement."""
    n_channels, n_events, _ = centers.shape
    n_freq = len(spec_freqs)
    rows = []
    for ci in range(n_channels):
        block = normed_specs[ci * n_freq : (ci + 1) * n_freq]
        for ev in range(n_events):
            center_time, center_freq_index = centers[ci, ev]
            if not np.isfinite(center_time) or not np.isfinite(center_freq_index):
                continue
            ti = int(np.argmin(np.abs(spec_times - float(center_time))))
            fi = int(np.clip(round(float(center_freq_index)), 0, n_freq - 1))
            rows.append(
                {
                    "channel_index": int(ci),
                    "event_index_within_panel": int(ev),
                    "time_within_event_sec": float(center_time - ev * float(window_sec)),
                    "frequency_hz": float(
                        np.interp(
                            float(center_freq_index),
                            np.arange(n_freq, dtype=np.float64),
                            spec_freqs,
                        )
                    ),
                    "display_weight_at_nearest_cell": float(block[fi, ti]),
                }
            )
    if len(rows) != n_channels * n_events:
        raise ValueError(f"expected {n_channels * n_events} finite centroids, audited {len(rows)}")
    support = np.asarray([row["display_weight_at_nearest_cell"] for row in rows])
    if float(np.min(support)) < float(acceptance_threshold):
        raise ValueError(
            "centroid/display misregistration: "
            f"minimum nearest-cell support={np.min(support):.3f}"
        )
    return {
        "n_centroids": int(len(rows)),
        "minimum_display_weight_at_nearest_cell": float(np.min(support)),
        "median_display_weight_at_nearest_cell": float(np.median(support)),
        "acceptance_threshold": float(acceptance_threshold),
        "all_centroids_pass": True,
        "centroids": rows,
    }
