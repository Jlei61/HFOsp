from __future__ import annotations

import numpy as np

from src.group_event_analysis import (
    EventWindow,
    build_stitched_window_signal,
    compute_centroid_matrix_spectrogram,
    compute_stitched_spectrogram_centroids_legacy,
    filter_windows_for_legacy_segment_loop,
)


def _burst_signal(
    sfreq: float,
    duration_sec: float,
    centers_sec: list[float],
    sigma_sec: float = 0.015,
    freq_hz: float = 120.0,
) -> np.ndarray:
    t = np.arange(int(round(duration_sec * sfreq)), dtype=np.float64) / sfreq
    x = np.zeros_like(t)
    for c in centers_sec:
        env = np.exp(-0.5 * ((t - c) / sigma_sec) ** 2)
        x += env * np.sin(2.0 * np.pi * freq_hz * t)
    return x[None, :]


def test_filter_windows_for_legacy_segment_loop_drops_cross_boundary_windows() -> None:
    windows = [
        EventWindow(199.70, 199.95, 0),
        EventWindow(199.90, 200.20, 1),
        EventWindow(200.30, 200.55, 2),
    ]

    kept = filter_windows_for_legacy_segment_loop(
        windows,
        segment_duration_sec=200.0,
        record_last_sec=399.999,
        sfreq=800.0,
    )

    assert [w.event_id for w in kept] == [0, 2]


def test_build_stitched_window_signal_uses_inclusive_endpoints() -> None:
    x = np.arange(10, dtype=np.float64)[None, :]
    windows = [EventWindow(1.0, 3.0, 0), EventWindow(5.0, 6.0, 1)]

    stitched, split_border_t = build_stitched_window_signal(x, windows, sfreq=1.0)

    assert np.array_equal(stitched, np.array([[1.0, 2.0, 3.0, 5.0, 6.0]]))
    assert np.allclose(split_border_t, np.array([3.0, 5.0]))


def test_stitched_centroid_matches_per_window_for_single_window() -> None:
    sfreq = 800.0
    x = _burst_signal(sfreq, 0.30, [0.15])
    windows = [EventWindow(0.0, 0.30, 0)]
    detections = {"A1": np.array([[0.0, 0.30]], dtype=np.float64)}

    per_window, events_bool = compute_centroid_matrix_spectrogram(
        windows=windows,
        detections=detections,
        ch_names=["A1"],
        x_band=x,
        sfreq=sfreq,
        start_sec=0.0,
        centroid_power=3.0,
    )
    stitched, split_border_t = build_stitched_window_signal(x, windows, sfreq=sfreq)
    legacy = compute_stitched_spectrogram_centroids_legacy(
        stitched,
        split_border_t,
        sfreq=sfreq,
        centroid_power=3.0,
    )

    assert events_bool[0, 0]
    assert np.isfinite(per_window[0, 0])
    assert np.isfinite(legacy[0, 0])
    assert abs(legacy[0, 0] - per_window[0, 0]) < 1e-6


def test_stitched_centroid_keeps_concat_timeline_across_windows() -> None:
    sfreq = 800.0
    x = _burst_signal(sfreq, 1.50, [0.12, 1.12])
    windows = [EventWindow(0.0, 0.30, 0), EventWindow(1.0, 1.30, 1)]

    stitched, split_border_t = build_stitched_window_signal(x, windows, sfreq=sfreq)
    legacy = compute_stitched_spectrogram_centroids_legacy(
        stitched,
        split_border_t,
        sfreq=sfreq,
        centroid_power=3.0,
    )

    assert legacy.shape == (1, 2)
    assert 0.05 < legacy[0, 0] < 0.25
    assert legacy[0, 1] > 0.30
    assert legacy[0, 1] > legacy[0, 0]
