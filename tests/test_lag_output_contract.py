from __future__ import annotations

from pathlib import Path

import numpy as np

from src.group_event_analysis import (
    lag_rank_from_centroids,
    load_group_analysis_results,
    save_group_analysis_results,
)


def test_lag_abs_and_lag_raw_contract_roundtrip(tmp_path: Path) -> None:
    ch_names = ["A1", "B1"]
    event_windows = np.array([[1.0, 1.5], [3.0, 3.5]], dtype=np.float64)
    centroids = np.array([[0.02, 0.03], [0.04, 0.06]], dtype=np.float64)
    events_bool = np.ones((2, 2), dtype=bool)
    lag_raw, lag_rank = lag_rank_from_centroids(centroids, events_bool, align="first_centroid")
    lag_abs = centroids + event_windows[:, 0][None, :]

    out_npz = tmp_path / "group_analysis_contract.npz"
    save_group_analysis_results(
        str(out_npz),
        sfreq=1000.0,
        band="ripple",
        ch_names=ch_names,
        event_windows=event_windows,
        centroid_time=centroids,
        events_bool=events_bool,
        lag_abs=lag_abs,
        lag_raw=lag_raw,
        lag_rank=lag_rank,
    )

    loaded = load_group_analysis_results(str(out_npz))
    assert "lag_abs" in loaded
    assert "lag_raw" in loaded
    assert "lag_rank" in loaded
    assert np.allclose(loaded["lag_abs"], lag_abs)
    assert np.allclose(loaded["lag_raw"], lag_raw)
    assert np.array_equal(loaded["lag_rank"], lag_rank)


def test_load_legacy_file_derives_lag_abs(tmp_path: Path) -> None:
    centroids = np.array([[0.01, 0.02], [0.03, 0.04]], dtype=np.float64)
    event_windows = np.array([[10.0, 10.5], [20.0, 20.5]], dtype=np.float64)
    legacy_npz = tmp_path / "legacy_group_analysis.npz"
    np.savez_compressed(
        str(legacy_npz),
        sfreq=np.array([1000.0], dtype=np.float64),
        band=np.array(["ripple"], dtype=object),
        ch_names=np.array(["A1", "B1"], dtype=object),
        window_sec=np.array([0.5], dtype=np.float64),
        n_events=np.array([2], dtype=np.int64),
        n_channels=np.array([2], dtype=np.int64),
        event_windows=event_windows,
        centroid_time=centroids,
        events_bool=np.ones((2, 2), dtype=bool),
        lag_raw=np.array([[0.0, 0.0], [0.02, 0.02]], dtype=np.float64),
        lag_rank=np.array([[0, 0], [1, 1]], dtype=np.int64),
    )

    loaded = load_group_analysis_results(str(legacy_npz))
    expected = centroids + event_windows[:, 0][None, :]
    assert np.allclose(loaded["lag_abs"], expected)


def test_lag_freq_alias_from_tf_centroid_freq(tmp_path: Path) -> None:
    ch_names = ["A1", "B1"]
    event_windows = np.array([[0.0, 0.5]], dtype=np.float64)
    centroids = np.array([[0.01], [0.02]], dtype=np.float64)
    events_bool = np.ones((2, 1), dtype=bool)
    lag_raw, lag_rank = lag_rank_from_centroids(centroids, events_bool, align="first_centroid")
    tf_freq = np.array([[120.0], [130.0]], dtype=np.float64)

    out_npz = tmp_path / "group_analysis_freq.npz"
    save_group_analysis_results(
        str(out_npz),
        sfreq=1000.0,
        band="ripple",
        ch_names=ch_names,
        event_windows=event_windows,
        centroid_time=centroids,
        events_bool=events_bool,
        lag_raw=lag_raw,
        lag_rank=lag_rank,
        tf_centroid_freq=tf_freq,
    )

    loaded = load_group_analysis_results(str(out_npz))
    assert "lag_freq" in loaded
    assert np.allclose(loaded["lag_freq"], tf_freq)
