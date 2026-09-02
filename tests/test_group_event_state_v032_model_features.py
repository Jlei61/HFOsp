"""Task 3: event token features + TRAIN-only standardisation (design §2, F1-F6)."""

from __future__ import annotations

import numpy as np

from src.topic5_group_event_state.v032_model.features import (
    FEATURE_VERSION,
    TrainStandardizer,
    event_token_features,
)


def _toy_event_arrays():
    n, c, b, p = 5, 4, 2, 1
    participation = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 1],   # single participant
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    )
    delay = np.full((n, c), np.nan)
    delay[0, [0, 1]] = [0.0, 0.005]
    delay[1, [0, 1, 2]] = [0.0, 0.02, 0.05]
    delay[2, 3] = 0.0
    delay[3, [0, 2, 3]] = [0.0, 0.03, 0.03]
    delay[4, :] = [0.0, 0.01, 0.02, 0.06]
    tied = np.full((n, c), -1, dtype=np.int16)
    tied[0, [0, 1]] = [0, 0]
    tied[1, [0, 1, 2]] = [0, 1, 2]
    tied[2, 3] = 0
    tied[3, [0, 2, 3]] = [0, 1, 1]
    tied[4, :] = [0, 0, 1, 2]
    rng = np.random.default_rng(0)
    band_features = rng.normal(size=(n, c, b, 5)).astype(np.float32)
    band_features[~participation] = np.nan
    cross = rng.normal(size=(n, c, p)).astype(np.float32)
    coords = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0], [10, 10, 10]], dtype=np.float32)
    return dict(
        participation=participation,
        relative_delay=delay,
        tied_group_id=tied,
        band_features=band_features,
        cross_band_lag=cross,
        contact_valid=np.array([True, True, True, True]),
        coords=coords,
        core_seconds=np.array([0.2, 0.3, 0.25, 0.4, 0.5], dtype=np.float32),
        has_waveform=np.array([1, 1, 1, 0, 1], dtype=bool),
        band_available=(True, True),
    )


def test_f1_f5_feature_families_present_and_no_dt_column():
    x, names = event_token_features(**_toy_event_arrays())
    assert x.shape == (5, len(names)) and x.dtype == np.float32
    assert FEATURE_VERSION.startswith("v032")
    lowered = [n.lower() for n in names]
    assert not any(("dt" in n.split("_")) or "interval" in n or "delta_t" in n for n in lowered)
    prefixes = (
        "participation[", "leader[", "extent_", "tied_", "delay_", "dispersion_",
        "band_", "crossband_", "confidence_", "coverage_",
    )
    for prefix in prefixes:
        assert any(n.startswith(prefix) for n in names), prefix
    # participation block is the raw boolean pattern
    part_cols = [i for i, n in enumerate(names) if n.startswith("participation[")]
    assert np.array_equal(x[:, part_cols] > 0.5, _toy_event_arrays()["participation"])


def test_f6_single_participant_event_has_zero_dispersion_and_flag():
    x, names = event_token_features(**_toy_event_arrays())
    disp = [i for i, n in enumerate(names) if n.startswith("dispersion_") and "flag" not in n]
    flag = names.index("dispersion_single_participant_flag")
    assert np.all(x[2, disp] == 0.0) and x[2, flag] == 1.0
    assert x[4, flag] == 0.0 and np.any(x[4, disp] > 0.0)
    # pairwise mean distance of contacts 0 and 1 is exactly 3 mm
    mean_pair = names.index("dispersion_mean_pairwise_mm")
    assert abs(x[0, mean_pair] - 3.0) < 1e-5


def test_f2_f3_f4_standardizer_uses_train_rows_only_and_zeroes_degenerate_columns():
    x = np.array(
        [[1.0, 5.0, np.nan], [3.0, 5.0, 1.0], [5.0, 5.0, 2.0], [100.0, 9.0, 3.0]],
        dtype=np.float32,
    )
    train = np.array([True, True, True, False])
    std = TrainStandardizer.fit(x, train)
    assert np.allclose(std.mean[0], 3.0) and np.allclose(std.scale[0], np.std([1, 3, 5]))
    z = std.transform(x)
    assert np.allclose(z[:3, 0], (np.array([1, 3, 5]) - 3.0) / np.std([1, 3, 5]))
    assert std.zero_variance[1] and np.all(z[:, 1] == 0.0)          # TRAIN-constant column
    assert z[0, 2] == 0.0 and np.isfinite(z).all()                    # NaN -> 0
    rebuilt = TrainStandardizer.from_dict(std.to_dict())
    assert np.allclose(rebuilt.transform(x), z)
