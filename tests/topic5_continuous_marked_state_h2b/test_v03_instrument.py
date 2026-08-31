from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_instrument import (
    effective_rank,
    lagged_decoder_autocorrelation,
    standardise_decoder,
    reset_phase_explained_variance,
    shuffled_temporal_structure_null,
)


def test_effective_rank_distinguishes_collapsed_and_two_dimensional_signal() -> None:
    collapsed = np.ones((20, 3))
    assert effective_rank(collapsed)["effective_rank"] == 0.0
    angle = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    signal = np.column_stack([np.sin(angle), np.cos(angle)])
    observed = effective_rank(signal)
    assert observed["effective_rank"] > 1.9
    assert observed["top_pc_share"] < 0.55


def test_standardise_decoder_drops_constant_dimensions() -> None:
    train = np.column_stack([np.arange(10), np.ones(10)])
    target = train[:3]
    standard, _, _, active = standardise_decoder(train, target)
    assert active.tolist() == [True, False]
    assert standard.shape == (3, 1)


def test_decoder_autocorrelation_estimates_finite_tau_for_decaying_signal() -> None:
    rng = np.random.default_rng(4)
    n = 1000
    value = np.zeros(n)
    for index in range(1, n):
        value[index] = 0.90 * value[index - 1] + rng.normal(scale=0.3)
    observed = lagged_decoder_autocorrelation(
        np.arange(n, dtype=float) * 30.0,
        np.zeros(n, dtype=np.int64),
        value[:, None],
        lag_minutes=(0.5, 1.0, 2.0, 5.0, 10.0),
    )
    assert observed["empirical_tau_minutes"] is not None
    assert 1.0 < observed["empirical_tau_minutes"] < 10.0


def test_decoder_autocorrelation_does_not_treat_static_offset_as_memory() -> None:
    observed = lagged_decoder_autocorrelation(
        np.arange(100, dtype=float) * 30.0,
        np.zeros(100, dtype=np.int64),
        np.full((100, 1), 50.0),
        lag_minutes=(0.5, 1.0, 2.0),
    )
    assert all(row["correlation"] is None for row in observed["lags"])


def test_temporal_shuffle_detects_smooth_decoder_trace() -> None:
    time = np.arange(300, dtype=float) * 30.0
    value = np.column_stack([
        np.sin(np.arange(300) / 25.0), np.cos(np.arange(300) / 31.0),
    ])
    result = shuffled_temporal_structure_null(
        time, np.zeros(300, dtype=np.int64), value,
        n_permutations=40, rng=np.random.default_rng(4),
    )
    assert result["temporally_smoother_than_shuffled"] is True
    assert result["lower_tail_monte_carlo_p"] <= 0.05


def test_reset_phase_r2_flags_reset_dominated_signal() -> None:
    local_time = np.arange(100, dtype=float) * 30.0
    time = np.concatenate([local_time, local_time + 100_000.0])
    elapsed = np.tile(local_time / 60.0, 2)
    value = np.log1p(elapsed)[:, None]
    r2 = reset_phase_explained_variance(
        time, np.repeat([0, 1], 100).astype(np.int64), value,
    )
    assert r2 is not None and r2 > 0.99


def test_reset_phase_is_not_identifiable_from_one_segment() -> None:
    time = np.arange(200, dtype=float) * 30.0
    value = np.log1p(time / 60.0)[:, None]
    assert reset_phase_explained_variance(
        time, np.zeros(200, dtype=np.int64), value,
    ) is None


def test_decoder_autocorrelation_removes_between_segment_offsets() -> None:
    local = np.tile(np.asarray([0.0, 1.0, 0.0, -1.0]), 20)
    time = np.tile(np.arange(len(local), dtype=float) * 30.0, 2)
    session = np.repeat([0, 1], len(local)).astype(np.int64)
    values = np.concatenate([local, local + 10_000.0])[:, None]
    observed = lagged_decoder_autocorrelation(
        time, session, values, lag_minutes=(0.5,),
    )
    assert observed["lags"][0]["correlation"] < 0.1
