from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_instrument import (
    effective_rank,
    lagged_decoder_autocorrelation,
    standardise_decoder,
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
