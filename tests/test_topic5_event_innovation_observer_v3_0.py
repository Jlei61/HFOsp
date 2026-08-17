from __future__ import annotations

import numpy as np

from src.topic5_event_innovation_observer_v3_0 import (
    blocked_innovation_validity,
    coherent_block_permutation,
    concatenate_feature_ladder,
    fit_standardized_masked_observer,
    masked_rank_mse,
)


def test_observer_recovers_masked_linear_rank_signal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1000, 3))
    y = np.column_stack([x[:, 0] + 2 * x[:, 1], -x[:, 2]])
    participation = np.ones_like(y, dtype=bool)
    participation[:300, 1] = False
    y[:300, 1] = 999
    observer = fit_standardized_masked_observer(
        x, y, participation, alpha=1e-8, feature_name="test"
    )
    assert masked_rank_mse(observer.predict(x), y, participation) < 1e-12


def test_feature_ladder_has_only_predeclared_nested_families():
    values = {name: np.ones((10, 2)) for name in (
        "pre20", "pre40", "pre80", "lag0_20", "lag20_40", "lag40_60", "lag60_80"
    )}
    ladder = concatenate_feature_ladder(values, np.ones((10, 3)))
    assert {name: value.shape[1] for name, value in ladder.items()} == {
        "pre20": 2,
        "pre20_40_80": 6,
        "four_lag_bins": 8,
        "four_lag_bins_plus_time": 11,
    }


def test_block_permutation_is_coherent_and_does_not_cross_sequence():
    residual = np.arange(24).reshape(12, 2)
    valid = np.ones_like(residual, dtype=bool)
    groups = np.repeat([0, 1], 6)
    permuted, permuted_valid = coherent_block_permutation(
        residual, valid, groups, block_size=2, rng=np.random.default_rng(2)
    )
    assert set(map(tuple, permuted[:6])) == set(map(tuple, residual[:6]))
    assert set(map(tuple, permuted[6:])) == set(map(tuple, residual[6:]))
    assert np.array_equal(permuted_valid, valid)


def test_blocked_validity_rejects_predictable_residual_and_accepts_noise():
    rng = np.random.default_rng(4)
    features = rng.normal(size=(2000, 3))
    valid = np.ones((2000, 2), dtype=bool)
    groups = np.repeat(np.arange(4), 500)
    predictable = np.column_stack([features[:, 0], -features[:, 1]])
    noise = rng.normal(size=(2000, 2))
    bad = blocked_innovation_validity(
        features, predictable, valid, groups, block_size=20, n_null=50, seed=5
    )
    good = blocked_innovation_validity(
        features, noise, valid, groups, block_size=20, n_null=50, seed=5
    )
    assert not bad["valid"]
    assert good["valid"]


def test_blocked_validity_fails_closed_when_every_group_has_one_block():
    rng = np.random.default_rng(23)
    features = rng.normal(size=(30, 2))
    residual = rng.normal(size=(30, 2))
    result = blocked_innovation_validity(
        features,
        residual,
        np.ones_like(residual, dtype=bool),
        np.zeros(30, dtype=int),
        block_size=20,
        n_null=20,
        seed=4,
    )
    assert result["valid"] is False
    assert result["n_eligible_groups"] == 0


def test_blocked_validity_drops_short_groups_before_permutation():
    rng = np.random.default_rng(29)
    features = rng.normal(size=(120, 2))
    residual = rng.normal(size=(120, 2))
    groups = np.r_[np.zeros(100, dtype=int), np.ones(20, dtype=int)]
    result = blocked_innovation_validity(
        features,
        residual,
        np.ones_like(residual, dtype=bool),
        groups,
        block_size=20,
        n_null=20,
        seed=6,
    )
    assert result["n_eligible_groups"] == 1
    assert result["n_null_finite"] == 20
