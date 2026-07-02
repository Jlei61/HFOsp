import numpy as np
import pytest

from src.topic5_v2_criticality import (
    activations_from_z,
    avalanche_atm,
    atm_direction_index,
    atm_forward_displacement,
    atm_rank_coupling_spearman,
    block_shuffle_surrogate,
    branching_ratio,
    contact_susceptibility,
    cv_one_step_r2,
    leading_eigvec,
    load_phase2_config,
    phase_randomize_surrogate,
    prepare_var_window,
    recovery_tau,
    spectral_radius,
    var1_ridge,
    var_window_ok,
)


def test_load_phase2_config_defaults_to_exploratory_state_band():
    cfg = load_phase2_config()
    assert cfg["tier"] == "exploratory"
    assert cfg["state_band"] == "legacy_bb_1_45"
    assert cfg["susceptibility"]["features"] == [
        "variance",
        "lag1_autocorr",
        "line_length_rate",
    ]
    assert {"phase_randomize", "block_shuffle"} <= set(cfg["dynamics"]["surrogates"])


def test_susceptibility_line_length_rate_is_length_normalized():
    n = 400
    rng = np.random.default_rng(0)
    early = rng.standard_normal((2, n)) * 0.2
    late = np.cumsum(rng.standard_normal((2, n)) * 1.0, axis=1)
    out = contact_susceptibility(
        np.concatenate([early, late], axis=1),
        (0, n),
        (n, 2 * n),
    )
    assert np.all(out["variance"] > 0)
    assert np.all(out["lag1_autocorr"] > 0)
    assert "line_length_rate" in out
    assert "line_length_sum" in out
    assert np.all(np.isfinite(out["line_length_rate"]))
    assert np.all(out["line_length_sum"] > out["line_length_rate"])


def test_susceptibility_nan_when_either_window_has_too_few_finite_samples():
    env = np.array(
        [
            [1.0, np.nan, np.nan, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 1.0, np.nan, np.nan],
        ]
    )
    out = contact_susceptibility(env, (0, 3), (3, 6))
    for values in out.values():
        assert np.isnan(values).all()


def test_activations_from_z_uses_strict_threshold():
    z = np.array([[1.0, 2.0, 2.1], [3.0, 1.9, 2.0]])
    active = activations_from_z(z, 2.0)
    assert active.dtype == bool
    assert active.tolist() == [[False, False, True], [True, False, False]]


def test_branching_ratio_skips_zero_source_bins():
    active = np.array(
        [
            [False, True, True, False],
            [False, True, False, True],
            [False, False, True, False],
        ],
        dtype=bool,
    )
    assert branching_ratio(active) == pytest.approx(0.75)
    assert np.isnan(branching_ratio(np.zeros((3, 4), dtype=bool)))


def test_avalanche_atm_row_normalizes_and_keeps_empty_rows_zero():
    active = np.array(
        [
            [True, False, False],
            [False, True, False],
            [False, True, True],
            [False, False, False],
        ],
        dtype=bool,
    )
    atm = avalanche_atm(active)
    expected = np.array(
        [
            [0.0, 0.5, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert atm == pytest.approx(expected)
    assert atm.sum(axis=1).tolist() == pytest.approx([1.0, 1.0, 1.0, 0.0])


def test_forward_displacement_not_fooled_by_self_persistence():
    rank = np.array([0.0, 1.0, 2.0, 3.0])

    persist = np.eye(4)
    assert abs(atm_forward_displacement(persist, rank)) < 1e-9
    assert atm_rank_coupling_spearman(persist, rank) > 0.9

    z = np.zeros((4, 9))
    for t, c in enumerate([0, 1, 2, 3, None, 0, 1, 2, 3]):
        if c is not None:
            z[c, t] = 3.0
    atm = avalanche_atm(activations_from_z(z, 2.0))
    assert atm_forward_displacement(atm, rank) > 0.3
    assert atm_direction_index(atm, rank) > 0.3


def test_backward_flow_has_negative_direction_metrics():
    rank = np.array([0.0, 1.0, 2.0, 3.0])
    z = np.zeros((4, 9))
    for t, c in enumerate([3, 2, 1, 0, None, 3, 2, 1, 0]):
        if c is not None:
            z[c, t] = 3.0
    atm = avalanche_atm(activations_from_z(z, 2.0))
    assert atm_forward_displacement(atm, rank) < -0.3
    assert atm_direction_index(atm, rank) < -0.3


def test_direction_metrics_ignore_nan_rank_channels():
    rank = np.array([0.0, 1.0, np.nan, 3.0])
    atm = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    assert atm_forward_displacement(atm, rank) == pytest.approx(1.0)
    assert atm_direction_index(atm, rank) == pytest.approx(1.0)
    assert np.isnan(atm_rank_coupling_spearman(atm, rank))


def test_var_preprocessing_and_fit():
    rng = np.random.default_rng(0)
    n = 2000
    A_true = np.array([[0.9, 0.0], [0.2, 0.8]])
    X = np.zeros((2, n))
    for t in range(1, n):
        X[:, t] = A_true @ X[:, t - 1] + 0.1 * rng.standard_normal(2)

    Xp = prepare_var_window(X + 5.0)
    assert abs(Xp.mean()) < 1e-6
    assert var_window_ok(2, n)

    A = var1_ridge(Xp, 1e-3)
    lambda_max = spectral_radius(A)
    loading = leading_eigvec(A)
    assert cv_one_step_r2(Xp, 1e-3, 5) > 0.2
    assert recovery_tau(lambda_max, 0.1) > 0
    assert loading.shape == (2,)
    assert np.all(loading >= 0)
    assert np.linalg.norm(loading) == pytest.approx(1.0)


def test_var_window_ok_requires_time_over_channels_and_margin():
    assert not var_window_ok(10, 49, min_t_over_ch=5)
    assert var_window_ok(10, 50, min_t_over_ch=5)
    assert not var_window_ok(2, 11, min_t_over_ch=5)
    assert var_window_ok(2, 12, min_t_over_ch=5)


def test_block_shuffle_surrogate_permutes_samples_by_common_blocks():
    X = np.arange(24, dtype=float).reshape(3, 8)
    Y = block_shuffle_surrogate(X, block_len=2, rng=np.random.default_rng(0))

    assert Y.shape == X.shape
    assert not np.array_equal(Y, X)
    for ch in range(X.shape[0]):
        assert np.array_equal(np.sort(Y[ch]), np.sort(X[ch]))

    # Common block order across channels preserves the between-channel offset.
    assert np.all(Y[1] - Y[0] == 8)
    assert np.all(Y[2] - Y[1] == 8)


def test_phase_randomize_surrogate_preserves_per_channel_moments():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((4, 257))
    X[1] = 2.0 * X[1] + 3.0
    X[2] = np.sin(np.linspace(0, 8 * np.pi, X.shape[1]))

    Y = phase_randomize_surrogate(X, rng=np.random.default_rng(2))

    assert Y.shape == X.shape
    assert np.allclose(Y.mean(axis=1), X.mean(axis=1), atol=1e-12)
    assert np.allclose(Y.var(axis=1), X.var(axis=1), rtol=1e-12, atol=1e-12)
    assert not np.allclose(Y[0], X[0])
