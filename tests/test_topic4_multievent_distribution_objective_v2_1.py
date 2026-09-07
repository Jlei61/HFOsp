import numpy as np

from src.topic4_multievent_distribution_objective_v2_1 import (
    a_b_from_matched_and_off_diagonal,
    explicit_off_diagonal_mean_distance,
    off_diagonal_mean_distance,
)


def test_d_off_matches_explicit_i_not_equal_j_sum():
    rng = np.random.default_rng(9107)
    x = rng.normal(size=(17, 9))
    target = rng.normal(size=9)
    assert np.isclose(
        off_diagonal_mean_distance(x, target),
        explicit_off_diagonal_mean_distance(x, target),
        atol=1e-12,
    )


def test_d_off_retains_negative_value():
    x = np.asarray([[-1.0], [1.0]])
    assert off_diagonal_mean_distance(x, np.asarray([0.0])) == -1.0


def test_d_off_mathematical_minimum_is_two():
    assert off_diagonal_mean_distance(np.ones((1, 2)), np.zeros(2)) is None
    assert np.isfinite(off_diagonal_mean_distance(
        np.asarray([[0.0, 1.0], [1.0, 0.0]]), np.zeros(2)
    ))


def test_a_b_decomposition_is_per_run_identity():
    rng = np.random.default_rng(20260907)
    x = rng.normal(size=(23, 5))
    target = rng.normal(size=5)
    mean = x.mean(axis=0)
    a_expected = np.sum((mean - target) ** 2)
    variance = np.mean(np.sum((x - mean) ** 2, axis=1))
    b_expected = variance / (len(x) - 1)
    d_off = a_expected - b_expected
    d16 = a_expected + (len(x) - 16) / (16 * (len(x) - 1)) * variance
    a, b = a_b_from_matched_and_off_diagonal(d16, d_off, len(x), 16)
    assert np.isclose(a, a_expected)
    assert np.isclose(b, b_expected)
    assert np.isclose(d_off, a - b)
