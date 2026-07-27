import numpy as np

from src.topic4_zm_effective_rank import (
    assemble_paired_sensitivity,
    bootstrap_rank,
    rank_summary,
    standardize_sensitivity,
)


def test_paired_central_difference_uses_matching_noise_and_delta():
    rows = []
    for j, name in enumerate(("z", "m", "sg")):
        yp = np.array([1.0, 2.0]) + np.eye(3, 2)[j] * 0.1
        ym = np.array([1.0, 2.0]) - np.eye(3, 2)[j] * 0.1
        rows.extend([
            dict(coordinate=name, sign=+1, delta=0.1, bank_sha=f"b{j}", y=yp),
            dict(coordinate=name, sign=-1, delta=0.1, bank_sha=f"b{j}", y=ym),
        ])
    S = assemble_paired_sensitivity(rows, ("z", "m", "sg"))
    assert S.shape == (2, 3)
    assert np.allclose(S[:, :2], np.eye(2))
    assert np.allclose(S[:, 2], 0.0)


def test_paired_difference_rejects_unmatched_future_noise():
    rows = [
        dict(coordinate="z", sign=+1, delta=0.1, bank_sha="a", y=[1.0]),
        dict(coordinate="z", sign=-1, delta=0.1, bank_sha="b", y=[0.0]),
    ]
    with np.testing.assert_raises(ValueError):
        assemble_paired_sensitivity(rows, ("z",))


def test_standardized_rank_is_invariant_to_input_and_output_units():
    S = np.array([[2.0, 0.0], [0.0, 0.5]])
    q_scale = np.array([3.0, 4.0])
    y_scale = np.array([5.0, 6.0])
    base = standardize_sensitivity(S, q_scale, y_scale)

    # q' = aq and y' = by changes the raw derivative by b/a and the
    # trajectory scales by a and b, which must cancel exactly.
    a = np.array([1000.0, 0.01])
    b = np.array([0.1, 20.0])
    S_units = (b[:, None] / a[None, :]) * S
    got = standardize_sensitivity(S_units, a * q_scale, b * y_scale)
    assert np.allclose(got, base)
    assert np.allclose(rank_summary(got)["singular_values"],
                       rank_summary(base)["singular_values"])


def test_rank_summary_distinguishes_rank1_from_rank2():
    r1 = rank_summary(np.array([[2.0, 4.0, 6.0], [1.0, 2.0, 3.0]]))
    r2 = rank_summary(np.eye(2))
    assert r1["near_rank1_descriptive"]
    assert r1["s2_over_s1"] < 1e-10
    assert not r2["near_rank1_descriptive"]
    assert r2["effective_rank_participation"] == 2.0


def test_bootstrap_requires_rank1_uncertainty_interval_not_point_only():
    rng = np.random.default_rng(2)
    rank1_samples = np.stack([
        np.array([[1.0, 2.0], [0.5, 1.0]]) + 0.003 * rng.standard_normal((2, 2))
        for _ in range(30)
    ])
    rank2_samples = np.stack([
        np.eye(2) + 0.03 * rng.standard_normal((2, 2))
        for _ in range(30)
    ])
    a = bootstrap_rank(rank1_samples, n_boot=300, seed=3)
    b = bootstrap_rank(rank2_samples, n_boot=300, seed=3)
    assert a["rank1_supported"]
    assert a["s2_over_s1_ci"][1] < 0.2
    assert not b["rank1_supported"]
