import numpy as np

from src.topic5_axis_readback_v2_2 import (
    empirical_upper_percentile,
    frozen_random_axes_by_subject,
    line_axis_consensus,
    sign_invariant_cosine,
    sign_invariant_projection_spearman,
)


def test_line_consensus_is_sign_invariant():
    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-0.99, -0.1, 0.0],
            [0.98, -0.1, 0.0],
        ]
    )
    first = line_axis_consensus(axes)
    second = line_axis_consensus(axes * np.asarray([[-1.0], [1.0], [-1.0]]))
    assert np.isclose(abs(first @ second), 1.0)
    assert sign_invariant_cosine(first, [1.0, 0.0, 0.0]) > 0.99


def test_projection_readback_is_sign_invariant():
    coords = np.asarray(
        [
            [-2.0, 0.0, 0.0],
            [-1.0, 0.2, 0.0],
            [0.0, -0.1, 0.0],
            [1.0, 0.1, 0.0],
            [2.0, -0.2, 0.0],
        ]
    )
    value = sign_invariant_projection_spearman(
        coords, [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]
    )
    assert np.isclose(value, 1.0)


def test_random_axis_stream_matches_subject_order_and_is_unit_norm():
    first = frozen_random_axes_by_subject(["a", "b"])
    second = frozen_random_axes_by_subject(["a", "b"])
    reversed_order = frozen_random_axes_by_subject(["b", "a"])
    assert np.array_equal(first["a"], second["a"])
    assert np.allclose(np.linalg.norm(first["a"], axis=1), 1.0)
    assert np.array_equal(first["a"], reversed_order["b"])
    assert not np.array_equal(first["b"], reversed_order["b"])


def test_empirical_percentile_has_finite_sample_correction():
    null = np.asarray([0.1, 0.2, 0.3])
    assert empirical_upper_percentile(0.2, null) == 0.75
