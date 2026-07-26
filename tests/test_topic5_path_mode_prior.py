import numpy as np

from scripts.build_topic5_path_mode_prior import (
    _cosine_rows,
    _mode_fit_metrics,
    event_transition_vectors,
    factor_path_modes,
)
from scripts.build_topic5_transition_skeleton_prior import (
    _folded_transition_skeleton,
)


def test_forward_and_reverse_events_fold_to_the_same_canonical_edge():
    axis = np.array([-1.0, 1.0], np.float32)
    groups = np.array([[0, 1], [1, 0]], np.int16)
    vectors, rows = event_transition_vectors(groups, axis)
    assert rows.tolist() == [0, 1]
    np.testing.assert_allclose(vectors[0], vectors[1])
    np.testing.assert_allclose(vectors.sum(1), 1.0)


def test_k1_is_the_normalized_aggregate_transition_skeleton():
    axis = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    groups = np.array(
        [
            [0, 1, -1, 2],
            [2, 1, -1, 0],
            [0, -1, 1, 2],
            [2, -1, 1, 0],
        ],
        np.int16,
    )
    vectors, _ = event_transition_vectors(groups, axis)
    aggregate = _folded_transition_skeleton(groups, axis)
    bases, prior, metadata = factor_path_modes(
        vectors,
        mode_count=1,
        aggregate_skeleton=aggregate,
        seed=7,
        max_iter=200,
    )
    expected = aggregate.reshape(1, -1) / aggregate.sum()
    np.testing.assert_allclose(bases, expected)
    np.testing.assert_allclose(prior, [1.0])
    assert metadata["nmf_iterations"] == 0


def test_k2_recovers_two_nonnegative_path_families_without_labels():
    axis = np.linspace(-1.0, 1.0, 4, dtype=np.float32)
    path_left = np.array([0, 1, -1, 2], np.int16)
    path_right = np.array([0, -1, 1, 2], np.int16)
    groups = np.row_stack(
        [
            np.tile(path_left, (80, 1)),
            np.tile(path_right, (80, 1)),
        ]
    )
    vectors, _ = event_transition_vectors(groups, axis)
    aggregate = _folded_transition_skeleton(groups, axis)
    bases, prior, _ = factor_path_modes(
        vectors,
        mode_count=2,
        aggregate_skeleton=aggregate,
        seed=13,
        max_iter=1000,
    )
    assert bases.shape == (2, 16)
    assert np.all(bases >= 0)
    np.testing.assert_allclose(bases.sum(1), 1.0)
    np.testing.assert_allclose(prior.sum(), 1.0)
    assert np.all(prior > 0.1)
    similarity = _cosine_rows(bases, bases)
    assert similarity[0, 1] < 0.5
    metrics = _mode_fit_metrics(vectors, bases)
    assert 0.0 <= metrics["max_mode_cosine_median"] <= 1.0
    assert 0.0 <= metrics["soft_reconstruction_cosine_median"] <= 1.0
