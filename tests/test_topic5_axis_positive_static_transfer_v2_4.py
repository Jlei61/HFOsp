import numpy as np
import pytest

from src.topic5_axis_positive_static_transfer_v2_4 import (
    candidate_alignment_summary,
    empirical_rank_distribution,
    normalized_rank_distribution,
    paired_rollout_design,
    robust_patient_standardize,
    rollout_model_distribution,
    sign_invariant_cosine,
    weighted_ridge_predict,
)
from src.topic5_competitive_propagation_v2_3 import CompetitivePropagationRNN


def test_sign_invariant_cosine_ignores_axis_sign():
    assert sign_invariant_cosine([1, 0, 0], [-1, 0, 0]) == pytest.approx(1.0)
    assert sign_invariant_cosine([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_alignment_summary_uses_candidate_distribution():
    candidates = np.eye(3)
    result = candidate_alignment_summary(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        candidates,
    )
    assert result["selected_abs_cosine"] == pytest.approx(1.0)
    assert result["alignment_margin"] == pytest.approx(1.0)
    assert 0.0 < result["candidate_empirical_p_upper"] <= 1.0


def test_rank_distribution_includes_nonparticipation_and_bins():
    groups = np.array([[0, 1, -1], [0, -1, 1]], dtype=int)
    distribution = normalized_rank_distribution(groups)
    assert distribution.shape == (3, 11)
    np.testing.assert_allclose(distribution.sum(axis=1), 1.0)
    assert distribution[0, 0] == 0.0
    assert distribution[1, 0] == pytest.approx(0.5)
    assert distribution[2, 0] == pytest.approx(0.5)


def test_empirical_distribution_respects_indices():
    groups = np.array([[0, 1], [1, 0], [0, -1]], dtype=int)
    observed = empirical_rank_distribution(groups, np.array([0, 2]))
    expected = normalized_rank_distribution(groups[[0, 2]])
    np.testing.assert_allclose(observed, expected)


def test_paired_rollout_design_is_deterministic():
    groups = np.array([[0, 1, -1], [0, -1, 1]], dtype=int)
    first = paired_rollout_design(
        groups, np.array([0, 1]), n_rollouts=20, seed=17
    )
    second = paired_rollout_design(
        groups, np.array([0, 1]), n_rollouts=20, seed=17
    )
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])


def test_vectorized_rollout_distribution_is_normalized_and_deterministic():
    groups = np.array(
        [[0, 1, 2, -1], [0, -1, 1, 2], [-1, 0, 1, 2]], dtype=int
    )
    model = CompetitivePropagationRNN(
        coords=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float
        ),
        axis=np.array([1.0, 0.0, 0.0]),
        node_logit=np.zeros(4),
        rho_propagation=0.5,
        rho_competition=0.75,
    )
    sampled, uniforms = paired_rollout_design(
        groups, np.arange(3), n_rollouts=100, seed=17
    )
    first = rollout_model_distribution(model, groups, sampled, uniforms)
    second = rollout_model_distribution(model, groups, sampled, uniforms)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(first.sum(axis=1), 1.0)


def test_robust_patient_standardize_handles_nonconstant_field():
    values = np.array([1.0, 2.0, 4.0, np.nan])
    result = robust_patient_standardize(values)
    assert np.isnan(result[-1])
    assert np.nanmedian(result) == pytest.approx(0.0)


def test_robust_patient_standardize_rejects_constant_field():
    with pytest.raises(ValueError, match="constant"):
        robust_patient_standardize(np.ones(5))


def test_weighted_ridge_uses_training_only_and_returns_test_shape():
    train_x = np.array([[0.0], [1.0], [2.0], [3.0]])
    train_y = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = weighted_ridge_predict(
        train_x,
        train_y,
        np.ones(4),
        np.array([[4.0], [5.0]]),
        alpha=1.0,
    )
    assert prediction.shape == (2,)
    assert prediction[1] > prediction[0]
