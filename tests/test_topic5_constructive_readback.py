import numpy as np

from src.topic5_constructive_readback import (
    axis_distribution_errors,
    evaluate_axis_readback,
    evaluate_mode_readback,
    first_order_transition,
    fit_train_axis_readback,
    fit_train_mode_readback,
    group_feature_matrix,
    mode_distribution_errors,
    source_sink_displacements,
    transition_errors,
)


def _bidirectional_groups(n_events: int = 80, n_contacts: int = 6) -> np.ndarray:
    groups = np.full((n_events, n_contacts), -1, dtype=int)
    for event in range(n_events):
        if event % 2 == 0:
            groups[event] = np.arange(n_contacts)
        else:
            groups[event] = np.arange(n_contacts)[::-1]
    return groups


def test_group_features_are_event_by_contact_and_finite():
    groups = _bidirectional_groups(10)
    features = group_feature_matrix(groups)
    assert features.shape == groups.shape
    assert np.all(np.isfinite(features))


def test_train_mode_readback_is_frozen_and_recovers_two_directions():
    groups = _bidirectional_groups()
    readback = fit_train_mode_readback(groups)
    scored = evaluate_mode_readback(readback, groups)
    errors = mode_distribution_errors(scored, scored)
    assert readback.reliable
    assert readback.minimum_cluster_fraction == 0.5
    assert scored["template_match_to_train"] > 0.99
    assert errors["template_error"] < 1e-8
    assert errors["mode_prevalence_error"] == 0


def test_transition_matrix_and_errors_follow_next_contact_order():
    groups = _bidirectional_groups(20)
    transition = first_order_transition(groups)
    assert transition.shape == (6, 6)
    assert np.allclose(transition.sum(axis=1), 1.0)
    identical = transition_errors(groups, groups)
    reversed_contacts = groups[:, ::-1]
    assert identical["transition_mae"] == 0
    assert transition_errors(groups, reversed_contacts)["transition_mae"] >= 0


def test_transition_matrix_splits_tied_rank_sets_fractionally():
    groups = np.asarray([[0, 0, 1, 1, -1]], dtype=int)
    transition = first_order_transition(groups, event_chunk_size=1)
    expected = np.zeros((5, 5))
    expected[0, 2:4] = 0.5
    expected[1, 2:4] = 0.5
    assert np.allclose(transition, expected)


def test_axis_readback_uses_unsigned_pca_and_retains_both_signs():
    groups = _bidirectional_groups(100)
    count = np.full(groups.shape[0], groups.shape[1], dtype=int)
    coords = np.column_stack(
        [np.arange(groups.shape[1]), np.zeros(groups.shape[1]), np.zeros(groups.shape[1])]
    )
    displacement = source_sink_displacements(groups, count, coords)
    assert np.sum(displacement[:, 0] > 0) == 50
    assert np.sum(displacement[:, 0] < 0) == 50
    readback = fit_train_axis_readback(groups, count, coords)
    scored = evaluate_axis_readback(readback, groups, count, coords)
    errors = axis_distribution_errors(scored, scored)
    assert readback.reliable
    assert np.isclose(readback.explained_variance_fraction, 1.0)
    assert scored["positive_count"] == 50
    assert scored["negative_count"] == 50
    assert errors["signed_axis_wasserstein"] == 0


def test_missing_geometry_is_ineligible_without_crashing():
    groups = _bidirectional_groups()
    count = np.full(groups.shape[0], groups.shape[1], dtype=int)
    coords = np.full((groups.shape[1], 3), np.nan)
    readback = fit_train_axis_readback(groups, count, coords)
    scored = evaluate_axis_readback(readback, groups, count, coords)
    assert not readback.reliable
    assert np.asarray(scored["projection"]).size == 0
