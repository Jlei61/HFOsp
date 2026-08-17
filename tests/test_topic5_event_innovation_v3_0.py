import numpy as np

from src.topic5_event_innovation_v3_0 import (
    fit_masked_contact_ridge,
    fit_local_projection,
    fit_rank_state_basis,
    innovation_alignment,
    masked_rank_reconstruction_error,
    masked_window_rank_field,
    observable_impulse,
    pairwise_precedence_innovation,
    precedence_probability,
    rank_innovation,
    rolling_past_rank_fields,
    split_window_precedence_reliability,
    split_window_rank_reliability,
    uniform_cumulative_innovation,
)


def test_masked_window_rank_field_ignores_nonparticipating_phantom_values():
    ranks = np.array([[0.0, 99.0], [1.0, 0.5], [99.0, 0.0]])
    participation = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
    mean, support = masked_window_rank_field(
        ranks, participation, np.array([0, 1, 2])
    )
    np.testing.assert_allclose(mean, [0.5, 0.25])
    np.testing.assert_array_equal(support, [2, 2])


def test_rank_basis_and_precedence_are_observable():
    rng = np.random.default_rng(3)
    loading = np.array([[1.0, 0.0], [0.5, 0.5], [-0.5, 0.5], [-1.0, 0.0]])
    state = rng.normal(size=(200, 2))
    fields = np.array([0.1, 0.4, 0.6, 0.9]) + state @ loading.T
    basis = fit_rank_state_basis(fields, 2)
    reconstruction = basis.inverse(basis.transform(fields))
    assert np.mean((fields - reconstruction) ** 2) < 1e-12
    probability = precedence_probability(np.array([0.1, 0.9]))
    assert probability[0, 1] > 0.5
    assert probability[1, 0] < 0.5


def test_rank_basis_accepts_unit_balancing_weights():
    # Ten repeated rows from one unit and one row from another should have the
    # midpoint backbone when unit totals receive equal mass.
    fields = np.vstack([np.zeros((10, 2)), np.ones((1, 2))])
    weights = np.array([0.1] * 10 + [1.0])
    basis = fit_rank_state_basis(fields, 1, sample_weight=weights)
    np.testing.assert_allclose(basis.backbone, [0.5, 0.5])


def test_family_specific_innovations_respect_mask_and_ties():
    observed = np.array([0.0, 0.0, 1.0, 99.0])
    predicted = np.array([0.2, 0.3, 0.7, 0.8])
    participation = np.array([1, 1, 1, 0], dtype=bool)
    groups = np.array([0, 0, 1, -1])
    rank = rank_innovation(observed, participation, predicted)
    np.testing.assert_array_equal(rank.valid, [1, 1, 1, 0])
    assert rank.residual[-1] == 0.0
    pairwise = pairwise_precedence_innovation(
        observed, groups, participation, predicted
    )
    # Contact 0/1 are tied, so only their pairs with contact 2 remain.
    assert len(pairwise.residual) == 2


def test_local_projection_recovers_event_increment_and_observable_map():
    rng = np.random.default_rng(4)
    pre = rng.normal(size=(1000, 2))
    innovation = rng.normal(size=(1000, 2))
    autonomous = np.array([[0.8, 0.1], [0.0, 0.7]])
    impulse = np.array([[0.5, 0.0], [0.0, -0.4]])
    future = pre @ autonomous.T + innovation @ impulse.T
    fitted = fit_local_projection(pre, future, innovation, alpha=1e-8)
    np.testing.assert_allclose(fitted.autonomous, autonomous, atol=1e-6)
    np.testing.assert_allclose(fitted.impulse, impulse, atol=1e-6)
    fields = rng.normal(size=(200, 4))
    basis = fit_rank_state_basis(fields, 2)
    mapped = observable_impulse(basis, fitted.impulse)
    assert mapped.shape == (4, 2)


def test_local_projection_supports_one_dimensional_state():
    rng = np.random.default_rng(41)
    pre = rng.normal(size=(300, 1))
    innovation = rng.normal(size=(300, 1))
    future = 0.7 * pre + 0.2 * innovation
    fitted = fit_local_projection(pre, future, innovation, alpha=1e-8)
    assert fitted.autonomous.shape == (1, 1)
    assert fitted.impulse.shape == (1, 1)


def test_cumulative_innovation_distinguishes_alignment_and_cancellation():
    aligned = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    cancelling = np.array([[1.0, 0.0], [-1.0, 0.0]] * 2 + [[1.0, 0.0]])
    assert innovation_alignment(aligned) == 1.0
    assert innovation_alignment(cancelling) < 0.3
    cumulative = uniform_cumulative_innovation(aligned, 3)
    assert np.isnan(cumulative[:2]).all()
    np.testing.assert_allclose(cumulative[2:], [[3.0, 0.0]] * 3)


def test_rolling_past_fields_are_future_blind_and_reset_at_sequence_boundary():
    ranks = np.arange(12, dtype=float).reshape(6, 2)
    participation = np.ones_like(ranks, dtype=bool)
    fields, support = rolling_past_rank_fields(
        ranks,
        participation,
        [np.array([0, 1, 2]), np.array([3, 4, 5])],
        start_offset=0,
        stop_offset=2,
    )
    assert np.isnan(fields[[0, 1, 3, 4]]).all()
    np.testing.assert_allclose(fields[2], ranks[[0, 1]].mean(axis=0))
    np.testing.assert_allclose(fields[5], ranks[[3, 4]].mean(axis=0))
    np.testing.assert_array_equal(support[2], [2, 2])


def test_masked_reconstruction_error_ignores_unsupported_entries():
    observed = np.array([[0.0, np.nan], [1.0, 1.0]])
    reconstructed = np.array([[0.5, 99.0], [1.0, 0.0]])
    support = np.array([[2, 0], [1, 1]])
    # Weighted squared errors: 2*0.25 + 1*0 + 1*1, divided by 4.
    assert masked_rank_reconstruction_error(observed, reconstructed, support) == 0.375


def test_split_window_reliability_separates_contact_scaffold_from_dynamics():
    # Large stable contact offsets inflate raw reliability, while alternating
    # window-specific residuals are deliberately inconsistent between halves.
    backbone = np.array([0.0, 10.0, 20.0])
    ranks = []
    for window in range(20):
        for half in range(2):
            jitter = (1 if (window + half) % 2 else -1) * np.array([1.0, -1.0, 0.5])
            ranks.append(backbone + jitter)
    ranks = np.asarray(ranks)
    participation = np.ones_like(ranks, dtype=bool)
    windows = [np.array([2 * index, 2 * index + 1]) for index in range(20)]
    reliability = split_window_rank_reliability(
        ranks, participation, windows, contact_backbone=backbone
    )
    assert reliability.raw > 0.9
    assert reliability.contact_residualized < -0.9


def test_masked_contact_ridge_uses_only_participating_targets():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(500, 2))
    target = np.column_stack(
        [1.0 + 2.0 * features[:, 0], -1.0 + 3.0 * features[:, 1]]
    )
    participation = np.ones_like(target, dtype=bool)
    participation[:250, 1] = False
    target[:250, 1] = 999.0
    fitted = fit_masked_contact_ridge(
        features, target, participation, alpha=1e-8, minimum_observations=20
    )
    prediction = fitted.predict(features)
    assert np.mean((prediction[:, 0] - (1 + 2 * features[:, 0])) ** 2) < 1e-12
    assert np.mean((prediction[:, 1] - (-1 + 3 * features[:, 1])) ** 2) < 1e-12


def test_pairwise_reliability_conditions_on_coparticipation_and_keeps_ties():
    ranks = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )
    participation = np.ones_like(ranks, dtype=bool)
    result = split_window_precedence_reliability(
        ranks,
        participation,
        [np.arange(4)],
        contact_backbone=np.array([0.0, 0.5, 1.5]),
    )
    assert result.n_windows == 1
    assert result.n_paired_entries == 3
    # Odd/even halves contain the same tied and ordered pair outcomes.
    assert np.isclose(result.raw, 1.0)
