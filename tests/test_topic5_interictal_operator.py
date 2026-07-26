import numpy as np
import pytest

from src.topic5_interictal_operator import (
    CONTACT_FEATURE_NAMES,
    ContactQueryGRU,
    StaticContactQuery,
    build_contact_features,
    contact_query_loss,
    encode_recruitment_matrix,
    fit_empirical_template_baseline,
    fit_first_order_markov,
    masked_local_ranks,
    pairwise_rank_concordance,
    prefix_targets,
    recruitment_groups,
)


def test_masked_local_rank_removes_phantom_values():
    ranks = np.array([[0.0, 999.0], [1.0, 0.0], [88.0, 1.0], [2.0, 77.0]])
    participation = np.array(
        [[True, False], [True, True], [False, True], [True, False]]
    )
    local = masked_local_ranks(ranks, participation)
    assert np.isnan(local[2, 0])
    assert np.isnan(local[0, 1])
    assert np.isnan(local[3, 1])
    np.testing.assert_allclose(local[[0, 1, 3], 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(local[[1, 2], 1], [0.0, 1.0])


def test_exact_lag_ties_form_a_set_without_ordering_nonparticipants():
    rank = np.array([0.0, 0.5, 1.0, np.nan])
    participation = np.array([True, True, True, False])
    lag = np.array([0.01, 0.01, 0.03, 123.0])
    groups, n_groups = recruitment_groups(
        rank, participation, lag_raw=lag, tie_tolerance_seconds=0.0
    )
    np.testing.assert_array_equal(groups, [0, 0, 1, -1])
    assert n_groups == 2


def test_positive_tie_tolerance_is_explicit_and_not_default():
    rank = np.array([0.0, 0.5, 1.0])
    participation = np.ones(3, bool)
    lag = np.array([0.01, 0.011, 0.020])
    exact, _ = recruitment_groups(rank, participation, lag_raw=lag)
    tolerant, _ = recruitment_groups(
        rank, participation, lag_raw=lag, tie_tolerance_seconds=0.002
    )
    np.testing.assert_array_equal(exact, [0, 1, 2])
    np.testing.assert_array_equal(tolerant, [0, 0, 1])


def test_event_matrix_contract_is_event_by_contact_and_masked():
    ranks = np.array([[0.0, 3.0], [1.0, 0.0], [2.0, 1.0]])
    participation = np.array([[True, False], [True, True], [True, True]])
    lag = np.array([[0.0, 8.0], [0.1, 0.0], [0.2, 0.1]])
    local, groups, count = encode_recruitment_matrix(ranks, participation, lag)
    assert local.shape == groups.shape == (2, 3)
    assert np.isnan(local[1, 0])
    assert groups[1, 0] == -1
    np.testing.assert_array_equal(count, [3, 2])


def test_prefix_targets_do_not_turn_nonparticipants_into_late_contacts():
    groups = np.array([0, 1, -1, 2])
    mid = prefix_targets(groups, tau=1)
    np.testing.assert_array_equal(mid["recruited"], [True, False, False, False])
    np.testing.assert_array_equal(mid["next_set"], [False, True, False, False])
    np.testing.assert_array_equal(mid["remaining"], [False, True, False, True])
    np.testing.assert_array_equal(mid["suffix_group"], [-1, 0, -1, 1])
    assert mid["terminal"] is False

    terminal = prefix_targets(groups, tau=3)
    assert terminal["terminal"] is True
    assert not terminal["next_set"].any()
    assert not terminal["remaining"].any()


def test_contact_features_do_not_expose_channel_or_shaft_strings():
    support = np.array([0.9, 0.6, 0.4, 0.2])
    coords = np.array([[0, 0, 0], [0, 0, 2], [4, 0, 0], [4, 0, 2]], float)
    first, meta = build_contact_features(["A1", "A2", "B1", "B2"], support, coords)
    renamed, _ = build_contact_features(["X1", "X2", "Y1", "Y2"], support, coords)
    np.testing.assert_allclose(first, renamed)
    assert first.shape == (4, len(CONTACT_FEATURE_NAMES))
    assert meta["string_identifiers_exposed_to_model"] is False


def test_markov_baseline_learns_first_order_direction():
    groups = np.array(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [2, 1, 0],
        ]
    )
    model = fit_first_order_markov(groups, laplace_alpha=0.01)
    score = model.scores(last_set=np.array([True, False, False]), recruited=np.array([True, False, False]))
    assert score[1] > score[2]
    assert score[0] <= 1e-8


def test_empirical_template_baseline_recovers_opposing_static_orders():
    forward = np.tile(np.array([0, 1, 2, 3]), (40, 1))
    reverse = np.tile(np.array([3, 2, 1, 0]), (40, 1))
    groups = np.row_stack([forward, reverse])
    model = fit_empirical_template_baseline(groups)
    score_forward, utility_forward = model.scores(forward[0], tau=1)
    score_reverse, utility_reverse = model.scores(reverse[0], tau=1)
    assert np.argmax(score_forward) == 1
    assert np.argmax(score_reverse) == 2
    assert pairwise_rank_concordance(
        utility_forward, prefix_targets(forward[0], 1)["suffix_group"]
    ) == pytest.approx(1.0)
    assert pairwise_rank_concordance(
        utility_reverse, prefix_targets(reverse[0], 1)["suffix_group"]
    ) == pytest.approx(1.0)


def test_pairwise_suffix_rank_concordance_handles_ties():
    utility = np.array([3.0, 3.0, 1.0, -2.0])
    suffix = np.array([0, 0, 1, -1])
    assert pairwise_rank_concordance(utility, suffix) == pytest.approx(1.0)


def test_contact_query_gru_shapes_masks_and_gradients():
    torch = pytest.importorskip("torch")
    model = ContactQueryGRU(8, hidden_size=16, contact_embedding_dim=12)
    features = torch.randn(2, 5, 8)
    contact_mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    prefix_sets = torch.zeros(2, 3, 5)
    prefix_sets[0, 0, 0] = 1
    prefix_sets[0, 1, 1] = 1
    prefix_sets[1, 0, [0, 1]] = 1
    step_mask = torch.tensor([[True, True, False], [True, False, False]])
    recruited = prefix_sets.bool().any(1)
    out = model(features, contact_mask, prefix_sets, step_mask, recruited)
    assert out["next_logits"].shape == (2, 5)
    assert out["stop_logit"].shape == (2,)
    assert out["remaining_participation_logits"].shape == (2, 5)
    assert out["suffix_utility"].shape == (2, 5)
    assert out["next_logits"][0, 0] < -1e8
    assert out["next_logits"][0, 4] < -1e8

    batch = {
        "contact_mask": contact_mask,
        "recruited": recruited,
        "next_set": torch.tensor(
            [[False, False, True, False, False], [False, False, False, False, False]]
        ),
        "terminal": torch.tensor([False, True]),
        "remaining": torch.tensor(
            [[False, False, True, True, False], [False, False, False, False, False]]
        ),
        "suffix_group": torch.tensor(
            [[-1, -1, 0, 1, -1], [-1, -1, -1, -1, -1]]
        ),
    }
    loss = contact_query_loss(out, batch)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("use_last_set", [False, True])
def test_static_contact_query_controls_have_no_recurrent_interface(use_last_set):
    torch = pytest.importorskip("torch")
    model = StaticContactQuery(
        8,
        hidden_size=16,
        contact_embedding_dim=12,
        use_last_set=use_last_set,
    )
    features = torch.randn(2, 5, 8)
    contact_mask = torch.ones(2, 5, dtype=torch.bool)
    prefix_sets = torch.zeros(2, 3, 5)
    prefix_sets[0, 0, 0] = 1
    prefix_sets[0, 1, 1] = 1
    prefix_sets[1, 0, [0, 1]] = 1
    step_mask = torch.tensor([[True, True, False], [True, False, False]])
    recruited = prefix_sets.bool().any(1)
    out = model(features, contact_mask, prefix_sets, step_mask, recruited)
    assert out["next_logits"].shape == (2, 5)
    assert out["suffix_utility"].shape == (2, 5)
    assert not hasattr(model, "gru")
