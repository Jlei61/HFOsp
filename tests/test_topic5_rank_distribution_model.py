import numpy as np
import pytest

pytest.importorskip("torch")

from src.topic5_rank_distribution import (
    FullHistorySequenceGRU,
    LinearStateSequenceRNN,
    LowRankLeakySequenceRNN,
    StaticSequenceContactQuery,
    VanillaRateSequenceRNN,
    WindowedHistorySequenceGRU,
    contact_rank_distribution,
    distribution_errors,
    next_set_stop_loss,
)


def test_sequence_loss_is_event_balanced_and_tie_set_invariant():
    torch = pytest.importorskip("torch")
    groups = torch.tensor([[0, 0, 1, -1], [0, 1, -1, -1]])
    counts = torch.tensor([2, 2])
    contact_logits = torch.zeros(2, 3, 4, requires_grad=True)
    stop_logits = torch.zeros(2, 3, requires_grad=True)
    outputs = {
        "contact_logits": contact_logits,
        "stop_logits": stop_logits,
        "candidate_mask": torch.ones(2, 3, 4, dtype=torch.bool),
    }
    loss = next_set_stop_loss(outputs, groups, counts)
    assert loss["event_nll"].shape == (2,)
    assert torch.isfinite(loss["total"])
    loss["total"].backward()
    assert contact_logits.grad is not None
    # Swapping the two exactly tied contacts leaves the set target unchanged.
    swapped = groups.clone()
    swapped[0, 0], swapped[0, 1] = groups[0, 1], groups[0, 0]
    other = next_set_stop_loss(outputs, swapped, counts)
    assert torch.allclose(loss["total"], other["total"])


@pytest.mark.parametrize(
    "kind", ["gru", "static", "unordered", "last_set"]
)
def test_models_emit_all_prefix_actions_and_mask_recruited_contacts(kind):
    torch = pytest.importorskip("torch")
    kwargs = {
        "hidden_size": 12,
        "contact_embedding_dim": 10,
        "contact_encoder_hidden": 9,
        "local_offset_dim": 4,
    }
    if kind == "gru":
        model = FullHistorySequenceGRU(8, **kwargs)
    else:
        mode = {
            "static": "static",
            "unordered": "unordered",
            "last_set": "last_set",
        }[kind]
        model = StaticSequenceContactQuery(8, mode=mode, **kwargs)
    features = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    groups = torch.tensor([[0, 1, 2, -1, -1], [1, 0, -1, 2, -1]])
    counts = torch.tensor([3, 3])
    offset = torch.zeros(5, 4)
    output = model(features, mask, groups, counts, offset)
    assert output["contact_logits"].shape == (2, 4, 5)
    assert output["stop_logits"].shape == (2, 4)
    assert torch.isfinite(output["stop_logits"]).all()
    assert torch.isfinite(output["contact_logits"][output["contact_logits"] > -1e8]).all()
    assert output["contact_logits"][0, 1, 0] < -1e8
    loss = next_set_stop_loss(output, groups, counts)
    loss["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_free_rollout_never_repeats_a_contact_and_can_stop():
    torch = pytest.importorskip("torch")
    model = FullHistorySequenceGRU(
        8,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    groups, counts = model.rollout(
        torch.zeros(1, 6, 8),
        torch.ones(1, 6, dtype=torch.bool),
        torch.zeros(6, 4),
        n_events=128,
        seed=17,
        batch_size=64,
    )
    assert groups.shape == (128, 6)
    assert np.all(counts <= 6)
    for event, count in zip(groups, counts):
        used = event[event >= 0]
        assert len(np.unique(used)) == int(count)


def test_windowed_history_matches_full_when_window_covers_event():
    torch = pytest.importorskip("torch")
    kwargs = {
        "hidden_size": 12,
        "contact_embedding_dim": 10,
        "contact_encoder_hidden": 9,
        "local_offset_dim": 4,
    }
    full = FullHistorySequenceGRU(8, **kwargs)
    windowed = WindowedHistorySequenceGRU(
        8, history_window=8, **kwargs
    )
    windowed.load_state_dict(full.state_dict())
    features = torch.randn(3, 6, 8)
    mask = torch.ones(3, 6, dtype=torch.bool)
    groups = torch.tensor(
        [
            [0, 1, 2, 3, -1, -1],
            [1, 0, -1, 2, 3, -1],
            [0, 0, 1, -1, -1, -1],
        ]
    )
    counts = torch.tensor([4, 4, 2])
    offset = torch.zeros(6, 4)
    full_output = full(features, mask, groups, counts, offset)
    windowed_output = windowed(features, mask, groups, counts, offset)
    torch.testing.assert_close(
        full_output["contact_logits"], windowed_output["contact_logits"]
    )
    torch.testing.assert_close(
        full_output["stop_logits"], windowed_output["stop_logits"]
    )
    assert torch.equal(
        full_output["candidate_mask"], windowed_output["candidate_mask"]
    )


def test_history_one_forgets_earlier_order_but_keeps_full_prefix_mask():
    torch = pytest.importorskip("torch")
    model = WindowedHistorySequenceGRU(
        8,
        history_window=1,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    features = torch.randn(1, 6, 8).expand(2, -1, -1).clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    # At step 3 both events have the same recruited set and the same last set,
    # but contacts 0 and 1 appeared in opposite earlier orders.
    groups = torch.tensor(
        [
            [0, 1, 2, -1, -1, -1],
            [1, 0, 2, -1, -1, -1],
        ]
    )
    counts = torch.tensor([3, 3])
    output = model(features, mask, groups, counts, torch.zeros(6, 4))
    torch.testing.assert_close(
        output["contact_logits"][0, 3],
        output["contact_logits"][1, 3],
    )
    torch.testing.assert_close(
        output["stop_logits"][0, 3],
        output["stop_logits"][1, 3],
    )
    assert torch.equal(
        output["candidate_mask"][0, 3],
        torch.tensor([False, False, False, True, True, True]),
    )
    loss = next_set_stop_loss(output, groups, counts)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])


def test_windowed_history_logits_do_not_read_future_event_length_or_identity():
    torch = pytest.importorskip("torch")
    model = WindowedHistorySequenceGRU(
        8,
        history_window=2,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    features = torch.randn(1, 6, 8).expand(2, -1, -1).clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    # Both rows share ranks 0 and 1. Their future identities and final lengths
    # differ, so equality at step 2 is an explicit no-future-length check.
    groups = torch.tensor(
        [
            [0, 1, 2, 3, -1, -1],
            [0, 1, -1, -1, 2, -1],
        ]
    )
    counts = torch.tensor([4, 3])
    output = model(features, mask, groups, counts, torch.zeros(6, 4))
    torch.testing.assert_close(
        output["contact_logits"][0, 2],
        output["contact_logits"][1, 2],
    )
    torch.testing.assert_close(
        output["stop_logits"][0, 2],
        output["stop_logits"][1, 2],
    )


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_low_rank_leaky_rnn_has_explicit_rank_and_trajectory_outputs(rank):
    torch = pytest.importorskip("torch")
    model = LowRankLeakySequenceRNN(
        8,
        recurrent_rank=rank,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    features = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    groups = torch.tensor([[0, 1, 2, -1, -1], [1, 0, -1, 2, -1]])
    counts = torch.tensor([3, 3])
    offset = torch.zeros(5, 4)
    output = model(features, mask, groups, counts, offset)
    loss = next_set_stop_loss(output, groups, counts)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert 0.05 < float(model.alpha.detach()) < 0.95
    assert torch.all(model.decay.detach() > 0)
    trajectory = model.hidden_trajectory(
        features, mask, groups, counts, offset
    )
    assert trajectory["hidden_states"].shape == (2, 4, 12)
    assert trajectory["mode_coordinates"].shape == (2, 4, rank)
    loading = model.contact_mode_loadings(features[0], offset)
    assert loading["u_output_loading"].shape == (5, rank)


@pytest.mark.parametrize(
    "model_class", [LinearStateSequenceRNN, VanillaRateSequenceRNN]
)
def test_architecture_controls_reset_per_event_and_have_finite_gradients(
    model_class,
):
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    model = model_class(
        8,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    one = torch.randn(1, 6, 8)
    features = one.expand(2, -1, -1).clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    groups = torch.tensor(
        [
            [0, 1, 2, -1, -1, -1],
            [0, 1, 2, -1, -1, -1],
        ]
    )
    counts = torch.tensor([3, 3])
    output = model(features, mask, groups, counts, torch.zeros(6, 4))
    # Identical events in a batch must remain identical. This catches hidden
    # state accidentally carried across event rows.
    torch.testing.assert_close(
        output["contact_logits"][0], output["contact_logits"][1]
    )
    torch.testing.assert_close(
        output["stop_logits"][0], output["stop_logits"][1]
    )
    loss = next_set_stop_loss(output, groups, counts)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert all(
        parameter.grad is None or torch.all(torch.isfinite(parameter.grad))
        for parameter in model.parameters()
    )


def test_linear_state_persistence_is_stable():
    torch = pytest.importorskip("torch")
    model = LinearStateSequenceRNN(
        8,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    assert torch.all(model.persistence > 0)
    assert torch.all(model.persistence < 0.995)


def test_distribution_keeps_nonparticipation_separate_from_rank():
    observed = np.array([[0, 1, -1], [0, -1, 1], [-1, 0, 1]])
    counts = np.array([2, 2, 2])
    summary = contact_rank_distribution(observed, counts, bins=5)
    np.testing.assert_allclose(
        summary["participation_probability"], [2 / 3, 2 / 3, 2 / 3]
    )
    np.testing.assert_allclose(summary["rank_histogram"].sum(1), 1.0)
    error = distribution_errors(observed, counts, observed, counts, bins=5)
    assert error["participation_mae"] == pytest.approx(0.0)
    assert error["rank_wasserstein"] == pytest.approx(0.0)
    assert error["precedence_mae"] == pytest.approx(0.0)


def test_rank_wasserstein_uses_normalized_bin_width_once():
    predicted = np.array([[0, 1]])
    observed = np.array([[1, 0]])
    counts = np.array([2])
    error = distribution_errors(
        predicted, counts, observed, counts, bins=10
    )
    assert error["rank_wasserstein"] == pytest.approx(0.9)


def test_participant_count_is_not_confused_with_tied_rank_set_count():
    predicted = np.array([[0, 0, 1, -1]])
    observed = np.array([[0, 1, -1, -1]])
    predicted_count = np.array([2])
    observed_count = np.array([2])
    error = distribution_errors(
        predicted,
        predicted_count,
        observed,
        observed_count,
        bins=5,
    )
    assert error["participant_count_mean_error"] == pytest.approx(1.0)
    assert error["event_length_mean_error"] == pytest.approx(0.0)
