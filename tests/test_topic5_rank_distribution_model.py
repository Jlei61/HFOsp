import numpy as np
import pytest

from src.topic5_rank_distribution import (
    FullHistorySequenceGRU,
    LowRankLeakySequenceRNN,
    StaticSequenceContactQuery,
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
