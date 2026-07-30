import numpy as np
import torch

from src.topic5_rnn_internal_state import (
    deterministic_event_sample,
    fit_pca,
    linear_cka,
    pca_summary,
    prefix_intervention_outputs,
    prefix_observables,
    project_reconstruct,
    readout_relevant_local_memory,
    split_train80,
    subspace_overlap,
)
from src.topic5_rank_distribution import (
    FullHistorySequenceGRU,
    VanillaRateSequenceRNN,
)


def test_chronological_split_and_sample_are_deterministic():
    train, validation = split_train80(np.arange(80))
    assert np.array_equal(train, np.arange(60))
    assert np.array_equal(validation, np.arange(60, 80))
    sampled = deterministic_event_sample(np.arange(100), 7)
    assert len(sampled) == 7
    assert sampled[0] == 0 and sampled[-1] == 99
    assert np.all(np.diff(sampled) > 0)


def test_prefix_observables_do_not_use_future_length_as_input():
    groups = np.array([[0, 2, -1, 1], [1, 0, 2, -1]])
    counts = np.array([3, 3])
    result = prefix_observables(
        groups,
        counts,
        event_index=np.array([0, 1]),
        step=np.array([1, 2]),
    )
    assert np.array_equal(result["recruited"][0], [1, 0, 0, 0])
    assert np.array_equal(result["last_set"][1], [1, 0, 0, 0])
    assert result["next_action"].tolist() == [3, 2]
    assert np.array_equal(result["future_participation"][0], [0, 1, 0, 1])


def test_pca_reconstruction_and_alignment_metrics():
    rng = np.random.default_rng(11)
    latent = rng.standard_normal((500, 2))
    loading = rng.standard_normal((2, 8))
    values = latent @ loading + 0.01 * rng.standard_normal((500, 8))
    pca = fit_pca(values)
    summary = pca_summary(pca)
    assert summary["k95"] == 2
    reconstructed = project_reconstruct(values, pca, 2)
    assert np.mean((values - reconstructed) ** 2) < 0.001
    assert linear_cka(values, values @ rng.standard_normal((8, 8))) > 0.1
    assert np.isclose(
        subspace_overlap(pca.components[:2], pca.components[:2]), 1.0
    )


def test_ordered_prefix_intervention_matches_native_forward():
    torch.manual_seed(23)
    model = FullHistorySequenceGRU(
        8,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    features = torch.randn(6, 8)
    groups = torch.tensor(
        [
            [0, 1, 2, -1, -1, -1],
            [1, 0, -1, 2, -1, -1],
        ]
    )
    counts = torch.tensor([3, 3])
    offset = torch.zeros(6, 4)
    native = model(
        features.unsqueeze(0).expand(2, -1, -1),
        torch.ones(2, 6, dtype=torch.bool),
        groups,
        counts,
        offset,
    )
    replay = prefix_intervention_outputs(
        model,
        features,
        offset,
        groups,
        counts,
        intervention="ordered",
    )
    torch.testing.assert_close(
        native["contact_logits"], replay["contact_logits"]
    )
    torch.testing.assert_close(native["stop_logits"], replay["stop_logits"])
    assert torch.equal(native["candidate_mask"], replay["candidate_mask"])


def test_drop_earliest_changes_state_but_not_candidate_mask():
    torch.manual_seed(29)
    model = VanillaRateSequenceRNN(
        8,
        hidden_size=12,
        contact_embedding_dim=10,
        contact_encoder_hidden=9,
        local_offset_dim=4,
    )
    features = torch.randn(6, 8)
    groups = torch.tensor([[0, 1, 2, -1, -1, -1]])
    counts = torch.tensor([3])
    offset = torch.zeros(6, 4)
    ordered = prefix_intervention_outputs(
        model,
        features,
        offset,
        groups,
        counts,
        intervention="ordered",
    )
    dropped = prefix_intervention_outputs(
        model,
        features,
        offset,
        groups,
        counts,
        intervention="drop_earliest",
    )
    assert torch.equal(
        ordered["candidate_mask"], dropped["candidate_mask"]
    )
    assert not torch.allclose(
        ordered["contact_logits"][:, 2:],
        dropped["contact_logits"][:, 2:],
    )


def test_reverse_prefix_progress_follows_intervened_replay_order():
    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def _encode(self, features, offset):
            del offset
            return features, features

        def _initial_hidden(self, embedding, mask):
            return torch.zeros((embedding.shape[0], 1), device=embedding.device)

        def _advance(self, embedding, current, recruited, hidden, mask):
            del embedding, current, hidden
            return (
                recruited.sum(1, keepdim=True).to(torch.float32)
                / mask.sum(1, keepdim=True)
            )

        def _decode(self, embedding, encoder_input, hidden, candidate):
            del encoder_input
            logits = hidden.expand(-1, embedding.shape[1]).masked_fill(
                ~candidate, -1e9
            )
            return logits, hidden[:, 0]

    model = Recorder()
    features = torch.zeros((4, 1))
    offset = torch.zeros((4, 0))
    # Unequal tie-set sizes distinguish replay-order progress from the token's
    # original-rank progress.
    groups = torch.tensor([[0, 1, 1, 2]])
    counts = torch.tensor([3])
    reversed_output = prefix_intervention_outputs(
        model,
        features,
        offset,
        groups,
        counts,
        intervention="reverse_prefix",
    )
    # Before STOP, reversed replay has still accumulated all four contacts.
    torch.testing.assert_close(
        reversed_output["stop_logits"][0, 3], torch.tensor(1.0)
    )


def test_linear_local_memory_matches_exact_persistence():
    torch.manual_seed(31)
    from src.topic5_rank_distribution import LinearStateSequenceRNN

    model = LinearStateSequenceRNN(
        3,
        hidden_size=4,
        contact_embedding_dim=4,
        contact_encoder_hidden=5,
        local_offset_dim=2,
    )
    features = torch.randn(5, 3)
    offset = torch.zeros(5, 2)
    groups = np.array([[0, 1, 2, -1, -1], [1, 0, -1, 2, -1]])
    counts = np.array([3, 3])
    result = readout_relevant_local_memory(
        model,
        features,
        offset,
        groups,
        counts,
        np.array([0, 1]),
        max_events=2,
    )
    expected = float(model.persistence[0])
    assert np.isclose(result["readout_retention_median"], expected)
    assert np.isclose(result["readout_alignment_median"], 1.0)
    assert np.isclose(result["local_spectral_radius_median"], expected)
