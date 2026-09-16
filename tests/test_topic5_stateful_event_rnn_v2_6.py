from __future__ import annotations

import numpy as np
import pytest
import torch

from src.topic5_stable_repertoire_event_history_v2_4 import FamilyScales, score_v24
from src.topic5_stable_repertoire_event_rnn import (
    StableTemplateEncoder,
    repertoire_descriptor,
)
from src.topic5_stateful_event_rnn_v2_6 import (
    StatefulEventRNN,
    StatefulEventSequence,
    StatefulProfile,
    build_stateful_sequences,
    fit_stateful_event_rnn,
    rollout_sequences,
)


def _encoder():
    return StableTemplateEncoder(
        centers=np.asarray([[0.0], [1.0]]),
        feature_mean=np.zeros(1),
        feature_scale=np.ones(1),
        rank_prior=np.asarray([0.5]),
        n_modes=2,
    )


def test_vectorized_future_targets_match_canonical_descriptor():
    rng = np.random.default_rng(2)
    n = 70
    rank = rng.uniform(size=(n, 1))
    participation = rng.random((n, 1)) > 0.25
    modes = rng.integers(0, 2, n)
    tokens = np.concatenate(
        [
            np.where(participation, rank, 0.5),
            participation.astype(float),
            np.eye(2)[modes],
        ],
        axis=1,
    )
    encoder = _encoder()
    sequence = build_stateful_sequences(
        tokens,
        modes,
        rank,
        participation,
        encoder,
        {"source": np.arange(n)},
        horizon=20,
        warmup_events=20,
    )[0]
    for anchor in np.flatnonzero(sequence.formal_mask):
        expected = repertoire_descriptor(
            np.arange(anchor + 1, anchor + 21),
            modes,
            rank,
            participation,
            encoder,
        )
        assert np.allclose(sequence.targets[anchor], expected)
        assert np.array_equal(
            sequence.target_event_indices[anchor], np.arange(anchor + 1, anchor + 21)
        )


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
def test_chunked_rollout_equals_full_rollout_when_state_is_carried(cell):
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(43, 5)).astype(np.float32)
    targets = np.tile(np.asarray([1.0, 0.4, 0.6]), (43, 1)).astype(np.float32)
    valid = np.ones(43, bool)
    sequence = StatefulEventSequence(
        source_id="s",
        tokens=tokens,
        targets=targets,
        valid_mask=valid,
        formal_mask=valid,
        event_indices=np.arange(43),
        target_event_indices=np.tile(np.arange(2), (43, 1)),
    )
    profile = StatefulProfile(
        cell=cell,
        hidden_size=7,
        num_layers=1,
        normalization="none",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer="adam",
        learning_rate=0.001,
        weight_decay=0.0,
        gradient_clip=1.0,
        tbptt_length=5,
        update_chunks=1,
    )
    torch.manual_seed(5)
    model = StatefulEventRNN(5, 3, 1, profile, np.asarray([1.0, 0.4, 0.6]))
    chunked = rollout_sequences(
        model,
        [sequence],
        mean=np.zeros(5),
        scale=np.ones(5),
        chunk_length=5,
        carry_state=True,
        formal=True,
        return_states=True,
    )
    full = rollout_sequences(
        model,
        [sequence],
        mean=np.zeros(5),
        scale=np.ones(5),
        chunk_length=100,
        carry_state=True,
        formal=True,
        return_states=True,
    )
    assert np.allclose(chunked[0], full[0], atol=1e-7)
    assert np.allclose(chunked[3], full[3], atol=1e-7)


def _long_memory_sequences(seed: int, n_sources: int):
    rng = np.random.default_rng(seed)
    output = []
    block = 24
    delay = 17
    for source in range(n_sources):
        n_blocks = 28
        tokens = np.zeros((n_blocks * block, 3), np.float32)
        tokens[:, 0] = 0.5
        tokens[:, 1:] = 1.0
        targets = np.tile(np.asarray([1.0, 0.5, 1.0]), (len(tokens), 1)).astype(np.float32)
        valid = np.zeros(len(tokens), bool)
        for index in range(n_blocks):
            start = index * block
            bit = float(rng.integers(0, 2))
            tokens[start, 0] = bit
            anchor = start + delay
            targets[anchor, 1] = bit
            valid[anchor] = True
        output.append(
            StatefulEventSequence(
                source_id=f"s{source}",
                tokens=tokens,
                targets=targets,
                valid_mask=valid,
                formal_mask=valid,
                event_indices=np.arange(len(tokens)) + source * 10000,
                target_event_indices=np.tile(np.arange(2), (len(tokens), 1)),
            )
        )
    return output


@pytest.mark.parametrize("cell", ["rnn", "gru"])
def test_multiple_recurrent_cells_learn_long_memory_with_state_carry(cell):
    train = _long_memory_sequences(10, 5)
    validation = _long_memory_sequences(11, 2)
    test = _long_memory_sequences(12, 2)
    profile = StatefulProfile(
        cell=cell,
        hidden_size=16,
        num_layers=1,
        normalization="zscore",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer="adam",
        learning_rate=0.005,
        weight_decay=0.0,
        gradient_clip=5.0,
        tbptt_length=32,
        update_chunks=1,
    )
    fitted = fit_stateful_event_rnn(
        train,
        validation,
        profile=profile,
        scales=FamilyScales(1.0, 0.25, 1.0),
        n_modes=1,
        n_contacts=1,
        seed=13,
        maximum_epochs=60,
        minimum_epochs=10,
        patience=12,
        carry_state=True,
    )
    prediction, target, _ = fitted.predict(test, checkpoint="trained", formal=True)
    score = score_v24(
        target,
        prediction,
        n_modes=1,
        n_contacts=1,
        scales=FamilyScales(1.0, 0.25, 1.0),
    )
    assert score.rank < 0.20


def test_resetting_each_tbptt_chunk_destroys_long_memory_signal():
    train = _long_memory_sequences(20, 5)
    validation = _long_memory_sequences(21, 2)
    test = _long_memory_sequences(22, 2)
    profile = StatefulProfile(
        cell="rnn",
        hidden_size=16,
        num_layers=1,
        normalization="zscore",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer="adam",
        learning_rate=0.005,
        weight_decay=0.0,
        gradient_clip=5.0,
        tbptt_length=32,
        update_chunks=1,
    )
    carried = fit_stateful_event_rnn(
        train,
        validation,
        profile=profile,
        scales=FamilyScales(1.0, 0.25, 1.0),
        n_modes=1,
        n_contacts=1,
        seed=23,
        maximum_epochs=60,
        minimum_epochs=10,
        patience=12,
        carry_state=True,
    )
    carried_prediction, target, _ = rollout_sequences(
        carried.trained_model,
        test,
        mean=carried.feature_mean,
        scale=carried.feature_scale,
        chunk_length=8,
        carry_state=True,
        formal=True,
        return_states=False,
    )
    reset_prediction, _, _ = rollout_sequences(
        carried.trained_model,
        test,
        mean=carried.feature_mean,
        scale=carried.feature_scale,
        chunk_length=8,
        carry_state=False,
        formal=True,
        return_states=False,
    )
    carried_score = score_v24(
        target,
        carried_prediction,
        n_modes=1,
        n_contacts=1,
        scales=FamilyScales(1.0, 0.25, 1.0),
    )
    reset_score = score_v24(
        target,
        reset_prediction,
        n_modes=1,
        n_contacts=1,
        scales=FamilyScales(1.0, 0.25, 1.0),
    )
    assert carried_score.rank < 0.5 * reset_score.rank
