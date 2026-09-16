from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.topic5_stable_repertoire_event_history_v2_4 import (
    EventHistoryDataset,
    FamilyScales,
    score_v24,
)
from src.topic5_trainable_event_rnn_v2_5 import (
    RecurrentProfile,
    ResidualEventRNN,
    aggregate_event_steps,
    fit_fixed_feature_baseline,
    fit_input_normalization,
    fit_trainable_residual_rnn,
    prepare_recurrent_inputs,
    window_balanced_source_partition,
)


ROOT = Path(__file__).resolve().parents[1]


def _toy_dataset(histories: np.ndarray, targets: np.ndarray) -> EventHistoryDataset:
    n, length, _ = histories.shape
    descriptor = np.tile(np.asarray([1.0, 0.5, 1.0]), (n, 1))
    history_indices = np.arange(n * length).reshape(n, length)
    target_indices = np.arange(n * length, n * length + n * 2).reshape(n, 2)
    history_positions = np.tile(np.arange(length), (n, 1))
    target_positions = np.tile(np.arange(length, length + 2), (n, 1))
    history_times = history_indices.astype(float)
    target_times = target_indices.astype(float)
    return EventHistoryDataset(
        histories=histories,
        recent_descriptors=descriptor.copy(),
        history_descriptors=descriptor.copy(),
        targets=targets,
        last_mode=np.zeros(n, int),
        source_ids=np.zeros(n, int),
        history_start=np.zeros(n, int),
        history_stop=np.full(n, length, int),
        target_start=np.full(n, length, int),
        target_stop=np.full(n, length + 2, int),
        history_event_indices=history_indices,
        target_event_indices=target_indices,
        history_positions=history_positions,
        target_positions=target_positions,
        history_event_times=history_times,
        target_event_times=target_times,
        source_lengths=np.full(n, length + 2, int),
        origin_rows=np.arange(n),
        donor_rows=np.arange(n),
        time_features=np.zeros((n, 4), float),
    )


def _order_task(seed: int = 0):
    rng = np.random.default_rng(seed)
    histories = []
    targets = []
    for _ in range(180):
        order = rng.permutation(np.r_[np.zeros(10), np.ones(10)])
        histories.append(np.stack([order, np.ones(20), np.ones(20)], axis=1))
        targets.append([1.0, order[-1], 1.0])
    histories = np.asarray(histories, float)
    targets = np.asarray(targets, float)
    return (
        _toy_dataset(histories[:120], targets[:120]),
        _toy_dataset(histories[120:150], targets[120:150]),
        _toy_dataset(histories[150:], targets[150:]),
    )


def test_event_block_aggregation_preserves_causality_and_shape():
    values = np.arange(2 * 20 * 3, dtype=float).reshape(2, 20, 3)
    blocked = aggregate_event_steps(values, 5)
    assert blocked.shape == (2, 4, 3)
    assert np.allclose(blocked[:, 0], values[:, :5].mean(axis=1))
    assert np.allclose(blocked[:, -1], values[:, -5:].mean(axis=1))
    with pytest.raises(ValueError):
        aggregate_event_steps(values, 3)


def test_normalization_is_fit_from_supplied_training_values_only():
    train = np.arange(24, dtype=float).reshape(2, 4, 3)
    center, scale = fit_input_normalization(train, "zscore")
    assert np.allclose(center, train.reshape(-1, 3).mean(axis=0))
    shifted = train + 10_000
    assert not np.allclose(center, shifted.reshape(-1, 3).mean(axis=0))
    robust_center, robust_scale = fit_input_normalization(train, "robust")
    assert np.all(np.isfinite(robust_center))
    assert np.all(robust_scale > 0)


@pytest.mark.parametrize("cell", ["rnn", "gru", "lstm"])
@pytest.mark.parametrize("optimizer", ["adam", "adamw", "rmsprop"])
def test_all_recurrent_cells_and_optimizers_train_finitely(cell, optimizer):
    train, validation, _ = _order_task(3)
    baseline = fit_fixed_feature_baseline(
        train,
        feature_name="unordered_l",
        decay=None,
        alpha=1.0,
        rank_prior=np.asarray([0.5]),
        n_modes=1,
    )
    profile = RecurrentProfile(
        cell=cell,
        hidden_size=4,
        num_layers=1,
        block_size=1,
        normalization="zscore",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer=optimizer,
        learning_rate=0.003,
        batch_size=32,
        weight_decay=0.0,
        gradient_clip=1.0,
    )
    fitted = fit_trainable_residual_rnn(
        train,
        baseline=baseline,
        profile=profile,
        scales=FamilyScales(1.0, 0.25, 1.0),
        n_modes=1,
        n_contacts=1,
        seed=11,
        maximum_epochs=3,
        patience=2,
        minimum_epochs=2,
        validation=validation,
    )
    assert fitted.trace.finite
    assert fitted.n_parameters > 0
    assert np.all(np.isfinite(fitted.predict(validation)))


def test_gru_solves_same_composition_order_task_beyond_unordered_baseline():
    train, validation, test = _order_task(19)
    scales = FamilyScales(1.0, 0.25, 1.0)
    baseline = fit_fixed_feature_baseline(
        train,
        feature_name="unordered_l",
        decay=None,
        alpha=1.0,
        rank_prior=np.asarray([0.5]),
        n_modes=1,
    )
    profile = RecurrentProfile(
        cell="gru",
        hidden_size=8,
        num_layers=1,
        block_size=1,
        normalization="zscore",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer="adam",
        learning_rate=0.01,
        batch_size=32,
        weight_decay=0.0,
        gradient_clip=1.0,
    )
    fitted = fit_trainable_residual_rnn(
        train,
        baseline=baseline,
        profile=profile,
        scales=scales,
        n_modes=1,
        n_contacts=1,
        seed=7,
        maximum_epochs=60,
        patience=10,
        minimum_epochs=10,
        validation=validation,
    )
    baseline_score = score_v24(
        test.targets, baseline.predict(test), n_modes=1, n_contacts=1, scales=scales
    )
    rnn_score = score_v24(
        test.targets, fitted.predict(test), n_modes=1, n_contacts=1, scales=scales
    )
    assert rnn_score.propagation < 0.35 * baseline_score.propagation


def test_nested_checkpoint_can_retain_exact_untrained_baseline():
    train, validation, _ = _order_task(23)
    baseline = fit_fixed_feature_baseline(
        train,
        feature_name="unordered_l",
        decay=None,
        alpha=1.0,
        rank_prior=np.asarray([0.5]),
        n_modes=1,
    )
    profile = RecurrentProfile(
        cell="gru",
        hidden_size=4,
        num_layers=1,
        block_size=1,
        normalization="zscore",
        input_layer_norm=False,
        hidden_layer_norm=False,
        optimizer="adam",
        learning_rate=0.0,
        batch_size=32,
        weight_decay=0.0,
        gradient_clip=1.0,
    )
    fitted = fit_trainable_residual_rnn(
        train,
        baseline=baseline,
        profile=profile,
        scales=FamilyScales(1.0, 0.25, 1.0),
        n_modes=1,
        n_contacts=1,
        seed=5,
        maximum_epochs=3,
        patience=2,
        minimum_epochs=2,
        validation=validation,
    )
    assert fitted.trace.best_epoch == -1
    assert fitted.trace.best_is_untrained_baseline
    assert np.allclose(fitted.predict(validation), baseline.predict(validation))


def test_recurrent_checkpoint_state_dict_reload_has_prediction_parity(tmp_path):
    train, validation, test = _order_task(31)
    baseline = fit_fixed_feature_baseline(
        train,
        feature_name="unordered_l",
        decay=None,
        alpha=1.0,
        rank_prior=np.asarray([0.5]),
        n_modes=1,
    )
    profile = RecurrentProfile(
        cell="gru",
        hidden_size=8,
        num_layers=1,
        block_size=1,
        normalization="zscore",
        input_layer_norm=True,
        hidden_layer_norm=False,
        optimizer="rmsprop",
        learning_rate=0.001,
        batch_size=32,
        weight_decay=0.0,
        gradient_clip=1.0,
    )
    fitted = fit_trainable_residual_rnn(
        train,
        baseline=baseline,
        profile=profile,
        scales=FamilyScales(1.0, 0.25, 1.0),
        n_modes=1,
        n_contacts=1,
        seed=13,
        maximum_epochs=6,
        patience=3,
        minimum_epochs=3,
        validation=validation,
    )
    checkpoint = tmp_path / "state.pt"
    torch.save(fitted.model.state_dict(), checkpoint)
    restored = ResidualEventRNN(
        input_dim=train.histories.shape[-1],
        target_dim=train.targets.shape[-1],
        profile=profile,
    )
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    values = prepare_recurrent_inputs(
        test.histories,
        block_size=profile.block_size,
        mean=fitted.feature_mean,
        scale=fitted.feature_scale,
    )
    tensor = torch.as_tensor(values, dtype=torch.float32)
    fitted.model.eval()
    restored.eval()
    with torch.no_grad():
        expected = fitted.model(tensor).cpu().numpy()
        observed = restored(tensor).cpu().numpy()
    assert np.array_equal(observed, expected)


def test_l20_window_balanced_partition_covers_all_34_subjects():
    data_root = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
    map_root = ROOT / "results/topic5_event_indexed_evolving_rank_field/development/input_audit/per_subject"
    subjects = sorted(path.stem for path in data_root.glob("*.npz"))
    assert len(subjects) == 34
    for subject in subjects:
        raw = np.load(data_root / f"{subject}.npz", allow_pickle=False)
        mapping = np.load(map_root / f"{subject}.npz", allow_pickle=False)
        eligible = np.flatnonzero(np.asarray(raw["event_split"], int) == 0)
        partition = window_balanced_source_partition(
            np.asarray(mapping["event_source_block_id"]),
            np.asarray(raw["event_abs_time"], float),
            eligible,
            history_length=20,
            horizon=20,
        )
        assert partition.formal_window_counts["train"] >= 1
        assert partition.formal_window_counts["test"] >= 1
        if partition.strategy == "window_balanced_source_disjoint":
            assert partition.formal_window_counts["validation"] >= 1
