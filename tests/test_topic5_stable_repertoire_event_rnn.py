import numpy as np

from src.topic5_stable_repertoire_event_rnn import (
    FutureWindowDataset,
    build_future_window_dataset,
    chronological_source_partition,
    circularly_shift_targets,
    fit_stable_templates,
    fit_gru_event_state,
    fit_residual_linear_event_state,
    fit_residual_gru_event_state,
    score_predictions,
    shuffled_histories,
    verify_dataset_contract,
    train_to_partition_template_stability,
)


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    n_source, per_source, n_contact = 5, 80, 5
    source = np.repeat(np.arange(n_source), per_source)
    time = np.arange(len(source), dtype=float)
    mode = np.tile(np.repeat([0, 1], per_source // 2), n_source)
    rank = np.empty((len(source), n_contact), float)
    part = rng.random(rank.shape) > 0.15
    for event, label in enumerate(mode):
        base = np.linspace(0, 1, n_contact)
        rank[event] = (base if label == 0 else base[::-1]) + rng.normal(0, 0.02, n_contact)
    rank = np.clip(rank, 0, 1)
    rank[~part] = np.nan
    return rank, part, source, time


def test_source_partition_is_disjoint_and_chronological():
    rank, part, source, time = _toy()
    split = chronological_source_partition(source, time, np.arange(len(time)))
    assert set(split.train_sources).isdisjoint(split.validation_sources)
    assert set(split.train_sources).isdisjoint(split.test_sources)
    assert max(split.train_sources) < min(split.validation_sources)
    assert max(split.validation_sources) < min(split.test_sources)


def test_source_partition_never_reintroduces_ineligible_tail_from_shared_source():
    source = np.repeat(np.arange(5), 10)
    time = np.arange(len(source), dtype=float)
    eligible = np.arange(47)  # source 4 straddles the eligibility cutoff
    split = chronological_source_partition(source, time, eligible)
    test_indices = split.indices(source, "test", eligible)
    assert np.all(test_indices < 47)
    assert not np.any(np.isin(np.arange(47, 50), test_indices))


def test_templates_and_future_windows_are_future_blind():
    rank, part, source, time = _toy()
    split = chronological_source_partition(source, time, np.arange(len(time)))
    train_idx = split.indices(source, "train")
    encoder = fit_stable_templates(rank, part, train_idx, seed=2)
    tokens, modes = encoder.event_tokens(rank, part)
    data = build_future_window_dataset(
        tokens, modes, rank, part, source, time, train_idx, encoder,
        history_length=20, horizon=10,
    )
    checks = verify_dataset_contract(data, 10)
    assert all(checks.values())
    assert np.all(data.history_stop <= data.target_start)
    assert np.all(data.target_start - data.history_start == 20)
    assert set(data.source_ids).issubset(set(split.train_sources))
    stability = train_to_partition_template_stability(
        rank, part, train_idx, split.indices(source, "validation"), encoder
    )
    assert stability["grade"] in {"strong", "moderate"}


def test_order_and_pairing_controls_preserve_values_but_break_alignment():
    rank, part, source, time = _toy()
    encoder = fit_stable_templates(rank, part, np.arange(240), seed=0)
    tokens, modes = encoder.event_tokens(rank, part)
    data = build_future_window_dataset(
        tokens, modes, rank, part, source, time, np.arange(len(time)), encoder,
        history_length=20, horizon=10,
    )
    shuffled = shuffled_histories(data, 4)
    shifted = circularly_shift_targets(data)
    assert np.allclose(np.sort(shuffled.histories, axis=1), np.sort(data.histories, axis=1))
    assert np.allclose(shuffled.targets, data.targets)
    assert np.allclose(shifted.histories, data.histories)
    assert np.allclose(np.sort(shifted.targets, axis=0), np.sort(data.targets, axis=0))


def test_family_balanced_score_does_not_weight_contacts_more_than_occupancy():
    target = np.zeros((2, 8))
    prediction = target.copy()
    prediction[:, :2] = 1.0
    score = score_predictions(target, prediction, n_modes=2, n_contacts=3)
    assert score.occupancy == 1.0
    assert score.rank == 0.0
    assert score.participation == 0.0
    assert np.isclose(score.composite, 1.0 / 3.0)


def test_gru_event_state_returns_valid_repertoire_probabilities():
    rank, part, source, time = _toy()
    split = chronological_source_partition(source, time, np.arange(len(time)))
    encoder = fit_stable_templates(rank, part, split.indices(source, "train"), seed=0)
    tokens, modes = encoder.event_tokens(rank, part)
    datasets = {}
    for name in ("train", "validation", "test"):
        datasets[name] = build_future_window_dataset(
            tokens, modes, rank, part, source, time, split.indices(source, name), encoder,
            history_length=20, horizon=10,
        )
    result = fit_gru_event_state(
        datasets["train"], datasets["validation"],
        hidden_size_grid=[4], weight_decay_grid=[1e-4], learning_rate=1e-2,
        batch_size=16, maximum_epochs=30, patience=5,
        n_modes=2, n_contacts=rank.shape[1], seed=0,
    )
    prediction = result.predict(datasets["test"].histories)
    assert prediction.shape == datasets["test"].targets.shape
    assert np.all(np.isfinite(prediction))
    assert np.allclose(np.sum(prediction[:, :2], axis=1), 1.0, atol=1e-6)
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_nested_linear_state_preserves_valid_recent_plus_history_output():
    rank, part, source, time = _toy()
    split = chronological_source_partition(source, time, np.arange(len(time)))
    encoder = fit_stable_templates(rank, part, split.indices(source, "train"), seed=0)
    tokens, modes = encoder.event_tokens(rank, part)
    datasets = {
        name: build_future_window_dataset(
            tokens, modes, rank, part, source, time, split.indices(source, name), encoder,
            history_length=20, horizon=10,
        )
        for name in ("train", "validation", "test")
    }
    model = fit_residual_linear_event_state(
        datasets["train"], datasets["validation"],
        dimension_grid=[2], decay_grid=[0.5, 0.9], alpha_grid=[0.1, 1.0],
        n_modes=2, n_contacts=rank.shape[1], seed=0,
    )
    prediction = model.predict(datasets["test"])
    assert prediction.shape == datasets["test"].targets.shape
    assert np.allclose(prediction[:, :2].sum(axis=1), 1.0)
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_nested_gru_state_is_a_valid_correction_on_unordered_history():
    rank, part, source, time = _toy()
    split = chronological_source_partition(source, time, np.arange(len(time)))
    encoder = fit_stable_templates(rank, part, split.indices(source, "train"), seed=0)
    tokens, modes = encoder.event_tokens(rank, part)
    datasets = {
        name: build_future_window_dataset(
            tokens, modes, rank, part, source, time, split.indices(source, name), encoder,
            history_length=20, horizon=10,
        )
        for name in ("train", "validation", "test")
    }
    model = fit_residual_gru_event_state(
        datasets["train"], datasets["validation"], hidden_size_grid=[4],
        weight_decay_grid=[1e-4], alpha_grid=[0.1, 1.0], learning_rate=1e-2,
        batch_size=16, maximum_epochs=30, patience=5,
        n_modes=2, n_contacts=rank.shape[1], seed=0,
    )
    prediction = model.predict(datasets["test"])
    assert prediction.shape == datasets["test"].targets.shape
    assert np.allclose(prediction[:, :2].sum(axis=1), 1.0, atol=1e-6)
    assert np.all((prediction >= 0.0) & (prediction <= 1.0))


def test_nested_linear_state_recovers_known_order_signal_beyond_fixed_composition():
    rng = np.random.default_rng(11)

    def make_dataset(n, offset):
        histories = []
        targets = []
        descriptors = []
        for _ in range(n):
            binary = np.array([0.0] * 10 + [1.0] * 10)
            rng.shuffle(binary)
            history = np.stack([binary, 1.0 - binary], axis=1)
            weights = 0.72 ** np.arange(19, -1, -1)
            p = float(np.sum(weights * binary) / np.sum(weights))
            target = np.array([p, 1 - p, p, 1 - p, p, 1 - p])
            histories.append(history)
            targets.append(target)
            descriptors.append(np.array([0.5] * 6))
        indices = np.arange(offset, offset + n)
        event_indices = np.arange(offset * 20, (offset + n) * 20).reshape(n, 20)
        target_indices = np.arange(10_000 + offset * 2, 10_000 + (offset + n) * 2).reshape(n, 2)
        return FutureWindowDataset(
            histories=np.stack(histories),
            recent_descriptors=np.stack(descriptors),
            history_descriptors=np.stack(descriptors),
            targets=np.stack(targets),
            last_mode=np.zeros(n, int),
            source_ids=indices,
            history_start=event_indices[:, 0],
            history_stop=event_indices[:, -1] + 1,
            target_start=target_indices[:, 0],
            target_stop=target_indices[:, -1] + 1,
            history_event_indices=event_indices,
            target_event_indices=target_indices,
        )

    train = make_dataset(300, 0)
    validation = make_dataset(100, 300)
    test = make_dataset(100, 400)
    ordered = fit_residual_linear_event_state(
        train, validation, dimension_grid=[1, 2], decay_grid=[0.5, 0.72, 0.9],
        alpha_grid=[0.01, 0.1, 1.0], n_modes=2, n_contacts=2, seed=0,
    )
    shuffled_train = shuffled_histories(train, 1)
    shuffled_validation = shuffled_histories(validation, 2)
    shuffled_test = shuffled_histories(test, 3)
    shuffled = fit_residual_linear_event_state(
        shuffled_train, shuffled_validation,
        dimension_grid=[1, 2], decay_grid=[0.5, 0.72, 0.9],
        alpha_grid=[0.01, 0.1, 1.0], n_modes=2, n_contacts=2, seed=0,
    )
    ordered_score = score_predictions(
        test.targets, ordered.predict(test), n_modes=2, n_contacts=2
    ).composite
    shuffled_score = score_predictions(
        shuffled_test.targets, shuffled.predict(shuffled_test), n_modes=2, n_contacts=2
    ).composite
    assert ordered_score < 0.25 * shuffled_score
