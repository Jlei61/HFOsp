import numpy as np

from src.topic5_stable_repertoire_event_rnn import fit_stable_templates
from src.topic5_stable_repertoire_event_history_v2_4 import (
    build_event_history_dataset,
    chronological_sequences,
    descriptor_from_token_groups,
    family_scales_from_train,
    feature_matrix,
    fit_low_dimensional_state,
    fit_matched_recency_baselines,
    safe_circular_target_pairing,
    score_v24,
    source_coherent_block_shuffle,
    split_half_reliability_v24,
    verify_event_history_contract,
    verify_target_values,
)


def _toy(seed=0, n_sources=5, per_source=260, n_contacts=6):
    rng = np.random.default_rng(seed)
    source = np.repeat(np.arange(n_sources), per_source)
    time = np.arange(len(source), dtype=float) * 2.0
    mode = ((np.arange(len(source)) // 20) % 2).astype(int)
    part = rng.random((len(source), n_contacts)) > 0.10
    rank = np.empty_like(part, dtype=float)
    for index, label in enumerate(mode):
        base = np.linspace(0, 1, n_contacts)
        rank[index] = (base if label == 0 else base[::-1]) + rng.normal(0, 0.02, n_contacts)
    rank = np.clip(rank, 0, 1)
    rank[~part] = np.nan
    eligible = np.arange(len(source))
    encoder = fit_stable_templates(rank, part, eligible[: per_source * 3], seed=0)
    tokens, modes = encoder.event_tokens(rank, part)
    return rank, part, source, time, eligible, encoder, tokens, modes


def _dataset(seed=0, horizon=20):
    rank, part, source, time, eligible, encoder, tokens, modes = _toy(seed)
    sequences = chronological_sequences(source, time, eligible)
    dataset = build_event_history_dataset(
        tokens,
        modes,
        rank,
        part,
        time,
        encoder,
        sequences,
        history_length=80,
        horizon=horizon,
    )
    return dataset, rank, part, source, time, eligible, encoder, tokens, modes, sequences


def test_block_shuffle_is_one_coherent_pseudosequence_per_source():
    dataset, rank, part, source, time, eligible, encoder, tokens, modes, sequences = _dataset()
    shuffled_sequences, metadata = source_coherent_block_shuffle(
        sequences, block_size=20, seed=4
    )
    shuffled = build_event_history_dataset(
        tokens,
        modes,
        rank,
        part,
        time,
        encoder,
        shuffled_sequences,
        history_length=80,
        horizon=20,
        surrogate_kind="source_block_shuffle",
    )
    checks = verify_event_history_contract(
        shuffled,
        raw_source_ids=source,
        raw_event_time=time,
        eligible_indices=eligible,
        horizon=20,
        require_future=True,
    )
    assert all(checks.values())
    assert all(order != list(range(len(order))) for order in metadata.values())
    assert not np.array_equal(shuffled.histories, dataset.histories)


def test_safe_circular_pairing_moves_values_and_all_provenance_together():
    dataset, rank, part, source, time, eligible, encoder, _, modes, _ = _dataset()
    shifted, shifts = safe_circular_target_pairing(
        dataset, shift_fraction=0.5, horizon=20
    )
    checks = verify_event_history_contract(
        shifted,
        raw_source_ids=source,
        raw_event_time=time,
        eligible_indices=eligible,
        horizon=20,
        require_future=False,
    )
    assert shifts
    assert all(checks.values())
    assert verify_target_values(shifted, modes, rank, part, encoder)
    assert np.all(shifted.origin_rows != shifted.donor_rows)
    assert all(
        np.intersect1d(history, target).size == 0
        for history, target in zip(
            shifted.history_event_indices, shifted.target_event_indices
        )
    )


def test_stale_circular_metadata_is_detected():
    dataset, rank, part, _, _, _, encoder, _, modes, _ = _dataset()
    stale = dataset.take(np.arange(len(dataset)))
    donor = np.roll(np.arange(len(stale)), 2)
    object.__setattr__(stale, "targets", stale.targets[donor])
    assert not verify_target_values(stale, modes, rank, part, encoder)


def test_equal_count_and_recency_features_have_expected_shapes():
    dataset, _, _, _, _, _, encoder, _, _, _ = _dataset()
    n_modes = encoder.n_modes
    n_contacts = len(encoder.rank_prior)
    first = feature_matrix(dataset, "first_h", encoder.rank_prior, n_modes)
    recent = feature_matrix(dataset, "recent_h", encoder.rank_prior, n_modes)
    full = feature_matrix(
        dataset, "full_token_ewma", encoder.rank_prior, n_modes, decay=0.8
    )
    descriptor = feature_matrix(
        dataset, "descriptor_ewma", encoder.rank_prior, n_modes, decay=0.8
    )
    binned = feature_matrix(dataset, "binned_lag", encoder.rank_prior, n_modes)
    assert first.shape == recent.shape == descriptor.shape
    assert first.shape[1] == n_modes + 2 * n_contacts
    assert full.shape[1] == dataset.histories.shape[-1]
    assert binned.shape[1] == 4 * first.shape[1]
    assert not np.allclose(first, recent)


def test_time_nuisance_is_future_blind():
    dataset, _, _, _, _, _, encoder, _, _, _ = _dataset()
    before = feature_matrix(dataset, "time_nuisance", encoder.rank_prior, encoder.n_modes)
    changed = dataset.take(np.arange(len(dataset)))
    object.__setattr__(changed, "target_event_times", changed.target_event_times + 10_000.0)
    after = feature_matrix(changed, "time_nuisance", encoder.rank_prior, encoder.n_modes)
    assert np.array_equal(before, after)


def test_propagation_score_is_not_hidden_by_participation():
    target = np.zeros((10, 8))
    prediction = target.copy()
    prediction[:, -3:] = 1.0
    scales = family_scales_from_train(
        np.tile(np.linspace(0, 1, 8), (20, 1)) + np.random.default_rng(0).normal(0, 0.1, (20, 8)),
        n_modes=2,
        n_contacts=3,
    )
    score = score_v24(
        target, prediction, n_modes=2, n_contacts=3, scales=scales
    )
    assert score.propagation == 0.0
    assert score.recruitment > 0.0


def test_low_dimensional_state_is_compared_to_validation_selected_recency_model():
    dataset, _, _, _, _, _, encoder, _, _, _ = _dataset(seed=3)
    n = len(dataset)
    train = dataset.take(np.arange(0, n // 2))
    validation = dataset.take(np.arange(n // 2, 3 * n // 4))
    test = dataset.take(np.arange(3 * n // 4, n))
    n_modes = encoder.n_modes
    n_contacts = len(encoder.rank_prior)
    scales = family_scales_from_train(
        train.targets, n_modes=n_modes, n_contacts=n_contacts
    )
    candidates, base = fit_matched_recency_baselines(
        train,
        validation,
        rank_prior=encoder.rank_prior,
        decay_grid=[0.5, 0.8, 0.95],
        alpha_grid=[0.1, 1.0],
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )
    state = fit_low_dimensional_state(
        train,
        validation,
        base_model=base,
        dimension_grid=[2, 4],
        decay_grid=[0.5, 0.8, 0.95],
        alpha_grid=[0.1, 1.0],
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
        seed=0,
    )
    prediction = state.predict(test)
    assert set(candidates) == {"full_token_ewma", "descriptor_ewma", "binned_lag"}
    assert state.base_model.feature_name in candidates
    assert prediction.shape == test.targets.shape
    assert np.all(np.isfinite(prediction))


def test_residualized_reliability_is_reported_separately():
    dataset, rank, part, _, _, _, encoder, _, modes, _ = _dataset(seed=7)
    reliability = split_half_reliability_v24(
        dataset,
        modes,
        rank,
        part,
        encoder,
        train_target_mean=np.mean(dataset.targets, axis=0),
        repeats=4,
        seed=1,
    )
    for family in ("occupancy", "rank", "participation"):
        assert set(reliability[family]) == {"raw", "train_mean_residualized"}
        for values in reliability[family].values():
            assert "split_half_spearman_median" in values
            assert "variance_reliability_median" in values


def test_single_event_descriptor_matches_group_contract():
    dataset, _, _, _, _, _, encoder, _, _, _ = _dataset()
    one = descriptor_from_token_groups(
        dataset.histories[:, :1], encoder.rank_prior, encoder.n_modes
    )
    assert one.shape == dataset.targets.shape
    assert np.allclose(one[:, : encoder.n_modes].sum(axis=1), 1.0)
