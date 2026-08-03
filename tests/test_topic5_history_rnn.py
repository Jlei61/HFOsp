import numpy as np
import pandas as pd
import pytest

from src.topic5_history_bridge import (
    causal_ewma_contact_fields,
    causal_contact_features,
    leave_one_seizure_out_residual,
    patient_balanced_contact_weights,
    weighted_ridge_fit,
    weighted_ridge_predict,
)

from src.topic5_history_rnn import (
    build_continuous_segment_ids,
    center_contact_field,
    exact_contact_join,
    prefix_matched_order_indices,
    select_causal_prefix,
)


def test_exact_contact_join_reorders_target_contacts_without_fuzzy_matching():
    joined = exact_contact_join(["A1", "B2"], ["B2", "A1", "C3"])
    np.testing.assert_array_equal(joined, [1, 0])
    with pytest.raises(ValueError):
        exact_contact_join(["A1"], ["A01"])


def test_segment_ids_reset_on_skipped_block_or_recording_change():
    metadata = {
        "r1_0000": {
            "recording_id": "r1",
            "block_no": 0,
            "block_start_epoch": 0.0,
            "block_end_epoch": 10.0,
        },
        "r1_0001": {
            "recording_id": "r1",
            "block_no": 1,
            "block_start_epoch": 11.0,
            "block_end_epoch": 20.0,
        },
        "r1_0003": {
            "recording_id": "r1",
            "block_no": 3,
            "block_start_epoch": 31.0,
            "block_end_epoch": 40.0,
        },
        "r2_0000": {
            "recording_id": "r2",
            "block_no": 0,
            "block_start_epoch": 41.0,
            "block_end_epoch": 50.0,
        },
    }
    segment, reset = build_continuous_segment_ids(
        ["r1_0000", "r1_0000", "r1_0001", "r1_0003", "r2_0000"], metadata
    )
    np.testing.assert_array_equal(segment, [0, 0, 0, 1, 2])
    np.testing.assert_array_equal(reset, [True, False, False, True, True])


def test_segment_ids_can_cross_file_boundary_only_when_time_is_contiguous():
    metadata = {
        "a": {
            "recording_id": "r1",
            "block_no": 0,
            "sequence_index": 0,
            "block_start_epoch": 0.0,
            "block_end_epoch": 10.0,
        },
        "b": {
            "recording_id": "r2",
            "block_no": 0,
            "sequence_index": 1,
            "block_start_epoch": 11.0,
            "block_end_epoch": 20.0,
        },
    }
    strict, _ = build_continuous_segment_ids(["a", "b"], metadata)
    permissive, _ = build_continuous_segment_ids(
        ["a", "b"], metadata, allow_cross_recording_contiguous=True
    )
    np.testing.assert_array_equal(strict, [0, 1])
    np.testing.assert_array_equal(permissive, [0, 0])


def test_causal_prefix_applies_recording_guard_postictal_and_final_segment():
    prefix = select_causal_prefix(
        [100.0, 200.0, 300.0, 400.0, 500.0],
        [0, 0, 1, 1, 1],
        ["r0", "r0", "r1", "r1", "r1"],
        seizure_recording_id="r1",
        clinical_onset_epoch=700.0,
        guard_seconds=100.0,
        previous_postictal_end_epoch=350.0,
    )
    np.testing.assert_array_equal(prefix.event_indices, [3, 4])
    assert prefix.segment_id == 1
    assert prefix.last_event_index == 4


def test_contact_field_centering_is_shift_invariant():
    value = np.asarray([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(
        center_contact_field(value), center_contact_field(value + 99.0)
    )


def test_prefix_matched_order_control_never_reads_target_or_future():
    start, indices = prefix_matched_order_indices(
        10, window=4, rng=np.random.default_rng(7)
    )
    assert start == 6
    assert sorted(indices.tolist()) == [6, 7, 8, 9]
    assert int(indices.max()) < 10
    assert not np.array_equal(indices, np.arange(6, 10))
    assert int(indices[-1]) == 9


def test_causal_contact_features_ignore_future_events():
    base = np.zeros((3, 8), dtype=np.float32)
    prefix = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=np.uint8)
    features, support = causal_contact_features(base, prefix)
    expected = (prefix.sum(0) + 0.5) / 3.0
    np.testing.assert_allclose(support, expected)
    np.testing.assert_allclose(features[:, 1], expected - expected.mean())


def test_causal_ewma_weights_recent_events_more_strongly():
    participation = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    rank = np.asarray([[0.0, np.nan], [np.nan, 0.0]])
    field, _ = causal_ewma_contact_fields(
        participation,
        rank,
        np.asarray([0.0, 3600.0]),
        cutoff_epoch=3600.0,
        half_life_hours=1.0,
    )
    assert field[1] > field[0]


def test_leave_one_seizure_out_residual_removes_other_seizure_mean():
    fields = np.asarray([[1.0, 3.0], [2.0, 5.0], [4.0, 7.0]])
    residual = leave_one_seizure_out_residual(fields)
    np.testing.assert_allclose(residual[0], fields[0] - fields[1:].mean(0))


def test_patient_balanced_ridge_has_equal_patient_total_weight_and_predicts():
    patient = np.asarray(["a", "a", "a", "a", "b", "b"])
    seizure = np.asarray(["a1", "a1", "a2", "a2", "b1", "b1"])
    count = np.asarray([2, 2, 2, 2, 2, 2])
    weight = patient_balanced_contact_weights(patient, seizure, count)
    np.testing.assert_allclose(weight[patient == "a"].sum(), 0.5)
    np.testing.assert_allclose(weight[patient == "b"].sum(), 0.5)
    x = np.arange(6, dtype=float)[:, None]
    y = 2.0 * x[:, 0] + 1.0
    model = weighted_ridge_fit(x, y, weight, alpha=0.0)
    np.testing.assert_allclose(weighted_ridge_predict(model, x), y, atol=1e-8)


def test_direct_r2_zero_state_ablation_preserves_baseline_features():
    from scripts.run_topic5_history_rnn_early_ictal_fold_v0_1 import (
        _features,
        _r2_zero_state_features,
    )

    frame = pd.DataFrame({
        "geometry_0": [0.1, -0.1],
        "scaffold_field_a": [0.2, -0.2],
        "static": [0.3, -0.3],
        "m1_part": [0.4, -0.4],
        "m1_rank": [0.5, -0.5],
        "history_part": [9.0, 8.0],
        "history_rank": [7.0, 6.0],
    })
    original, names = _features(frame, "R2")
    ablated = _r2_zero_state_features(frame)
    history_index = [names.index("history_part"), names.index("history_rank")]
    baseline_index = [index for index in range(len(names)) if index not in history_index]
    np.testing.assert_allclose(ablated[:, baseline_index], original[:, baseline_index])
    np.testing.assert_allclose(ablated[:, history_index], 0.0)


def test_early_ictal_field_score_is_invariant_to_arbitrary_global_shifts():
    from scripts.run_topic5_history_rnn_early_ictal_fold_v0_1 import _score

    prediction = np.asarray([1.0, 4.0, 2.0, 7.0])
    target = np.asarray([2.0, 5.0, 1.0, 8.0])
    reference = _score(prediction, target)
    shifted = _score(prediction + 100.0, target - 37.0)
    np.testing.assert_allclose(reference, shifted, rtol=0, atol=1e-12)


def test_early_ictal_reader_requires_explicit_multiseed_g1_pass(tmp_path):
    import json

    from scripts.run_topic5_history_rnn_early_ictal_fold_v0_1 import _require_g1

    summary = tmp_path / "G1_MULTI_SEED_SUMMARY.json"
    with pytest.raises(RuntimeError, match="absent"):
        _require_g1(tmp_path)
    summary.write_text(
        json.dumps(
            {
                "status": "G1_MULTI_SEED_FAIL_KEEP_ICTAL_TARGET_SEALED",
                "target_values_read": False,
            }
        )
    )
    with pytest.raises(RuntimeError, match="locked"):
        _require_g1(tmp_path)
    summary.write_text(
        json.dumps(
            {
                "status": "G1_MULTI_SEED_PASS_OPEN_G2",
                "target_values_read": False,
            }
        )
    )
    _require_g1(tmp_path)


def test_direct_transfer_contract_can_authorize_target_after_proxy_failure(tmp_path):
    import json

    from scripts.run_topic5_history_rnn_early_ictal_fold_v0_1 import (
        _authorize_target_access,
    )

    (tmp_path / "G1_MULTI_SEED_SUMMARY.json").write_text(
        json.dumps({
            "status": "G1_MULTI_SEED_FAIL_KEEP_ICTAL_TARGET_SEALED",
            "target_values_read": False,
        })
    )
    contract = tmp_path / "direct.json"
    contract.write_text(json.dumps({
        "status": "DIRECT_TRANSFER_AUTHORIZED_INDEPENDENT_OF_G1",
        "endpoint": "clinical_onset_[0,10]s_1-150Hz_contact_energy",
        "g1_role": "PARALLEL_PROXY_EVIDENCE_NOT_HARD_GATE",
    }))
    assert _authorize_target_access(tmp_path, contract).startswith("DIRECT_V0_2:")


def test_patient_contrast_treats_float_residue_as_a_tie_and_keeps_the_exact_null():
    """Ties must not be signed observations and must not trigger the normal null.

    Eleven exactly tied patients plus one 1e-17 residue and three real positives
    is an n=3 all-positive signed-rank test (exact p=0.125).  SciPy's ``auto``
    method would report 0.028 here because it falls back to the normal
    approximation as soon as it has to drop zeros itself.
    """

    from scripts.summarize_topic5_history_rnn_direct_early_ictal_transfer_v0_2 import (
        _contrast,
    )

    values = pd.Series(
        [0.0] * 11 + [5.551115123125783e-17, 0.05, 0.01, 0.2]
    )
    contrast = _contrast(values, 1)
    assert contrast["n_positive"] == 3
    assert contrast["n_negative"] == 0
    assert contrast["n_tied"] == 12
    assert contrast["n_patients"] == 15
    np.testing.assert_allclose(contrast["one_sided_wilcoxon_p"], 0.125)


torch = pytest.importorskip("torch")
from src.topic5_history_rnn import (  # noqa: E402
    MatchedUnorderedHistory,
    NextEventFieldHeads,
    TimeDecayHistoryGRU,
    encode_within_event,
    next_event_field_loss,
)
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402


def test_time_decay_is_monotonic_and_segment_reset_erases_previous_state():
    model = TimeDecayHistoryGRU(3, 4, initial_half_life_hours=1.0)
    state = torch.ones(2, 4)
    decayed = model.decay(state, torch.tensor([0.0, 3600.0]))
    torch.testing.assert_close(decayed[0], state[0])
    assert torch.all(decayed[1] < state[1])
    event = torch.zeros(2, 3)
    reset_state = model.step(
        event,
        state,
        torch.zeros(2),
        reset=torch.tensor([True, False]),
    )
    zero_reference = model.step(
        event[:1], torch.zeros(1, 4), torch.zeros(1), reset=torch.tensor([False])
    )
    torch.testing.assert_close(reset_state[:1], zero_reference)


def test_unordered_pool_is_invariant_to_history_permutation():
    torch.manual_seed(1)
    model = MatchedUnorderedHistory(5, 7)
    events = torch.randn(2, 4, 5)
    mask = torch.ones(2, 4, dtype=torch.bool)
    scalar = torch.zeros(2, 3)
    # Keep the frozen last-event covariate fixed and permute only preceding events.
    permuted = events[:, [2, 0, 1, 3]]
    torch.testing.assert_close(
        model(events, mask, scalar), model(permuted, mask, scalar)
    )


def test_event_encoder_resets_hidden_for_every_event():
    torch.manual_seed(2)
    model = LinearStateSequenceRNN(
        4,
        hidden_size=5,
        contact_embedding_dim=6,
        contact_encoder_hidden=7,
        local_offset_dim=0,
    )
    features = torch.randn(2, 3, 4)
    groups = torch.tensor([[0, 1, -1], [0, 1, -1]])
    count = torch.tensor([2, 2])
    together, _ = encode_within_event(model, features, groups, count)
    separate = []
    for index in range(2):
        state, _ = encode_within_event(
            model,
            features[index : index + 1],
            groups[index : index + 1],
            count[index : index + 1],
        )
        separate.append(state)
    torch.testing.assert_close(together, torch.cat(separate))


def test_masked_history_carries_state_but_ignores_padding():
    torch.manual_seed(4)
    model = TimeDecayHistoryGRU(3, 5)
    events = torch.randn(2, 4, 3)
    delta = torch.zeros(2, 4)
    reset = torch.zeros(2, 4, dtype=torch.bool)
    reset[:, 0] = True
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    states, final = model.forward_masked(events, delta, reset, mask)
    torch.testing.assert_close(states[0, 1], states[0, 3])
    torch.testing.assert_close(final[0], states[0, 1])


def test_bptt_chunk_detach_does_not_reset_history_state():
    torch.manual_seed(6)
    model = TimeDecayHistoryGRU(3, 5)
    events = torch.randn(2, 6, 3)
    delta = torch.rand(2, 6) * 20.0
    delta[:, 0] = 0.0
    reset = torch.zeros(2, 6, dtype=torch.bool)
    reset[:, 0] = True
    mask = torch.ones(2, 6, dtype=torch.bool)
    full, _ = model.forward_masked(events, delta, reset, mask)
    left, carried = model.forward_masked(
        events[:, :3], delta[:, :3], reset[:, :3], mask[:, :3]
    )
    right, _ = model.forward_masked(
        events[:, 3:],
        delta[:, 3:],
        reset[:, 3:],
        mask[:, 3:],
        initial_state=carried.detach(),
    )
    torch.testing.assert_close(full, torch.cat([left, right], dim=1))


def test_prefix_matched_order_control_uses_the_chronological_model_composition():
    """The control must differ from M2 only by event order, not by model terms.

    ``window=1`` replays the true last event from the true earlier state, so a
    correctly composed control has to reproduce the chronological arm decision
    by decision.  Any extra readout term (for example the unordered residual)
    breaks this identity and turns the order contrast into a model contrast.
    """

    from scripts.audit_topic5_history_rnn_gate1_order_controls_v0_1 import (
        _prefix_matched_shuffle,
    )
    from scripts.run_topic5_history_rnn_gate1_sequential_fold_v0_1 import (
        MatchedSequentialModel,
        ResidualSequentialModel,
        Segment,
        UnorderedResidualSequentialModel,
        _causal_unordered_summary,
        _evaluate_condition,
    )

    torch.manual_seed(11)
    rng = np.random.default_rng(11)
    event_dim, contact_dim, state_dim, n_events, n_contacts = 4, 6, 3, 12, 5
    embedding = rng.normal(size=(n_events, event_dim)).astype(np.float32)
    event_time = np.cumsum(rng.uniform(30.0, 600.0, size=n_events))
    event_time = event_time - event_time[0]
    event_split = np.zeros(n_events, dtype=np.uint8)
    event_split[-4:] = 1
    segment = Segment(
        subject="test_subject",
        dataset="epilepsiae",
        original_index=np.arange(n_events),
        embedding=embedding,
        unordered_summary=_causal_unordered_summary(embedding, event_time),
        participation=(rng.random((n_events, n_contacts)) < 0.5).astype(np.uint8),
        relative_rank=rng.normal(size=(n_events, n_contacts)).astype(np.float32),
        event_time=event_time,
        event_split=event_split,
        contact_embedding=rng.normal(size=(n_contacts, contact_dim)).astype(np.float32),
        static_logit=rng.normal(size=n_contacts).astype(np.float32),
        train_decision_weight=1.0,
    )
    device = torch.device("cpu")
    matched = MatchedSequentialModel(event_dim, state_dim, contact_dim).eval()
    unordered = UnorderedResidualSequentialModel(event_dim, state_dim, contact_dim).eval()
    chronological = ResidualSequentialModel(
        event_dim, state_dim, contact_dim, initial_half_life_hours=2.0
    ).eval()

    reference = _evaluate_condition(
        matched,
        unordered,
        chronological,
        [segment],
        condition="chronological_history",
        batch_segments=1,
        chunk_length=8,
        rank_weight=0.2,
        seed=0,
        device=device,
    )
    control = _prefix_matched_shuffle(
        matched,
        chronological,
        [segment],
        window=1,
        batch_size=8,
        rank_weight=0.2,
        seed=0,
        device=device,
    )
    merged = reference.merge(
        control, on="event_index", suffixes=("_reference", "_control")
    )
    assert len(merged) == int(np.sum(event_split == 1))
    np.testing.assert_allclose(
        merged.participation_bce_control.to_numpy(),
        merged.participation_bce_reference.to_numpy(),
        atol=1e-6,
    )


def test_direct_order_control_reassigns_the_entire_causal_prefix():
    """The order control must permute the whole prefix, not a recent window.

    A recent-window permutation leaves the long tail of the history in true
    chronological order.  With a two-hour half-life and prefixes of hundreds to
    thousands of events that control would be nearly identical to the true arm,
    so a null order effect would be uninformative rather than evidence.
    """

    from scripts.run_topic5_history_rnn_early_ictal_fold_v0_1 import (
        _history_final,
        _history_final_order_control,
    )
    from scripts.run_topic5_history_rnn_gate1_sequential_fold_v0_1 import (
        ResidualSequentialModel,
    )
    from src.topic5_history_rnn import prefix_matched_order_indices

    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    n_events, event_dim = 700, 4
    embedding = rng.normal(size=(n_events, event_dim)).astype(np.float32)
    event_time = np.cumsum(rng.uniform(1.0, 120.0, size=n_events))
    device = torch.device("cpu")
    model = ResidualSequentialModel(
        event_dim, 5, 6, initial_half_life_hours=2.0
    ).eval()

    control = _history_final_order_control(
        model, embedding, event_time, seed=17, device=device
    )
    start, order = prefix_matched_order_indices(
        n_events, window=n_events, rng=np.random.default_rng(17)
    )
    assert start == 0
    assert int(order[-1]) == n_events - 1
    assert sorted(order.tolist()) == list(range(n_events))
    # The whole prefix moves; a 64-event window could displace at most 64.
    assert int(np.sum(order != np.arange(n_events))) > 64
    expected = _history_final(
        model, embedding[order], event_time, device=device
    )
    torch.testing.assert_close(control, expected)


def test_sequence_contact_heads_and_padding_aware_loss():
    torch.manual_seed(5)
    heads = NextEventFieldHeads(4, 6)
    prediction = heads(torch.randn(2, 3, 4), torch.randn(2, 5, 6))
    assert prediction["participation_logits"].shape == (2, 3, 5)
    participation = torch.zeros(2, 3, 5)
    rank = torch.full((2, 3, 5), float("nan"))
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )
    loss = next_event_field_loss(
        prediction, participation, rank, contact_mask=mask
    )
    assert torch.isfinite(loss["total"])


def test_all_contact_channel_null_is_patient_folded_and_deterministic():
    from scripts.summarize_topic5_history_rnn_direct_early_ictal_transfer_v0_2 import (
        MODELS,
        _channel_null_rows,
    )

    rows = []
    target = np.arange(6, dtype=float)
    for model in MODELS:
        for seizure_index in range(2):
            for contact_index, value in enumerate(target):
                rows.append({
                    "seizure_id": f"s{seizure_index}",
                    "contact_index": contact_index,
                    "model": model,
                    "prediction": value,
                    "target_z": value,
                })
    frame = pd.DataFrame(rows)
    first = pd.DataFrame(
        _channel_null_rows(frame, subject="epilepsiae_test", n_perm=300)
    ).sort_values("model").reset_index(drop=True)
    second = pd.DataFrame(
        _channel_null_rows(frame, subject="epilepsiae_test", n_perm=300)
    ).sort_values("model").reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(MODELS)
    assert np.allclose(first.observed_patient_median_rho, 1.0)
    assert np.all(first.patient_permutation_p < 0.05)
    assert np.all(first.n_seizures == 2)


def test_data_aligned_dual_field_design_preserves_ab_and_history_controls():
    from scripts.run_topic5_history_rnn_data_aligned_fold_v0_3 import _design

    frame = pd.DataFrame({
        "scaffold_field_a": [1.0, -1.0],
        "scaffold_field_b": [2.0, -2.0],
        "scaffold_earliness_a": [3.0, -3.0],
        "scaffold_earliness_b": [4.0, -4.0],
        "scaffold_axis_magnitude": [0.5, -0.5],
        "scaffold_support_mean": [0.2, -0.2],
        "scaffold_support_difference": [0.4, -0.4],
        "static": [0.1, -0.1],
        "m1_part": [0.3, -0.3], "m1_rank": [0.6, -0.6],
        "history_part": [0.7, -0.7], "history_rank": [0.8, -0.8],
        "history_shuffle_part": [0.9, -0.9],
        "history_shuffle_rank": [1.0, -1.0],
    })
    true_a, true_b, names = _design(frame, "STATIC_RNN")
    shuffled_a, shuffled_b, _ = _design(
        frame, "STATIC_RNN", control="order_shuffle"
    )
    zero_a, zero_b, _ = _design(frame, "STATIC_RNN", control="zero_state")
    assert names[:2] == ["branch_field", "branch_earliness"]
    np.testing.assert_allclose(true_a[:, 0], frame.scaffold_field_a)
    np.testing.assert_allclose(true_b[:, 0], frame.scaffold_field_b)
    np.testing.assert_allclose(true_a[:, 4], -true_b[:, 4])
    np.testing.assert_allclose(shuffled_a[:, -2], frame.history_shuffle_part)
    np.testing.assert_allclose(shuffled_b[:, -1], frame.history_shuffle_rank)
    np.testing.assert_allclose(zero_a[:, -2:], 0.0)
    np.testing.assert_allclose(zero_b[:, -2:], 0.0)


def test_data_aligned_readout_learns_static_candidate_without_signed_target_contract():
    from scripts.run_topic5_history_rnn_data_aligned_fold_v0_3 import (
        _fit_readout,
        _predict,
        _score_candidates,
    )

    rows = []
    for patient in range(4):
        for contact in range(8):
            base = float(contact - 3.5)
            rows.append({
                "subject": f"p{patient}", "seizure_id": f"s{patient}",
                "seizure_idx": patient, "contact": f"c{contact}",
                "target_z": (-1.0 if patient % 2 else 1.0) * base,
                "scaffold_field_a": base,
                "scaffold_field_b": -0.2 * base + (contact % 2),
                "scaffold_earliness_a": 0.0,
                "scaffold_earliness_b": 0.0,
                "scaffold_axis_magnitude": 0.0,
                "scaffold_support_mean": 0.0,
                "scaffold_support_difference": 0.0,
                "static": 0.0,
                "m1_part": 0.0, "m1_rank": 0.0,
                "history_part": 0.0, "history_rank": 0.0,
                "history_shuffle_part": 0.0, "history_shuffle_rank": 0.0,
            })
    frame = pd.DataFrame(rows)
    fit = _fit_readout(
        frame, "STATIC_LEARNED", seeds=(3,), steps=100,
        learning_rate=0.03, weight_decay=0.01,
    )
    pred_a, pred_b = _predict(frame, fit)
    score = _score_candidates(frame, pred_a, pred_b, "STATIC_LEARNED")
    assert np.all(score.maxab_abs_rho > 0.99)


def test_v04_static_residual_zero_gain_and_tiny_residual_are_safe():
    import torch
    from src.topic5_static_anchored_history_residual import (
        compose_static_residual,
        safe_unit_residual,
        unit_eps,
    )

    static = torch.tensor([3.0, -1.0, 2.0, 0.5])
    residual = torch.tensor([0.2, -0.1, 0.5, -0.3])
    torch.testing.assert_close(
        compose_static_residual(static, residual, 0.0),
        unit_eps(static),
        rtol=0,
        atol=0,
    )
    tiny = torch.full((4,), 1e-12)
    torch.testing.assert_close(
        safe_unit_residual(tiny, norm_threshold=1e-6),
        torch.zeros_like(tiny),
        rtol=0,
        atol=0,
    )


def test_v04_soft_maxab_is_ab_and_sign_invariant():
    import torch
    from src.topic5_static_anchored_history_residual import soft_maxab_score

    a = torch.tensor([-2.0, -1.0, 1.0, 3.0])
    b = torch.tensor([1.0, -2.0, 4.0, 0.0])
    target_rank = torch.tensor([1.0, 2.0, 3.0, 4.0])
    reference = soft_maxab_score(a, b, target_rank)
    torch.testing.assert_close(reference, soft_maxab_score(b, a, target_rank))
    torch.testing.assert_close(reference, soft_maxab_score(-a, -b, target_rank))


def test_v04_fixed_time_summary_has_frozen_2h_recency_weighting():
    import torch
    from src.topic5_static_anchored_history_residual import fixed_time_aware_summary

    embedding = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    event_time = torch.tensor([0.0, 7200.0])
    summary = fixed_time_aware_summary(
        embedding, event_time, cutoff_time=7200.0, tau_hours=2.0
    )
    weight_old = np.exp(-1.0)
    expected_ewma = torch.tensor(
        [weight_old / (weight_old + 1.0), 2.0 / (weight_old + 1.0)],
        dtype=summary.dtype,
    )
    torch.testing.assert_close(summary[:2], expected_ewma)
    assert summary.shape == (10,)  # 4 * event_dim + count + span


def test_v04_history_state_is_decayed_from_last_event_to_cutoff():
    import torch
    from src.topic5_history_rnn import TimeDecayHistoryGRU
    from src.topic5_static_anchored_history_residual import run_history_to_cutoff

    torch.manual_seed(1)
    history = TimeDecayHistoryGRU(3, 4, initial_half_life_hours=2.0)
    embedding = torch.randn(5, 3)
    event_time = torch.arange(5, dtype=torch.float32) * 60.0
    at_last = run_history_to_cutoff(
        history, embedding, event_time, cutoff_time=event_time[-1], chunk_events=2
    )
    later = run_history_to_cutoff(
        history,
        embedding,
        event_time,
        cutoff_time=event_time[-1] + 7200.0,
        chunk_events=2,
    )
    assert torch.linalg.vector_norm(later) < torch.linalg.vector_norm(at_last)


def test_v04_gain_is_near_static_but_not_gradient_saturated():
    import torch
    from src.topic5_static_anchored_history_residual import DualCandidateResidualHead

    head = DualCandidateResidualHead(4, 3, initial_gain=1e-3)
    torch.testing.assert_close(head.gains, torch.full((2,), 1e-3))
    head.gains.sum().backward()
    assert torch.all(torch.abs(head.raw_gain.grad) > 0.05)


def test_v04_endpoint_and_no_retrain_sensitivity_are_frozen_in_config():
    import json
    from pathlib import Path

    config = json.loads(
        Path("config/topic5_history_conditioned_field_refinement_v0_4.json").read_text()
    )
    assert config["primary_target"] == "clinical_onset_[0,10]s_1-45Hz_contact_energy"
    assert "1-150Hz" in config["sensitivity_target"]
    assert "no_retrain" in config["sensitivity_target"]
    assert config["target_seeds"] == [11, 29, 47]


def test_v04_seed_ensemble_averages_candidate_fields_before_scoring():
    from scripts.summarize_topic5_history_conditioned_field_v0_4 import _ensemble_true
    from scipy.stats import spearmanr

    rows = []
    target = np.arange(6, dtype=float)
    predictions = {
        11: np.array([0, 1, 2, 3, 5, 4], dtype=float),
        29: np.array([0, 1, 2, 4, 3, 5], dtype=float),
        47: np.array([0, 1, 2, 3, 4, 5], dtype=float),
    }
    for seed, prediction in predictions.items():
        for contact, value in enumerate(prediction):
            rows.append(
                {
                    "subject": "p1",
                    "seizure_id": "s1",
                    "seizure_idx": 0,
                    "contact": f"c{contact}",
                    "model": "M3_JOINT_RNN",
                    "seed": seed,
                    "draw": -1,
                    "prediction_a": value,
                    "prediction_b": -value,
                    "target_1_45": target[contact],
                    "target_1_150": target[contact],
                }
            )
    ensemble, metrics = _ensemble_true(pd.DataFrame(rows))
    expected = np.mean(np.stack(list(predictions.values())), axis=0)
    np.testing.assert_allclose(
        ensemble.sort_values("contact").prediction_a.to_numpy(), expected
    )
    expected_score = abs(spearmanr(expected, target).statistic)
    assert metrics.maxab_1_45.iloc[0] == pytest.approx(expected_score)


def test_v04_exact_signed_rank_keeps_zero_ties_out_of_null_distribution():
    from scripts.summarize_topic5_history_conditioned_field_v0_4 import (
        _exact_signed_rank,
    )

    result = _exact_signed_rank(np.array([1.0, 1.0, 0.0, 5e-12]))
    assert result["n_positive"] == 2
    assert result["n_negative"] == 0
    assert result["n_tie"] == 2
    assert result["n_nonzero"] == 2
    assert result["p_two_sided_exact"] == pytest.approx(0.5)
