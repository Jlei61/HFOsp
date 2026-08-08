from __future__ import annotations

import numpy as np
import torch
import json
import sys
from pathlib import Path

from src.topic5_rnn_motif_v0_4 import (
    MODEL_SPECS,
    RolloutSizeHead,
    rollout_with_size_head,
    shuffle_rank_sets,
    teacher_forced_size_examples,
)
from src.topic5_wiring_economy_rnn import WEConfig, WEModel, build_event_tensors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from launch_topic5_rnn_motif_v0_4 import build_jobs  # noqa: E402
from build_topic5_rnn_motif_fields_v0_4 import (  # noqa: E402
    aggregate_records,
    derive_common_contrast,
    split_half_stability,
)
from analyse_topic5_rnn_motif_interictal_v0_4 import seed_removed_sequence_agreement  # noqa: E402
from score_topic5_rnn_motif_early_ictal_v0_4 import permutation_indices  # noqa: E402


def _static_model(n_contacts: int = 6) -> WEModel:
    model = WEModel(WEConfig(arm="STATIC_CONTACT", n_contacts=n_contacts, seed=0))
    with torch.no_grad():
        model.contact_bias.copy_(torch.arange(n_contacts, dtype=torch.float32))
        for parameter in model.stop_head.parameters():
            parameter.zero_()
        model.stop_head[-1].bias.fill_(-10.0)
    return model


def test_factorial_models_differ_only_in_growth_and_cost_components():
    square = {key: MODEL_SPECS[key] for key in (
        "M2_UNIFORM_SET", "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID"
    )}
    assert square["M2_UNIFORM_SET"].arm == "RANDOM_SET"
    assert square["M4_SPATIAL_GROWTH"].arm == "SPATIAL_SET_NOCOST"
    assert square["M6_SPATIAL_MID"].arm == "SPATIAL_SET"
    assert square["M8_UNIFORM_COST_MID"].arm == "RANDOM_SET_COST"
    assert square["M2_UNIFORM_SET"].eta == square["M4_SPATIAL_GROWTH"].eta == 0.0
    assert square["M6_SPATIAL_MID"].eta == square["M8_UNIFORM_COST_MID"].eta == 0.03
    assert all(len(spec.seeds) == 3 for spec in square.values())


def test_primary_shuffle_keeps_first_rank_and_whole_tie_sets():
    ranks = np.array([
        [0, 0, 1, 2, 2, 3],
        [2, 0, 1, 1, 3, 0],
    ], dtype=np.int16)
    shuffled = shuffle_rank_sets(ranks, seed=4, keep_first=True)
    assert np.array_equal(shuffled == 0, ranks == 0)
    for before, after in zip(ranks, shuffled):
        before_sets = sorted(sorted(np.flatnonzero(before == rank).tolist()) for rank in np.unique(before))
        after_sets = sorted(sorted(np.flatnonzero(after == rank).tolist()) for rank in np.unique(after))
        assert before_sets == after_sets
    assert not np.array_equal(shuffled, ranks)


def test_full_shuffle_can_change_first_rank_but_keeps_sets():
    ranks = np.array([[0, 0, 1, 2, 2, 3]], dtype=np.int16)
    changed = False
    for seed in range(20):
        shuffled = shuffle_rank_sets(ranks, seed=seed, keep_first=False)
        changed |= not np.array_equal(shuffled == 0, ranks == 0)
    assert changed


def test_free_rollout_uses_predicted_multi_contact_size_and_masks_repeats():
    model = _static_model(6)
    head = RolloutSizeHead(6)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.network[-1].bias[1] = 10.0  # K=2 at every continuing step
    generated = rollout_with_size_head(model, head, [np.array([0])], torch.device("cpu"))[0]
    flat = [contact for rank_set in generated for contact in rank_set]
    assert any(len(rank_set) == 2 for rank_set in generated[1:])
    assert len(flat) == len(set(flat)) == 6
    assert generated[1] == [5, 4]


def test_launcher_builds_the_locked_1426_units(tmp_path):
    fits = [{"fit_id": f"p{i}__shared", "n_contacts": 10 + i} for i in range(31)]
    (tmp_path / "INPUT_MANIFEST.json").write_text(json.dumps({"fits": fits}))
    counts = {stage: len(build_jobs(tmp_path, stage)) for stage in ("core", "dose", "gru")}
    assert counts == {"core": 744, "dose": 217, "gru": 465}
    assert sum(counts.values()) == 1426


def test_launcher_order_control_preserves_seed_and_full_control_does_not(tmp_path):
    fits = [{"fit_id": "p__shared", "n_contacts": 10}]
    (tmp_path / "INPUT_MANIFEST.json").write_text(json.dumps({"fits": fits}))
    order = [job for job in build_jobs(tmp_path, "core") if job["spec_id"] == "C_ORDER_SHUFFLED"]
    full = [job for job in build_jobs(tmp_path, "dose") if job["spec_id"] == "C_FULL_RANK_SHUFFLED"]
    assert len(order) == 3 and len(full) == 1


def test_chunked_size_features_are_identical_to_one_event_chunks():
    model = _static_model(6)
    tensors = build_event_tensors(np.array([
        [0, 1, 2, -1, -1, -1],
        [0, 0, 1, 2, 3, -1],
        [1, 2, 0, 3, 4, 5],
    ], dtype=np.int16))
    index = np.arange(3)
    one_x, one_y = teacher_forced_size_examples(
        model, tensors, index, torch.device("cpu"), batch_size=1
    )
    all_x, all_y = teacher_forced_size_examples(
        model, tensors, index, torch.device("cpu"), batch_size=16
    )
    assert torch.equal(one_y, all_y)
    assert torch.allclose(one_x, all_x, atol=0, rtol=0)


def test_seed_removed_field_uses_missing_seed_and_nonseed_denominator():
    records = [
        {"generated_rank_sets": [[0], [1], [2]], "event_abs_time": 1.0, "kept_event_index": 0},
        {"generated_rank_sets": [[1], [2], [0]], "event_abs_time": 2.0, "kept_event_index": 1},
    ]
    field = aggregate_records(records, 3)
    assert field["canonical_full"].shape == (3,)
    assert np.array_equal(field["seed_removed_denominator"], np.array([1, 1, 2]))
    assert np.allclose(field["seed_removed"], np.array([0.0, 1.0, 0.5]))


def test_common_and_contrast_are_exactly_derived_on_common_support():
    common, contrast = derive_common_contrast(np.array([1.0, 0.0]), np.array([0.0, 0.5]))
    assert np.allclose(common, np.array([0.5, 0.25]))
    assert np.allclose(contrast, np.array([1.0, -0.5]))
    with np.testing.assert_raises(ValueError):
        derive_common_contrast(np.ones(2), np.ones(3))


def test_split_half_stability_sorts_by_real_event_time():
    records = [
        {"generated_rank_sets": [[0], [1], [2]], "event_abs_time": time, "kept_event_index": index}
        for index, time in enumerate([4.0, 1.0, 3.0, 2.0])
    ]
    stability = split_half_stability(records, 3)
    assert np.isclose(stability["canonical_full"], 1.0)


def test_rollout_agreement_does_not_credit_the_supplied_seed():
    observed = np.array([0, 1, 2, 3])
    correct = [[0], [1], [2], [3]]
    reversed_postseed = [[0], [3], [2], [1]]
    assert np.isclose(seed_removed_sequence_agreement(observed, correct), 1.0)
    assert np.isclose(seed_removed_sequence_agreement(observed, reversed_postseed), -1.0)


def test_early_ictal_permutations_are_synchronized_and_shaft_preserving():
    eligible = np.arange(6)
    shafts = ["A", "A", "A", "B", "B", "B"]
    first = permutation_indices(6, eligible, shafts, 20, 7, True)
    second = permutation_indices(6, eligible, shafts, 20, 7, True)
    assert np.array_equal(first, second)
    assert all(set(row[:3]) == {0, 1, 2} and set(row[3:]) == {3, 4, 5} for row in first)
    all_contact = permutation_indices(6, eligible, shafts, 20, 7, False)
    assert any(set(row[:3]) != {0, 1, 2} for row in all_contact)
