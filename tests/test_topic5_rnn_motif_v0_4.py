from __future__ import annotations

import csv
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
    aggregate_patient_fields,
    derive_common_contrast,
    split_half_stability,
)
from analyse_topic5_rnn_motif_interictal_v0_4 import (  # noqa: E402
    event_pair_reliability,
    seed_removed_sequence_agreement,
)
from score_topic5_rnn_motif_early_ictal_v0_4 import (  # noqa: E402
    conditional_effects,
    compute_dose_trend,
    compute_factorial_effects,
    permutation_indices,
    permutation_support,
)
from score_topic5_rnn_motif_lesion_early_ictal_v0_4 import patient_fields  # noqa: E402
from summarize_topic5_rnn_motif_theory_v0_4 import (  # noqa: E402
    candidate_distance_classes,
    pairwise_seed_stability,
)
from run_topic5_rnn_motif_matched_lesions_v0_4 import (  # noqa: E402
    edge_descriptor_matches,
)


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


def test_event_pair_reliability_is_only_the_repeated_event_reference():
    ranks = np.tile(np.array([0, 1, 2, -1]), (24, 1))
    assert np.isclose(event_pair_reliability(ranks, seed=3, n_pairs=100), 1.0)


def test_early_ictal_permutations_are_synchronized_and_shaft_preserving():
    eligible = np.arange(6)
    shafts = ["A", "A", "A", "B", "B", "B"]
    first = permutation_indices(6, eligible, shafts, 20, 7, True)
    second = permutation_indices(6, eligible, shafts, 20, 7, True)
    assert np.array_equal(first, second)
    assert all(set(row[:3]) == {0, 1, 2} and set(row[3:]) == {3, 4, 5} for row in first)
    all_contact = permutation_indices(6, eligible, shafts, 20, 7, False)
    assert any(set(row[:3]) != {0, 1, 2} for row in all_contact)
    assert permutation_support(eligible, shafts) == {
        "n_eligible_contacts": 6,
        "n_shafts": 2,
        "n_within_shaft_permutable_contacts": 6,
        "n_within_shaft_permutable_groups": 2,
    }


def test_factorial_effects_use_one_complete_patient_denominator():
    lookup = {}
    for subject in ("p1", "p2"):
        for value, model in enumerate((
            "M2_UNIFORM_SET", "M4_SPATIAL_GROWTH",
            "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
        )):
            lookup[(subject, model, "rnn", "canonical_full")] = {
                "all_contact_margin": float(value)
            }
    del lookup[("p2", "M8_UNIFORM_COST_MID", "rnn", "canonical_full")]
    result = compute_factorial_effects(lookup, ["p1", "p2"], "canonical_full")
    assert result["complete_patients"] == ["p1"]
    assert result["excluded_incomplete_patients"] == ["p2"]
    assert result["growth_at_zero"]["n"] == 1
    assert set(result["holm_family"]) == {
        "growth_at_zero", "growth_at_mid", "cost_uniform", "cost_spatial", "interaction"
    }
    assert all("holm_q_factorial_family" in result[name] for name in result["holm_family"])


def test_early_ictal_dose_trend_uses_three_preassigned_positive_costs():
    lookup = {}
    models = ("M5_SPATIAL_LOW", "M6_SPATIAL_MID", "M7_SPATIAL_HIGH")
    for subject, offset in (("p1", 0.0), ("p2", 0.2)):
        for value, model in enumerate(models):
            lookup[(subject, model, "rnn", "canonical_full")] = {
                "all_contact_margin": offset + value,
            }
    result = compute_dose_trend(lookup, ["p1", "p2", "missing"], "canonical_full")
    assert result["models"] == list(models)
    assert result["complete_patients"] == ["p1", "p2"]
    assert result["excluded_incomplete_patients"] == ["missing"]
    assert result["median"] > 0


def test_conditional_early_model_refits_patient_clusters_and_permutations():
    models = (
        "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
        "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID",
    )
    patient_rows, fidelity_rows = [], []
    gamma = {model: 0.0 for model in models}
    gamma["M6_SPATIAL_MID"] = 0.30
    for patient_index in range(5):
        subject = f"p{patient_index}"
        for model_index, model in enumerate(models):
            fidelity = (0.04 * model_index + 0.015 * patient_index * model_index
                        + 0.01 * (patient_index % 2))
            margin = 0.2 * patient_index + 0.5 * fidelity + gamma[model]
            patient_rows.append({
                "subject": subject, "primary": True, "endpoint": "canonical_full",
                "cell": "rnn", "model": model, "all_contact_margin": margin,
            })
            fidelity_rows.append({
                "subject": subject, "cell": "rnn", "model": model,
                "matched_empirical_r": fidelity,
            })
    first = conditional_effects(patient_rows, fidelity_rows, draws=200, seed=12)
    second = conditional_effects(patient_rows, fidelity_rows, draws=200, seed=12)
    effect = first["contrasts"]["M6_SPATIAL_MID_vs_M0_NO_REC"]
    assert first == second
    assert first["n_complete_patients"] == 5
    assert first["design_rank"] == first["n_parameters"]
    assert np.isclose(effect["estimate"], 0.30, atol=1e-8)
    assert len(effect["patient_cluster_bootstrap_95ci"]) == 2
    assert 0.0 < effect["patient_label_permutation_p"] <= 1.0


def test_lesion_fields_keep_noncollinear_a_and_b_producers_separate():
    def record(scope, fit_id, template, values):
        payload = {
            "status": "inference_available", "field_contacts": ["A1", "B1"],
            "baseline_fields": {template: values},
            "targeted_fields": {template: [value / 2 for value in values]},
        }
        return {
            "subject": "p1", "model": "M6_SPATIAL_MID", "scope": scope,
            "fit_id": fit_id, "lesions": {"connector_nodes": payload},
        }
    resolved = patient_fields([
        record("own_a", "p1__own_a", "A", [1.0, 0.0]),
        record("own_b", "p1__own_b", "B", [0.0, 1.0]),
    ])[("p1", "M6_SPATIAL_MID", "connector_nodes")]
    assert resolved["baseline"]["producers"] == {"A": "p1__own_a", "B": "p1__own_b"}
    assert np.allclose(resolved["targeted"]["A"], [0.5, 0.0])


def test_effective_operator_seed_stability_keeps_inactive_edges(tmp_path):
    first = np.array([[0.0, 4.0, 0.0], [1.0, 0.0, 3.0], [0.0, 2.0, 0.0]])
    second = 2.0 * first
    paths = []
    for index, value in enumerate((first, second)):
        path = tmp_path / f"seed{index}.npz"
        np.savez_compressed(path, edge_effective_influence=value)
        paths.append(path)
    assert np.isclose(pairwise_seed_stability(paths), 1.0)

    # Changing which edges are inactive must lower stability; intersecting only
    # the surviving active edges would incorrectly hide this instability.
    third = np.array([[0.0, 0.0, 4.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    path = tmp_path / "seed2.npz"
    np.savez_compressed(path, edge_effective_influence=third)
    assert pairwise_seed_stability([paths[0], path]) < 1.0


def test_motif_distance_thresholds_do_not_create_long_edges_in_local_mask():
    position = np.arange(8, dtype=float)
    distance = np.abs(position[:, None] - position[None, :])
    mask = (distance == 1.0)
    np.fill_diagonal(mask, False)
    local, long, q50, q75 = candidate_distance_classes(mask, distance)
    assert q50 > 1.0 and q75 > q50
    assert local.all()
    assert not long.any()


def test_matched_edge_lesion_does_not_collapse_in_and_out_degree():
    target = {
        "total_weight": 10.0, "mean_length": 5.0,
        "mean_in_degree": 1.0, "mean_out_degree": 4.0, "extent": 20.0,
    }
    valid = {
        "total_weight": 10.5, "mean_length": 4.8,
        "mean_in_degree": 1.5, "mean_out_degree": 3.5, "extent": 19.0,
    }
    # The total degree is identical to the target, but its direction is
    # reversed.  The pre-fix matcher accepted this invalid control.
    direction_swapped = {
        "total_weight": 10.0, "mean_length": 5.0,
        "mean_in_degree": 4.0, "mean_out_degree": 1.0, "extent": 20.0,
    }
    assert edge_descriptor_matches(valid, target)
    assert not edge_descriptor_matches(direction_swapped, target)


def test_patient_field_aggregation_keeps_seed_rows_distinct_from_pair_correlations(tmp_path):
    field_root = tmp_path / "empirical"
    field_root.mkdir()
    (tmp_path / "INPUT_MANIFEST.json").write_text(json.dumps({
        "input_roots": {"field": str(field_root)}
    }))
    (field_root / "p1.json").write_text(json.dumps({"interictal_field": {
        "contact_order": ["A1", "A2", "A3"],
        "rank_a": [0.0, 1.0, 2.0], "rank_b": [2.0, 1.0, 0.0],
    }}))
    rows, fields = [], {}
    for seed in (0, 1, 2):
        for template, value in (("A", [0.0, 0.5, 1.0]), ("B", [1.0, 0.5, 0.0])):
            rows.append({
                "subject": "p1", "fit_id": "p1__shared", "scope": "shared",
                "model": "M6_SPATIAL_MID", "cell": "rnn", "seed": seed,
                "template": template, "canonical_empirical_r": 1.0,
                "seed_removed_empirical_r": 1.0, "hit_epoch_ceiling": False,
            })
            fields[("p1__shared", "M6_SPATIAL_MID", "rnn", seed, template)] = {
                "contacts": np.array(["A1", "A2", "A3"]),
                "canonical_full": np.asarray(value), "seed_removed": np.asarray(value),
                "participation": np.asarray(value), "seed_removed_denominator": np.ones(3),
                "canonical_full_split_half_stability": np.array([1.0]),
                "seed_removed_split_half_stability": np.array([1.0]),
            }
    patient_rows, _ = aggregate_patient_fields(tmp_path, rows, fields)
    assert len(patient_rows) == 1
    with (tmp_path / "model_field_fit_metrics.csv").open(newline="") as handle:
        fit_rows = list(csv.DictReader(handle))
    assert len(fit_rows) == 2
    assert all(row["n_seeds"] == "3" for row in fit_rows)
