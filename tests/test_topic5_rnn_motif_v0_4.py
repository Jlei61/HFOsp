from __future__ import annotations

import csv
import numpy as np
import torch
import json
import sys
from pathlib import Path

import yaml

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
    holm as interictal_holm,
    paired_test,
    seed_removed_sequence_agreement,
)
from analyse_topic5_rnn_motif_influence_v0_4 import (  # noqa: E402
    contact_pair_observation_count,
    contact_orientation_summary,
    contact_response_summary,
)
from score_topic5_rnn_motif_early_ictal_v0_4 import (  # noqa: E402
    aggregate_patients,
    conditional_effects,
    compute_dose_trend,
    compute_factorial_effects,
    locked_target_artifacts,
    permutation_indices,
    permutation_support,
    paired_summary,
    target_artifact_recheck_payload,
)
from score_topic5_rnn_motif_lesion_early_ictal_v0_4 import patient_fields  # noqa: E402
from summarize_topic5_rnn_motif_theory_v0_4 import (  # noqa: E402
    candidate_distance_classes,
    pairwise_array_seed_stability,
    pairwise_seed_stability,
)
from run_topic5_rnn_motif_matched_lesions_v0_4 import (  # noqa: E402
    choose_units,
    complete_patient_fit_set,
    edge_descriptor_matches,
    perturbation_damage,
)
from build_topic5_rnn_motif_common_observables_v0_4 import (  # noqa: E402
    patient_level_vectors,
)
from export_topic5_rnn_motif_unit_contracts_v0_4 import (  # noqa: E402
    export as export_unit_contracts,
    sha256,
)
from plot_topic5_rnn_motif_figures_v0_4 import (  # noqa: E402
    lesion_display_values,
    patient_level_effective_reach,
    selected_metrics,
)
from finalize_topic5_rnn_motif_v0_4 import (  # noqa: E402
    audit_figure_sources,
    target_artifact_recheck_ok,
    target_contract_trace_ok,
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


def test_interictal_factorial_holm_is_monotone():
    adjusted = interictal_holm({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_target_artifact_recheck_is_required_before_unseal():
    payload = {
        "status": "PASS",
        "n_artifacts": 26,
        "artifact_sha256_mismatches": 0,
        "metadata_target_values_read": False,
        "model_field_manifest_target_values_read": False,
        "target_access_audit_existed_before_recheck": False,
    }
    assert target_artifact_recheck_ok(payload)
    assert not target_artifact_recheck_ok({**payload, "artifact_sha256_mismatches": 1})
    assert not target_artifact_recheck_ok({**payload, "target_access_audit_existed_before_recheck": True})


def test_target_artifact_recheck_payload_is_value_blind(tmp_path):
    metadata = {
        "actual_primary_join": ["p1", "p2"],
        "supportive_subject": "s1",
        "target_values_read": False,
        "target_energy_arrays_deserialized": False,
    }
    artifacts = {
        "p1": [tmp_path / "a.npz", tmp_path / "b.npz"],
        "p2": [tmp_path / "c.npz"],
        "s1": [tmp_path / "d.npz"],
    }
    payload = target_artifact_recheck_payload(tmp_path, metadata, artifacts)
    assert payload["n_artifacts"] == 4
    assert payload["n_primary_seizure_files"] == 3
    assert payload["n_supportive_seizure_files"] == 1
    assert payload["metadata_target_energy_arrays_deserialized"] is False
    assert payload["target_access_audit_existed_before_recheck"] is False


def test_target_contract_trace_matches_paper_endpoint():
    payload = {
        "status": "PASS",
        "target_key": "target_1_150",
        "anchor": "clinical_onset",
        "post_onset_window_seconds": [0.0, 10.0],
        "frequency_band_hz": [1.0, 150.0],
        "primary_field_endpoint": "canonical_full_maxAB",
        "primary_null": "5000 synchronized all-contact permutations with support rebuilt",
        "sensitivity_null": "within-shaft permutations",
        "target_values_read_during_trace": False,
        "producer_chain": [{}, {}, {}, {}],
    }
    assert target_contract_trace_ok(payload)
    assert not target_contract_trace_ok({**payload, "anchor": "eeg_onset"})
    assert not target_contract_trace_ok({**payload, "frequency_band_hz": [1.0, 45.0]})


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


def test_early_ictal_null_is_folded_patient_first_draw_by_draw():
    rows = []
    nulls = {}
    for seizure, observed, null in (
        ("s1", 0.2, np.array([0.1, 0.6])),
        ("s2", 0.8, np.array([0.3, 0.2])),
    ):
        keys = {name: f"{seizure}_{name}" for name in
                ("all", "shaft", "common_all", "common_shaft")}
        for key in keys.values():
            nulls[key] = null
        rows.append({
            "subject": "p1", "model": "M6_SPATIAL_MID", "cell": "rnn",
            "endpoint": "canonical_full", "seizure_id": seizure,
            "observed": observed, "common_observed": observed,
            "n_contacts": 8, "n_within_shaft_permutable_contacts": 8,
            "n_within_shaft_permutable_groups": 2,
            "null_key_all": keys["all"], "null_key_shaft": keys["shaft"],
            "null_key_common_all": keys["common_all"],
            "null_key_common_shaft": keys["common_shaft"],
        })
    patients, patient_nulls = aggregate_patients(rows, nulls, supportive="supportive")
    assert len(patients) == 1
    assert np.isclose(patients[0]["observed"], 0.5)
    assert np.isclose(patients[0]["all_contact_null_median"], 0.3)
    assert np.isclose(patients[0]["all_contact_margin"], 0.2)
    key = "p1|M6_SPATIAL_MID|rnn|canonical_full|maxab"
    assert np.allclose(patient_nulls[key], np.array([0.2, 0.4]))


def test_early_ictal_scorer_reads_only_hash_locked_target_artifacts(tmp_path):
    target_root = tmp_path / "targets"
    target_root.mkdir()
    artifact = target_root / "epilepsiae_p1__s1.npz"
    artifact.write_bytes(b"frozen target bytes")
    inventory = tmp_path / "early_ictal_metadata_inventory.csv"
    with inventory.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject", "artifact_path", "artifact_sha256"))
        writer.writeheader()
        writer.writerow({"subject": "epilepsiae_p1", "artifact_path": str(artifact),
                         "artifact_sha256": sha256(artifact)})
    metadata = {
        "target_cache_root": str(target_root),
        "inventory_csv_sha256": sha256(inventory),
        "seizure_file_counts_filename_only": {"epilepsiae_p1": 1},
    }
    resolved = locked_target_artifacts(tmp_path, target_root, metadata)
    assert resolved == {"epilepsiae_p1": [artifact.resolve()]}
    artifact.write_bytes(b"mutated after target freeze")
    with np.testing.assert_raises_regex(RuntimeError, "hash changed"):
        locked_target_artifacts(tmp_path, target_root, metadata)


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


def test_patient_wilcoxon_removes_ties_before_requesting_exact_distribution():
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 0.0, 5.0e-17])
    interictal = paired_test(values)
    early = paired_summary(values, draws=100, seed=9)
    assert (interictal["positive"], interictal["negative"], interictal["tied"]) == (4, 0, 2)
    assert (early["positive"], early["negative"], early["tied"]) == (4, 0, 2)
    assert np.isclose(interictal["p_two_sided"], 0.125)
    assert np.isclose(early["wilcoxon_p"], 0.125)


def test_lesion_fields_keep_noncollinear_a_and_b_producers_separate():
    def record(scope, fit_id, template, values, status="inference_available"):
        payload = {
            "status": status, "field_contacts": ["A1", "B1"],
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
    assert resolved["matched_inference_available"] is True

    unresolved = patient_fields([
        record("own_a", "p1__own_a", "A", [1.0, 0.0]),
        record("own_b", "p1__own_b", "B", [0.0, 1.0],
               status="matched_inference_unavailable"),
    ])[('p1', 'M6_SPATIAL_MID', 'connector_nodes')]
    assert unresolved["matched_inference_available"] is False


def test_matched_lesion_requires_both_noncollinear_fits():
    expected = {"p1__own_a", "p1__own_b"}
    one_side = [{"fit_id": "p1__own_a", "status": "inference_available"}]
    assert not complete_patient_fit_set(one_side, expected)
    both = one_side + [{"fit_id": "p1__own_b", "status": "inference_available"}]
    assert complete_patient_fit_set(both, expected)
    unavailable = [both[0], {"fit_id": "p1__own_b",
                             "status": "matched_inference_unavailable"}]
    assert not complete_patient_fit_set(unavailable, expected)


def test_lesion_figure_uses_estimable_frozen_components_without_effect_selection(
        tmp_path):
    records = []
    for lesion, n_rows, base in (
        ("local_backbone_edges", 5, -10.0),
        ("long_range_high_influence_edges", 6, -20.0),
        ("connector_nodes", 4, 100.0),
    ):
        for index in range(n_rows):
            records.append({
                "subject": f"p{index}", "model": "M6_SPATIAL_MID",
                "cell": "rnn", "lesion": lesion,
                "all_inference_available": True,
                "specificity_contact_nll": base + index,
            })
    with (tmp_path / "matched_lesion_patient_metrics.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    selected = lesion_display_values(tmp_path)
    assert [label for label, _ in selected] == [
        "Local\nbackbone", "Long-range\nedges"
    ]
    assert np.allclose(selected[0][1], np.arange(5) - 10.0)

    # Connector magnitude is deliberately huge, but it must not be selected
    # until its patient denominator reaches the fixed threshold.
    assert all("Connector" not in label for label, _ in selected)


def test_lesion_figure_denominator_counts_unique_patients(tmp_path):
    records = []
    for lesion in ("local_backbone_edges", "long_range_high_influence_edges"):
        for subject in ("p1", "p2", "p3", "p4"):
            records.append({
                "subject": subject, "model": "M6_SPATIAL_MID", "cell": "rnn",
                "lesion": lesion, "all_inference_available": True,
                "specificity_contact_nll": 1.0,
            })
        records.append({**records[-1], "specificity_contact_nll": 100.0})
    with (tmp_path / "matched_lesion_patient_metrics.csv").open(
            "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    assert lesion_display_values(tmp_path) == []


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

    pulse_a = np.stack([first, first * 2, first * 3])
    pulse_b = np.stack([second, second * 2, second * 3])
    pa = tmp_path / "pulse0.npz"; pb = tmp_path / "pulse1.npz"
    np.savez_compressed(pa, open_loop_pulse_lag123=pulse_a)
    np.savez_compressed(pb, open_loop_pulse_lag123=pulse_b)
    assert np.isclose(pairwise_array_seed_stability(
        [pa, pb], "open_loop_pulse_lag123", array_index=2
    ), 1.0)


def test_motif_distance_thresholds_do_not_create_long_edges_in_local_mask():
    position = np.arange(8, dtype=float)
    distance = np.abs(position[:, None] - position[None, :])
    mask = (distance == 1.0)
    np.fill_diagonal(mask, False)
    local, long, q50, q75 = candidate_distance_classes(mask, distance)
    assert q50 > 1.0 and q75 > q50
    assert local.all()
    assert not long.any()


def test_contact_pulse_summary_separates_axis_and_transverse_pairs():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    pulse = np.zeros((3, 3, 3), float)
    pulse[:, 0, 1] = 2.0
    pulse[:, 1, 0] = 2.0
    pulse[:, 0, 2] = 0.5
    pulse[:, 2, 0] = 0.5
    summary = contact_orientation_summary(pulse, xy)
    assert summary["lag1_axis_aligned_abs"] > summary["lag1_transverse_abs"]
    assert summary["lag1_axis_to_transverse_ratio"] > 1.0
    assert set(summary) == {
        f"lag{lag}_{name}" for lag in (1, 2, 3)
        for name in ("axis_aligned_abs", "transverse_abs", "axis_to_transverse_ratio")
    }
    complete = contact_response_summary(np.ones((3, 3)), pulse, xy, ["A", "A", "B"])
    for key in (
        "tf_lag1_signed_influence", "tf_lag1_abs_influence",
        "lag1_signed_influence", "lag1_abs_influence",
        "lag2_to_lag1_abs_ratio", "lag3_to_lag1_abs_ratio",
        "lag2_same_shaft_abs", "lag3_cross_shaft_abs",
        "lag1_distance_q1_abs", "lag1_distance_q4_signed",
    ):
        assert key in complete and np.isfinite(complete[key])


def test_unobserved_contact_pairs_are_not_treated_as_zero_influence():
    ranks = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 3, 2, 4],
    ], dtype=np.int16)
    count = contact_pair_observation_count(ranks, np.array([2, 2]), max_prefixes=32)
    assert count[0, 1] == 0  # contact 0 is already recruited at every eligible prefix
    matrix = np.zeros((5, 5), float)
    matrix[count == 0] = np.nan
    pulse = np.stack([matrix, matrix, matrix])
    summary = contact_response_summary(matrix, pulse, np.c_[np.arange(5), np.zeros(5)], ["A"] * 5)
    assert summary["lag1_abs_influence"] == 0.0
    assert np.isnan(matrix[0, 1])


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


def test_common_observable_effective_reach_is_patient_first():
    rows = []
    for fit in ("own_a", "own_b"):
        for seed in range(3):
            rows.append({
                "subject": "p_noncollinear", "fit_id": fit, "seed": str(seed),
                "lag1_reach_mm": "1", "lag2_reach_mm": "2", "lag3_reach_mm": "3",
            })
    for seed in range(3):
        rows.append({
            "subject": "p_shared", "fit_id": "shared", "seed": str(seed),
            "lag1_reach_mm": "10", "lag2_reach_mm": "20", "lag3_reach_mm": "30",
        })
    output = patient_level_vectors(
        rows, ("lag1_reach_mm", "lag2_reach_mm", "lag3_reach_mm")
    )
    assert set(output) == {"p_noncollinear", "p_shared"}
    np.testing.assert_allclose(output["p_noncollinear"], [1, 2, 3])
    np.testing.assert_allclose(output["p_shared"], [10, 20, 30])
    np.testing.assert_allclose(np.median(np.asarray(list(output.values())), axis=0),
                               [5.5, 11.0, 16.5])


def test_unit_contract_export_is_lossless_and_idempotent(tmp_path):
    out = tmp_path / "run"
    unit = out / "per_subject" / "p1__shared" / "M6_SPATIAL_MID__rnn" / "seed0"
    cache = out / "cache" / "p1__shared"
    unit.mkdir(parents=True); cache.mkdir(parents=True)
    manifest = out / "INPUT_MANIFEST.json"
    manifest.write_text(json.dumps({"cohort": "frozen"}))
    for name, content in (("plane.npz", b"plane"), ("events.npz", b"events")):
        (cache / name).write_bytes(content)
    (cache / "provenance.json").write_text(json.dumps({"subject": "p1"}))
    metrics = {
        "fit_id": "p1__shared", "subject": "p1", "fit_scope": "shared",
        "model_id": "M6_SPATIAL_MID__rnn", "arm": "SPATIAL_SET", "cell": "rnn",
        "seed": 0, "shuffled_targets": False, "shuffle_mode": "none",
        "config": {"lr": 0.01, "state_dim": 32},
        "rollout_decoder": {"n_epochs": 4}, "config_sha256": "recorded-config",
        "producer_hashes": {"input_manifest": sha256(manifest), "trainer": "abc"},
    }
    metrics_path = unit / "metrics.json"
    metrics_path.write_text(json.dumps(metrics))

    first = export_unit_contracts(out)
    config_before = (unit / "config.json").read_bytes()
    hashes_before = (unit / "input_hashes.json").read_bytes()
    second = export_unit_contracts(out)

    assert first == second
    assert first["n_formal_training_units"] == 1
    assert first["n_smoke_training_units"] == 0
    assert config_before == (unit / "config.json").read_bytes()
    assert hashes_before == (unit / "input_hashes.json").read_bytes()
    config = json.loads(config_before)
    hashes = json.loads(hashes_before)
    assert config["training_config"] == metrics["config"]
    assert hashes["input_manifest"]["sha256"] == sha256(manifest)
    assert hashes["fit_cache"]["events.npz"]["sha256"] == sha256(cache / "events.npz")


def test_executed_yaml_contract_matches_the_frozen_model_and_split_contracts():
    path = ROOT / "config/topic5_rnn_motif_cross_state_v0_4.yaml"
    contract = yaml.safe_load(path.read_text())

    assert contract["contract_role"] == "EXECUTED_CONTRACT_EXPORT"
    assert contract["geometry_status"] == "RETROSPECTIVE_TEST_INFORMED_PROPAGATION_PLANE"
    assert contract["cohort"]["n_patients"] == 21
    assert contract["cohort"]["n_fits"] == 31
    assert len(contract["cohort"]["shared_fits"]) == 11
    assert len(contract["cohort"]["split_fits"]) == 20
    assert contract["training"]["formal_training_units"] == 1426

    split = contract["split"]
    assert split["source_pool"] == "canonical_train80_only"
    assert sum(split[key] for key in (
        "train_fraction_within_train80",
        "validation_fraction_within_train80",
        "test_fraction_within_train80",
    )) == 1.0
    assert split["old_outer_heldout20_status"].startswith("BURNED")

    exported = contract["model_matrix"]
    for model, spec in MODEL_SPECS.items():
        assert exported[model]["arm"] == spec.arm
        assert exported[model]["eta"] == spec.eta
        assert exported[model]["seeds"] == list(spec.seeds)
    assert contract["rollout_decoder"]["observed_future_set_size_read"] is False
    assert contract["statistics"]["primary_unit"] == "patient"


def test_effective_reach_plot_input_is_patient_first(tmp_path):
    path = tmp_path / "effective_influence_fit_seed.csv"
    rows = []
    for fit_id in ("p1__own_a", "p1__own_b"):
        for seed in (0, 1, 2):
            rows.append({
                "subject": "p1", "fit_id": fit_id, "model": "M6_SPATIAL_MID",
                "cell": "rnn", "seed": seed,
                "lag1_reach_mm": 1, "lag2_reach_mm": 2, "lag3_reach_mm": 3,
            })
    for seed in (0, 1, 2):
        rows.append({
            "subject": "p2", "fit_id": "p2__shared", "model": "M6_SPATIAL_MID",
            "cell": "rnn", "seed": seed,
            "lag1_reach_mm": 10, "lag2_reach_mm": 20, "lag3_reach_mm": 30,
        })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    reach = patient_level_effective_reach(tmp_path)
    assert set(reach) == {"p1", "p2"}
    np.testing.assert_allclose(reach["p1"], [1, 2, 3])
    np.testing.assert_allclose(reach["p2"], [10, 20, 30])


def test_figure_representative_checkpoint_is_median_seed_not_best_seed(tmp_path):
    values = (1.0, 2.0, 10.0)
    for seed, value in enumerate(values):
        directory = tmp_path / "per_subject" / "p1__shared" / "M6_SPATIAL_MID__rnn" / f"seed{seed}"
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "validation": {"contact_nll": value},
        }))
    selected = selected_metrics(tmp_path, "p1", "M6_SPATIAL_MID")
    assert selected.parent.name == "seed1"


def test_figure_source_manifest_verifies_every_panel_byte(tmp_path):
    records = {}
    for panel in "ABCDEF":
        path = tmp_path / f"panel_{panel}.csv"
        path.write_text(f"source,{panel}\n")
        records[panel] = [{"path": str(path), "sha256": sha256(path)}]
    manifest = {
        "_contract": "topic5_figure6_source_manifest_v0_4",
        "_representative_selection": {
            "patient": "epilepsiae_1146",
            "role": "supportive visualization; excluded from primary p-values",
            "checkpoint_rule": "choose validation contact NLL nearest the seed median",
        },
        **records,
    }
    passed, errors = audit_figure_sources(manifest)
    assert passed and errors == []
    (tmp_path / "panel_E.csv").write_text("target bytes changed\n")
    passed, errors = audit_figure_sources(manifest)
    assert not passed
    assert "panel_E_0_hash" in errors


def test_matched_lesion_unit_is_median_seed_not_motif_selected(tmp_path):
    values = (0.5, 1.0, 8.0)
    for seed, value in enumerate(values):
        metrics_dir = (tmp_path / "per_subject" / "p1__shared"
                       / "M6_SPATIAL_MID__rnn" / f"seed{seed}")
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "metrics.json").write_text(json.dumps({
            "fit_id": "p1__shared", "validation": {"contact_nll": value},
        }))
        influence_dir = (tmp_path / "effective_influence" / "p1__shared"
                         / "M6_SPATIAL_MID__rnn" / f"seed{seed}")
        influence_dir.mkdir(parents=True)
        np.savez(influence_dir / "influence.npz", marker=np.asarray([seed]))
    selected = choose_units(tmp_path)
    assert len(selected) == 1
    assert selected[0][0].parent.name == "seed1"
    assert selected[0][1].parent.name == "seed1"


def test_matched_lesion_damage_uses_each_metric_optimum():
    assert np.isclose(perturbation_damage("contact_nll", 1.4, 1.0), 0.4)
    assert np.isclose(perturbation_damage("stop_bce", 0.8, 0.5), 0.3)
    assert np.isclose(perturbation_damage("rollout_spearman", 0.2, 0.7), 0.5)
    assert np.isclose(perturbation_damage("interictal_field_fidelity", 0.1, 0.6), 0.5)
    # Both shorter and longer rollouts are harmful relative to the ideal ratio 1.
    assert np.isclose(perturbation_damage("postseed_length_ratio", 0.6, 0.9), 0.3)
    assert np.isclose(perturbation_damage("postseed_length_ratio", 1.4, 1.1), 0.3)
