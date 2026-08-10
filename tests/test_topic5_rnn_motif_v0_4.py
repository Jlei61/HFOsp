from __future__ import annotations

import csv
import numpy as np
import pytest
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
from analyse_topic5_rnn_motif_influence_v0_4 import (  # noqa: E402
    contact_pair_observation_count,
    contact_orientation_summary,
    contact_response_summary,
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
    pairwise_array_seed_stability,
    pairwise_seed_stability,
)
from run_topic5_rnn_motif_matched_lesions_v0_4 import (  # noqa: E402
    edge_descriptor_matches,
)
from closeout_topic5_rnn_motif_review_v0_4 import (  # noqa: E402
    field_decomposition,
    graph_wiring_metrics,
    median_by_fit_then_patient,
    sequence_metrics,
    target_leave_one_seizure_out,
    wiring_decomposition,
    write_figure_readme,
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


def test_review_wiring_metrics_distinguish_total_from_mean_cost():
    graph = {
        "mask": np.array([[0, 1], [1, 0]], dtype=np.uint8),
        "strength": np.array([[0.0, 2.0], [1.0, 0.0]]),
        "D_mm": np.array([[0.0, 10.0], [20.0, 0.0]]),
    }
    result = graph_wiring_metrics(graph, d0_mm=10.0)
    assert result["edge_count"] == 2
    assert np.isclose(result["total_geometric_length_mm"], 30.0)
    assert np.isclose(result["total_strength_weighted_length_mm"], 40.0)
    assert np.isclose(result["mean_edge_strength_weighted_length_over_d0"], 2.0)
    assert np.isclose(result["strength_normalized_mean_length_mm"], 40.0 / 3.0)


def test_review_sequence_diagnostics_remove_supplied_seed():
    observed = np.array([0, 1, 2, 3, -1])
    correct = sequence_metrics(observed, [[0], [1], [2], [3]])
    reverse = sequence_metrics(observed, [[0], [3], [2], [1]])
    assert np.isclose(correct["kendall_tau_b"], 1.0)
    assert np.isclose(correct["normalized_rank_mae"], 0.0)
    assert np.isclose(correct["participation_jaccard"], 1.0)
    assert np.isclose(reverse["kendall_tau_b"], -1.0)


def test_review_loo_target_reliability_uses_other_seizures_only(tmp_path):
    names = np.array(["A1", "A2", "A3", "A4"])
    for index, values in enumerate(([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])):
        np.savez(tmp_path / f"p__s{index}.npz", contact_names=names,
                 target_1_150=np.asarray(values))
    rows = target_leave_one_seizure_out(sorted(tmp_path.glob("*.npz")))
    assert len(rows) == 2
    assert all(np.isclose(row["loo_spearman"], 1.0) for row in rows)


def test_review_field_decomposition_keeps_registered_full_field_primary():
    rows = []
    values = {
        ("M6_SPATIAL_MID", "rnn", "canonical_full"): 0.30,
        ("M6_SPATIAL_MID", "rnn", "seed_removed"): 0.20,
        ("M0_NO_REC", "rnn", "canonical_full"): 0.10,
        ("C_ORDER_SHUFFLED", "rnn", "canonical_full"): 0.15,
        ("M4_SPATIAL_GROWTH", "rnn", "canonical_full"): 0.25,
        ("EMPIRICAL_REFERENCE", "reference", "canonical_full"): 0.40,
    }
    for (model, cell, endpoint), value in values.items():
        rows.append({"subject": "p1", "primary": "True", "model": model,
                     "cell": cell, "endpoint": endpoint, "all_contact_margin": str(value)})
    patients, summary = field_decomposition(rows)
    assert summary["primary_endpoint_unchanged"] == "FIELD_CANONICAL_FULL"
    assert np.isclose(patients[0]["source_contribution"], 0.10)
    assert np.isclose(patients[0]["recurrence_increment"], 0.20)
    assert np.isclose(patients[0]["order_specific_increment"], 0.15)
    assert np.isclose(patients[0]["wiring_cost_increment"], 0.05)


def test_review_patient_aggregation_weights_fits_not_seeds():
    rows = [{"subject": "p1", "fit_id": "own_a", "model": "M", "cell": "rnn", "value": v}
            for v in (1.0, 1.0, 1.0)]
    rows += [{"subject": "p1", "fit_id": "own_b", "model": "M", "cell": "rnn", "value": 9.0}]
    patients = median_by_fit_then_patient(rows, ["value"])
    # Flat median over the four runs would be 1.0; the frozen rule gives each fit
    # equal weight, so the patient value sits between the two fit medians.
    assert np.isclose(patients[0]["value"], 5.0)


def _write_wiring_unit(out_root, subject, scope, model, seed, strength, c_wiring):
    unit = out_root / "per_subject" / f"{subject}__{scope}" / f"{model}__rnn" / f"seed{seed}"
    unit.mkdir(parents=True)
    np.savez(unit / "graph.npz", mask=np.array([[0, 1], [0, 0]], dtype=np.uint8),
             strength=np.array([[0.0, strength], [0.0, 0.0]]),
             D_mm=np.array([[0.0, 10.0], [10.0, 0.0]]))
    (unit / "metrics.json").write_text(json.dumps({
        "subject": subject, "fit_id": f"{subject}__{scope}", "fit_scope": scope,
        "cell": "rnn", "seed": seed, "c_wiring": c_wiring, "config": {"d0_mm": 10.0},
    }))


def test_review_wiring_drops_smoke_units_and_must_match_the_frozen_table(tmp_path):
    _write_wiring_unit(tmp_path, "p1", "own_a", "M6_SPATIAL_MID", 0, 2.0, 2.0)
    _write_wiring_unit(tmp_path, "p1", "own_a", "SMOKE_M6_SPATIAL_MID", 0, 90.0, 90.0)
    (tmp_path / "interictal_per_patient.csv").write_text(
        "subject,model,cell,c_wiring\np1,M6_SPATIAL_MID,rnn,2.0\n"
    )
    rows, summary = wiring_decomposition(tmp_path)
    assert [row["model"] for row in rows] == ["M6_SPATIAL_MID"]
    assert summary["n_smoke_graph_runs_excluded"] == 1
    assert summary["registered_c_wiring_parity"]["maximum_absolute_difference"] < 1e-9

    (tmp_path / "interictal_per_patient.csv").write_text(
        "subject,model,cell,c_wiring\np1,M6_SPATIAL_MID,rnn,0.5\n"
    )
    with pytest.raises(RuntimeError, match="frozen interictal table"):
        wiring_decomposition(tmp_path)


def test_review_closeout_readme_section_survives_a_later_figure_rewrite(tmp_path):
    marker = "<!-- topic5-rnn-motif-v0.4-stage-and-final-figures -->"
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / "README.md").write_text(f"# head\n\n{marker}\n\n### main figure\n")
    write_figure_readme(tmp_path)
    text = (figures / "README.md").read_text()
    assert text.index("topic5_rnn_motif_review_closeout.png") < text.index(marker)
    # The figure script keeps everything before its own marker verbatim.
    assert "topic5_rnn_motif_review_closeout.png" in text.split(marker)[0]


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
