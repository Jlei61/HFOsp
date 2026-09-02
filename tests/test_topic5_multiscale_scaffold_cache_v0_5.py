from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import scripts.build_topic5_multiscale_scaffold_cache_v0_5 as builder
import scripts.run_topic5_v0_5_target_free as embargo
import scripts.build_topic5_train_only_modes_suffix_null_v0_5 as modes_builder
import scripts.analyse_topic5_prefix_template_v0_5 as template_control
import scripts.analyse_topic5_multiscale_interictal_v0_5 as interictal_analysis
import scripts.build_topic5_multiscale_fields_v0_5 as field_builder
import scripts.analyse_topic5_multiscale_mechanism_v0_5 as mechanism
import scripts.run_topic5_mode_flow_attenuation_v0_5 as mode_flow
import scripts.run_topic5_multiscale_attenuation_v0_5 as arm_attenuation
import scripts.score_topic5_multiscale_early_ictal_v0_5 as early_scorer
import scripts.prepare_topic5_multiscale_target_unseal_v0_5 as target_freezer
import scripts.audit_topic5_multiscale_closeout_v0_5 as closeout_audit
import scripts.freeze_topic5_preunseal_analysis_metrics_v0_5 as metric_freezer
import scripts.repair_topic5_multiscale_train_mixture_v0_5 as mixture_repair
import scripts.hotfill_topic5_attenuation_deduplicated_rollouts_v0_5 as attenuation_hotfill
from src.topic5_multiscale_scaffold_v0_5 import (
    construct_macro_matched_nonlocal,
    distance_decile_labels,
    exact_macro_match_audit,
)
from src.topic5_lbss_rnn_v0_2 import (
    LBSSConfig,
    LBSSModel,
    build_pool_contract,
    source_balanced_sample,
)
from scripts.build_topic5_crossfit_nonlocality_v0_5 import (
    contact_path_distance_matrix,
    event_sensitivities,
    fit_nonnegative_beta,
    top_mass_support,
    weighted_quantile,
)


def test_densify_groups_preserves_missing_and_removes_rank_gaps():
    raw = np.asarray([[4, -1, 9, 4], [-1, 7, 3, 7]], dtype=np.int16)
    dense = builder.densify_groups(raw)
    assert dense.tolist() == [[0, -1, 1, 0], [-1, 1, 0, 1]]


def test_development_split_is_chronological_inside_frozen_train80():
    frozen = np.asarray([0] * 20 + [1] * 5, dtype=np.uint8)
    split = builder.development_split(frozen)
    assert np.all(split[:14] == 0)
    assert np.all(split[14:17] == 1)
    assert np.all(split[17:20] == 2)
    assert np.all(split[20:] == -1)


def test_solved_scopes_prefers_shared_and_requires_both_noncollinear_planes():
    shared = {
        "status": "ok", "contact_order": ["A"],
        "planes": {"shared": {"status": "ok"}},
    }
    scopes, reason = builder.solved_scopes(shared)
    assert reason is None
    assert set(scopes) == {"shared"}

    incomplete = {
        "status": "ok", "contact_order": ["A"],
        "planes": {
            "shared": {"status": "not_available"},
            "own_a": {"status": "ok"},
            "own_b": {"status": "not_available"},
        },
    }
    scopes, reason = builder.solved_scopes(incomplete)
    assert scopes == {}
    assert reason == "UNSOLVED_OWN_B"


def test_geometry_dimension_separates_planar_from_collinear_contacts():
    line = np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    plane = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    line_rank, line_ratio, line_area = builder.geometry_dimension(line)
    plane_rank, plane_ratio, plane_area = builder.geometry_dimension(plane)
    assert (line_rank, line_ratio, line_area) == (1, 0.0, 0.0)
    assert plane_rank == 2
    assert plane_ratio > 0.05
    assert plane_area > 0.0


def test_routing_copy_deserializes_only_frozen_non_target_columns(tmp_path, monkeypatch):
    rows = []
    for patient in range(17):
        for seizure in range(10 if patient < 14 else 9):
            rows.append({
                "dataset": "d", "subject": f"p{patient}",
                "seizure_idx": seizure, "group_id": "all_phenotype_matched",
                "phenotype": "x", "band": "b", "permutation_seed": 7,
                "observed": 0.99, "energy": 123.0, "null_median": 0.4,
            })
    assert len(rows) == 167
    source = tmp_path / "event.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    monkeypatch.setattr(builder, "FIG3_EVENT_TABLE", source)
    result = builder.copy_routing_metadata(tmp_path / "out")
    copied = pd.read_csv(result["path"])
    assert copied.subject.nunique() == 17
    assert len(copied) == 167
    assert list(copied.columns) == list(builder.ROUTING_COLUMNS)
    assert {"observed", "energy", "null_median"}.isdisjoint(copied.columns)
    assert result["target_numeric_values_read"] is False


def test_builder_source_has_no_target_reader_or_numeric_target_root():
    source = Path(builder.__file__).read_text()
    forbidden = (
        "score_topic5_lbss_full_tissue_early_ictal",
        "load_target", "load_early_ictal", "t0_feature_cache_bb150_1_150",
        "v2_band_scan/cache",
    )
    assert all(token not in source for token in forbidden)


def test_physical_embargo_covers_numeric_and_previous_target_derived_roots():
    roots = {str(path) for path in embargo.protected_roots()}
    assert any("t0_feature_cache_bb150_1_150" in path for path in roots)
    assert any("v2_band_scan/cache" in path for path in roots)
    assert any("tspectral_field_concordance" in path for path in roots)
    assert any("lbss_full_tissue_rnn_v0_3/early_ictal" in path for path in roots)
    assert any("rnn_full_cohort_field_transfer_v0_1" in path for path in roots)


def _synthetic_two_mode_ranks(n_events=30):
    ranks = np.full((n_events, 6), -1, dtype=np.int16)
    for event in range(n_events):
        if event % 2 == 0:
            ranks[event] = [0, 1, 2, 3, 4, 5]
        else:
            ranks[event] = [5, 4, 3, 2, 1, 0]
    return ranks


def test_train_only_prefix_posterior_does_not_read_heldout_suffix():
    ranks = _synthetic_two_mode_ranks()
    split = np.asarray([0] * 24 + [1] * 3 + [2] * 3, dtype=np.int8)
    # Two heldout rows have the same first three rank sets but opposing suffixes.
    ranks[24] = [0, 1, 2, 3, 4, 5]
    ranks[25] = [0, 1, 2, 5, 4, 3]
    result = modes_builder.train_only_modes(ranks, split)
    assert np.array_equal(result["prefix_posterior"][24], result["prefix_posterior"][25])
    assert result["full_train_mode"][24] == -1
    assert result["full_train_mode"][25] == -1


def test_scope_split_never_filters_prediction_task_by_mode():
    base = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int8)
    modes = {
        "full_train_mode": np.asarray([0, 1, -1, -1, -1, -1], dtype=np.int8),
        "prefix_mode": np.asarray([1, 0, 0, 1, 0, 1], dtype=np.int8),
    }
    split = modes_builder.scope_split(base, "own_a", modes, own_cluster=0)
    assert split.tolist() == base.tolist()


def test_suffix_null_never_crosses_split_and_preserves_recipient_prefix():
    ranks = np.tile(np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int16), (8, 1))
    # Give suffixes distinguishable donor identities while retaining length.
    for event in range(8):
        ranks[event, 3:] = np.roll(ranks[event, 3:], event % 3)
    split = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int8)
    mode = np.zeros(8, dtype=np.int8)
    mapping, audit = modes_builder.suffix_mapping(ranks, split, mode, seed=9)
    assert audit["effectively_reassigned"] == 8
    assert np.all(mapping != np.arange(8))
    donors = mapping >= 0
    assert np.all(split[mapping[donors]] == split[donors])
    shuffled = modes_builder.apply_suffix_mapping(ranks, mapping, seed=9)
    assert np.array_equal(shuffled[:, :3], ranks[:, :3])
    audit2 = modes_builder.suffix_distribution_audit(ranks, shuffled, split, mapping)
    assert audit2["distribution_audit_events"] == 8
    assert audit2["mean_suffix_kendall_distance"] >= 0
    assert audit2["mean_suffix_tie_block_shift"] == 0
    assert audit["same_suffix_rank_count_for_all_changed"] is True
    assert audit["prefix_suffix_overlap_for_any_changed"] is False


def test_l2m_double_edge_swap_preserves_all_frozen_macro_statistics():
    coordinates = np.stack(np.meshgrid(np.arange(6.0), np.arange(6.0)), axis=-1).reshape(-1, 2)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    pools = build_pool_contract(
        distance, density=0.16, added_fraction=0.35, r_local_multiplier=1.5
    )
    reference = source_balanced_sample(pools.nonlocal_pool, pools.k_added, seed=91)
    matched = construct_macro_matched_nonlocal(
        reference, pools.nonlocal_pool, distance, seed=203,
        max_restarts=20, attempts_per_restart=3000, minimum_disruption_fraction=0.35,
    )
    audit = exact_macro_match_audit(
        reference, matched.mask, pools.nonlocal_pool, matched.bin_labels
    )
    assert audit["all_exact"]
    assert audit["pairing_disruption_fraction"] >= 0.35


def test_l2m_model_requires_and_uses_frozen_nonlocal_mask():
    coordinates = np.stack(np.meshgrid(np.arange(5.0), np.arange(5.0)), axis=-1).reshape(-1, 2)
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=-1)
    pools = build_pool_contract(
        distance, density=0.20, added_fraction=0.25, r_local_multiplier=1.5
    )
    fixed = source_balanced_sample(pools.nonlocal_pool, pools.k_added, seed=7)
    observation = np.eye(5, len(coordinates), dtype=np.float32)
    config = LBSSConfig(
        arm="L2M_MACRO_MATCHED_RANDOM_LR", n_contacts=5, n_nodes=len(coordinates),
        observation_operator=observation, node_distance_mm=distance,
        local_mask=pools.local_mask, extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool, k_added=pools.k_added, seed=3,
        fixed_added_mask=fixed,
    )
    model = LBSSModel(config)
    assert np.array_equal(model.added_mask.detach().cpu().numpy().astype(np.uint8), fixed)
    assert model.arm not in model.REWIRING_ARMS


def test_distance_deciles_never_label_outside_pool():
    distance = np.abs(np.arange(12)[:, None] - np.arange(12)[None, :]).astype(float)
    pool = (distance >= 3).astype(np.uint8)
    labels, cuts = distance_decile_labels(distance, pool)
    assert cuts.shape == (9,)
    assert np.all(labels[~pool.astype(bool)] == -1)
    assert np.all((labels[pool.astype(bool)] >= 0) & (labels[pool.astype(bool)] <= 9))


def test_h_support_truncation_discards_small_long_tail_mass():
    nodes, weights = top_mass_support(np.asarray([0.80, 0.15, 0.04, 0.01]))
    assert nodes.tolist() == [0, 1]
    assert np.isclose(weights.sum(), 1.0)


def test_weighted_path_quantile_does_not_use_raw_minimum_tail():
    values = np.asarray([1.0, 10.0, 11.0])
    weights = np.asarray([0.01, 0.49, 0.50])
    assert weighted_quantile(values, weights, 0.10) == 10.0


def test_contact_path_distance_uses_h_support_and_directed_local_graph():
    H = np.asarray([[0.90, 0.10, 0.0], [0.0, 0.10, 0.90]], dtype=float)
    local = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    D = np.asarray([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    result = contact_path_distance_matrix(H, local, D)
    assert result.shape == (2, 2)
    assert np.isfinite(result).all()
    assert result[1, 0] >= 1.0


def test_negative_local_wave_slope_is_retained_as_zero_not_excluded():
    beta = fit_nonnegative_beta(np.asarray([1.0, 2.0, 3.0]), np.asarray([3.0, 2.0, 1.0]))
    assert beta == 0.0


def test_vectorized_rank_tau_matches_scipy_tau_b():
    from scipy.stats import kendalltau
    frame = pd.DataFrame({
        "path_distance_mm": [1.0, 2.0, 2.0, 4.0],
        "rank": [0, 1, 2, 2],
        "relative_latency": [0.1, 0.4, 0.2, 0.8],
    })
    one_minus_tau, violation = event_sensitivities(frame)
    expected = kendalltau(frame.path_distance_mm, frame["rank"], variant="b").statistic
    assert np.isclose(one_minus_tau, 1.0 - expected)
    assert 0.0 <= violation <= 1.0


def test_prefix_template_probability_masks_recruited_contacts():
    templates = np.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]])
    log_prob = template_control.template_log_prob(
        templates, np.asarray([0.7, 0.3]), np.asarray([False, True, True]), 1.0,
    )
    assert np.isneginf(log_prob[0])
    assert np.isclose(np.exp(log_prob[1:]).sum(), 1.0)


def test_template_decisions_begin_after_three_rank_prefixes():
    ranks = np.asarray([[0, 1, 2, 3, 4]], dtype=np.int16)
    frame = template_control.template_decisions(
        ranks, np.asarray([2]), np.asarray([[0.5, 0.5]]), np.asarray([0.69]),
        np.asarray([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]], dtype=float),
        1.0, selected_split=2,
    )
    assert frame.rank_index.tolist() == [2, 3]


def test_precedence_uses_support_shrinkage_and_returns_weighted_node_score():
    ranks = np.asarray([
        [0, 1, 2], [0, 1, 2], [0, 2, 1], [0, 1, 2], [0, 1, 2],
    ], dtype=np.int16)
    score, support, q = mechanism.precedence(ranks, np.arange(len(ranks)))
    assert support[0, 1] == 5
    assert 0 < score[0, 1] < 1  # beta-binomial shrinkage prevents a hard +1
    assert q[0] > q[2]


def test_mode_flow_bundle_inventory_is_train_only():
    import torch
    tensors = {
        "valid": torch.ones((3, 3), dtype=torch.bool),
        "is_last": torch.tensor([[False, False, True]] * 3),
        "available": torch.ones((3, 3, 4), dtype=torch.bool),
    }
    split = np.asarray([0, 1, 2], dtype=np.int8)
    train = mechanism.prefix_inventory_for_split(tensors, split, 0, 20)
    test = mechanism.prefix_inventory_for_split(tensors, split, 2, 20)
    assert {event for event, _ in train} == {0}
    assert {event for event, _ in test} == {2}


def test_oracle_repertoire_score_is_signed_not_absolute():
    candidate = {
        "a": np.asarray([0.0, 1.0, 2.0]),
        "b": np.asarray([2.0, 1.0, 0.0]),
        "oracle": True,
    }
    target = np.asarray([0.0, 1.0, 2.0])
    score, _ = early_scorer.score_candidate(
        candidate, target, np.asarray([[0, 1, 2], [2, 1, 0]])
    )
    assert score["observed"] == 1.0
    assert score["mode_a_r"] == 1.0
    assert score["mode_b_r"] == -1.0


def test_oracle_mode_selection_is_nan_safe_and_secondary_metrics_follow_choice():
    candidate = {
        "a": np.asarray([0.0, 1.0, 2.0, 3.0]),
        "b": np.ones(4),
        "oracle": True,
    }
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    score, null = early_scorer.score_candidate(
        candidate, target, np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
    )
    assert score["selected_mode"] == "A"
    assert score["identifiable"] is True
    assert score["observed"] == 1.0
    assert np.isclose(score["rank_weighted_concordance"], 1.0)
    assert np.isfinite(null).all()


def test_constant_oracle_field_is_not_identifiable_and_has_no_empirical_p():
    candidate = {"a": np.ones(4), "b": np.ones(4), "oracle": True}
    target = np.asarray([0.0, 1.0, 2.0, 3.0])
    score, null = early_scorer.score_candidate(
        candidate, target, np.asarray([[0, 1, 2, 3], [3, 2, 1, 0]])
    )
    assert score["selected_mode"] == "NOT_IDENTIFIABLE"
    assert score["identifiable"] is False
    assert np.isnan(score["observed"])
    assert np.isnan(null).all()
    p_value, n_finite = early_scorer.empirical_null_p(score["observed"], null)
    assert np.isnan(p_value)
    assert n_finite == 0


def test_target_first_unlock_record_is_exclusive_and_attempts_are_append_only(tmp_path):
    import json
    early = tmp_path / "early"
    early.mkdir()
    authorization = tmp_path / "authorization.json"
    authorization.write_text('{"authorized": true}\n')
    first = early_scorer.record_target_unlock(early, authorization, {"fields": 2})
    original = first.read_text()
    early_scorer.record_target_unlock(early, authorization, {"fields": 999})
    assert first.read_text() == original
    ledger = (early / "TARGET_SCORER_ATTEMPT_LEDGER.jsonl").read_text().splitlines()
    assert len(ledger) == 2
    assert all(json.loads(row)["first_unlock_record_sha256"] for row in ledger)


def test_spatial_null_interaction_uses_synchronized_patient_draws():
    subjects = [f"p{i}" for i in range(6)]
    J = pd.Series(np.arange(6, dtype=float), index=subjects)
    delta = pd.Series(np.arange(6, dtype=float), index=subjects)
    l3 = {subject: np.asarray([index, -index], float)
          for index, subject in enumerate(subjects)}
    l2m = {subject: np.zeros(2) for subject in subjects}
    result = early_scorer.spatial_null_interaction(J, delta, l3, l2m)
    assert result["status"] == "IDENTIFIABLE"
    assert result["finite_spatial_null_draws"] == 2
    assert result["spearman_rho"] == 1.0


def test_mode_flow_descriptor_changes_when_source_target_pairing_changes():
    added = np.zeros((4, 4), dtype=bool)
    added[1, 0] = True; added[3, 2] = True
    alternate = np.zeros_like(added)
    alternate[3, 0] = True; alternate[1, 2] = True
    flow = np.arange(16, dtype=float).reshape(4, 4) + 1.0
    strength = np.ones((4, 4), dtype=float)
    distance = np.abs(np.arange(4)[:, None] - np.arange(4)[None, :]).astype(float)
    H = np.eye(4)
    left = mode_flow._descriptor(added, flow, strength, distance, H)
    right = mode_flow._descriptor(alternate, flow, strength, distance, H)
    assert not np.array_equal(left, right)
    assert mode_flow._inside_match_calipers(left, left, 4, 4)
    assert not mode_flow._inside_match_calipers(right, left, 4, 4)


def test_v05_primary_distance_estimand_uses_r_local_not_q80():
    root = Path(__file__).resolve().parents[1]
    analysis = (root / "scripts/analyse_topic5_multiscale_interictal_v0_5.py").read_text()
    attenuation = (root / "scripts/run_topic5_multiscale_attenuation_v0_5.py").read_text()
    mode_flow = (root / "scripts/run_topic5_mode_flow_attenuation_v0_5.py").read_text()
    assert 'row["frontier_distance_mm"] > pools.r_local_mm' in analysis
    assert 'float(pools.r_local_mm), float(pools.r_local_mm)' in attenuation
    assert 'r_local_mm = float(provenance["r_local_mm"])' in mode_flow
    assert 'row["frontier_distance_mm"] > r_local_mm' in mode_flow
    assert 'row["frontier_distance_mm"] <= r_local_mm' in mode_flow
    assert "distance_decision_support_sha256" in analysis


def test_early_scorer_verifies_every_payload_not_only_manifests(tmp_path):
    payload = tmp_path / "field.npz"
    np.savez_compressed(payload, value=np.asarray([1.0]))
    digest = early_scorer.sha256_file(payload)
    contracts = (
        ("MODEL_FIELD_MANIFEST.csv", "file_sha256"),
        ("TEMPLATE_FIELD_MANIFEST.csv", "file_sha256"),
        ("ATTENUATED_FIELD_MANIFEST.csv", "file_sha256"),
        ("GAIN_ADJUSTED_FIELD_MANIFEST.csv", "sha256"),
        ("NULL_INDEX_MAP_MANIFEST.csv", "sha256"),
    )
    for name, hash_column in contracts:
        pd.DataFrame([{
            "path": str(payload), hash_column: digest, "target_values_read": False,
        }]).to_csv(tmp_path / name, index=False)
    checked = early_scorer.verify_frozen_payload_manifests(tmp_path)
    assert set(checked) == {name for name, _ in contracts}
    payload.write_bytes(b"changed")
    import pytest
    with pytest.raises(RuntimeError, match="frozen payload hash mismatch"):
        early_scorer.verify_frozen_payload_manifests(tmp_path)


def test_early_secondary_endpoints_follow_locked_high_energy_contract():
    target = np.asarray([4.0, 3.0, 2.0, 1.0])
    aligned = np.asarray([1.0, 0.7, 0.3, 0.0])
    reversed_field = aligned[::-1]
    assert early_scorer.weighted_concordance(aligned, target) > 0.99
    assert early_scorer.weighted_concordance(reversed_field, target) < -0.99
    xy = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    tied_prediction = np.asarray([1.0, 1.0, 0.0, 0.0])
    tied_target = np.asarray([0.0, 1.0, 1.0, 0.0])
    assert np.isclose(early_scorer.tied_peak_distance(tied_prediction, tied_target, xy), 2.0)
    distance = early_scorer.sinkhorn_distance(aligned, target, xy)
    assert 0.0 <= distance <= 1.0


def test_arm_attenuation_eligibility_survives_patient_aggregation():
    rows = []
    for seed in (0, 1):
        for draw, eligible in ((0, True), (1, False)):
            rows.append({
                "subject": "p", "fit_id": "p__shared", "scope": "shared",
                "target": "L3_MATCHED_LOCAL", "alpha": 0.25, "seed": seed,
                "draw": draw, "local_damage": 0.1, "distal_damage": 0.2,
                "distal_selectivity": 0.1, "contact_nll": 1.0,
                "rollout_spearman": 0.5, "inferential_eligible": eligible,
                "n_valid_matched_draws": 4,
            })
    patient, auc = arm_attenuation.aggregate_metrics(pd.DataFrame(rows))
    assert not bool(patient.iloc[0].inferential_eligible)
    assert not bool(auc.iloc[0].inferential_eligible)


def test_v05_rollout_metric_never_credits_the_supplied_first_rank():
    observed = np.asarray([0, 1, 2, 3], dtype=int)
    assert np.isclose(
        interictal_analysis.seed_removed_sequence_agreement(
            observed, [[0], [1], [2], [3]]
        ),
        1.0,
    )
    assert np.isclose(
        interictal_analysis.seed_removed_sequence_agreement(
            observed, [[0], [3], [2], [1]]
        ),
        -1.0,
    )
    # A perfect supplied start with fewer than three generated post-seed
    # contacts is intentionally not scoreable.
    assert np.isnan(
        interictal_analysis.seed_removed_sequence_agreement(
            observed, [[0], [1], [2]]
        )
    )


def test_primary_early_interaction_uses_raw_signed_correspondence_not_null_margin():
    l3, l2m = "L3", "L2m"
    patient = pd.DataFrame([
        {"subject": "p", "condition": l3, "endpoint": "canonical_full",
         "observed": 0.60, "all_contact_margin": 0.10},
        {"subject": "p", "condition": l2m, "endpoint": "canonical_full",
         "observed": 0.40, "all_contact_margin": 0.30},
    ])
    delta = early_scorer.primary_raw_delta(patient, "canonical_full", l3, l2m)
    assert np.isclose(delta.loc["p"], 0.20)
    assert not np.isclose(delta.loc["p"], -0.20)


def test_train_prevalence_mixture_tracks_ab_label_alignment():
    direct = field_builder.ab_prevalence(np.asarray([8, 2]), {0: "A", 1: "B"})
    swapped = field_builder.ab_prevalence(np.asarray([8, 2]), {0: "B", 1: "A"})
    assert direct == {"A": 0.8, "B": 0.2}
    assert swapped == {"A": 0.2, "B": 0.8}


def test_noncollinear_train_mixture_uses_mode_specific_components():
    prevalence = {"A": 0.75, "B": 0.25}
    a_component = np.asarray([1.0, 0.5, 0.0])
    b_component = np.asarray([0.0, 0.25, 1.0])
    expected = prevalence["A"] * a_component + prevalence["B"] * b_component
    np.testing.assert_allclose(expected, [0.75, 0.4375, 0.25])
    assert mixture_repair.target_label("own_a") == "A"
    assert mixture_repair.target_label("own_b") == "B"


def test_train_mixture_is_invariant_to_ab_label_swap():
    a = np.asarray([1.0, 0.2, 0.0])
    b = np.asarray([0.0, 0.4, 1.0])
    direct = mixture_repair.weighted_mixture(a, b, {"A": 0.7, "B": 0.3})
    swapped = mixture_repair.weighted_mixture(b, a, {"A": 0.3, "B": 0.7})
    np.testing.assert_allclose(direct, swapped)


def test_mixture_repair_atomic_savez_preserves_unmodified_oracle_vectors(tmp_path):
    path = tmp_path / "field.npz"
    payload = {
        "contacts": np.asarray(["A1", "A2"], dtype="U64"),
        "A_canonical_full": np.asarray([1.0, 0.0]),
        "B_canonical_full": np.asarray([0.0, 1.0]),
        "canonical_full_train_prevalence_mixture": np.asarray([0.5, 0.5]),
    }
    before = {
        key: field_builder.vector_sha256(value)
        for key, value in payload.items() if key.startswith(("A_", "B_"))
    }
    payload["canonical_full_train_prevalence_mixture"] = np.asarray([0.8, 0.2])
    mixture_repair.atomic_savez(path, payload)
    with np.load(path, allow_pickle=False) as frozen:
        after = {
            key: field_builder.vector_sha256(frozen[key])
            for key in before
        }
        np.testing.assert_allclose(
            frozen["canonical_full_train_prevalence_mixture"], [0.8, 0.2]
        )
    assert before == after


def test_deduplicated_rollout_expands_unique_starts_in_original_order():
    starts = [np.asarray([2]), np.asarray([1, 3]), np.asarray([2]), np.asarray([1, 3])]
    calls = []

    def fake_rollout(_model, _head, unique, _device):
        calls.append([tuple(value.tolist()) for value in unique])
        return [[value.tolist(), [int(value[0]) + 10]] for value in unique]

    expanded = attenuation_hotfill.deduplicated_rollout_with_size_head(
        None, None, starts, None, rollout_fn=fake_rollout
    )
    assert calls == [[(2,), (1, 3)]]
    assert expanded == [
        [[2], [12]], [[1, 3], [11]], [[2], [12]], [[1, 3], [11]],
    ]


def test_prefix_template_preparation_matches_registered_scalar_decisions():
    ranks = np.asarray([
        [0, 1, 2, 3, -1],
        [1, 0, 3, 2, 4],
        [0, 2, 1, 3, 4],
    ], dtype=np.int16)
    split = np.asarray([1, 1, 2], dtype=np.int8)
    posterior = np.asarray([[0.8, 0.2], [0.35, 0.65], [0.5, 0.5]], dtype=np.float32)
    entropy = np.asarray([0.5, 0.7, 0.9], dtype=np.float32)
    templates = np.asarray([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0, 0.0],
    ], dtype=np.float32)
    temperature = 0.7
    vectorized = template_control.template_decisions(
        ranks, split, posterior, entropy, templates, temperature, selected_split=1,
    )
    scalar = []
    for event_index in np.flatnonzero(split == 1):
        recruited: set[int] = set()
        row = ranks[event_index]
        for rank_index in range(int(row[row >= 0].max())):
            recruited.update(np.flatnonzero(row == rank_index).tolist())
            if rank_index < 2:
                continue
            target = np.flatnonzero(row == rank_index + 1)
            available = np.ones(ranks.shape[1], dtype=bool)
            available[list(recruited)] = False
            log_prob = template_control.template_log_prob(
                templates, posterior[event_index], available, temperature,
            )
            scalar.append(float(-np.mean(log_prob[target])))
    assert np.allclose(vectorized.template_nll, scalar, rtol=0.0, atol=1e-7)


def test_mechanism_mode_mapping_respects_noncollinear_fit_scope():
    assert mechanism.mode_mapping(Path("unused"), {"scope": "own_a"}) == {0: "A", 1: "A"}
    assert mechanism.mode_mapping(Path("unused"), {"scope": "own_b"}) == {0: "B", 1: "B"}


def test_mode_flow_same_cross_analysis_is_shared_fit_only():
    source = Path(mode_flow.__file__).read_text()
    assert 'if str(metrics["scope"]) == "shared"' in source
    assert "expected 42 shared-fit L3 units" in source


def test_vectorized_spearman_null_is_exact_with_ties():
    prediction = np.asarray([0.0, 0.5, 0.5, 1.0, 0.2, 0.9])
    target = np.asarray([4.0, 2.0, 2.0, 1.0, 3.0, 4.0])
    rng = np.random.default_rng(20260813)
    permutations = np.stack([rng.permutation(len(target)) for _ in range(100)])
    observed, vectorized = early_scorer.signed_spearman_permutations(
        prediction, target, permutations
    )
    expected_observed = early_scorer.signed_spearman(prediction, target)
    expected = np.asarray([
        early_scorer.signed_spearman(prediction, target[row]) for row in permutations
    ])
    assert np.isclose(observed, expected_observed, rtol=0, atol=1e-12)
    assert np.allclose(vectorized, expected, rtol=0, atol=1e-7)


def test_spectral_surrogate_preserves_laplacian_power_and_is_deterministic():
    xy = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 1], [3, 0], [3, 1]], float)
    basis, meta = target_freezer.laplacian_basis(xy)
    assert meta["eligible"]
    target = np.asarray([1, 5, 2, 4, 3, 8, 7, 6], float)
    signs = np.ones((2, len(target)), dtype=np.int8)
    signs[1, 1::2] = -1
    surrogate = early_scorer.spectral_surrogates(target, basis, signs)
    coefficient = basis.T @ (target - target.mean())
    assert np.allclose(surrogate[0], target, rtol=0, atol=1e-10)
    assert np.allclose(np.square(basis.T @ (surrogate[1] - surrogate[1].mean())),
                       np.square(coefficient), rtol=0, atol=1e-9)


def test_variogram_surrogates_restore_target_marginal_exactly():
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    xy = np.c_[np.cos(theta), np.sin(theta)]
    target = np.asarray([2, 5, 1, 9, 7, 3, 11, 4, 6, 12, 10, 8], float)
    normals = np.random.default_rng(7).standard_normal((20, len(target)))
    surrogate, fitted_range = early_scorer.variogram_surrogates(target, xy, normals)
    assert fitted_range > 0
    assert np.allclose(np.sort(surrogate, axis=1), np.sort(target)[None, :])


def test_closeout_audit_reconstructs_patient_first_numeric_aggregation():
    observed = pd.DataFrame([
        {"subject": "p", "arm": "a", "metric": 1.5},
        {"subject": "q", "arm": "a", "metric": 3.0},
    ])
    expected = observed.copy()
    result = closeout_audit.compare_numeric_tables(
        observed, expected, ["subject", "arm"]
    )
    assert result["pass"]
    changed = expected.copy()
    changed.loc[0, "metric"] += 0.1
    result = closeout_audit.compare_numeric_tables(
        observed, changed, ["subject", "arm"]
    )
    assert not result["pass"]
    assert any(value.startswith("NUMERIC_MISMATCH") for value in result["failures"])


def test_closeout_audit_detects_snapshot_source_drift(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("value = 1\n")
    digest = closeout_audit.sha256_file(source)
    snapshot = {"source_hashes": {"source": digest}}
    assert closeout_audit.verify_source_snapshot(
        snapshot, {"source": source}
    )["pass"]
    source.write_text("value = 2\n")
    result = closeout_audit.verify_source_snapshot(snapshot, {"source": source})
    assert not result["pass"]
    assert result["failures"] == ["HASH_MISMATCH:source"]


def test_closeout_audit_requires_actual_validation_gain_matching():
    rows = []
    arms = ("L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR")
    for pair in range(126):
        for arm in arms:
            rows.append({
                "subject": f"p{pair % 28}", "fit_id": f"fit{pair // 3}",
                "scope": "shared", "seed": pair % 3, "arm": arm,
                "validation_G3_intact": 1.2 if arm.startswith("L3") else 1.0,
                "validation_G3_matched": 1.0,
                "recurrent_scale": 0.8 if arm.startswith("L3") else 1.0,
            })
    frame = pd.DataFrame(rows)
    assert closeout_audit.assess_gain_matching(frame)["pass"]
    frame.loc[
        (frame.fit_id == "fit0") & frame.arm.str.startswith("L3"),
        "validation_G3_matched",
    ] = 1.02
    failed = closeout_audit.assess_gain_matching(frame)
    assert not failed["pass"]
    assert failed["maximum_relative_error"] > 0.01


def test_closeout_audit_proves_masked_rank_lineage_and_detects_phantom_reentry(tmp_path):
    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    dataset.mkdir()
    fit = cache / "p__shared"
    fit.mkdir(parents=True)
    groups = np.asarray([
        [0, -1, 2, 0],
        [-1, 4, 1, 4],
    ], dtype=np.int16)
    participation = groups >= 0
    source = dataset / "p.npz"
    np.savez_compressed(
        source,
        event_group_ids=groups,
        event_participation=participation.astype(np.uint8),
        contact_names=np.asarray(["A", "B", "C", "D"]),
    )
    expected = closeout_audit.densify_groups(groups[:, [0, 1, 2, 3]])
    np.savez_compressed(fit / "events_raw.npz", ranks=expected)
    (fit / "provenance.json").write_text(__import__("json").dumps({
        "fit_id": "p__shared", "subject": "p",
        "joint_contacts": ["A", "B", "C", "D"],
        "dataset_sha256": closeout_audit.sha256_file(source),
    }))
    # The helper's formal cohort-size gate is intentionally strict, but all
    # rank-lineage diagnostics must be clean in this one-fit fixture.
    result = closeout_audit.assess_masked_rank_lineage(dataset, cache)
    assert result["source_participation_mask_mismatches"] == 0
    assert result["cache_value_mismatches"] == 0
    assert result["nondense_cache_events"] == 0
    with np.load(source, allow_pickle=False) as payload:
        fields = {name: payload[name] for name in payload.files}
    fields["event_participation"] = np.ones_like(participation, dtype=np.uint8)
    np.savez_compressed(source, **fields)
    (fit / "provenance.json").write_text(__import__("json").dumps({
        "fit_id": "p__shared", "subject": "p",
        "joint_contacts": ["A", "B", "C", "D"],
        "dataset_sha256": closeout_audit.sha256_file(source),
    }))
    failed = closeout_audit.assess_masked_rank_lineage(dataset, cache)
    assert failed["source_participation_mask_mismatches"] > 0


def test_preunseal_metric_freezer_fails_closed_on_gain_or_attenuation_gaps():
    gain_rows = []
    arms = ("L2M_MACRO_MATCHED_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR")
    for pair in range(126):
        for arm in arms:
            gain_rows.append({
                "subject": f"p{pair % 28}", "fit_id": f"fit{pair // 3}",
                "scope": "shared", "seed": pair % 3, "arm": arm,
                "validation_G3_intact": 1.2 if arm.startswith("L3") else 1.0,
                "validation_G3_matched": 1.0,
                "recurrent_scale": 0.8 if arm.startswith("L3") else 1.0,
            })
    gain = pd.DataFrame(gain_rows)
    assert metric_freezer.assess_gain_matching(gain)["pass"]
    gain.loc[
        (gain.fit_id == "fit0") & gain.arm.str.startswith("L3"),
        "validation_G3_matched",
    ] = 1.02
    assert not metric_freezer.assess_gain_matching(gain)["pass"]

    targets = ("L1_ADDED", "L2M_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL")
    attenuation_rows = []
    for fit_index in range(42):
        for target in targets:
            for seed in range(3):
                for alpha in (0.25, 0.50, 0.75, 1.00):
                    attenuation_rows.append({
                        "subject": f"p{fit_index % 28}", "fit_id": f"fit{fit_index}",
                        "target": target, "alpha": alpha, "seed": seed,
                        "target_values_read": False,
                    })
    attenuation = pd.DataFrame(attenuation_rows)
    assert metric_freezer.assess_attenuation_coverage(attenuation)["pass"]
    assert not metric_freezer.assess_attenuation_coverage(
        attenuation.iloc[:-1]
    )["pass"]


def test_closeout_audit_proves_formal_suffix_control_uses_frozen_cross_event_null(
    tmp_path,
):
    out = tmp_path / "out"
    fit_id = "p__shared"
    cache = out / "cache" / fit_id
    cache.mkdir(parents=True)
    ranks = np.asarray([[0, 1, 2], [0, 2, 1], [0, 1, 2]], dtype=np.int16)
    split = np.asarray([0, 1, 2], dtype=np.int8)
    np.savez_compressed(cache / "events.npz", ranks=ranks, split=split)
    files = {}
    metrics = []
    for seed in range(3):
        null_ranks = ranks.copy()
        null_ranks[0, 1:] = null_ranks[0, 1:][::-1]
        np.savez_compressed(
            cache / f"events_suffix_null_seed{seed}.npz",
            ranks=null_ranks,
            split=split,
        )
        path = cache / f"events_suffix_null_seed{seed}.npz"
        files[path.name] = {"sha256": closeout_audit.sha256_file(path)}
        metrics.append({
            "fit_id": fit_id, "seed": seed, "arm": "C_L3_ORDER_SHUFFLED",
            "shuffle_audit": {
                "scope": "precomputed_suffix_pairing_train_and_validation_only",
                "events_file_name": path.name,
                "heldout_test_unchanged": True,
                "null_events_sha256": closeout_audit.sha256_file(path),
            },
        })
    manifest = {"cache_records": [{"fit_id": fit_id, "files": files}]}
    result = closeout_audit.assess_suffix_control_lineage(
        out, manifest, metrics, expected_fits=1,
    )
    assert result["pass"]
    assert result["heldout_test_exact_units"] == 3
    broken = [dict(row) for row in metrics]
    broken[0] = dict(broken[0], shuffle_audit=dict(broken[0]["shuffle_audit"]))
    broken[0]["shuffle_audit"]["scope"] = "train_and_validation_only"
    assert not closeout_audit.assess_suffix_control_lineage(
        out, manifest, broken, expected_fits=1,
    )["pass"]


def test_closeout_audit_proves_own_views_share_the_same_all_event_task(tmp_path):
    out = tmp_path / "out"
    rows = []
    for scope in ("own_a", "own_b"):
        fit_id = f"p__{scope}"
        cache = out / "cache" / fit_id
        cache.mkdir(parents=True)
        np.savez_compressed(
            cache / "events.npz",
            ranks=np.asarray([[0, 1, -1], [1, -1, 0]], dtype=np.int16),
            split=np.asarray([0, 2], dtype=np.int8),
            mode=np.asarray([0, 1], dtype=np.int8),
        )
        rows.append({"subject": "p", "fit_id": fit_id, "scope": scope})
    census = pd.DataFrame(rows)
    result = closeout_audit.assess_geometry_view_event_scope(out, census)
    # Production freezes 14 patients.  This one-patient fixture proves exact
    # equality but deliberately fails only the cohort denominator.
    assert result["exact_all_event_geometry_view_pairs"] == 1
    assert not result["pass"]
    with np.load(out / "cache/p__own_b/events.npz", allow_pickle=False) as payload:
        altered = {name: payload[name] for name in payload.files}
    altered["mode"] = np.asarray([1, 1], dtype=np.int8)
    np.savez_compressed(out / "cache/p__own_b/events.npz", **altered)
    failed = closeout_audit.assess_geometry_view_event_scope(out, census)
    assert failed["exact_all_event_geometry_view_pairs"] == 0
    assert any(value.endswith(":mode") for value in failed["failures"])


def _attenuation_semantic_fixture():
    rows = []
    for target, n_valid in (("L3_ADDED", 1), ("L3_MATCHED_LOCAL", 240)):
        draw_ids = [0] if target == "L3_ADDED" else list(range(16))
        for alpha in (0.25, 0.50, 0.75, 1.00):
            for draw in draw_ids:
                rows.append({
                    "subject": "p", "fit_id": "p__shared", "target": target,
                    "seed": 0, "alpha": alpha, "draw": draw,
                    "n_valid_matched_draws": n_valid,
                    "inferential_eligible": True,
                    "target_mask_sha256": f"mask-{draw}",
                    "contact_nll": 1.0, "local_nll": 1.0, "distal_nll": 1.1,
                    "local_damage": 0.01, "distal_damage": 0.02,
                    "distal_selectivity": 0.01,
                    "rollout_spearman": np.nan if (alpha == 1.0 and draw == 0) else 0.5,
                    "rollout_spearman_n": 0 if (alpha == 1.0 and draw == 0) else 10,
                    "target_values_read": False,
                })
    return pd.DataFrame(rows)


def test_attenuation_draw_semantics_distinguish_rollout_collapse_from_metric_failure():
    frame = _attenuation_semantic_fixture()
    frozen = metric_freezer.assess_attenuation_draw_semantics(frame)
    audited = closeout_audit.assess_attenuation_draw_semantics(frame)
    assert frozen["pass"] and audited["pass"]
    assert frozen["rollout_undefined_rows"] == 2
    broken = frame.copy()
    broken.loc[broken.index[0], "contact_nll"] = np.nan
    assert not metric_freezer.assess_attenuation_draw_semantics(broken)["pass"]
    assert not closeout_audit.assess_attenuation_draw_semantics(broken)["pass"]


def test_attenuation_draw_semantics_reject_missing_dose_or_mask_drift():
    frame = _attenuation_semantic_fixture()
    dropped = frame.drop(frame[(frame.target == "L3_ADDED") & (frame.alpha == 0.75)].index)
    assert not metric_freezer.assess_attenuation_draw_semantics(dropped)["pass"]
    drift = frame.copy()
    index = drift[(drift.target == "L3_MATCHED_LOCAL") & (drift.draw == 0)].index[0]
    drift.loc[index, "target_mask_sha256"] = "different"
    assert not closeout_audit.assess_attenuation_draw_semantics(drift)["pass"]


def test_closeout_audit_reconstructs_broadband_target_from_registered_source(tmp_path):
    out = tmp_path / "out"
    source_root = tmp_path / "source"
    early = out / "early_ictal"
    early.mkdir(parents=True)
    source_root.mkdir()
    routing = pd.DataFrame([
        {"subject": "p", "seizure_idx": 4},
        {"subject": "p", "seizure_idx": 9},
    ])
    routing.to_csv(out / "EARLY_ICTAL_ROUTING_METADATA.csv", index=False)
    meta = {
        "channels": ["A", "B", "C"],
        "band_broad_1_150": [1.0, 150.0],
        "t_window": [0.0, 10.0],
        "line_noise_masked_1_150": True,
        "feature": "bb150_auc_0_10s (mean baseline-robust-z 1-150Hz over [0,10]s)",
    }
    import json
    (source_root / "p.json").write_text(json.dumps(meta))
    source_values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.savez_compressed(
        source_root / "p.npz",
        channels=np.asarray(meta["channels"]),
        bb150_auc__4=source_values[0], bb150_auc__9=source_values[1],
    )
    target = early / "p.npz"
    np.savez_compressed(
        target, contacts=np.asarray(["C", "A"]),
        all_seizure_broadband_energy=source_values[:, [2, 0]],
        median_broadband_energy=np.median(source_values[:, [2, 0]], axis=0),
        n_seizures=np.asarray(2, dtype=np.int32),
        time_window_s=np.asarray([0.0, 10.0], dtype=np.float32),
        frequency_band_hz=np.asarray([1.0, 150.0], dtype=np.float32),
    )
    pd.DataFrame([{
        "subject": "p", "path": str(target),
        "sha256": closeout_audit.sha256_file(target),
    }]).to_csv(early / "EARLY_ICTAL_TARGET_MANIFEST.csv", index=False)
    result = closeout_audit.assess_broadband_target_lineage(
        out, source_root, expected_patients=1, expected_seizures=2,
    )
    assert result["pass"]
    with np.load(target, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    arrays["all_seizure_broadband_energy"] = arrays["all_seizure_broadband_energy"] + 1
    np.savez_compressed(target, **arrays)
    manifest = pd.read_csv(early / "EARLY_ICTAL_TARGET_MANIFEST.csv")
    manifest.loc[0, "sha256"] = closeout_audit.sha256_file(target)
    manifest.to_csv(early / "EARLY_ICTAL_TARGET_MANIFEST.csv", index=False)
    failed = closeout_audit.assess_broadband_target_lineage(
        out, source_root, expected_patients=1, expected_seizures=2,
    )
    assert not failed["pass"]
    assert any(value.startswith("TARGET_VALUE_MISMATCH") for value in failed["failures"])


def test_distance_bin_null_preserves_each_frozen_group():
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radii = np.linspace(1.0, 2.0, 12)
    xy = np.c_[radii * np.cos(theta), radii * np.sin(theta)]
    groups, meta = target_freezer.distance_bin_groups(xy)
    assert meta["eligible"]
    permutation = target_freezer.grouped_permutations(12, groups, 30, 11)
    for row in permutation:
        for group in groups:
            assert set(row[group]) == set(group)


def test_local_control_descriptor_includes_full_directed_degree_profiles():
    local = np.zeros((5, 5), dtype=bool)
    local[1, 0] = True
    local[2, 0] = True
    local[3, 2] = True
    descriptor = arm_attenuation.match_local_control_subsets.__globals__["edge_set_descriptors"](
        local, np.ones((5, 5)), np.c_[np.arange(5), np.zeros(5)], np.eye(5)
    )
    assert descriptor["source_degree_profile"] == [0, 0, 1, 2, 0] or sorted(
        descriptor["source_degree_profile"]
    ) == [0, 0, 0, 1, 2]
    assert descriptor["source_degree_profile"] == sorted(descriptor["source_degree_profile"])
    assert descriptor["target_degree_profile"] == sorted(descriptor["target_degree_profile"])
    assert sum(descriptor["source_degree_profile"]) == 3
    assert sum(descriptor["target_degree_profile"]) == 3
