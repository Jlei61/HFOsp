from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.analyse_topic5_lbss_interictal_v0_2 import (
    aggregate_patient,
    paired_test,
    require_no_rec_equivalence,
)
from scripts.complete_topic5_lbss_closeout_audits_v0_2 import adjudicate_detectability
from scripts.plot_topic5_lbss_figure6_v0_2 import assert_no_label_overlap
from scripts.launch_topic5_lbss_v0_2 import acquire_stage_lock
from scripts.score_topic5_lbss_early_ictal_v0_2 import align
from scripts.train_topic5_lbss_unit_v0_2 import decision_rows
from scripts.run_topic5_lbss_detectability_v0_2 import (
    build_ground_truth_model,
    simulate_events,
)
from scripts.run_topic5_lbss_attenuation_v0_2 import (
    attenuation_unit_cache_path,
    load_attenuation_unit_cache,
    sha256_file,
    write_attenuation_unit_cache,
)
from scripts.run_topic5_lbss_spatial_search_v0_4 import (
    ARMS,
    choose_confirmation,
    DEV_FITS,
    SCREEN_CONFIGS,
    SEEDS,
    metric_path,
    screen_jobs,
    summarize_confirmation,
    summarize_screen,
    target_must_be_sealed,
)
from scripts.freeze_topic5_lbss_postprocess_snapshot_v0_2 import dependency_closure
from scripts.summarize_topic5_lbss_claims_v0_2 import (
    attenuation_damage_auc,
    holm,
    main as claim_summary_main,
)
from scripts.summarize_topic5_lbss_claims_v0_3 import (
    L0,
    L1,
    L2,
    L3,
    summarize as summarize_claims_v0_3,
)
from scripts.summarize_topic5_lbss_topology_plateau_v0_3 import (
    summarize as summarize_topology_plateau_v0_3,
)
from src.topic5_lbss_rnn_v0_2 import (
    LBSSConfig,
    LBSSModel,
    build_pool_contract,
    checkpoint_is_eligible,
    clear_recurrent_optimizer_state,
    derange_rank_sets,
    derange_training_validation_only,
    semantic_snapshot_epochs,
    source_balanced_sample,
    strong_component_audit,
    transition_frontier_distance,
)
from src.topic5_lbss_analysis_v0_2 import (
    attenuate_mask,
    upsert_figure_readme,
    endpoint_density,
    match_local_control_subsets,
)
from src.topic5_wiring_economy_rnn import build_event_tensors


def line_distance(n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)
    return np.abs(x[:, None] - x[None, :])


def test_local_backbone_is_symmetric_strong_and_pool_disjoint():
    contract = build_pool_contract(line_distance(16), density=0.2)
    mask = contract.local_mask.astype(bool)
    assert np.array_equal(mask, mask.T)
    audit = strong_component_audit(mask)
    assert audit["all_nodes_one_strong_component"]
    assert audit["contact_supported_pairwise_reachability"] == 1.0
    assert audit["minimum_in_degree"] >= 1
    assert audit["minimum_out_degree"] >= 1
    assert not np.any(mask & contract.extra_local_pool.astype(bool))
    assert not np.any(mask & contract.nonlocal_pool.astype(bool))
    assert not np.any(contract.extra_local_pool.astype(bool) & contract.nonlocal_pool.astype(bool))


def test_nonlocal_cutoff_multiplier_changes_only_candidate_partition():
    distance = line_distance(20)
    narrow = build_pool_contract(
        distance, density=0.15, added_fraction=0.10, r_local_multiplier=1.5,
    )
    wide = build_pool_contract(
        distance, density=0.15, added_fraction=0.10, r_local_multiplier=2.5,
    )
    assert np.array_equal(narrow.local_mask, wide.local_mask)
    assert narrow.k_added == wide.k_added
    assert narrow.r_local_mm < wide.r_local_mm
    assert int(narrow.nonlocal_pool.sum()) > int(wide.nonlocal_pool.sum())
    assert int(narrow.extra_local_pool.sum()) < int(wide.extra_local_pool.sum())
    with pytest.raises(ValueError, match="greater than one"):
        build_pool_contract(distance, r_local_multiplier=1.0)


def test_target_free_spatial_screen_is_frozen_and_balanced():
    jobs = screen_jobs()
    assert len(SCREEN_CONFIGS) == 13
    assert len(jobs) == 13 * 3 * 3
    assert {job["arm"] for job in jobs} == {"L3_LOCAL_PLUS_LEARNED_LR"}
    assert all(job["phase"] == "screen" for job in jobs)
    counts = {}
    for job in jobs:
        counts[job["config_id"]] = counts.get(job["config_id"], 0) + 1
    assert set(counts.values()) == {9}


def test_target_free_spatial_search_refuses_any_target_marker(tmp_path):
    target_must_be_sealed(tmp_path)
    (tmp_path / "TARGET_ACCESS_AUDIT.json").write_text("{}\n")
    with pytest.raises(RuntimeError, match="must precede early-ictal access"):
        target_must_be_sealed(tmp_path)


def test_spatial_screen_selects_only_a_target_free_interictal_improvement(tmp_path):
    out = tmp_path
    search_name = "development_spatial_search_v0_4"
    search = out / search_name
    (search / "configs").mkdir(parents=True)
    for config_id in SCREEN_CONFIGS:
        for fit in DEV_FITS:
            for seed in SEEDS:
                path = metric_path(
                    out, search_name, "screen", config_id, fit,
                    "L3_LOCAL_PLUS_LEARNED_LR", seed,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                gain = 0.05 if config_id == "radius_1p5" else 0.0
                path.write_text(json.dumps({
                    "subject": fit.split("__")[0],
                    "test": {"contact_nll": 1.5 - gain},
                    "distance_bins": {"distal": {"contact_nll": 1.7 - gain}},
                    "rollout": {"seed_removed_spearman_median": 0.5},
                    "converged": True,
                    "best_checkpoint_eligible": True,
                    "hit_ceiling": False,
                    "target_values_read": False,
                }))
    decision = summarize_screen(out, search, search_name)
    assert decision["screen_units"] == 13 * 3 * 3
    assert decision["retained_one_factor_levels"] == {
        "r_local_multiplier": "radius_1p5",
    }
    assert decision["joint_config"]["r_local_multiplier"] == 1.5
    assert decision["target_values_read"] is False


def test_spatial_screen_collapses_seeds_before_development_fits(tmp_path, monkeypatch):
    """Optimization seeds must not be treated as nine biological replicates."""
    out = tmp_path
    search_name = "development_spatial_search_v0_4"
    search = out / search_name
    (search / "configs").mkdir(parents=True)
    gains = {
        DEV_FITS[0]: [0.0, 100.0, 101.0],  # fit median 100
        DEV_FITS[1]: [1.0, 2.0, 3.0],      # fit median 2
        DEV_FITS[2]: [4.0, 5.0, 6.0],      # fit median 5
    }
    for config_id in SCREEN_CONFIGS:
        for fit in DEV_FITS:
            for seed, gain in zip(SEEDS, gains[fit]):
                if config_id != "radius_1p5":
                    gain = 0.0
                path = metric_path(
                    out, search_name, "screen", config_id, fit,
                    "L3_LOCAL_PLUS_LEARNED_LR", seed,
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "subject": fit.split("__")[0],
                    "test": {"contact_nll": 200.0 - gain},
                    "distance_bins": {"distal": {"contact_nll": 200.0 - gain}},
                    "rollout": {"seed_removed_spearman_median": 0.5},
                    "converged": True,
                    "best_checkpoint_eligible": True,
                    "hit_ceiling": False,
                    "target_values_read": False,
                }))
    monkeypatch.setattr(
        "scripts.run_topic5_lbss_spatial_search_v0_4.plot_screen",
        lambda *_args, **_kwargs: None,
    )
    summarize_screen(out, search, search_name)
    summary = pd.read_csv(search / "screen_summary.csv").set_index("config_id")
    # Median of per-fit medians is 5; a pooled 9-seed median would be 4.
    assert summary.loc["radius_1p5", "median_distal_gain"] == 5.0
    assert summary.loc["radius_1p5", "n_development_fits"] == 3


def test_base_diagnostic_cannot_trigger_duplicate_formal_confirmation(tmp_path, monkeypatch):
    """The already-completed v0.3 base is not a searched replacement model."""
    out = tmp_path
    search_name = "development_spatial_search_v0_4"
    search = out / search_name
    search.mkdir()
    (search / "DEVELOPMENT_SPATIAL_SELECTION.json").write_text(json.dumps({
        "selected_for_matched_confirmation": ["base"],
        "target_values_read": False,
    }))
    for fit in DEV_FITS:
        for arm in ARMS:
            for seed in SEEDS:
                path = metric_path(out, search_name, "confirm", "base", fit, arm, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                selected = arm == "L3_LOCAL_PLUS_LEARNED_LR"
                path.write_text(json.dumps({
                    "subject": fit.split("__")[0],
                    "test": {"contact_nll": 1.0 if selected else 1.1},
                    "distance_bins": {"distal": {"contact_nll": 1.0 if selected else 1.1}},
                    "rollout": {"seed_removed_spearman_median": 0.6 if selected else 0.5},
                    "converged": True,
                    "best_checkpoint_eligible": True,
                    "hit_ceiling": False,
                    "target_values_read": False,
                }))
    monkeypatch.setattr(
        "scripts.run_topic5_lbss_spatial_search_v0_4.plot_confirmation",
        lambda *_args, **_kwargs: None,
    )
    decision = summarize_confirmation(out, search, search_name)
    assert decision["configurations"][0]["selective_nonlocal_confirmed"] is True
    assert decision["selected_config_id"] is None
    assert decision["full_cohort_confirmation_required"] is False
    assert decision["verdict"] == "NO_SELECTIVE_NONLOCAL_CONFIGURATION_IN_FROZEN_SEARCH"


def test_confirmation_candidates_require_prespecified_distal_gain(tmp_path):
    search = tmp_path / "development_spatial_search_v0_4"
    search.mkdir()
    pd.DataFrame([
        {
            "config_id": "base", "eligible": True, "median_distal_gain": 0.0,
            "median_overall_gain": 0.0, "median_rollout_gain": 0.0,
        },
        {
            "config_id": "radius_1p5", "eligible": True, "median_distal_gain": 0.0019,
            "median_overall_gain": 0.02, "median_rollout_gain": 0.02,
        },
    ]).to_csv(search / "screen_summary.csv", index=False)
    (search / "SCREEN_DECISION.json").write_text(json.dumps({
        "joint_config_id": "base", "target_values_read": False,
    }))
    decision = choose_confirmation(
        tmp_path, search, "development_spatial_search_v0_4"
    )
    assert decision["selected_for_matched_confirmation"] == ["base"]


@pytest.mark.parametrize("state_dim", [2, 4])
def test_lbss_multi_state_node_keeps_the_same_spatial_mask_and_readout(state_dim):
    distance = line_distance(8)
    contract = build_pool_contract(distance, density=0.30, added_fraction=0.25)
    model = LBSSModel(LBSSConfig(
        arm="L3_LOCAL_PLUS_LEARNED_LR",
        n_contacts=8,
        n_nodes=8,
        state_dim=state_dim,
        observation_operator=np.eye(8, dtype=np.float32),
        node_distance_mm=distance,
        local_mask=contract.local_mask,
        extra_local_pool=contract.extra_local_pool,
        nonlocal_pool=contract.nonlocal_pool,
        k_added=contract.k_added,
        seed=3,
    ))
    x = torch.zeros(2, 3, 8)
    x[:, 0, 0] = 1
    recruited = torch.zeros_like(x)
    valid = torch.ones(2, 3, dtype=torch.bool)
    logits, stop = model(x, recruited, valid)
    assert logits.shape == (2, 3, 8)
    assert stop.shape == (2, 3)
    assert model.recurrent.shape[-1] == 8 * state_dim
    assert model.node_mask.shape == (8, 8)


def test_source_balanced_sampler_uses_only_pool_and_is_deterministic():
    pool = build_pool_contract(line_distance(20), density=0.15).nonlocal_pool
    a = source_balanced_sample(pool, 12, seed=5)
    b = source_balanced_sample(pool, 12, seed=5)
    assert np.array_equal(a, b)
    assert int(a.sum()) == 12
    assert np.all(a.astype(bool) <= pool.astype(bool))


def _make_model(arm: str, seed: int = 2) -> LBSSModel:
    distance = line_distance(8)
    contract = build_pool_contract(distance, density=0.30, added_fraction=0.25)
    return LBSSModel(LBSSConfig(
        arm=arm,
        n_contacts=8,
        n_nodes=8,
        observation_operator=np.eye(8, dtype=np.float32),
        node_distance_mm=distance,
        local_mask=contract.local_mask,
        extra_local_pool=contract.extra_local_pool,
        nonlocal_pool=contract.nonlocal_pool,
        k_added=contract.k_added,
        seed=seed,
    ))


def test_batched_distance_decisions_match_single_event_evaluation():
    model = _make_model("L3_LOCAL_PLUS_LEARNED_LR")
    ranks = np.asarray([
        [0, 1, 2, 3, -1, -1, -1, -1],
        [0, 1, 1, 2, 3, -1, -1, -1],
        [0, 1, 2, 2, 3, 4, -1, -1],
        [0, 1, 2, 3, 4, 5, 6, 7],
    ], dtype=np.int16)
    tensors = build_event_tensors(ranks)
    indices = np.arange(len(ranks))
    xy = np.c_[np.arange(8, dtype=float), np.zeros(8)]
    one = decision_rows(model, tensors, ranks, indices, xy, torch.device("cpu"), batch_size=1)
    many = decision_rows(model, tensors, ranks, indices, xy, torch.device("cpu"), batch_size=4)
    assert len(one) == len(many)
    for left, right in zip(one, many):
        assert left.keys() == right.keys()
        for key in left:
            if isinstance(left[key], float):
                assert left[key] == pytest.approx(right[key], abs=1e-7)
            else:
                assert left[key] == right[key]


def test_lbss_arms_share_parameters_and_l2_l3_initial_mask():
    models = {arm: _make_model(arm) for arm in (
        "L0_LOCAL_ONLY",
        "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR",
        "L3_LOCAL_PLUS_LEARNED_LR",
    )}
    assert torch.equal(models["L2_LOCAL_PLUS_RANDOM_LR"].added_mask,
                       models["L3_LOCAL_PLUS_LEARNED_LR"].added_mask)
    for model in models.values():
        assert torch.equal(model.recurrent, models["L0_LOCAL_ONLY"].recurrent)
        assert torch.equal(model.input_gain, models["L0_LOCAL_ONLY"].input_gain)
        assert torch.equal(model.contact_bias, models["L0_LOCAL_ONLY"].contact_bias)
        assert torch.all(model.node_mask >= model.local_mask)


def test_rewiring_preserves_local_budget_and_new_edge_grace():
    model = _make_model("L3_LOCAL_PLUS_LEARNED_LR")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model.recurrent.sum().backward()
    optimizer.step()
    local_before = model.local_mask.clone()
    n_added = int(model.added_mask.sum())
    result = model.rewire_added(0.5, np.random.default_rng(4))
    clear_recurrent_optimizer_state(model, optimizer, result["touched"])
    assert result["n_drop"] > 0
    assert torch.equal(model.local_mask, local_before)
    assert int(model.added_mask.sum()) == n_added
    assert not bool(((model.added_mask > 0) & (model.local_mask > 0)).any())
    newly_grown = result["touched"] & (model.added_mask > 0)
    assert torch.all(model.edge_age[newly_grown] == 0)
    second = model.rewire_added(1.0, np.random.default_rng(5))
    assert not bool((second["touched"] & newly_grown).any())


@pytest.mark.parametrize("state_dim", [2, 4])
def test_rewiring_resets_full_multistate_blocks(state_dim):
    """A changed node edge owns every state-channel-to-channel weight."""
    distance = line_distance(8)
    contract = build_pool_contract(distance, density=0.30, added_fraction=0.25)
    model = LBSSModel(LBSSConfig(
        arm="L3_LOCAL_PLUS_LEARNED_LR",
        n_contacts=8,
        n_nodes=8,
        state_dim=state_dim,
        observation_operator=np.eye(8, dtype=np.float32),
        node_distance_mm=distance,
        local_mask=contract.local_mask,
        extra_local_pool=contract.extra_local_pool,
        nonlocal_pool=contract.nonlocal_pool,
        k_added=contract.k_added,
        seed=2,
    ))
    result = model.rewire_added(0.5, np.random.default_rng(4))
    assert result["n_drop"] > 0
    expanded = torch.kron(
        result["touched"].float(), torch.ones(state_dim, state_dim)
    ).bool()
    assert tuple(expanded.shape) == tuple(model.recurrent.shape[-2:])
    assert torch.count_nonzero(model.recurrent[:, expanded]) == 0


def test_runtime_state_restores_mask_age_and_counter():
    model = _make_model("L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL")
    model.rewire_added(0.5, np.random.default_rng(4))
    state = model.runtime_state()
    restored = _make_model("L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL")
    restored.restore_runtime_state(state)
    assert torch.equal(restored.added_mask, model.added_mask)
    assert torch.equal(restored.edge_age, model.edge_age)
    assert int(restored.rewire_counter) == int(model.rewire_counter)


def test_derangement_keeps_first_and_has_no_later_fixed_rank():
    ranks = np.array([
        [0, 0, 1, 2, 2, 3],
        [0, 1, -1, -1, -1, -1],
    ], dtype=np.int16)
    shuffled, audit = derange_rank_sets(ranks, seed=7)
    assert np.array_equal(shuffled == 0, ranks == 0)
    for old_rank in (1, 2, 3):
        assert not np.array_equal(shuffled[0] == old_rank, ranks[0] == old_rank)
    assert np.array_equal(shuffled[1], ranks[1])
    assert audit["n_effectively_shuffled"] == 1
    assert audit["n_unchanged_due_to_length_2"] == 1
    assert audit["mean_kendall_distance_from_true_order"] > 0


def test_order_shuffle_never_changes_heldout_test_targets():
    ranks = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 3],
        [0, 3, 1, 2],
    ], dtype=np.int16)
    split = np.array([0, 1, 2], dtype=np.int8)
    shuffled, audit = derange_training_validation_only(ranks, split, seed=9)
    assert not np.array_equal(shuffled[:2], ranks[:2])
    assert np.array_equal(shuffled[2], ranks[2])
    assert audit["heldout_test_unchanged"]
    assert audit["heldout_test_sha256_before"] == audit["heldout_test_sha256_after"]


def test_frontier_distance_uses_novel_contacts_not_centroid_shift():
    xy = np.array([[-10.0, 0.0], [10.0, 0.0], [0.0, 10.0], [0.0, 0.0]])
    value = transition_frontier_distance([0, 1], [0, 1], [2], xy)
    assert np.isclose(value, np.sqrt(200.0))
    assert np.isnan(transition_frontier_distance([0, 1], [0, 1, 2], [2], xy))


def test_snapshot_and_checkpoint_contract():
    epochs = semantic_snapshot_epochs(10, 40)
    assert epochs["SNAPSHOT_AFTER_WARMUP"] == 9
    assert epochs["SNAPSHOT_REWIRE_1_3"] == 23
    assert epochs["SNAPSHOT_REWIRE_2_3"] == 36
    assert epochs["SNAPSHOT_MASK_FREEZE"] == 49
    assert not checkpoint_is_eligible(48, epochs["SNAPSHOT_MASK_FREEZE"])
    assert checkpoint_is_eligible(49, epochs["SNAPSHOT_MASK_FREEZE"])


def test_detectability_generator_uses_real_geometry_schema():
    distance = line_distance(8).astype(np.float32)
    plane = {
        "D_mm": distance,
        "H": np.eye(8, dtype=np.float32),
    }
    model, planted = build_ground_truth_model(plane, seed=3)
    ranks = simulate_events(model, n_events=12, seed=4)
    assert ranks.shape == (12, 8)
    assert int(planted.sum()) == model.lbss_config.k_added
    assert np.all((ranks >= 0).sum(axis=1) >= 5)
    assert np.all((ranks == 0).sum(axis=1) == 1)


def test_patient_aggregation_is_seed_then_fit_then_patient():
    table = pd.DataFrame({
        "subject": ["p"] * 6,
        "fit_id": ["a"] * 3 + ["b"] * 3,
        "arm": ["L3"] * 6,
        "value": [1.0, 2.0, 100.0, 5.0, 6.0, 7.0],
    })
    result = aggregate_patient(table, ["value"])
    # median fit a=2, median fit b=6, then patient mean=4.
    assert np.isclose(result.loc[0, "value"], 4.0)


def test_paired_test_separates_numerical_ties():
    result = paired_test(np.array([1.0, -1.0, 1e-12, 0.0]))
    assert result["n_positive"] == 1
    assert result["n_negative"] == 1
    assert result["n_tied"] == 2


def test_launcher_lock_blocks_duplicate_and_recovers_stale(tmp_path):
    lock = acquire_stage_lock(tmp_path, "formal")
    with np.testing.assert_raises(RuntimeError):
        acquire_stage_lock(tmp_path, "formal")
    lock.write_text('{"pid": 999999999, "stage": "formal"}\n')
    recovered = acquire_stage_lock(tmp_path, "formal")
    assert recovered.exists()
    assert (tmp_path / "FORMAL_STALE_LOCK_RECOVERY.json").exists()


def test_attenuation_only_changes_requested_active_edges():
    model = _make_model("L3_LOCAL_PLUS_LEARNED_LR")
    before = model.recurrent.detach().clone()
    target = model.added_mask.detach().cpu().numpy().astype(bool)
    attenuate_mask(model, target, 0.5)
    assert torch.allclose(model.recurrent[:, model.added_mask.bool()],
                          0.5 * before[:, model.added_mask.bool()])
    assert torch.equal(model.recurrent[:, model.local_mask.bool()],
                       before[:, model.local_mask.bool()])


def test_endpoint_density_uses_weighted_source_and_target_semantics():
    strength = np.zeros((3, 3), float)
    strength[2, 0] = 2.0  # source column 0 -> target row 2
    result = endpoint_density(strength, strength > 0, np.eye(3))
    assert np.argmax(result["source_node"]) == 0
    assert np.argmax(result["target_node"]) == 2
    assert np.isclose(result["source_contact"].sum(), 1.0)


def test_local_control_matching_is_deterministic_and_uses_active_local_edges():
    n = 12
    xy = np.c_[np.arange(n, dtype=float), np.zeros(n)]
    local = np.zeros((n, n), bool)
    for source in range(n):
        local[(source - 1) % n, source] = True
        local[(source + 1) % n, source] = True
    target = np.zeros_like(local)
    for source in range(0, n, 3):
        target[(source + 5) % n, source] = True
    kwargs = dict(
        local_mask=local,
        target_mask=target,
        strength=np.ones((n, n)),
        nodes_xy_mm=xy,
        observation_operator=np.eye(n),
        seed=9,
        max_candidate_draws=2_000,
        keep_valid=50,
        evaluate_best=5,
    )
    first = match_local_control_subsets(**kwargs)
    second = match_local_control_subsets(**kwargs)
    assert first["selected_hashes"] == second["selected_hashes"]
    assert first["selected_masks"].shape == (5, n, n)
    assert np.all(first["selected_masks"].astype(bool) <= local)
    assert np.all(first["selected_masks"].sum(axis=(1, 2)) == int(target.sum()))


def test_seed_removed_scoring_keeps_fixed_contact_support():
    names = np.asarray(["A1", "A2", "A3"])
    values = np.asarray([0.8, np.nan, 0.2])
    assert np.array_equal(
        align(values, names, ["A3", "A2", "A1"], "seed_removed"),
        np.asarray([0.2, 0.0, 0.8]),
    )
    canonical = align(values, names, ["A3", "A2", "A1"], "canonical_full")
    assert np.isnan(canonical[1])


def test_claim_family_holm_is_monotone_in_sorted_pvalues():
    adjusted = holm({"a": 0.01, "b": 0.03, "c": 0.20})
    assert np.isclose(adjusted["a"], 0.03)
    assert np.isclose(adjusted["b"], 0.06)
    assert np.isclose(adjusted["c"], 0.20)


def test_early_ictal_attenuation_auc_uses_each_arms_own_intact_field():
    rows = []
    for target, arm, base in (
        ("L1_ADDED", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL", 0.20),
        ("L2_ADDED", "L2_LOCAL_PLUS_RANDOM_LR", 0.30),
        ("L3_ADDED", "L3_LOCAL_PLUS_LEARNED_LR", 0.40),
        ("L3_MATCHED_LOCAL", "L3_LOCAL_PLUS_LEARNED_LR", 0.40),
    ):
        rows.append({"subject": "p", "primary": True, "endpoint": "seed_removed",
                     "condition": f"INTACT|{arm}", "all_contact_margin": base})
        for alpha in (0.25, 0.50, 0.75, 1.00):
            rows.append({"subject": "p", "primary": True, "endpoint": "seed_removed",
                         "condition": f"ATTEN|{target}|{alpha:.2f}",
                         "all_contact_margin": base - alpha})
    result = attenuation_damage_auc(pd.DataFrame(rows), "seed_removed")
    assert set(result.target) == {"L1_ADDED", "L2_ADDED", "L3_ADDED", "L3_MATCHED_LOCAL"}
    assert np.allclose(result.damage_auc, 0.5)


def test_attenuation_eligibility_uses_minimum_across_seeds_and_fits(tmp_path):
    from scripts.run_topic5_lbss_attenuation_v0_2 import aggregate_metrics

    (tmp_path / "attenuation").mkdir()
    rows = []
    for seed, eligible, draws in ((0, True, 500), (1, False, 3), (2, True, 500)):
        for alpha in (0.25, 0.50, 0.75, 1.00):
            rows.append({
                "subject": "p", "fit_id": "p__shared", "scope": "shared",
                "arm": "L3_LOCAL_PLUS_LEARNED_LR", "seed": seed,
                "target": "L3_MATCHED_LOCAL", "alpha": alpha,
                "contact_nll": 1.0, "top1": 0.5,
                "local_nll": 1.0, "intermediate_nll": 1.0, "distal_nll": 1.1,
                "rollout_spearman": 0.5, "rollout_reach_mm": 10.0,
                "intact_contact_nll": 0.9, "intact_local_nll": 0.9,
                "intact_intermediate_nll": 0.9, "intact_distal_nll": 0.9,
                "intact_rollout_spearman": 0.6, "inferential_eligible": eligible,
                "n_valid_matched_draws": draws,
            })
    aggregate_metrics(tmp_path, pd.DataFrame(rows), pd.DataFrame())
    auc = pd.read_csv(tmp_path / "attenuation" / "attenuation_patient_auc.csv")
    assert len(auc) == 1
    assert not bool(auc.iloc[0].inferential_eligible)
    assert int(auc.iloc[0].n_valid_matched_draws_min) == 3


def test_attenuation_unit_cache_is_atomic_restart_safe_and_input_bound(tmp_path):
    metrics_path = tmp_path / "fit" / "L3" / "seed2" / "metrics.json"
    metrics_path.parent.mkdir(parents=True)
    metrics_path.write_text(json.dumps({"fit_id": "fit_a", "seed": 2}) + "\n")
    field = tmp_path / "field.npz"
    np.savez_compressed(field, vector=np.arange(4, dtype=float))
    rollout = tmp_path / "rollout.json.gz"
    import gzip
    with gzip.open(rollout, "wt", encoding="utf-8") as stream:
        json.dump({"records": [1, 2]}, stream)
    rows = [{"contact_nll": 1.2, "target_values_read": False}]
    fields = [{
        "path": str(field), "field_sha256": sha256_file(field),
        "rollout_path": str(rollout), "rollout_sha256": sha256_file(rollout),
        "target_values_read": False,
    }]
    target = "L3_ADDED"
    cache = attenuation_unit_cache_path(tmp_path, metrics_path, target)
    write_attenuation_unit_cache(cache, metrics_path, target, rows, fields)
    assert not list(cache.parent.glob("*.tmp.*"))
    assert load_attenuation_unit_cache(cache, metrics_path, target) == (rows, fields)

    # Never mix cached counterfactuals from a changed checkpoint producer.
    metrics_path.write_text(
        json.dumps({"fit_id": "fit_a", "seed": 2, "changed": True}) + "\n"
    )
    assert load_attenuation_unit_cache(cache, metrics_path, target) is None


def test_postprocess_snapshot_dependency_closure_is_self_contained():
    closure = set(dependency_closure(("scripts/run_topic5_lbss_attenuation_v0_2.py",)))
    assert {
        Path("scripts/train_topic5_lbss_unit_v0_2.py"),
        Path("scripts/build_topic5_rnn_motif_fields_v0_4.py"),
        Path("src/topic5_lbss_analysis_v0_2.py"),
        Path("src/topic5_lbss_rnn_v0_2.py"),
    }.issubset(closure)


def test_claim_adjudicator_runs_on_patient_first_tables(tmp_path, monkeypatch):
    for marker in (
        "INTERICTAL_ANALYSIS_COMPLETE.json", "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATION_COMPLETE.json", "EARLY_ICTAL_SCORING_COMPLETE.json",
    ):
        (tmp_path / marker).write_text("{}\n")
    arms = (
        "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
        "C_L3_ORDER_SHUFFLED",
    )
    interictal = []
    for subject in ("p1", "p2"):
        for index, arm in enumerate(arms):
            interictal.append({
                "subject": subject, "arm": arm,
                "no_rec_contact_nll": 2.0,
                "test_contact_nll": 1.8 - 0.02 * index,
                "distal_contact_nll": 1.9 - 0.03 * index,
                "distal_n": 25,
            })
    pd.DataFrame(interictal).to_csv(tmp_path / "interictal_per_patient.csv", index=False)
    pathway = tmp_path / "pathway_analysis"; pathway.mkdir()
    pd.DataFrame({
        "subject": ["p1", "p2"],
        "endpoint_dissimilarity_beyond_proposal": [0.2, 0.1],
        "effective_dissimilarity_beyond_proposal": [0.3, 0.2],
    }).to_csv(pathway / "true_vs_shuffle_patient_patterns.csv", index=False)
    attenuation = tmp_path / "attenuation"; attenuation.mkdir()
    auc_rows = []
    for subject in ("p1", "p2"):
        for target, value in (
            ("L1_ADDED", 0.05), ("L2_ADDED", 0.04),
            ("L3_ADDED", 0.20), ("L3_MATCHED_LOCAL", 0.01),
        ):
            auc_rows.append({
                "subject": subject, "target": target, "auc_distal_selectivity": value,
                "inferential_eligible": True, "n_valid_matched_draws_min": 500,
            })
    pd.DataFrame(auc_rows).to_csv(attenuation / "attenuation_patient_auc.csv", index=False)
    early = tmp_path / "early_ictal"; early.mkdir()
    early_rows = []
    target_arm = {
        "L1_ADDED": arms[1], "L2_ADDED": arms[2],
        "L3_ADDED": arms[3], "L3_MATCHED_LOCAL": arms[3],
    }
    for subject in ("p1", "p2"):
        for endpoint in ("canonical_full", "seed_removed"):
            for index, arm in enumerate(arms[:4]):
                early_rows.append({
                    "subject": subject, "primary": True, "endpoint": endpoint,
                    "condition": f"INTACT|{arm}", "all_contact_margin": 0.1 + 0.02 * index,
                })
            for target, arm in target_arm.items():
                base = next(row["all_contact_margin"] for row in early_rows
                            if row["subject"] == subject and row["endpoint"] == endpoint
                            and row["condition"] == f"INTACT|{arm}")
                for alpha in (0.25, 0.50, 0.75, 1.00):
                    early_rows.append({
                        "subject": subject, "primary": True, "endpoint": endpoint,
                        "condition": f"ATTEN|{target}|{alpha:.2f}",
                        "all_contact_margin": base - alpha * (0.2 if target == "L3_ADDED" else 0.05),
                    })
    pd.DataFrame(early_rows).to_csv(early / "early_ictal_per_patient_condition.csv", index=False)
    monkeypatch.setattr("sys.argv", ["claim-summary", "--out-root", str(tmp_path)])
    claim_summary_main()
    result = json.loads((tmp_path / "LBSS_CLAIM_SUMMARY.json").read_text())
    assert result["n_interictal_patients"] == 2
    assert set(result["claim_B_holm_family"]) == {
        "L3_vs_L0_LOCAL_ONLY_distal",
        "L3_vs_L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL_distal",
        "L3_vs_L2_LOCAL_PLUS_RANDOM_LR_distal",
    }
    assert result["claim_C_holm_family"]["selected_nonlocal_vs_matched_local_attenuation_dd"]["median"] > 0


def test_order_shuffle_is_a_cyclic_rotation_not_a_uniform_derangement():
    """Pin the control's real strength so nobody reads it as a uniform shuffle.

    ``derange_rank_sets`` rolls each event's rank labels by a random non-zero
    offset.  That is fixed-point free, but it preserves the relative cyclic
    order and every adjacent transition except the wrap, so it destroys less
    order information than a uniform random derangement would.
    """
    ranks = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int16)
    shuffled, _ = derange_rank_sets(ranks, seed=3)
    n_later = int((ranks[0] > 0).sum())
    rotations = {
        tuple(np.roll(np.arange(1, n_later + 1), shift))
        for shift in range(1, n_later)
    }
    assert tuple(shuffled[0][1:].tolist()) in rotations


def test_v03_claim_summary_keeps_d2_controls_separate(tmp_path):
    """D2 must not test a biased patient-wise minimum against zero."""
    for marker in (
        "INTERICTAL_ANALYSIS_COMPLETE.json", "PATHWAY_ANALYSIS_COMPLETE.json",
        "ATTENUATION_COMPLETE.json", "EARLY_ICTAL_SCORING_COMPLETE.json",
    ):
        (tmp_path / marker).write_text("{}\n")
    arms = (L0, L1, L2, L3, "C_L3_ORDER_SHUFFLED")
    interictal = []
    for subject in ("p1", "p2"):
        for index, arm in enumerate(arms):
            interictal.append({
                "subject": subject, "arm": arm, "no_rec_contact_nll": 2.0,
                "test_contact_nll": 1.8 - 0.01 * index,
                "distal_contact_nll": 1.9 - 0.02 * index, "distal_n": 25,
            })
    pd.DataFrame(interictal).to_csv(tmp_path / "interictal_per_patient.csv", index=False)
    pathway = tmp_path / "pathway_analysis"; pathway.mkdir()
    pd.DataFrame({
        "subject": ["p1", "p2"],
        "endpoint_dissimilarity_beyond_proposal": [0.2, 0.1],
        "effective_dissimilarity_beyond_proposal": [0.3, 0.2],
    }).to_csv(pathway / "true_vs_shuffle_patient_patterns.csv", index=False)
    attenuation = tmp_path / "attenuation"; attenuation.mkdir()
    pd.DataFrame([
        {"subject": subject, "target": target,
         "auc_distal_selectivity": value, "inferential_eligible": True}
        for subject in ("p1", "p2")
        for target, value in (("L1_ADDED", 0.05), ("L2_ADDED", 0.04),
                              ("L3_ADDED", 0.20), ("L3_MATCHED_LOCAL", 0.01))
    ]).to_csv(attenuation / "attenuation_patient_auc.csv", index=False)
    early = tmp_path / "early_ictal"; early.mkdir()
    rows = []
    for subject in ("p1", "p2"):
        for endpoint in ("canonical_full", "seed_removed"):
            for index, arm in enumerate(arms[:4]):
                rows.append({
                    "subject": subject, "primary": True, "endpoint": endpoint,
                    "family": "intact", "arm": arm,
                    "condition": f"INTACT|{arm}", "all_contact_margin": 0.1 + 0.02 * index,
                })
            for target, arm in (("L1_ADDED", L1), ("L2_ADDED", L2),
                                ("L3_ADDED", L3), ("L3_MATCHED_LOCAL", L3)):
                base = next(row["all_contact_margin"] for row in rows
                            if row["subject"] == subject and row["endpoint"] == endpoint
                            and row["condition"] == f"INTACT|{arm}")
                for alpha in (0.25, 0.50, 0.75, 1.00):
                    rows.append({
                        "subject": subject, "primary": True, "endpoint": endpoint,
                        "family": "attenuated", "arm": arm,
                        "condition": f"ATTEN|{target}|{alpha:.2f}",
                        "all_contact_margin": base - alpha * (0.2 if target == "L3_ADDED" else 0.05),
                    })
    pd.DataFrame(rows).to_csv(early / "early_ictal_per_patient_condition.csv", index=False)

    result = summarize_claims_v0_3(tmp_path)
    keys = set(result["claim_D_holm_family"])
    assert "D2_L3_seed_removed_better_than_all_controls" not in keys
    assert {
        "D2_L3_vs_L0_LOCAL_ONLY_seed_removed",
        "D2_L3_vs_L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL_seed_removed",
        "D2_L3_vs_L2_LOCAL_PLUS_RANDOM_LR_seed_removed",
    }.issubset(keys)
    claim_c = result["claim_C_holm_family"]
    assert "coarse_pattern_difference_beyond_proposal" not in claim_c
    assert {
        "endpoint_pattern_difference_beyond_proposal",
        "effective_pattern_difference_beyond_proposal",
    }.issubset(claim_c)
    assert claim_c["endpoint_pattern_difference_beyond_proposal"]["median"] == pytest.approx(0.15)
    assert claim_c["effective_pattern_difference_beyond_proposal"]["median"] == pytest.approx(0.25)
    assert result["minimum_of_controls_used_for_inference"] is False


def test_topology_plateau_summary_is_patient_first_and_target_free(tmp_path):
    arms = (L0, L1, L2, L3)
    interictal, fields = [], []
    for subject, offset in (("p1", 0.0), ("p2", 0.1), ("p3", -0.1)):
        for index, arm in enumerate(arms):
            interictal.append({
                "subject": subject, "arm": arm,
                "test_contact_nll": 2.0 - 0.1 * index + offset,
                "distal_contact_nll": 2.2 - 0.1 * index + offset,
                "rollout_spearman": 0.3 + 0.1 * index - offset,
            })
            fields.append({
                "subject": subject, "arm": arm,
                "canonical_empirical_r": 0.2 + 0.1 * index,
                "seed_removed_empirical_r": 0.1 + 0.1 * index,
                "canonical_contrast_empirical_r": 0.3 + 0.1 * index,
                "seed_removed_contrast_empirical_r": 0.2 + 0.1 * index,
            })
    pd.DataFrame(interictal).to_csv(tmp_path / "interictal_per_patient.csv", index=False)
    pd.DataFrame(fields).to_csv(tmp_path / "model_field_patient_metrics.csv", index=False)
    result = summarize_topology_plateau_v0_3(tmp_path)
    assert result["n_patients"] == 3
    assert result["target_values_read"] is False
    assert result["early_ictal_values_used"] is False
    assert set(result["endpoints"]) == {
        "overall_contact_nll", "distal_contact_nll", "free_rollout_spearman",
        "canonical_interictal_field_r", "seed_removed_interictal_field_r",
        "canonical_ab_contrast_r", "seed_removed_ab_contrast_r",
    }
    # Keys are emitted in combination order, so L0-vs-L3 has a negative
    # advantage when L3 is better.
    reverse = result["endpoints"]["overall_contact_nll"][
        "pairwise_positive_means_left_arm_better"
    ][f"{L0}_vs_{L3}"]
    assert reverse["median"] < 0


def test_readme_upsert_replaces_one_section_and_keeps_the_others(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("### a.png\n\nfirst\n\n### b.png\n\nsecond\n")
    upsert_figure_readme(readme, "b.png", "### b.png\n\nrewritten\n")
    text = readme.read_text()
    assert text.count("### a.png") == 1 and "first" in text
    assert text.count("### b.png") == 1 and "rewritten" in text and "second" not in text
    upsert_figure_readme(readme, "b.png", "### b.png\n\nrewritten\n")
    assert readme.read_text().count("### b.png") == 1


def test_figure_label_overlap_check_catches_a_label_drawn_over_another_panel():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(3.0, 2.0))
    axes[0].plot([0, 1], [0, 1])
    axes[1].plot([0, 1], [0, 1])
    axes[1].set_ylabel("a label long enough to reach the neighbouring panel")
    axes[1].yaxis.set_label_coords(-1.4, 0.5)
    with pytest.raises(RuntimeError, match="overlap"):
        assert_no_label_overlap(fig)
    axes[1].yaxis.set_label_coords(-0.18, 0.5)
    axes[1].set_ylabel("short")
    fig.subplots_adjust(wspace=0.9)
    assert_no_label_overlap(fig)
    plt.close(fig)


def test_detectability_sensitivity_requires_every_geometry_and_a_margin(tmp_path):
    """A control that fails one geometry, or matches the real effect, is not sensitivity."""
    root = tmp_path / "synthetic_detectability"
    root.mkdir(parents=True)
    rows = [
        {"fit_id": "g0", "l3_minus_l0_distal_gain": 0.004, "l3_minus_l1_distal_gain": 0.30,
         "l3_minus_l2_distal_gain": 0.30, "true_minus_shuffle_distal_gain": 0.20,
         "l3_attenuation_distal_nll_increase": 0.10},
        {"fit_id": "g1", "l3_minus_l0_distal_gain": 0.100, "l3_minus_l1_distal_gain": 0.30,
         "l3_minus_l2_distal_gain": 0.30, "true_minus_shuffle_distal_gain": 0.20,
         "l3_attenuation_distal_nll_increase": 0.10},
        {"fit_id": "g2", "l3_minus_l0_distal_gain": -0.020, "l3_minus_l1_distal_gain": 0.30,
         "l3_minus_l2_distal_gain": 0.30, "true_minus_shuffle_distal_gain": -0.10,
         "l3_attenuation_distal_nll_increase": 0.10},
    ]
    (root / "FUNCTIONAL_DETECTABILITY_SUMMARY.json").write_text(json.dumps({
        "rows": rows, "functional_class_detected": True,
    }))
    real = {"comparisons": {
        "L3_vs_L0_distal": {"median": 0.0023},
        "L3_vs_L1_distal": {"median": 0.0056},
        "L3_vs_L2_distal": {"median": -0.0060},
    }}
    verdict = adjudicate_detectability(tmp_path, real)
    # One geometry is negative for both of these, so neither may be claimed.
    assert "l3_minus_l0_distal_gain" in verdict["sensitivity_not_demonstrated_for"]
    assert "true_minus_shuffle_distal_gain" in verdict["sensitivity_not_demonstrated_for"]
    assert "l3_minus_l1_distal_gain" in verdict["sensitivity_demonstrated_for"]
    assert verdict["criteria"]["l3_minus_l0_distal_gain"]["n_geometries_positive"] == 2


def test_claim_a_refuses_the_imported_comparator_without_its_equivalence_audit(tmp_path):
    with pytest.raises(RuntimeError, match="only-comparator-audit"):
        require_no_rec_equivalence(tmp_path)
    (tmp_path / "NO_REC_EQUIVALENCE_AUDIT.json").write_text(json.dumps({"verdict": "NOT_EQUIVALENT"}))
    with pytest.raises(RuntimeError, match="NOT_EQUIVALENT"):
        require_no_rec_equivalence(tmp_path)
    (tmp_path / "NO_REC_EQUIVALENCE_AUDIT.json").write_text(
        json.dumps({"verdict": "EQUIVALENT_ENOUGH_FOR_MATCHED_CONTRAST"}))
    assert require_no_rec_equivalence(tmp_path)["verdict"] == "EQUIVALENT_ENOUGH_FOR_MATCHED_CONTRAST"


def test_figure_closeout_audit_marker_follows_selected_primary_root():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_topic5_lbss_full_tissue_figure_closeout_v0_3.py").read_text()
    assert 'primary / "CLOSEOUT_AUDIT.json"' in source
    assert 'out / "CLOSEOUT_AUDIT.json"' not in source


def test_spatial_decision_watcher_resolves_search_before_waiting_for_pretarget():
    repo = Path(__file__).resolve().parents[1]
    source = (repo / "scripts" / "run_topic5_lbss_spatial_decision_watcher_v0_4.py").read_text()
    assert "while not screen_marker.exists()" in source
    assert "while not pretarget_marker.exists()" in source
    assert "pretarget_marker.exists() and screen_marker.exists()" not in source
    assert '"spatial_screen_complete": screen_marker.exists()' in source
    assert 'stages = ("summarize-screen",)' in source
    assert 'stages = ("initialize", "screen", "summarize-screen")' not in source
    assert source.index("while not screen_marker.exists()") < source.index('stages = ("summarize-screen",)')
    assert source.index('stages = ("summarize-screen",)') < source.index("while not pretarget_marker.exists()")
