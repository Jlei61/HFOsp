from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from scripts.analyse_topic5_lbss_interictal_v0_2 import aggregate_patient, paired_test
from scripts.launch_topic5_lbss_v0_2 import acquire_stage_lock
from scripts.score_topic5_lbss_early_ictal_v0_2 import align
from scripts.run_topic5_lbss_detectability_v0_2 import (
    build_ground_truth_model,
    simulate_events,
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
    endpoint_density,
    match_local_control_subsets,
)


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
