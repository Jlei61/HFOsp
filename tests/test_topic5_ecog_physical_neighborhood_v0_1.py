from __future__ import annotations

import json

import numpy as np
import torch

from scripts.build_topic5_ecog_full_grid_rank_cache_v0_1 import densify_rank_sets
from scripts.run_topic5_ecog_patch_necessity_v0_1 import ensemble_nll, event_patch_coverage
from scripts.rerun_topic5_ecog_stale_training_units_v0_1 import is_stale
from scripts.train_topic5_ecog_graph_unit_v0_1 import top1_hits
from scripts.summarize_topic5_ecog_patch_necessity_v0_1 import stratified_randomization_test

from src.topic5_ecog_physical_neighborhood_v0_1 import (
    build_fixed_grid_model,
    degree_class_permutation,
    degree_preserving_random_mask,
    enumerate_square_patches,
    graph_audit,
    matched_dispersed_directed_edge_sets,
    matched_dispersed_edge_sets,
    patch_edge_mask,
    true_grid_mask,
)
from src.topic5_wiring_economy_rnn import build_event_tensors


def names_8x8() -> list[str]:
    return [f"G{row}{col}" for row in "ABCDEFGH" for col in range(1, 9)]


def test_true_four_neighbour_grid_edge_count() -> None:
    mask = true_grid_mask(names_8x8())
    assert mask.shape == (64, 64)
    assert int(mask.sum()) == 224
    assert np.array_equal(mask, mask.T)
    assert sorted(np.unique(mask.sum(axis=0)).tolist()) == [2, 3, 4]


def test_missing_gc1_is_not_bridged() -> None:
    names = [name for name in names_8x8() if name != "GC1"]
    mask = true_grid_mask(names)
    index = {name: idx for idx, name in enumerate(names)}
    assert mask[index["GB1"], index["GD1"]] == 0
    assert int(mask.sum()) == 218


def test_wrong_grid_preserves_per_contact_degree_and_spectrum() -> None:
    true = true_grid_mask(names_8x8())
    wrong = degree_class_permutation(true, seed=17)
    assert np.array_equal(true.sum(axis=0), wrong.sum(axis=0))
    assert np.allclose(np.linalg.eigvalsh(true), np.linalg.eigvalsh(wrong))
    assert graph_audit(wrong, true)["true_edge_overlap_fraction"] < 0.55


def test_degree_random_preserves_degree_and_connectivity() -> None:
    true = true_grid_mask(names_8x8())
    random = degree_preserving_random_mask(true, seed=19, swaps_per_edge=5)
    audit = graph_audit(random, true)
    assert np.array_equal(true.sum(axis=0), random.sum(axis=0))
    assert audit["connected"] is True
    assert audit["true_edge_overlap_fraction"] < 0.55


def test_two_microsteps_allow_current_input_to_reach_a_neighbour() -> None:
    names = names_8x8()
    mask = true_grid_mask(names)
    source = names.index("GD4")
    neighbour = names.index("GD5")
    x = torch.zeros(1, 1, 64, requires_grad=True)
    x.data[0, 0, source] = 1.0
    recruited = torch.zeros_like(x)
    recruited.data[0, 0, source] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)

    model_one = build_fixed_grid_model(names, mask, seed=3, microsteps=1)
    logits_one, _ = model_one(x, recruited, valid)
    grad_one = torch.autograd.grad(logits_one[0, 0, neighbour], x, retain_graph=True)[0]

    model_two = build_fixed_grid_model(names, mask, seed=3, microsteps=2)
    logits_two, _ = model_two(x, recruited, valid)
    grad_two = torch.autograd.grad(logits_two[0, 0, neighbour], x)[0]
    assert abs(float(grad_one[0, 0, source])) < 1e-12
    assert abs(float(grad_two[0, 0, source])) > 1e-8


def test_complete_patches_do_not_fill_missing_contact() -> None:
    full = enumerate_square_patches(names_8x8(), side=2)
    missing = enumerate_square_patches(
        [name for name in names_8x8() if name != "GC1"], side=2
    )
    assert len(full) == 49
    assert len(missing) == 47
    assert all(len(nodes) == 4 for _, nodes in missing)


def test_dispersed_controls_match_edge_count_and_degree_classes() -> None:
    names = names_8x8()
    graph = true_grid_mask(names)
    _, nodes = enumerate_square_patches(names, side=2)[24]
    lesion = patch_edge_mask(graph, nodes).edge_mask
    rng = np.random.default_rng(23)
    weight = rng.normal(size=graph.shape) * graph
    controls, audits = matched_dispersed_edge_sets(
        graph, weight, lesion, n_controls=4, seed=29, candidates_per_control=256
    )
    assert len(controls) == len(audits) == 4
    assert all(int(mask.sum()) == int(lesion.sum()) for mask in controls)
    assert all(int(audit["largest_component_nodes"]) <= 4 for audit in audits)


def test_vectorized_lesion_engine_matches_unmodified_model() -> None:
    names = names_8x8()
    graph = true_grid_mask(names)
    model = build_fixed_grid_model(names, graph, seed=31, microsteps=2).eval()
    ranks = np.full((3, 64), -1, dtype=np.int16)
    ranks[0, [0, 1, 9, 18]] = [0, 1, 2, 3]
    ranks[1, [10, 11, 12, 20]] = [0, 1, 1, 2]
    ranks[2, [30, 31, 39, 47, 55]] = [0, 1, 2, 3, 4]
    batch = build_event_tensors(ranks)
    with torch.no_grad():
        logits, _ = model(batch["x"], batch["recruited"], batch["valid"])
        masked = logits.masked_fill(~batch["available"], -1e9)
        expected = -(torch.log_softmax(masked, -1) * batch["target"]).sum(-1)
        expected /= batch["target"].sum(-1).clamp_min(1.0)
        observed = ensemble_nll(model, model.node_mask.unsqueeze(0), batch)[0]
    predict = batch["valid"] & ~batch["is_last"]
    assert torch.allclose(expected[predict], observed[predict], atol=1e-6, rtol=1e-6)


def test_tied_competition_ranks_become_consecutive_rank_sets() -> None:
    rank = np.asarray([
        [0, 0],
        [1, 2],
        [1, 4],
        [3, -1],
    ], dtype=np.int16)
    dense = densify_rank_sets(rank)
    assert dense[:, 0].tolist() == [0, 1, 1, 2]
    assert dense[:, 1].tolist() == [0, 1, 2, -1]


def test_top1_counts_any_contact_in_a_tied_next_set() -> None:
    logits = torch.tensor([[[0.1, 0.2, 3.0, 2.0]]])
    target = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    available = torch.ones_like(target, dtype=torch.bool)
    assert bool(top1_hits(logits, target, available)[0, 0])

    logits[0, 0, 0] = 4.0
    assert not bool(top1_hits(logits, target, available)[0, 0])


def test_patch_randomization_keeps_patch_by_seed_strata() -> None:
    rng = np.random.default_rng(41)
    strata = rng.normal(0.0, 0.05, size=(8, 3, 33))
    strata[:, :, 0] += 1.0
    observed, p_value, low, high = stratified_randomization_test(
        strata, seed=43, n_randomizations=2000
    )
    assert observed > high
    assert p_value < 0.01
    assert low < high


def test_inbound_directed_controls_match_orientation_classes_and_avoid_patch() -> None:
    names = names_8x8()
    graph = true_grid_mask(names)
    _, nodes = enumerate_square_patches(names, side=2)[24]
    involved = np.zeros(len(names), dtype=bool)
    involved[list(nodes)] = True
    lesion = graph.astype(bool) & involved[:, None] & ~involved[None, :]
    rng = np.random.default_rng(47)
    weights = rng.normal(size=graph.shape)
    controls, audits = matched_dispersed_directed_edge_sets(
        graph, weights, lesion, forbidden_nodes=nodes,
        n_controls=4, seed=53, candidates_per_control=256,
    )
    assert len(controls) == len(audits) == 4
    source_degree = graph.sum(axis=0)
    target_degree = graph.sum(axis=1)
    expected = sorted(
        (int(source_degree[source]), int(target_degree[target]))
        for target, source in np.argwhere(lesion)
    )
    for control in controls:
        observed = sorted(
            (int(source_degree[source]), int(target_degree[target]))
            for target, source in np.argwhere(control)
        )
        assert observed == expected
        assert not np.any(control[list(nodes), :])
        assert not np.any(control[:, list(nodes)])


def test_stale_training_detection_requires_current_cpu_contract(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    assert not is_stale(summary, "cpu")
    current = {
        "training_device_type": "cpu",
        "top1_contract": "top_prediction_is_any_member_of_tied_next_rank_set_v0.1",
        "batch_size": 512,
        "microsteps": 2,
        "state_dim": 1,
    }
    summary.write_text(json.dumps(current))
    assert not is_stale(summary, "cpu")
    current["top1_contract"] = None
    summary.write_text(json.dumps(current))
    assert is_stale(summary, "cpu")


def test_first_entry_coverage_excludes_later_reentry() -> None:
    # Patch node 0 is first reached at rank 1 and reappears at rank 3. Only the
    # rank-0 -> rank-1 decision is a first entry.
    ranks = np.asarray([[1, 3, 0, 2]], dtype=np.int16)
    event_count, entering = event_patch_coverage(
        ranks, patch_nodes=(0, 1), lesion_mode="inbound_first_entry"
    )
    assert event_count == 1
    assert entering == 1
