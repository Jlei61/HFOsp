"""Contract tests for the WE-SLP-RNN v0.3 model and its SET rewiring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_wiring_economy_rnn import (  # noqa: E402
    WEConfig,
    WEModel,
    active_edge_count,
    build_event_tensors,
    initial_mask,
    next_rank_stop_loss,
    zeta_schedule,
)


def _geometry(n_contacts=6, n_nodes=20, seed=0):
    rng = np.random.default_rng(seed)
    contacts = rng.uniform(-20, 20, size=(n_contacts, 2))
    nodes = rng.uniform(-20, 20, size=(n_nodes, 2))
    d = np.linalg.norm(contacts[:, None] - nodes[None], axis=-1)
    weights = np.exp(-(d ** 2) / (2 * 6.0 ** 2))
    weights[d > 18.0] = 0.0
    weights[weights.sum(1) == 0, 0] = 1.0
    H = weights / weights.sum(1, keepdims=True)
    D = np.linalg.norm(nodes[:, None] - nodes[None], axis=-1)
    return H.astype(np.float32), D.astype(np.float32)


def _model(arm="SPATIAL_SET", cell="rnn", state_dim=1, n_nodes=20, density=0.1, seed=0):
    H, D = _geometry(n_nodes=n_nodes, seed=seed)
    return WEModel(WEConfig(arm=arm, cell=cell, n_contacts=H.shape[0], n_nodes=n_nodes,
                            state_dim=state_dim, density=density, seed=seed,
                            observation_operator=H, node_distance_mm=D))


def test_edge_count_is_conserved_across_rewiring():
    model = _model()
    before = int(model.node_mask.sum())
    assert before == active_edge_count(20, 0.1)
    for _ in range(5):
        model.rewire(0.2)
        assert int(model.node_mask.sum()) == before


def test_frozen_mask_stops_rewiring():
    model = _model()
    model.freeze_mask()
    before = model.node_mask.clone()
    assert model.rewire(0.5) == 0
    assert torch.equal(model.node_mask, before)


def test_no_self_loops_anywhere():
    for arm in ("DENSE_TISSUE", "RANDOM_SET", "SPATIAL_SET"):
        model = _model(arm=arm)
        assert float(torch.diagonal(model.node_mask).sum()) == 0.0
        model.rewire(0.3)
        assert float(torch.diagonal(model.node_mask).sum()) == 0.0


def test_distance_biased_growth_makes_shorter_edges_than_uniform_growth():
    lengths = {}
    for arm in ("SPATIAL_SET", "RANDOM_SET"):
        per_seed = []
        for seed in range(6):
            model = _model(arm=arm, n_nodes=40, seed=seed)
            d = model.D_mm.numpy()
            per_seed.append(float(d[model.node_mask.numpy() > 0].mean()))
        lengths[arm] = np.mean(per_seed)
    assert lengths["SPATIAL_SET"] < 0.75 * lengths["RANDOM_SET"]


def test_initial_mask_density_matches_the_fixed_resource():
    _, D = _geometry(n_nodes=30)
    for spatial in (True, False):
        mask = initial_mask(30, 0.1, D, spatial=spatial, seed=3)
        assert mask.sum() == active_edge_count(30, 0.1)


def test_one_mask_drives_all_three_gru_matrices():
    model = _model(cell="gru")
    masked = model.masked_recurrent()
    zero = model.node_mask == 0
    for gate in range(3):
        assert float(masked[gate][zero].abs().sum()) == 0.0


def test_units_cannot_talk_except_through_the_mask():
    # Drive one contact and check the units that respond are exactly those the
    # observation operator touches -- no dense input path smuggles activity
    # across the plane before the recurrent mask gets a say.
    model = _model()
    with torch.no_grad():
        model.node_mask.zero_()
        x = torch.zeros(1, model.n_contacts)
        x[0, 0] = 1.0
        h = model._step(torch.zeros(1, model.n_nodes * model.state_dim), x)
        reached = (h.abs() > 1e-8).numpy().reshape(-1)
        touched = (model.H.numpy()[0] > 0)
    assert reached.tolist() == touched.tolist()


def test_wiring_cost_scales_inversely_with_the_cohort_distance_unit():
    H, D = _geometry(n_nodes=24)
    kwargs = dict(arm="SPATIAL_SET", n_contacts=H.shape[0], n_nodes=24, seed=1,
                  observation_operator=H, node_distance_mm=D)
    near = WEModel(WEConfig(d0_mm=10.0, **kwargs))
    far = WEModel(WEConfig(d0_mm=20.0, **kwargs))
    assert float(near.wiring_cost()) == pytest.approx(2.0 * float(far.wiring_cost()), rel=1e-5)


def test_state_dim_two_masks_by_block():
    model = _model(state_dim=2, n_nodes=12)
    expanded = model._expanded_mask()
    assert expanded.shape == (24, 24)
    for i in range(12):
        for j in range(12):
            block = expanded[2 * i:2 * i + 2, 2 * j:2 * j + 2]
            assert float(block.min()) == float(block.max()) == float(model.node_mask[i, j])


def test_edge_strength_pools_gru_gates_into_one_number_per_pair():
    model = _model(cell="gru", n_nodes=8)
    strength = model.edge_strength()
    expected = model.recurrent.detach().pow(2).sum(0).sqrt()
    assert torch.allclose(strength, expected, atol=1e-6)


def test_zeta_anneals_to_exactly_zero_before_the_freeze():
    values = [zeta_schedule(e, warmup=10, rewire_epochs=40, zeta0=0.2) for e in range(60)]
    assert all(v == 0.0 for v in values[:10])
    assert values[10] == pytest.approx(0.2)
    assert all(v == 0.0 for v in values[50:])
    assert values[49] < 1e-3


def test_grown_edges_start_from_zero_weight():
    model = _model(n_nodes=24)
    before = model.node_mask.clone()
    model.rewire(0.25)
    grown = (model.node_mask > 0) & (before == 0)
    assert int(grown.sum()) > 0
    strength = model.edge_strength()
    assert float(strength[grown].abs().max()) == 0.0


def test_event_tensors_never_ask_the_model_to_repredict_a_recruited_contact():
    ranks = np.array([[0, 1, 2, -1], [0, 0, 1, -1]], np.int16)
    t = build_event_tensors(ranks)
    assert not bool((t["available"] & (t["recruited"] > 0)).any())
    assert t["is_last"][0].tolist() == [False, False, True]
    assert t["is_last"][1].tolist() == [False, True, False]


def test_loss_ignores_contacts_outside_the_available_support():
    torch.manual_seed(0)
    t = build_event_tensors(np.array([[0, 1, 2, -1]], np.int16))
    logits = torch.zeros(1, 3, 4, requires_grad=True)
    loss, _, _ = next_rank_stop_loss(logits, torch.zeros(1, 3), t["target"],
                                     t["available"], t["valid"], t["is_last"])
    loss.backward()
    assert torch.isfinite(loss)
    assert float(logits.grad[0, 0, 0].abs()) == 0.0  # contact 0 already recruited at t=0


def test_static_arm_has_no_recurrence_and_no_graph():
    model = WEModel(WEConfig(arm="STATIC_CONTACT", n_contacts=5))
    assert model.graph_snapshot() == {}
    t = build_event_tensors(np.array([[0, 1, 2, -1, -1]], np.int16))
    logits, stops = model(t["x"], t["recruited"], t["valid"])
    assert logits.shape == (1, 3, 5) and stops.shape == (1, 3)
    assert torch.allclose(logits[0, 0], logits[0, 1])  # nothing carries over time


def test_dense_arm_keeps_every_off_diagonal_edge():
    model = _model(arm="DENSE_TISSUE", n_nodes=16)
    assert int(model.node_mask.sum()) == 16 * 15
