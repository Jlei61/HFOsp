"""TDD suite for the Topic 5 spatial latent propagation RNN v0.1.

Every test names the spec clause it guards.  Spec:
``docs/superpowers/specs/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.topic5_spatial_latent_rnn import (
    LATENT_ARMS,
    ModelConfig,
    SLPModel,
    build_event_tensors,
    next_set_stop_loss,
)
from src.topic5_virtual_seeg_operator import (
    MIN_NODES_PER_CONTACT,
    SUPPORT_SIGMA,
    build_observation_operator,
    hop_reachability,
    kernel_sigma_mm,
    knn_edge_mask,
    nearest_node,
    node_count,
    normalised_distance,
    resolve_node_count,
    sample_latent_nodes,
)


def _toy_plane(n_shafts: int = 3, per_shaft: int = 5, pitch: float = 4.0):
    """A small multi-shaft montage, the geometry this cohort actually has."""
    xy = []
    for s in range(n_shafts):
        for i in range(per_shaft):
            xy.append([i * pitch, s * 3.0 * pitch])
    return np.asarray(xy, float)


# --------------------------------------------------------------------------
# T1 / T2 -- observation operator (spec 4.2)
# --------------------------------------------------------------------------

def test_t1_operator_rows_sum_to_one():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, node_count(len(xy)), sigma, seed=0)
    H = build_observation_operator(xy, nodes, sigma)
    assert H.shape == (len(xy), len(nodes))
    np.testing.assert_allclose(H.sum(axis=1), 1.0, atol=1e-12)


def test_t2_operator_support_is_local():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, node_count(len(xy)), sigma, seed=0)
    H = build_observation_operator(xy, nodes, sigma)
    d = np.linalg.norm(xy[:, None, :] - nodes[None, :, :], axis=-1)
    assert np.all(H[d > SUPPORT_SIGMA * sigma] == 0.0)
    # and the operator must not have collapsed onto a single node per contact
    assert np.all((H > 0).sum(axis=1) >= 2)


def test_t2b_operator_raises_when_a_contact_sees_nothing():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    far_nodes = xy.mean(axis=0)[None, :] + np.array([[1e4, 1e4]])
    with pytest.raises(ValueError, match="observe no latent node"):
        build_observation_operator(xy, far_nodes, sigma)


# --------------------------------------------------------------------------
# T4 -- determinism (spec 4.1)
# --------------------------------------------------------------------------

def test_t4_node_sampling_is_seed_deterministic():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    a = sample_latent_nodes(xy, 24, sigma, seed=7)
    b = sample_latent_nodes(xy, 24, sigma, seed=7)
    c = sample_latent_nodes(xy, 24, sigma, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_node_count_follows_the_frozen_rule():
    assert node_count(5) == 24     # floor
    assert node_count(8) == 32     # 4C
    assert node_count(15) == 60    # 4C
    assert node_count(52) == 64    # cap


def test_nodes_stay_inside_the_dilated_domain():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, node_count(len(xy)), sigma, seed=3)
    d = np.linalg.norm(nodes[:, None, :] - xy[None, :, :], axis=-1)
    assert np.all(d.min(axis=1) <= SUPPORT_SIGMA * sigma + 1e-9)


# --------------------------------------------------------------------------
# wiring-cost helpers (spec 5) and the hop diagnostic (spec 4.4)
# --------------------------------------------------------------------------

def test_normalised_distance_has_unit_median_off_diagonal():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, 24, sigma, seed=1)
    d = normalised_distance(nodes)
    off = d[~np.eye(len(nodes), dtype=bool)]
    assert np.isclose(np.median(off), 1.0)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)


def test_knn_mask_has_exact_out_degree_and_no_self_loops():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, 24, sigma, seed=1)
    mask = knn_edge_mask(nodes, k=6)
    assert np.all(mask.sum(axis=1) == 6)
    assert not np.any(np.diag(mask))


def test_hop_reachability_grows_with_hops_on_a_chain():
    # a directed chain 0 -> 1 -> 2 -> 3: one hop reaches only neighbours
    n = 4
    mask = np.zeros((n, n), bool)
    for i in range(n - 1):
        mask[i, i + 1] = True
    anchors = list(range(n))
    transitions = [(0, 1), (0, 2), (0, 3)]
    r1 = hop_reachability(mask, anchors, transitions, k_hops=1)
    r2 = hop_reachability(mask, anchors, transitions, k_hops=2)
    r3 = hop_reachability(mask, anchors, transitions, k_hops=3)
    assert r1 == pytest.approx(1 / 3)
    assert r2 == pytest.approx(2 / 3)
    assert r3 == pytest.approx(1.0)


def test_nearest_node_is_an_index_per_contact():
    xy = _toy_plane()
    sigma = kernel_sigma_mm(xy)
    nodes = sample_latent_nodes(xy, 24, sigma, seed=1)
    idx = nearest_node(xy, nodes)
    assert idx.shape == (len(xy),)
    assert idx.min() >= 0 and idx.max() < len(nodes)


def test_resolve_node_count_guarantees_a_neighbourhood_when_contacts_are_dense():
    # 40 contacts on a fine lattice: the spec's 64-node cap leaves some contacts
    # reading a single node, which is the degeneracy resolve_node_count removes.
    xy = _toy_plane(n_shafts=8, per_shaft=5, pitch=3.0)
    sigma = kernel_sigma_mm(xy)
    n_nodes, nodes, H, nominal = resolve_node_count(xy, sigma, seed=0)
    assert nominal == node_count(len(xy))
    assert n_nodes >= nominal
    assert int((H > 0).sum(axis=1).min()) >= MIN_NODES_PER_CONTACT
    assert len(nodes) == n_nodes


# --------------------------------------------------------------------------
# cache-level contract: these run against the real built cache (spec 3.2-3.5)
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1/cache"
MANIFEST = ROOT / "results/topic5_spatial_latent_propagation_rnn_v0_1/INPUT_MANIFEST.json"

pytestmark_cache = pytest.mark.skipif(
    not (CACHE / "CACHE_SUMMARY.json").exists(), reason="cache not built"
)


def _cached_patients():
    if not (CACHE / "CACHE_SUMMARY.json").exists():
        return []
    return json.loads((CACHE / "CACHE_SUMMARY.json").read_text())["patients"]


@pytestmark_cache
def test_t2c_every_cached_contact_observes_a_neighbourhood():
    for patient in _cached_patients():
        assert patient["min_nodes_seen_per_contact"] >= MIN_NODES_PER_CONTACT, (
            f"{patient['subject']} has a contact reading fewer than "
            f"{MIN_NODES_PER_CONTACT} latent nodes"
        )


@pytestmark_cache
def test_t1b_cached_operator_rows_sum_to_one():
    for patient in _cached_patients():
        H = np.load(CACHE / patient["subject"] / "seeg_operator.npz")["H"]
        np.testing.assert_allclose(H.sum(axis=1), 1.0, atol=1e-12)


@pytestmark_cache
def test_t3_contact_order_is_the_exact_name_intersection_in_record_order():
    manifest = json.loads(MANIFEST.read_text())
    tree = manifest["primary_geometry_tree"]
    by_subject = {e["subject"]: e for e in manifest["subjects"]}
    for patient in _cached_patients():
        entry = by_subject[patient["subject"]]
        expected = entry["geometry"][tree]["joint_contacts"]
        cached = list(
            np.load(CACHE / patient["subject"] / "plane_coordinates.npz",
                    allow_pickle=True)["contact_names"]
        )
        assert [str(c) for c in cached] == expected
        # and every cached contact really came from the event record
        assert set(expected) <= set(entry["event_contacts"])


@pytestmark_cache
def test_t15_splits_are_disjoint_chronological_and_exclude_the_burned_heldout():
    for patient in _cached_patients():
        split = np.load(CACHE / patient["subject"] / "events.npz")["split"]
        train = np.flatnonzero(split == 0)
        validation = np.flatnonzero(split == 1)
        test = np.flatnonzero(split == 2)
        assert len(np.intersect1d(train, validation)) == 0
        assert len(np.intersect1d(train, test)) == 0
        assert len(np.intersect1d(validation, test)) == 0
        # chronological order: train entirely before validation before test
        assert train.max() < validation.min()
        assert validation.max() < test.min()
        # the burned old heldout20 is the tail of the record and must be unused
        n_used = len(train) + len(validation) + len(test)
        assert n_used <= patient["n_events_total"]
        assert test.max() < patient["n_events_total"]


# --------------------------------------------------------------------------
# T5 - T16 -- model contract
# --------------------------------------------------------------------------

def _toy_model(arm, seed=0, microsteps=3, hidden=4):
    xy = _toy_plane(n_shafts=2, per_shaft=4, pitch=4.0)
    sigma = kernel_sigma_mm(xy)
    n_nodes, nodes, H, _ = resolve_node_count(xy, sigma, seed=0)
    config = ModelConfig(
        arm=arm,
        n_contacts=len(xy),
        n_nodes=n_nodes,
        hidden=hidden,
        microsteps=microsteps,
        seed=seed,
        normalised_distance=normalised_distance(nodes)
        if arm != "CONTACT_GRAPH_RNN" else normalised_distance(xy),
        fixed_edge_mask=knn_edge_mask(nodes, k=6),
        observation_operator=H if arm in LATENT_ARMS else None,
    )
    return SLPModel(config), H, nodes, xy


def _toy_events(n_events=6, n_contacts=8, seed=0):
    rng = np.random.default_rng(seed)
    ranks = np.full((n_events, n_contacts), -1, np.int16)
    for e in range(n_events):
        order = rng.permutation(n_contacts)[: rng.integers(3, n_contacts + 1)]
        for t, c in enumerate(order):
            ranks[e, c] = t
    return ranks


def test_event_tensors_encode_the_support_mask_and_stop_position():
    ranks = np.array([[0, 1, 2, -1], [1, 0, -1, -1]], np.int16)
    ev = build_event_tensors(ranks)
    # event 0 runs three ranks, event 1 runs two
    assert ev.valid[0].sum() == 3 and ev.valid[1].sum() == 2
    assert bool(ev.is_last[0, 2]) and bool(ev.is_last[1, 1])
    # a contact recruited at step t is no longer available at step t
    assert not bool(ev.available[0, 0, 0])
    assert bool(ev.available[0, 0, 1])
    # target at step t is exactly rank t+1
    np.testing.assert_array_equal(ev.target[0, 0].numpy(), [0, 1, 0, 0])


def test_t5_latent_input_and_readout_use_the_same_operator():
    model, H, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN")
    # the module holds exactly one H buffer and both directions index it
    buffers = [n for n, _ in model.named_buffers() if n.endswith("H")]
    assert buffers == ["H"]
    np.testing.assert_allclose(model.H.numpy(), H.astype(np.float32), atol=1e-7)


def test_t6_zeroing_the_graph_degrades_the_latent_model_to_node_wise():
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN")
    model.eval()
    ev = build_event_tensors(_toy_events())
    with torch.no_grad():
        before = model(ev.x, ev.recruited, ev.valid)[0]
        model.graph.weight.data.zero_()
        after = model(ev.x, ev.recruited, ev.valid)[0]
        # with A = 0 the message is identically zero, so a hand-run node-wise
        # recurrence must reproduce the model exactly
        state = model.initial_state(ev.x.shape[0], ev.x.device)
        manual = []
        for t in range(ev.x.shape[1]):
            injection = torch.einsum("bc,cm->bm", ev.x[:, t], model.H)
            for k in range(model.config.microsteps):
                drive = injection if k == 0 else torch.zeros_like(injection)
                state = model.cell(state, drive, torch.zeros_like(state))
            emission = torch.nn.functional.softplus(model.emission(state).squeeze(-1))
            manual.append(model.contact_bias + model.logit_scale *
                          torch.einsum("bm,cm->bc", emission, model.H))
        np.testing.assert_allclose(
            after.numpy(), torch.stack(manual, 1).numpy(), atol=1e-5
        )
    assert not np.allclose(before.numpy(), after.numpy())


def test_t7_no_dense_contact_to_contact_path_exists():
    model, H, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN")
    model.eval()
    # a contact whose observation kernel shares no node with contact j cannot
    # influence j within a single microstep once the graph is removed
    model.graph.weight.data.zero_()
    ev = build_event_tensors(_toy_events())
    with torch.no_grad():
        base = model(ev.x, ev.recruited, ev.valid)[0]
        bumped = ev.x.clone()
        bumped[:, 0, 0] = 1.0
        moved = model(bumped, ev.recruited, ev.valid)[0]
    changed = (moved - base).abs().sum(dim=(0, 1)).numpy() > 1e-6
    shared_support = (H[0] > 0) @ (H.T > 0)
    assert np.all(changed <= shared_support)


def test_t8_node_coordinates_never_reach_the_prediction_head():
    model, _, nodes, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN")
    model.eval()
    ev = build_event_tensors(_toy_events())
    with torch.no_grad():
        before = model(ev.x, ev.recruited, ev.valid)[0]
        # permuting the wiring-cost geometry must not touch predictions
        perm = np.random.default_rng(0).permutation(len(nodes))
        model.edge_distance.copy_(model.edge_distance[np.ix_(perm, perm)])
        after = model(ev.x, ev.recruited, ev.valid)[0]
    np.testing.assert_allclose(before.numpy(), after.numpy(), atol=1e-12)


def test_t9_single_microstep_matches_a_hand_computed_step():
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", microsteps=1)
    model.eval()
    ev = build_event_tensors(_toy_events(n_events=2))
    with torch.no_grad():
        state = model.initial_state(2, ev.x.device)
        adjacency = model.graph.adjacency(1.0)
        injection = torch.einsum("bc,cm->bm", ev.x[:, 0], model.H)
        phi = torch.tanh(state)
        num = torch.einsum("bjd,ji->bid", phi, adjacency)
        den = adjacency.abs().sum(0).clamp_min(1e-6).view(1, -1, 1)
        manual_state = model.cell(state, injection, num / den)
        emission = torch.nn.functional.softplus(model.emission(manual_state).squeeze(-1))
        manual = model.contact_bias + model.logit_scale * torch.einsum(
            "bm,cm->bc", emission, model.H
        )
        got = model(ev.x, ev.recruited, ev.valid)[0][:, 0]
    np.testing.assert_allclose(got.numpy(), manual.numpy(), atol=1e-6)


def test_t10_the_forward_pass_never_reads_a_future_rank():
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN")
    model.eval()
    ranks = _toy_events(n_events=4, seed=1)
    ev = build_event_tensors(ranks)
    with torch.no_grad():
        base = model(ev.x, ev.recruited, ev.valid)[0]
        scrambled = ev.x.clone()
        scrambled[:, 2:] = scrambled[:, 2:].flip(dims=[1])
        moved = model(scrambled, ev.recruited, ev.valid)[0]
    # steps 0 and 1 saw no future information, so they must be untouched
    np.testing.assert_allclose(base[:, :2].numpy(), moved[:, :2].numpy(), atol=1e-12)


def test_t13_topology_freeze_is_reproducible_and_sparsifies():
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", seed=3)
    with torch.no_grad():
        model.graph.gate.log_alpha.copy_(
            torch.linspace(-4, 4, model.graph.n_nodes ** 2).view(
                model.graph.n_nodes, model.graph.n_nodes
            )
        )
    kept_a = model.graph.freeze_topology(temperature=0.5, edge_budget=6.0)
    mask_a = model.graph.frozen_mask.clone()
    model.graph.topology_frozen = False
    kept_b = model.graph.freeze_topology(temperature=0.5, edge_budget=6.0)
    assert kept_a == kept_b
    np.testing.assert_array_equal(mask_a.numpy(), model.graph.frozen_mask.numpy())
    assert 0 < kept_a < model.graph.n_nodes ** 2
    assert not np.any(np.diag(model.graph.frozen_mask.numpy()))
    # the frozen degree must equal the budget, not whatever a threshold happened
    # to leave behind -- an all-weak-gates graph used to freeze to nothing
    assert kept_a == pytest.approx(6.0 * model.graph.n_nodes, abs=1)


def test_t13b_freeze_survives_a_uniformly_weak_gate_field():
    """Every gate barely open must still leave a graph of the budgeted size."""
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", seed=4)
    with torch.no_grad():
        model.graph.gate.log_alpha.fill_(-6.0)
        model.graph.gate.log_alpha.add_(
            0.01 * torch.randn_like(model.graph.gate.log_alpha)
        )
    kept = model.graph.freeze_topology(temperature=0.3, edge_budget=6.0)
    assert kept == pytest.approx(6.0 * model.graph.n_nodes, abs=1)
    assert float(model.graph.adjacency(0.3).abs().sum()) > 0.0


def test_t14_identity_coordinate_shuffle_reproduces_the_compared_arm():
    """The ablation control must be isomorphic to the arm it is compared against.

    An earlier Topic 5 round reported an order effect that came from a control
    built differently from its own comparison arm; the identity permutation is
    the cheap guard against repeating that.
    """
    model_a, _, nodes, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", seed=11)
    model_b, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", seed=11)
    identity = np.arange(len(nodes))
    with torch.no_grad():
        model_b.edge_distance.copy_(
            torch.from_numpy(normalised_distance(nodes[identity]).astype(np.float32))
        )
    ev = build_event_tensors(_toy_events())
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        la = model_a(ev.x, ev.recruited, ev.valid)[0]
        lb = model_b(ev.x, ev.recruited, ev.valid)[0]
        wa = model_a.wiring_loss(1.0)
        wb = model_b.wiring_loss(1.0)
    np.testing.assert_array_equal(la.numpy(), lb.numpy())
    assert float(wa) == float(wb)


def test_t16_no_bias_variant_has_no_per_contact_parameter():
    xy = _toy_plane(n_shafts=2, per_shaft=4, pitch=4.0)
    sigma = kernel_sigma_mm(xy)
    n_nodes, nodes, H, _ = resolve_node_count(xy, sigma, seed=0)
    for arm, operator in (
        ("CONTACT_GRAPH_RNN", None),
        ("LATENT_LEARNED_SPATIAL_RNN", H),
    ):
        model = SLPModel(ModelConfig(
            arm=arm, n_contacts=len(xy), n_nodes=n_nodes, use_contact_bias=False,
            normalised_distance=normalised_distance(nodes if operator is not None else xy),
            observation_operator=operator,
        ))
        model.eval()
        ev = build_event_tensors(_toy_events())
        with torch.no_grad():
            model.contact_bias.add_(torch.randn_like(model.contact_bias))
            perturbed = model(ev.x, ev.recruited, ev.valid)[0]
            model.contact_bias.zero_()
            clean = model(ev.x, ev.recruited, ev.valid)[0]
        np.testing.assert_allclose(perturbed.numpy(), clean.numpy(), atol=1e-12)


def test_loss_ignores_already_recruited_contacts():
    ranks = np.array([[0, 1, 2, -1]], np.int16)
    ev = build_event_tensors(ranks)
    logits = torch.zeros_like(ev.x)
    # a wild logit on an already-recruited contact must not move the loss
    stop = torch.zeros(ev.valid.shape)
    total_a, _, _ = next_set_stop_loss(
        logits, stop, ev.target, ev.available, ev.valid, ev.is_last
    )
    logits2 = logits.clone()
    logits2[0, 1, 0] = 50.0  # contact 0 was recruited at rank 0
    total_b, _, _ = next_set_stop_loss(
        logits2, stop, ev.target, ev.available, ev.valid, ev.is_last
    )
    assert float(total_a) == pytest.approx(float(total_b))


def test_wiring_and_budget_losses_are_zero_for_non_learned_arms():
    for arm in ("STATIC_CONTACT", "ORDINARY_GRU", "LATENT_FIXED_LOCAL_RNN"):
        model, _, _, _ = _toy_model(arm)
        assert float(model.wiring_loss(1.0)) == 0.0
        assert float(model.edge_budget_loss(1.0)) == 0.0


def test_wiring_loss_rises_when_long_edges_are_opened():
    model, _, _, _ = _toy_model("LATENT_LEARNED_SPATIAL_RNN", seed=5)
    with torch.no_grad():
        model.graph.weight.data.fill_(0.5)
        far = model.edge_distance > model.edge_distance.median()
        model.graph.gate.log_alpha.data.fill_(-4.0)
        short_cost = float(model.wiring_loss(1.0))
        model.graph.gate.log_alpha.data[far] = 4.0
        long_cost = float(model.wiring_loss(1.0))
    assert long_cost > short_cost


def test_batch_size_gives_every_patient_the_same_number_of_updates():
    """The epoch budget must mean the same amount of optimisation for everyone.

    A fixed batch counts gradient steps in units of epochs, which starves the
    patients with few events: at 1024 a patient with 249 training events gets one
    update per epoch. That is how half this cohort ended up fitted far from its
    own optimum while the logs looked normal.
    """
    from scripts.train_topic5_slp_unit import MIN_BATCHES_PER_EPOCH

    configured = 1024
    for n_train in (249, 359, 1313, 4687, 12667, 69114):
        batch = int(np.clip(n_train // MIN_BATCHES_PER_EPOCH, 32, configured))
        batches_per_epoch = max(1, n_train // batch)
        assert batches_per_epoch >= MIN_BATCHES_PER_EPOCH - 1, (
            f"{n_train} training events give only {batches_per_epoch} updates per epoch"
        )
        assert batch <= configured


@pytestmark_cache
def test_ranks_are_dense_and_contacts_appear_at_most_once_per_event():
    for patient in _cached_patients():
        ranks = np.load(CACHE / patient["subject"] / "events.npz")["group_ids"]
        sample = ranks[:: max(1, len(ranks) // 200)]
        for row in sample:
            present = row[row >= 0]
            if not len(present):
                continue
            assert present.min() == 0
            assert set(np.unique(present).tolist()) == set(range(present.max() + 1))
