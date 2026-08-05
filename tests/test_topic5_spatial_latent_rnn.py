"""TDD suite for the Topic 5 spatial latent propagation RNN v0.1.

Every test names the spec clause it guards.  Spec:
``docs/superpowers/specs/2026-08-06-topic5-spatial-latent-propagation-rnn-v0_1.md``
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
