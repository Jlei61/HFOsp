"""TDD for src.topic4_hub_criticality — structural criticality probe.

All fixtures are hand-built sparse matrices / fake `net` dicts. No real SNN engine is
loaded; every expected value is hand-computed so the test would fail if the linear-algebra
semantics (gap_factor clip, E->E sub-block, global-via-hub zeroing) drift.
"""

import numpy as np
import scipy.sparse as sp
import pytest

from src.topic4_hub_criticality import (
    recruitment_operator,
    branching_ratio,
    crossing_branching,
    sigma_phase_map,
)


def _net_from_block(block_ee, NE, NI):
    """Embed an (NE x NE) E->E weight block into a full (NE+NI) AMPA matrix and wrap as net.

    [row=target, col=source]. Returns a net dict with a single ampa_by_delay entry.
    """
    N = NE + NI
    full = np.zeros((N, N), float)
    full[:NE, :NE] = block_ee
    return {"ampa_by_delay": [sp.csr_matrix(full)]}


# ---------------------------------------------------------------------------
# recruitment_operator + branching_ratio: feed-forward chain w/ a self edge
# ---------------------------------------------------------------------------

def test_branching_ratio_chain_with_selfloop_matches_handcomputed():
    NE, NI = 3, 0
    w = 0.5
    # chain 0->1->2 plus a self-recruitment 2->2 so the spectral radius is nonzero.
    block = np.zeros((3, 3))
    block[1, 0] = w   # source 0 -> target 1
    block[2, 1] = w   # source 1 -> target 2
    block[2, 2] = w   # source 2 -> target 2 (self loop)
    net = _net_from_block(block, NE, NI)

    drive_rest = 0.0
    vth_c = 2.0            # gap_factor = 1/(2-0) = 0.5, well under link_cap
    V_th = np.full(NE + NI, vth_c)

    M = recruitment_operator(net, V_th, NE, drive_rest)
    assert sp.issparse(M)
    assert M.shape == (NE, NE)

    g = 0.5  # 1/(vth_c - drive_rest)
    Mdense = M.toarray()
    expected_M = block * g
    np.testing.assert_allclose(Mdense, expected_M)

    # spectral radius of M = the self-loop entry = w*g (chain part is nilpotent).
    sigma = branching_ratio(M)
    hand = float(np.max(np.real(np.linalg.eigvals(expected_M))))
    assert hand == pytest.approx(w * g)
    assert sigma == pytest.approx(hand)
    assert sigma == pytest.approx(w * g)


def test_recruitment_operator_uses_only_EE_subblock_and_clip():
    NE, NI = 2, 2
    N = NE + NI
    full = np.zeros((N, N))
    full[1, 0] = 1.0     # E->E (kept)
    full[0, 2] = 9.0     # I-source -> E-target (col >= NE, dropped)
    full[3, 0] = 7.0     # E-source -> I-target (row >= NE, dropped)
    net = {"ampa_by_delay": [sp.csr_matrix(full)]}

    drive_rest = 0.0
    # target threshold tiny -> 1/(vth-drive) huge -> clipped to link_cap.
    V_th = np.array([1e-9, 1e-9, 5.0, 5.0])
    M = recruitment_operator(net, V_th, NE, drive_rest, link_cap=4.0)
    Md = M.toarray()
    assert Md.shape == (2, 2)
    # only the E->E edge survives, scaled by clipped gap_factor (link_cap=4.0).
    assert Md[1, 0] == pytest.approx(4.0)
    assert Md[0, 1] == 0.0
    assert Md[0, 0] == 0.0
    assert Md[1, 1] == 0.0


def test_recruitment_operator_sums_delay_matrices():
    NE, NI = 2, 0
    a = np.zeros((2, 2)); a[1, 0] = 0.3
    b = np.zeros((2, 2)); b[1, 0] = 0.2
    net = {"ampa_by_delay": [sp.csr_matrix(a), sp.csr_matrix(b)]}
    V_th = np.array([1.0, 1.0]); drive_rest = 0.0  # gap_factor = 1.0
    M = recruitment_operator(net, V_th, NE, drive_rest)
    assert M.toarray()[1, 0] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# branching_ratio submatrix selection + empty/zero handling
# ---------------------------------------------------------------------------

def test_branching_ratio_idx_subselects():
    block = np.zeros((4, 4))
    block[0, 0] = 0.9   # strong self loop in node 0
    block[3, 3] = 0.1   # weak self loop in node 3
    net = _net_from_block(block, 4, 0)
    V_th = np.ones(4); drive_rest = 0.0  # gap_factor 1.0
    M = recruitment_operator(net, V_th, 4, drive_rest)
    assert branching_ratio(M, idx=[0]) == pytest.approx(0.9)
    assert branching_ratio(M, idx=[3]) == pytest.approx(0.1)
    assert branching_ratio(M, idx=None) == pytest.approx(0.9)


def test_branching_ratio_empty_or_zero_returns_zero():
    block = np.zeros((3, 3))
    net = _net_from_block(block, 3, 0)
    V_th = np.ones(3); drive_rest = 0.0
    M = recruitment_operator(net, V_th, 3, drive_rest)
    assert branching_ratio(M, idx=[]) == 0.0
    assert branching_ratio(M, idx=None) == 0.0   # all-zero


def test_branching_ratio_large_sparse_path():
    # > 3 nodes drives the scipy.sparse.linalg.eigs branch (with dense fallback).
    n = 8
    block = np.zeros((n, n))
    for i in range(n):
        block[i, i] = 0.25  # diagonal -> spectral radius 0.25
    net = _net_from_block(block, n, 0)
    V_th = np.ones(n); drive_rest = 0.0
    M = recruitment_operator(net, V_th, n, drive_rest)
    assert branching_ratio(M) == pytest.approx(0.25, abs=1e-6)


# ---------------------------------------------------------------------------
# crossing_branching: global only reachable via hub
# ---------------------------------------------------------------------------

def _hub_net(corridor_hub_w, hub_global_w, corridor_global_w, self_w=0.0):
    """5-node E net: idx 0,1 corridor; 2 hub; 3,4 global.

    Edges (target<-source): hub<-corridor, global<-hub, global<-corridor(direct).
    """
    NE = 5
    block = np.zeros((NE, NE))
    block[2, 0] = corridor_hub_w      # hub <- corridor
    block[2, 1] = corridor_hub_w
    block[3, 2] = hub_global_w         # global <- hub
    block[4, 2] = hub_global_w
    block[3, 0] = corridor_global_w    # global <- corridor (direct, must be zeroed)
    block[4, 1] = corridor_global_w
    if self_w:
        for k in (3, 4):
            block[k, k] = self_w
    return _net_from_block(block, NE, 0)


def test_crossing_branching_disconnected_hub_is_zero():
    # No hub->global edges and no global self loops: even with strong direct
    # corridor->global, crossing == 0 because the direct edges are forced to zero
    # (path must go through hub) and nothing else closes a cycle into global.
    net = _hub_net(corridor_hub_w=0.0, hub_global_w=0.0, corridor_global_w=0.9, self_w=0.0)
    V_th = np.ones(5); drive_rest = 0.0
    M = recruitment_operator(net, V_th, 5, drive_rest)
    sigma = crossing_branching(M, corridor_idx=[0, 1], hub_idx=[2], global_idx=[3, 4])
    assert sigma == pytest.approx(0.0)


def test_crossing_branching_empty_hub_returns_zero():
    net = _hub_net(0.5, 0.5, 0.0)
    V_th = np.ones(5); drive_rest = 0.0
    M = recruitment_operator(net, V_th, 5, drive_rest)
    assert crossing_branching(M, [0, 1], [], [3, 4]) == 0.0


def test_crossing_branching_zeros_direct_corridor_global_edge():
    # A direct corridor->global self-amplifying loop would inflate spectral radius if kept.
    # global node 3 has self loop 0.8 AND a strong direct edge from corridor; zeroing the
    # direct edge leaves only the hub path (here a nilpotent chain) so sigma == self loop.
    net = _hub_net(corridor_hub_w=0.5, hub_global_w=0.5, corridor_global_w=0.99, self_w=0.8)
    V_th = np.ones(5); drive_rest = 0.0  # gap_factor 1.0
    M = recruitment_operator(net, V_th, 5, drive_rest)
    sigma = crossing_branching(M, [0, 1], [2], [3, 4])
    # the only cycle left is the global self loop (0.8); chain corridor->hub->global nilpotent.
    assert sigma == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# monotonicity
# ---------------------------------------------------------------------------

def test_branching_ratio_decreases_with_threshold():
    block = np.zeros((3, 3))
    block[0, 0] = 0.6
    net = _net_from_block(block, 3, 0)
    drive_rest = 0.0
    sigmas = []
    for vth in (1.0, 2.0, 4.0):
        V_th = np.full(3, vth)
        M = recruitment_operator(net, V_th, 3, drive_rest)
        sigmas.append(branching_ratio(M))
    assert sigmas[0] > sigmas[1] > sigmas[2]


def test_crossing_and_corridor_increase_with_weight_scale():
    drive_rest = 0.0
    V_th = np.ones(5)
    corr_sigmas, cross_sigmas = [], []
    for scale in (0.3, 0.6, 1.2):
        # corridor self loops grow + hub/global loops grow with scale.
        block = np.zeros((5, 5))
        block[0, 0] = 0.5 * scale     # corridor self loop
        block[1, 0] = 0.5 * scale
        block[2, 0] = 0.5 * scale     # hub <- corridor
        block[3, 2] = 0.5 * scale     # global <- hub
        block[3, 3] = 0.5 * scale     # global self loop
        net = _net_from_block(block, 5, 0)
        M = recruitment_operator(net, V_th, 5, drive_rest)
        corr_sigmas.append(branching_ratio(M, [0, 1]))
        cross_sigmas.append(crossing_branching(M, [0, 1], [2], [3, 4]))
    assert corr_sigmas[0] < corr_sigmas[1] < corr_sigmas[2]
    assert cross_sigmas[0] < cross_sigmas[1] < cross_sigmas[2]


# ---------------------------------------------------------------------------
# sigma_phase_map
# ---------------------------------------------------------------------------

def test_sigma_phase_map_shapes_and_content():
    NE, NI = 4, 0

    def build_fn(gain):
        block = np.zeros((4, 4))
        block[0, 0] = 0.5          # corridor self loop
        block[2, 0] = 0.4          # hub <- corridor
        block[3, 2] = gain         # global <- hub (long-range strength)
        block[3, 3] = 0.3          # global self loop
        return _net_from_block(block, NE, NI)

    def degnorm_fn(net, alpha):
        # uniform threshold bump scaled by alpha.
        return np.full(NE + NI, alpha)

    V_th0 = np.ones(NE + NI)
    drive_rest = 0.0
    alpha_grid = [0.0, 0.5]
    gain_grid = [0.1, 0.9]
    regions = {"corridor_idx": [0, 1], "hub_idx": [2], "global_idx": [3]}

    out = sigma_phase_map(build_fn, alpha_grid, gain_grid, regions,
                          V_th0, NE, drive_rest, degnorm_fn)
    assert set(out) == {"alpha_grid", "gain_grid", "sigma_corridor", "sigma_crossing"}
    assert out["sigma_corridor"].shape == (2, 2)
    assert out["sigma_crossing"].shape == (2, 2)
    np.testing.assert_array_equal(out["alpha_grid"], np.asarray(alpha_grid))
    np.testing.assert_array_equal(out["gain_grid"], np.asarray(gain_grid))

    # higher alpha -> higher threshold -> smaller corridor sigma (monotone down columns).
    assert out["sigma_corridor"][0, 0] > out["sigma_corridor"][1, 0]
    assert np.all(np.isfinite(out["sigma_corridor"]))
    assert np.all(np.isfinite(out["sigma_crossing"]))


def test_crossing_path_gain_two_stage_product():
    from scipy import sparse
    from src.topic4_hub_criticality import crossing_path_gain
    # corridor=[0], hub=[1], global=[2,3]; M[target, source]
    M = np.zeros((4, 4)); M[1, 0] = 2.0; M[2, 1] = 0.5; M[3, 1] = 0.5
    out = crossing_path_gain(sparse.csr_matrix(M), [0], [1], [2, 3])
    assert np.isclose(out["hub_recruit"], 2.0)         # corridor->hub drive into the 1 hub cell
    assert np.isclose(out["hub_broadcast"], 0.5)       # hub->global drive averaged over 2 global cells
    assert np.isclose(out["gain"], 1.0)                # product
    # no hub->global broadcast -> crossing gain 0 even though hub is recruited
    M2 = M.copy(); M2[2, 1] = 0; M2[3, 1] = 0
    assert crossing_path_gain(sparse.csr_matrix(M2), [0], [1], [2, 3])["gain"] == 0.0
    # empty hub / corridor / global -> 0
    assert crossing_path_gain(sparse.csr_matrix(M), [0], [], [2, 3])["gain"] == 0.0
