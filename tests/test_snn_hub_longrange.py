import hashlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params              # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick    # noqa: E402

M3_BASE_SHA = "da5fc18c27d5340a"


def _net(hub_gain=0.0, hub_mask_E=None, hub_long_range_C=0, l_hub_long=None):
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0,
                                 hub_mask_E=hub_mask_E, hub_long_range_C=hub_long_range_C,
                                 l_hub_long=l_hub_long, hub_gain=hub_gain)
    return p, net, pos, labels, NE, NI


def _ee_edges(net, NE, posE):
    A = sum(m.tocsr() for m in net["ampa_by_delay"]).tocoo()
    mask = (A.row < NE) & (A.col < NE)
    tgt, src = A.row[mask], A.col[mask]
    dist = np.linalg.norm(posE[tgt] - posE[src], axis=1)
    return tgt, src, dist


def _hub_mask(pos, NE, n_hub=5):
    posE = pos[:NE]
    order = np.argsort(posE[:, 0])          # deterministic: the n_hub rightmost-x E cells
    m = np.zeros(NE, bool)
    m[order[-n_hub:]] = True
    return m


def test_hub_gain0_default_is_bit_identical():
    p, net, pos, labels, NE, NI = _net(hub_gain=0.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0))
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == M3_BASE_SHA


def test_hub_gain_adds_longrange_edges_from_hub_sources():
    p, net0, pos, labels, NE, NI = _net(hub_gain=0.0)
    posE = pos[:NE]
    hub = _hub_mask(pos, NE, n_hub=5)
    _, net1, *_ = _net(hub_gain=0.5, hub_mask_E=hub, hub_long_range_C=20, l_hub_long=4.0)
    _, src0, dist0 = _ee_edges(net0, NE, posE)
    _, src1, dist1 = _ee_edges(net1, NE, posE)
    # long-range-from-hub edges: source in hub AND distance well beyond the local kernel
    lr0 = int(np.sum(hub[src0] & (dist0 > 2.0)))
    lr1 = int(np.sum(hub[src1] & (dist1 > 2.0)))
    assert lr1 > lr0 + 10            # many added broadcast edges from hub sources
    assert src1.size > src0.size     # total E->E edge count increased


def test_larger_l_hub_long_reaches_farther():
    # 30 SPREAD hubs with C=8 < n_hub so the distance kernel actually SELECTS which hubs
    # connect (with C>=n_hub all hubs are always taken and l_hub_long is irrelevant).
    p, _, pos, labels, NE, NI = _net(hub_gain=0.0)
    posE = pos[:NE]
    hub = np.zeros(NE, bool)
    hub[np.linspace(0, NE - 1, 30).astype(int)] = True
    _, net_s, *_ = _net(hub_gain=0.5, hub_mask_E=hub, hub_long_range_C=8, l_hub_long=1.0)
    _, net_l, *_ = _net(hub_gain=0.5, hub_mask_E=hub, hub_long_range_C=8, l_hub_long=6.0)
    _, src_s, dist_s = _ee_edges(net_s, NE, posE)
    _, src_l, dist_l = _ee_edges(net_l, NE, posE)
    # larger l_hub_long -> the broadcast reaches more FAR (>3mm) hub-source edges
    far_s = int(np.sum(hub[src_s] & (dist_s > 3.0)))
    far_l = int(np.sum(hub[src_l] & (dist_l > 3.0)))
    assert far_l > far_s


def test_hub_gain_requires_params():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    with pytest.raises(ValueError):
        build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0,
                               hub_gain=0.5)   # missing hub_mask_E / C / l
