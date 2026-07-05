"""M2 wide gates in connectivity_rot — bit-parity + edge-placement TDD.

Two ADDED default-OFF paths (each consumes NO extra rng draw when off -> bit-parity SHA
da5fc18c27d5340a):
  - I->E veto gate   (gate_scale/l_gate/C_gate):      extra wide I sources -> GABA edges on E targets.
  - E->I recruit gate (ei_gate_scale/l_ei_gate/C_ei_gate): extra wide E sources -> AMPA edges on I targets.
"""
import hashlib
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick

BASELINE_SHA = "da5fc18c27d5340a"


def _net(**gate_kw):
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0, **gate_kw)
    return p, net, NE, NI, pos


def _sha(**gate_kw):
    p, net, NE, NI, _ = _net(**gate_kw)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0))
    return hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]


def _ampa_counts(net, NE):
    e = i = 0
    for m in net["ampa_by_delay"]:
        coo = m.tocoo()
        e += int((coo.row < NE).sum())   # E targets -> E->E
        i += int((coo.row >= NE).sum())  # I targets -> E->I
    return e, i


def _gaba_counts(net, NE):
    e = i = 0
    for m in net["gaba_by_delay"]:
        coo = m.tocoo()
        e += int((coo.row < NE).sum())   # E targets -> I->E
        i += int((coo.row >= NE).sum())  # I targets -> I->I
    return e, i


def _mean_ei_dist(net, pos, NE):
    ds = []
    for m in net["ampa_by_delay"]:
        coo = m.tocoo()
        mask = coo.row >= NE   # I targets (E->I edges)
        if mask.any():
            ds.append(np.linalg.norm(pos[coo.col[mask]] - pos[coo.row[mask]], axis=1))
    return float(np.concatenate(ds).mean()) if ds else 0.0


# ---------------- E->I recruit gate (Task 4) ----------------
def test_ei_gate_off_is_bit_identical():
    assert _sha() == BASELINE_SHA                       # default ei_gate_scale=0 -> no extra draw


def test_ei_gate_on_adds_ampa_edges_to_I_targets_only():
    _, n0, NE, _, _ = _net()
    _, n1, _, _, _ = _net(ei_gate_scale=0.5, l_ei_gate=1.5, C_ei_gate=100)
    e0, i0 = _ampa_counts(n0, NE)
    e1, i1 = _ampa_counts(n1, NE)
    assert i1 > i0 and e1 == e0                          # adds E->I (I targets); E->E unchanged


def test_ei_gate_wider_kernel_increases_mean_distance():
    _, n_narrow, NE, _, pos = _net(ei_gate_scale=0.5, l_ei_gate=1.0, C_ei_gate=100)
    _, n_wide, _, _, _ = _net(ei_gate_scale=0.5, l_ei_gate=3.0, C_ei_gate=100)
    assert _mean_ei_dist(n_wide, pos, NE) > _mean_ei_dist(n_narrow, pos, NE)


def test_ei_gate_requires_l_and_C():
    with pytest.raises(ValueError):
        _net(ei_gate_scale=0.5)                          # missing l_ei_gate, C_ei_gate


# ---------------- I->E veto gate (lock the committed gate) ----------------
def test_ie_gate_off_is_bit_identical():
    assert _sha(gate_scale=0.0) == BASELINE_SHA


def test_ie_gate_on_adds_gaba_edges_to_E_targets_only():
    _, n0, NE, _, _ = _net()
    _, n1, _, _, _ = _net(gate_scale=0.5, l_gate=1.5, C_gate=100)
    e0, i0 = _gaba_counts(n0, NE)
    e1, i1 = _gaba_counts(n1, NE)
    assert e1 > e0 and i1 == i0                          # adds I->E (E targets); I->I unchanged


def test_ie_gate_requires_l_and_C():
    with pytest.raises(ValueError):
        _net(gate_scale=0.5)                             # missing l_gate, C_gate
