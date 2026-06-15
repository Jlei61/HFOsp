"""逐神经元 V_th 注入：传标量等价数组、传高门槛抑制发放。tiny smoke params。

Engine is the gitignored coworker scaffold under results/.../engine; we sys.path
into it (same as the existing two-electrode runner). Spec 2026-06-08 Task 2.
"""
import os
import sys

import numpy as np
import pytest

ENGINE = os.path.join("src", "snn_engine")
sys.path.insert(0, ENGINE)


@pytest.fixture(scope="module")
def net_and_p():
    from params import Params, compute_nu_theta
    from model import build_network
    p = Params(g=3.6, L=1.0, density=4000.0, T=220.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    net = build_network(p, verbose=False)
    nu_theta = compute_nu_theta(p)[0]
    return p, net, nu_theta


def test_array_equals_scalar_when_uniform(net_and_p):
    from kick_probe import simulate_kick
    p, net, nu_theta = net_and_p
    N = net["NE"] + net["NI"]
    net["rng"] = np.random.default_rng(7)
    a = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5])
    net["rng"] = np.random.default_rng(7)
    b = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5],
                      V_th_per_neuron=np.full(N, p.V_th))
    np.testing.assert_array_equal(a["rate_E"], b["rate_E"])   # uniform array == scalar


def test_high_threshold_suppresses_firing(net_and_p):
    from kick_probe import simulate_kick
    p, net, nu_theta = net_and_p
    N = net["NE"] + net["NI"]
    vth = np.full(N, p.V_th); vth[: net["NE"]] = 1e6      # E can never reach threshold
    net["rng"] = np.random.default_rng(7)
    c = simulate_kick(p, net, KICK_BOOST=2 * nu_theta, kick_center=[0.5, 0.5],
                      V_th_per_neuron=vth)
    assert c["rate_E"].max() == 0.0                        # no E spikes at all
