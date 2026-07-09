"""TDD for the M4-3A conductance a-shunt wired into kick_probe's M4 membrane update (Task 5, form A).

`simulate_kick`'s `if slow is not None:` branch previously computed the membrane update
UNCONDITIONALLY as `Vtmp = I_net + (V - I_net) * decay_V` (current-based; a-shunt ignored, even
when Task 4's SpatialSlowField had use_A/k_n/alpha_A set). Task 5 adds a `slow.uses_shunt()`-gated
conductance form: only when the shunt is genuinely engaged (use_A AND k_n!=0 AND alpha_A!=0,
Task 4's `uses_shunt()`) do E cells get a conductance g_A = alpha_A*a (clipped to g_A_max) that
pulls V toward the reversal E_A (= e_gaba = p.E_gaba by default); I cells always get g=0. When
uses_shunt() is False (every existing config today -- no caller sets use_A+k_n+alpha_A yet), the
ELSE branch is the LITERAL pre-existing line -> byte-parity with every current caller.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

DT = 0.1


def _build_kicked_net(seed_net=1, seed_rng=3):
    """A small E-I network (mirrors the _net() helper in test_a1c_feedback.py /
    test_m4_shared_inhibition.py) with the poisson-drive rng seeded AFTER construction so two
    calls with the same seeds produce byte-identical anatomy + stochastic drive."""
    p = Params(L=6.0, density=100.0, T=200.0, dt=DT, nu_ext_ratio=0.6, seed=seed_net)
    rng = np.random.default_rng(seed_net)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(seed_rng)
    return p, net, NE, NI


def _slow_for_net(p, net, **cfgkw):
    """SpatialSlowField sized/positioned to match the ACTUAL built network (not a disconnected
    population) so E/I indices line up with kick_probe -- mirrors test_m4_shared_inhibition.py's
    _slow_for()."""
    NE, NI = net["NE"], net["NI"]
    posE = net["pos"][net["labels"] == 0]
    posI = net["pos"][net["labels"] == 1]
    cfg = SpatialSlowFieldConfig(n_grid=8, **cfgkw)
    return SpatialSlowField(NE + NI, p.V_th, posE, posI, p.L, cfg=cfg)


def _run_pair(use_A):
    """Run an identical tiny E-I network once with slow=None (the pre-existing baseline) and once
    with a SpatialSlowField present whose use_A flag is `use_A`. With use_A=False, uses_shunt() is
    False (Task 4), so the new branch must fall through to the literal pre-change line and
    reproduce the slow=None run bit-for-bit."""
    p, net_a, NE, NI = _build_kicked_net()
    res_a = simulate_kick(p, net_a, KICK_BOOST=6.0, r_kick=1.5, V_th_per_neuron=np.full(NE + NI, 16.5))

    p, net_b, NE2, NI2 = _build_kicked_net()
    slow = _slow_for_net(p, net_b, use_A=use_A)
    res_b = simulate_kick(p, net_b, KICK_BOOST=6.0, r_kick=1.5,
                          V_th_per_neuron=np.full(NE2 + NI2, 16.5), slow=slow)
    return res_a, res_b


def test_shunt_off_matches_baseline_bit_exact():
    """use_A off (uses_shunt()==False): M4 path must be byte-identical to pre-change (slow=None)."""
    res_a, res_b = _run_pair(use_A=False)
    assert np.array_equal(res_a["E_spk_bool"], res_b["E_spk_bool"])
    assert np.array_equal(res_a["rate_E"], res_b["rate_E"])


def test_shunt_on_pulls_membrane_toward_reversal():
    """With g_A>0 the effective V_inf moves toward E_A (rest) vs the un-shunted drive."""
    I_net = np.array([5.0, 5.0]); V = np.array([5.0, 5.0]); decay_V = 0.9
    g = np.array([0.0, 4.0]); E_A = 0.0
    V_inf = (I_net + g * E_A) / (1.0 + g)
    Vtmp = V_inf + (V - V_inf) * decay_V ** (1.0 + g)
    assert Vtmp[1] < Vtmp[0]                    # shunted cell driven closer to E_A


def test_shunt_engaged_suppresses_relative_to_off():
    """When the shunt is genuinely engaged (uses_shunt()==True) and a_shunt is pre-set > 0 (bypassing
    Task 4's slow n-load ODE ramp-up so THIS test targets kick_probe's wiring, not the ODE's build-up
    dynamics), the conductance pulls E cells toward E_A = p.E_gaba (11.0, well below V_th=16.5) and
    firing is suppressed relative to the shunt disengaged (alpha_A=0 -> uses_shunt() False, matched
    control). Pre-fix, kick_probe never calls uses_shunt()/shunt_g_at_E(), so engaging the shunt has
    NO effect on the trajectory and this test is RED; post-fix the two runs must diverge (GREEN)."""
    def run(engaged):
        p, net, NE, NI = _build_kicked_net()
        alpha_A = 8.0 if engaged else 0.0        # alpha_A=0 -> uses_shunt() False (matched control)
        slow = _slow_for_net(p, net, use_A=True, k_n=1.0, alpha_A=alpha_A)
        if engaged:
            slow.n_load[:] = 5.0                 # pre-seed load so a_shunt > 0 from step 0
            slow.a_shunt[:] = slow.cfg.a_max
        res = simulate_kick(p, net, KICK_BOOST=6.0, r_kick=1.5,
                            V_th_per_neuron=np.full(NE + NI, 16.5), slow=slow)
        return res

    res_off = run(engaged=False)
    res_on = run(engaged=True)
    assert res_off["E_spk_bool"].sum() > 0                              # sanity: baseline actually spikes
    assert res_on["E_spk_bool"].sum() < res_off["E_spk_bool"].sum()     # pulled toward sub-threshold E_A
