"""Z/M migration parity: the per-neuron z+m ported into SpatialSlowField (slow_field.py) must be
byte-identical to the CANONICAL MZSlowVars (mz_slow_vars.py) when driven through the same kick_probe
with q_I/g_K/S_G/H/persist all off (the pure Z/M substrate: use_qI=False -> q_I==1 -> z*q_I*I_I == z*I_I).

This is the load-bearing verification for the user's "port faithfully + parity-check vs the canonical
implementation" requirement (Z/M migration 2026-07-22). Two clauses:
  - unit level: apply_currents composition (z scales E inhibition, eta_m*m subtracted, E-only) == mz
  - engine level: full simulate_kick E_spk_bool bit-identical to mz AND != baseline (z/m actually engaged)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

# aggressive-but-partial z/m params so z depletes + m accumulates visibly in T=200ms (engages the paths
# without saturating the raster); exact values are irrelevant to parity (both engines use the same).
ZM = dict(tau_z=200.0, I_th_EI=0.0, tau_adp=200.0, eta_m=0.5)


def _build_substrate(seed=1):
    from params import Params
    from connectivity import place_neurons, build_connectivity

    p = Params(L=1.0, density=400.0, T=200.0, dt=0.1, seed=seed, nu_ext_ratio=1.0)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    return p, net, pos, labels, NE, NI


def test_apply_currents_composition_matches_mz():
    """Unit: with q_I==1 (use_qI=False), slow_field's E-cell I_net == mz's for arbitrary z/m state."""
    N, NE = 10, 8
    core_mask_E = np.zeros(NE, bool); core_mask_E[:2] = True
    posE = np.random.default_rng(0).random((NE, 2)); posI = np.random.default_rng(1).random((N - NE, 2))
    sf = SpatialSlowField(N, 18.0, posE, posI, 1.0, core_mask_E=core_mask_E,
                          cfg=SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_z=True, use_m=True, **ZM))
    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_z=True, use_m=True, **ZM), NE=NE, core_mask_E=core_mask_E)
    # same non-uniform z/m state on both
    zst = np.linspace(0.2, 1.0, N); mst = np.linspace(0.0, 3.0, N)
    for eng in (sf, mz):
        eng.z = zst.copy(); eng.m = mst.copy()
        eng.z[NE:] = 1.0; eng.m[NE:] = 0.0  # I cells pinned (never updated)
    I_E = np.arange(N, dtype=float) + 1.0
    I_I = np.arange(N, dtype=float) * 0.5 + 0.3
    assert np.array_equal(sf.apply_currents(I_E, I_I), mz.apply_currents(I_E, I_I))


def test_engine_byte_parity_zm_matches_canonical_mz_and_engages():
    """Engine: full simulate_kick with z+m on -> E_spk_bool bit-identical to canonical MZSlowVars,
    and DIFFERENT from the slow=None baseline (proves z/m are actually engaged, not a no-op)."""
    from kick_probe import simulate_kick

    SEED = 1
    p, net, pos, labels, NE, NI = _build_substrate(SEED)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    center = np.array([p.L / 2, p.L / 2])
    core_mask_E = np.zeros(NE, bool)

    def run(slow):
        net["rng"] = np.random.default_rng(SEED)  # identical noise realization each run
        return simulate_kick(p, net, 5.0, slow=slow, kick_center=center, r_kick=0.3,
                             t_kick=50.0, V_th_per_neuron=vth, verbose=False)

    sf = SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L, core_mask_E=core_mask_E,
                          cfg=SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_z=True, use_m=True, **ZM))
    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_z=True, use_m=True, **ZM), NE=NE, core_mask_E=core_mask_E)

    res_base = run(None)
    res_sf = run(sf)
    res_mz = run(mz)

    # LOAD-BEARING: ported z+m == canonical z+m, bit-for-bit
    assert np.array_equal(res_sf["E_spk_bool"], res_mz["E_spk_bool"]), "slow_field z+m != canonical MZSlowVars"
    assert np.array_equal(res_sf["rate_E"], res_mz["rate_E"])
    assert np.array_equal(res_sf["rate_I"], res_mz["rate_I"])
    # non-trivial: z/m actually changed the dynamics vs the plain substrate
    assert res_sf["E_spk_bool"].sum() > 0
    assert not np.array_equal(res_sf["E_spk_bool"], res_base["E_spk_bool"]), "z+m had no effect (not engaged)"


def test_H_active_sensor_builds_on_localized_focus_while_global_starves():
    """Z/M migration fix: with a spatially-LOCALIZED persistence focus (the Z/M bursting focus is core-
    localized on the L=20 sheet), H_sensor='active' (mean Phi over cells >20% of peak) builds H, while
    'global' (spatial mean) is diluted by inactive cortex and starves. Exercises the real use_H branch:
    use_persist=False freezes p at the injected focus, so H integrates the sensor of a fixed field."""
    N, NE = 200, 160
    rng = np.random.default_rng(0)
    posE = rng.random((NE, 2)) * 20.0; posI = rng.random((N - NE, 2)) * 20.0

    def build(sensor):
        c = SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_SG=True, alpha_G=16.0,
                                   use_persist=False, use_H=True, alpha_H=16.0, tau_H=100.0, H_sensor=sensor)
        c.validate()
        return SpatialSlowField(N, 18.0, posE, posI, 20.0, cfg=c)

    Hs = {}
    for sensor in ("global", "active"):
        sf = build(sensor)
        sf.p[:] = 0.0; sf.p[:3, :3] = 0.9                 # localized focus (~1% of the n_grid lattice)
        for _ in range(1000):                             # p frozen (use_persist=False) -> H integrates the sensor
            sf.step(np.zeros(N, bool), None, 1.0)
        Hs[sensor] = sf.H
    assert Hs["active"] > 0.5 and Hs["global"] < 0.05, f"active must build, global must starve: {Hs}"


def test_H_sensor_invalid_raises():
    c = SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_SG=True, use_H=True, H_sensor="bogus")
    try:
        c.validate()
    except ValueError as e:
        assert "H_sensor" in str(e); return
    raise AssertionError("invalid H_sensor must raise")
