"""FCXR-ION B1: ion state class + IonHomeostaticMZAdapter (plan §7, spec §6/§13).

Numbering follows the plan's 16-row test list.  Byte-parity, the engine's hasattr guards, the
empty-voxel contract and the heterogeneous initializer are all locked here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src", "snn_engine"), os.path.join(ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from params import Params                                        # noqa: E402
from connectivity import place_neurons, build_connectivity       # noqa: E402
from kick_probe import simulate_kick                             # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig            # noqa: E402

import src.topic4_fcxr_ion as ION                                # noqa: E402
from src.snn_engine.ion_homeostasis import (                     # noqa: E402
    IonHomeostasis, IonHomeostasisConfig, IonHomeostaticMZAdapter, IonSafetyError,
    build_from_rate_field, cell_to_voxel, resting_state,
)

SEED = 1
DT = 0.1
Q1 = ION.q_ion_from_fprime(1.0)
ENGINE_VERSIONS = os.path.join(ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity",
                               "engine_versions.json")


# ------------------------------------------------------------------ substrates
def _net(L=6.0, T=250.0):
    p = Params(L=L, density=100.0, T=T, dt=DT, nu_ext_ratio=0.9, seed=SEED)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    return p, net, NE, NI


def _arm_c_cfg():
    import run_topic4_mz_fcxr as FCXR                            # noqa: E402
    return FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                        fail_on_clip=False, rec_sat_g=21.6)


def _mz(net, NE, cfg=None):
    N = len(net["labels"])
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**(cfg or _arm_c_cfg())), NE=NE)


def _grid_cfg(n_grid, L, **kw):
    return IonHomeostasisConfig(q_ion=Q1, n_grid=n_grid, dx_mm=L / n_grid, **kw)


def _uniform_voxel_map(n_cells, n_grid, per_voxel=None):
    """Deterministic even occupancy: cell i -> voxel i % nv."""
    nv = n_grid * n_grid
    if per_voxel is None:
        per_voxel = int(np.ceil(n_cells / nv))
    return np.repeat(np.arange(nv, dtype=np.int32), per_voxel)[:n_cells]


# =====================================================================================
#  1 / 4 / 4b / 5 / 5b -- adapter protocol and byte parity
# =====================================================================================
def _run(slow, p, net):
    net["rng"] = np.random.default_rng(SEED)
    return simulate_kick(p, net, 0.0, slow=slow, t_kick=1e9, early_stop_runaway=False)


def _fingerprint(res):
    return (hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest(),
            hashlib.sha1(np.asarray(res["rate_E"]).tobytes()).hexdigest(),
            hashlib.sha1(np.asarray(res["rate_I"]).tobytes()).hexdigest())


def test_1_adapter_off_is_byte_identical_to_bare_mz_slow_vars():
    p, net, NE, NI = _net()
    base = _fingerprint(_run(_mz(net, NE), p, net))
    assert _fingerprint(_run(IonHomeostaticMZAdapter(_mz(net, NE), None), p, net)) == base
    N = NE + NI
    cfg = _grid_cfg(4, p.L, enabled=False)
    ions = resting_state(N, NE, _uniform_voxel_map(N, 4), cfg)
    assert _fingerprint(_run(IonHomeostaticMZAdapter(_mz(net, NE), ions), p, net)) == base
    assert ions.n_updates == 0                      # a disabled layer must not integrate either


def test_2_existing_Z_M_X_update_order_is_unchanged():
    """Step-by-step values of the existing slow variables must be identical with the adapter in
    place -- the ion layer runs AFTER mz.step, never interleaved with it."""
    cfg = dict(_arm_c_cfg())
    cfg.update(use_z=True, use_m=True)
    p, net, NE, NI = _net()
    N = NE + NI
    labels = net["labels"]
    rng = np.random.default_rng(7)
    spikes = [rng.random(N) < 0.02 for _ in range(60)]

    bare = _mz(net, NE, cfg)
    wrapped_mz = _mz(net, NE, cfg)
    icfg = _grid_cfg(4, p.L)
    ad = IonHomeostaticMZAdapter(wrapped_mz, resting_state(N, NE, _uniform_voxel_map(N, 4), icfg))
    for spk in spikes:
        bare.step(spk, labels, DT)
        ad.step(spk, labels, DT)
        assert np.array_equal(bare.z, wrapped_mz.z)
        assert np.array_equal(bare.m, wrapped_mz.m)
        assert np.array_equal(bare.x_relay, wrapped_mz.x_relay)
        assert np.array_equal(bare.y, wrapped_mz.y)


def test_3_I_cell_coupling_is_a_current_not_a_conductance():
    """The engine discards g_rel/g_rev for I cells (spec §5).  The ion term must therefore appear in
    `drive` for I cells too, and must leave g_rel/g_rev untouched to the byte."""
    p, net, NE, NI = _net()
    N = NE + NI
    mz_a, mz_b = _mz(net, NE), _mz(net, NE)
    cfg = _grid_cfg(4, p.L, I_bias_E=0.3, I_bias_I=-0.2)
    ions = resting_state(N, NE, _uniform_voxel_map(N, 4), cfg)
    ions.K_o_grid += 0.4                              # move E_K so the term is non-zero
    ions._refresh_membrane_state()
    ad = IonHomeostaticMZAdapter(mz_a, ions)

    I_E = np.full(N, 3.0)
    I_I = np.full(N, 1.0)
    I_E_rec = np.full(N, 1.0)
    d0, g0, v0 = mz_b.membrane_terms(I_E, I_I, net["labels"], I_E_rec=I_E_rec)
    d1, g1, v1 = ad.membrane_terms(I_E, I_I, net["labels"], I_E_rec=I_E_rec)

    expect = ions.membrane_current()
    expect_bias = np.where(np.arange(N) < NE, 0.3, -0.2)
    assert np.allclose(d1 - d0, expect + expect_bias, rtol=0, atol=1e-15)
    assert np.abs(d1[NE:] - d0[NE:]).min() > 0        # every I cell really moved
    assert np.array_equal(g1, g0) and np.array_equal(v1, v0)
    assert np.all(g1[NE:] == 0.0) and np.all(v1[NE:] == 0.0)


def test_4_capability_passthrough():
    p, net, NE, NI = _net()
    mz = _mz(net, NE)
    ad = IonHomeostaticMZAdapter(mz, None)
    assert ad.NE == mz.NE
    assert ad.cfg is mz.cfg
    assert np.array_equal(ad.ee_relay_send, mz.ee_relay_send)
    assert ad.uses_conductance_membrane() == mz.uses_conductance_membrane()
    assert ad.uses_split_excitation() == mz.uses_split_excitation()
    assert ad.uses_ee_relay() == mz.uses_ee_relay()
    assert ad.threshold(18.0) == mz.threshold(18.0)


def test_4b_absent_attributes_stay_absent():
    """Synthesising nE / q_I / uses_shunt would flip the engine's hasattr guards onto a different
    execution path -- a silent contamination that no numeric test would catch."""
    p, net, NE, NI = _net()
    ad = IonHomeostaticMZAdapter(_mz(net, NE), None)
    for name in ("nE", "q_I", "uses_shunt"):
        assert not hasattr(ad, name), name
        with pytest.raises(AttributeError):
            getattr(ad, name)


def test_5_apply_currents_is_symmetric_with_membrane_terms():
    p, net, NE, NI = _net()
    N = NE + NI
    # a CURRENT-membrane config: the engine calls apply_currents, not membrane_terms
    cfg = dict(membrane_mode="current", use_z=True, use_m=False, use_phi=False)
    mz_a, mz_b = _mz(net, NE, cfg), _mz(net, NE, cfg)
    icfg = _grid_cfg(4, p.L, I_bias_E=0.25, I_bias_I=0.1)
    ions = resting_state(N, NE, _uniform_voxel_map(N, 4), icfg)
    ions.K_o_grid += 0.3
    ions._refresh_membrane_state()
    ad = IonHomeostaticMZAdapter(mz_a, ions)
    I_E, I_I = np.full(N, 2.5), np.full(N, 1.5)
    base = mz_b.apply_currents(I_E, I_I, net["labels"])
    got = ad.apply_currents(I_E, I_I, net["labels"])
    expect = ions.membrane_current() + np.where(np.arange(N) < NE, 0.25, 0.1)
    assert np.allclose(got - base, expect, rtol=0, atol=1e-15)


def test_5b_full_conductance_integration_through_simulate_kick():
    """Not a unit call: the real engine's full-conductance branch with the ion layer live."""
    p, net, NE, NI = _net(T=120.0)
    N = NE + NI
    cfg = _grid_cfg(4, p.L)
    ions = resting_state(N, NE, _uniform_voxel_map(N, 4), cfg)
    res = _run(IonHomeostaticMZAdapter(_mz(net, NE), ions), p, net)
    assert np.all(np.isfinite(res["rate_E"])) and res["E_spk_bool"].sum() > 0
    assert ions.n_updates == int(round(p.T / DT)) // ions.steps_per_block(DT)
    assert np.all(ions.Na_i_all > 0) and np.all(ions.K_o_grid > 0)
    assert ions.Na_i_all.max() > ION.NA_I0            # spikes really loaded Na
    assert ions.K_o_grid.max() > ION.K_O0


# =====================================================================================
#  6 / 6b / 7 / 7b / 8 / 11 -- the deviation form at the state-class level
# =====================================================================================
def test_6_resting_equilibrium_does_not_move():
    N, NE, ng = 400, 320, 4
    ions = resting_state(N, NE, _uniform_voxel_map(N, ng), _grid_cfg(ng, 2.5))
    for _ in range(50):
        ions.update()
    assert np.max(np.abs(ions.Na_i_all - ION.NA_I0)) < 1e-15
    assert np.max(np.abs(ions.K_o_grid - ION.K_O0)) < 1e-15


def test_6b_empty_voxel_resting_equilibrium_is_exact():
    """An empty voxel is a SAMPLING GAP: the pump term takes the unresolved tissue's resting value.
    rev3's masking would make it accumulate K at +0.28221 mM/s and Gate H would fail by design."""
    N, NE, ng = 300, 240, 4
    vox = _uniform_voxel_map(N, ng)
    vox = np.where(vox == 5, 6, vox)                  # voxel 5 is now empty
    ions = resting_state(N, NE, vox, _grid_cfg(ng, 2.5))
    assert ions.n_per_grid[5] == 0
    for _ in range(200):
        ions.update()
    assert np.max(np.abs(ions.K_o_grid - ION.K_O0)) < 1e-15


def test_7_single_spike_moves_Na_by_q_ion_and_voxel_K_by_beta_q_over_n():
    N, NE, ng = 400, 320, 4
    vox = _uniform_voxel_map(N, ng)
    ions = resting_state(N, NE, vox, _grid_cfg(ng, 2.5))
    spk = np.zeros(N, bool)
    spk[3] = True
    ions.accumulate(spk)
    ions.update()
    g = int(vox[3])
    n_g = int(ions.n_per_grid[g])
    assert ions.Na_i_all[3] - ION.NA_I0 == pytest.approx(Q1, rel=1e-12)
    assert np.max(np.abs(np.delete(ions.Na_i_all, 3) - ION.NA_I0)) < 1e-12
    flat = ions.K_o_grid.ravel()
    assert flat[g] - ION.K_O0 == pytest.approx(ION.BETA * Q1 / n_g, rel=1e-6)


def test_7b_deviation_form_is_exact_independently_of_occupancy():
    """dNa is EXACTLY zero: no averaging is involved, so the deviation cancels bit-for-bit.

    dK sits at float summation-rounding level instead, because the per-voxel pump term averages
    n_g values before subtracting I_pump_0.  The point of the gate is the fifteen-orders-of-
    magnitude separation from the two broken forms, which leave 0.28221 mM/s standing
    (locked by the reverse regressions in test_topic4_fcxr_ion.py and by test_11 below).
    """
    for per_voxel in (1, 7, 40):
        ng = 3
        N = ng * ng * per_voxel
        ions = resting_state(N, int(0.8 * N), _uniform_voxel_map(N, ng), _grid_cfg(ng, 2.0))
        dNa, dK = ions.derivatives(np.zeros(N))
        assert np.max(np.abs(dNa)) == 0.0
        assert np.max(np.abs(dK)) < 1e-12
        assert np.max(np.abs(dK)) < 1e-10 * 2.0 * ION.BETA * ION.I_PUMP_0


def test_8_pump_3_to_2_stoichiometry_at_the_state_level():
    N, NE, ng = 400, 320, 4
    ions = resting_state(N, NE, _uniform_voxel_map(N, ng), _grid_cfg(ng, 2.5))
    ions.Na_i_all[:] = 21.0
    ions._refresh_membrane_state()
    dNa, dK = ions.derivatives(np.zeros(N))
    Ip = ION.pump_flux(21.0, ION.K_O0)
    assert dNa[0] == pytest.approx(-3.0 * (Ip - ION.I_PUMP_0), rel=1e-12)
    pump_part = -2.0 * ION.BETA * (Ip - ION.I_PUMP_0)
    assert float(dK.ravel()[0]) == pytest.approx(pump_part, rel=1e-12)


def test_11_empty_voxel_zeroes_only_the_spike_excess():
    N, NE, ng = 300, 240, 4
    vox = np.where(_uniform_voxel_map(N, ng) == 5, 6, _uniform_voxel_map(N, ng))
    ions = resting_state(N, NE, vox, _grid_cfg(ng, 2.5))
    ions.K_o_grid[:] = 5.0                            # away from rest: clearance/glia must still act
    ions._refresh_membrane_state()
    K_before = ions.K_o_grid.ravel()[5]
    ions.update()
    K_after = ions.K_o_grid.ravel()[5]
    assert K_after < K_before                         # clearance + glia act on the empty voxel too
    dt_s = ions.cfg.dt_ion_ms * 1e-3
    expect = K_before + dt_s * (-ION.bath_clearance(5.0) - (ION.glia_uptake(5.0) - ION.I_GLIA_0))
    assert K_after == pytest.approx(expect, rel=1e-9)   # pump deviation is 0, NOT -J_K_0


# =====================================================================================
#  6c -- heterogeneous initializer on this network's own rate field
# =====================================================================================
def _smooth_rate_field(N, NE, vox, ng, scatter=0.0, seed=0):
    """Rate rises smoothly across the sheet; E and I differ.  scatter=0 isolates discretisation
    from per-cell sampling noise."""
    rng = np.random.default_rng(seed)
    col = (vox % ng) / max(ng - 1, 1)
    base = np.where(np.arange(N) < NE, 4.0, 9.0) * (1.0 + 1.2 * col)
    if scatter:
        base = base * rng.uniform(1 - scatter, 1 + scatter, N)
    return base[:NE], base[NE:]


def test_6c_heterogeneous_init_residual_passes_and_scalar_init_does_not():
    N, NE, ng = 4000, 3200, 10
    vox = _uniform_voxel_map(N, ng)
    rE, rI = _smooth_rate_field(N, NE, vox, ng, scatter=0.3, seed=2)
    cfg = _grid_cfg(ng, 6.3246)
    ions, rep = build_from_rate_field(N, NE, vox, cfg, rE, rI, return_report=True)
    dNa, dK = ions.derivatives(np.concatenate([rE, rI]))
    assert np.quantile(np.abs(dNa), 0.95) < 1e-10
    assert np.quantile(np.abs(dNa), 0.99) < 1e-10
    assert np.max(np.abs(dNa)) < 1e-8
    assert np.max(np.abs(dK)) < 1e-8
    assert rep["converged"] and rep["n_empty_voxels"] == 0

    scalar = ION.scalar_steady_state_init(rE, rI, vox[:NE], vox[NE:], n_grid=ng,
                                          q_ion=cfg.q_ion, dx_mm=cfg.dx_mm)
    assert scalar["q99_abs_dNa_dt"] > 1e5 * max(np.quantile(np.abs(dNa), 0.99), 1e-14)


# =====================================================================================
#  9 / 10 / 12 / 13 / 14 / 15 -- numerical contract
# =====================================================================================
def test_9_finite_volume_K_budget_closes_through_the_integrator():
    N, NE, ng = 900, 720, 3
    vox = _uniform_voxel_map(N, ng)
    ions = resting_state(N, NE, vox, _grid_cfg(ng, 2.0))
    rng = np.random.default_rng(5)
    dt_s = ions.cfg.dt_ion_ms * 1e-3
    for _ in range(60):
        spk = rng.random(N) < 0.05
        ions.accumulate(spk)
        K0 = ions.K_o_grid.copy()
        Ip = ions.pump_flux_all.copy()
        counts = ions._cell_spikes.astype(float).copy()
        n = ions.n_per_grid.astype(float)
        src = ION.BETA * ions.cfg.q_ion * (np.bincount(vox, weights=counts, minlength=ng * ng) / n)
        Ip_bar = np.bincount(vox, weights=Ip, minlength=ng * ng) / n
        sink = (-2.0 * ION.BETA * (Ip_bar.reshape(K0.shape) - ION.I_PUMP_0)
                - ION.bath_clearance(K0) - (ION.glia_uptake(K0) - ION.I_GLIA_0))
        dif = ION.diffusion_term(K0, dx_mm=ions.cfg.dx_mm)
        ions.update()
        delta = float(ions.K_o_grid.sum() - K0.sum())
        budget = float(src.sum()) + dt_s * (float(sink.sum()) + float(dif.sum()))
        assert abs(budget - delta) / max(abs(delta), 1e-12) < 1e-10
        assert abs(float(dif.sum())) < 1e-12          # zero-flux boundary: no net diffusive flux


def test_10_zero_flux_boundary_net_flux_is_zero_on_every_grid():
    rng = np.random.default_rng(11)
    for ng in (3, 5, 16, 32):
        K = 4.0 + rng.uniform(0, 3, (ng, ng))
        assert abs(float(ION.diffusion_term(K, dx_mm=20.0 / ng).sum())) < 1e-11


def test_12_grid_resolution_preserves_total_budget_and_the_coarse_grained_field():
    """Grid invariance of the finite-volume scheme (plan §7 row 12): the total extracellular K
    content and the coarse-grained field agree.  Per-voxel equality is NOT expected."""
    L, N, NE = 6.3246, 4000, 3200
    totals, fields = {}, {}
    for ng in (5, 10, 20):
        vox = _uniform_voxel_map(N, ng)
        rE, rI = _smooth_rate_field(N, NE, vox, ng, scatter=0.0)
        cfg = _grid_cfg(ng, L)
        ions = build_from_rate_field(N, NE, vox, cfg, rE, rI)
        rates = np.concatenate([rE, rI])
        for _ in range(400):
            ions._cell_spikes[:] = np.round(rates * cfg.dt_ion_ms * 1e-3 * 1000).astype(np.int32)
            ions._cell_spikes[:] = 0                  # relaxation only: the source is in the init
            ions.update()
        totals[ng] = ions.total_extracellular_K()
        fields[ng] = ions.K_o_grid.copy()
    ref = totals[10]
    for ng, tot in totals.items():
        assert abs(tot - ref) / ref < 1e-3, (ng, tot, ref)
    coarse = fields[20].reshape(5, 4, 5, 4).mean(axis=(1, 3))
    assert np.max(np.abs(coarse - fields[5])) / np.mean(fields[5]) < 1e-2


def test_13_multi_rate_convergence():
    """Numerical convergence of the ion sub-step.  This is a NUMERICS check only -- agreement under
    dt-halving says nothing about whether the equations themselves are right."""
    finals = {}
    for dt_ion in (2.0, 1.0, 0.5, 0.25):
        N, NE, ng = 900, 720, 3
        ions = resting_state(N, NE, _uniform_voxel_map(N, ng),
                             _grid_cfg(ng, 2.0, dt_ion_ms=dt_ion))
        ions.K_o_grid[1, 1] = 5.0                     # local perturbation, then relax
        ions._refresh_membrane_state()
        for _ in range(int(round(200.0 / dt_ion))):
            ions.update()
        finals[dt_ion] = ions.K_o_grid.copy()
    d_coarse = np.max(np.abs(finals[2.0] - finals[1.0]))
    d_mid = np.max(np.abs(finals[1.0] - finals[0.5]))
    d_fine = np.max(np.abs(finals[0.5] - finals[0.25]))
    assert d_fine < d_mid < d_coarse
    assert d_fine < 1e-4


def test_14_checkpoint_restart_identity():
    N, NE, ng = 900, 720, 3
    vox = _uniform_voxel_map(N, ng)
    rng = np.random.default_rng(3)
    spikes = [rng.random(N) < 0.04 for _ in range(80)]

    a = resting_state(N, NE, vox, _grid_cfg(ng, 2.0))
    for spk in spikes:
        a.accumulate(spk)
        a.update()

    b = resting_state(N, NE, vox, _grid_cfg(ng, 2.0))
    for spk in spikes[:40]:
        b.accumulate(spk)
        b.update()
    sd = json.loads(json.dumps(b.state_dict(), default=lambda x: x.tolist()))
    c = resting_state(N, NE, vox, _grid_cfg(ng, 2.0))
    c.load_state_dict({k: np.asarray(v) if isinstance(v, list) else v for k, v in sd.items()})
    for spk in spikes[40:]:
        c.accumulate(spk)
        c.update()

    assert np.array_equal(a.Na_i_all, c.Na_i_all)
    assert np.array_equal(a.K_o_grid, c.K_o_grid)
    assert a.n_updates == c.n_updates


def test_15_safety_bounds_fail_fast_and_never_saturate():
    N, NE, ng = 400, 320, 4
    cfg = _grid_cfg(ng, 2.5)
    cfg.q_ion = 5.0                                   # absurd loading: must RAISE, not clip
    ions = resting_state(N, NE, _uniform_voxel_map(N, ng), cfg)
    with pytest.raises(IonSafetyError):
        for _ in range(200):
            ions.accumulate(np.ones(N, bool))
            ions.update()
    # a normal trajectory never touches the band and never goes negative
    ok = resting_state(N, NE, _uniform_voxel_map(N, ng), _grid_cfg(ng, 2.5))
    rng = np.random.default_rng(9)
    for _ in range(200):
        ok.accumulate(rng.random(N) < 0.03)
        ok.update()
    assert ok.Na_i_all.min() > 0 and ok.K_o_grid.min() > 0
    assert ok.Na_i_all.max() < ok.cfg.na_bounds[1] and ok.K_o_grid.max() < ok.cfg.ko_bounds[1]


def test_16_blessed_engine_files_are_unmodified():
    recorded = json.load(open(ENGINE_VERSIONS))
    for rel, expected in recorded.items():
        got = hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()
        assert got == expected, rel


def test_cell_to_voxel_matches_the_grid_contract():
    pos = np.array([[0.0, 0.0], [19.99, 19.99], [10.0, 0.0], [0.0, 10.0]])
    v = cell_to_voxel(pos, 20.0, 32)
    assert v[0] == 0
    assert v[1] == 32 * 32 - 1
    assert v[2] == 16
    assert v[3] == 16 * 32
