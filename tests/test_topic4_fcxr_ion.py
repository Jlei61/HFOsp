"""FCXR-ION B0: constitutive Na/K ion math, provenance table, voltage-unit audit,
initiation-site readout.

Every test below is one clause of
  docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md (rev4)
  docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md (rev3)
Tests are written BEFORE the implementation (plan §5/§6 "tests first").
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.topic4_fcxr_ion as ION  # noqa: E402


# =====================================================================================
#  T3 -- pure ion math (plan §5).  Each test = one clause of spec §3 / §4.
# =====================================================================================
def test_pump_reproduces_the_reference_equation_and_resting_value():
    """Clause: spec §3.1 verbatim pump form; I_pump(18.0, 4.0) == 0.02016 mM/s (4 s.f.)."""
    got = ION.pump_flux(18.0, 4.0)
    rho, Na_half, s_Na, K_half, s_K = 1.25, 25.0, 3.0, 5.5, 1.0
    hand = rho / (1 + np.exp((Na_half - 18.0) / s_Na)) / (1 + np.exp((K_half - 4.0) / s_K))
    assert got == pytest.approx(hand, rel=1e-15)
    assert float(f"{got:.4g}") == 0.02016
    assert ION.I_PUMP_0 == pytest.approx(got, rel=1e-15)


def test_pump_is_monotone_in_both_ions_and_bounded_by_rho():
    Na = np.linspace(5.0, 60.0, 200)
    Ko = np.linspace(1.0, 40.0, 200)
    assert np.all(np.diff(ION.pump_flux(Na, 4.0)) > 0)
    assert np.all(np.diff(ION.pump_flux(18.0, Ko)) > 0)
    assert ION.pump_flux(60.0, 40.0) < ION.RHO           # strictly below the ceiling in range
    assert ION.pump_flux(1e4, 1e4) <= ION.RHO            # and approaches it asymptotically
    assert ION.pump_flux(1e4, 1e4) == pytest.approx(ION.RHO, rel=1e-9)


def test_glia_uptake_saturates_and_is_not_a_linear_term():
    """Regression: the saturating glial term must NOT be folded into a linear reservoir
    (spec §4.2 -- folding it in deletes a positive-feedback component route B depends on)."""
    assert ION.glia_uptake(1e3) == pytest.approx(ION.G_GLIA, rel=1e-9)
    lo = ION.glia_uptake(20.0) - ION.glia_uptake(10.0)
    hi = ION.glia_uptake(40.0) - ION.glia_uptake(30.0)
    assert hi < 0.05 * lo                       # strongly sublinear == saturating


def test_K_i_algebraic_closure():
    assert ION.K_i_from_Na_i(18.0) == pytest.approx(140.0)
    assert ION.K_i_from_Na_i(20.07) == pytest.approx(140.0 + (18.0 - 20.07))


def test_E_K_resting_value():
    assert ION.E_K(4.0, 140.0) == pytest.approx(-94.71, abs=5e-3)
    assert ION.E_K_0 == pytest.approx(ION.E_K(4.0, 140.0), rel=1e-15)


def test_background_fluxes_are_constants_independent_of_fprime():
    J_Na_0, J_K_0 = ION.background_fluxes()
    assert float(f"{J_Na_0:.5g}") == 0.060474
    assert float(f"{J_K_0:.5g}") == 0.28221
    assert J_Na_0 == pytest.approx(3.0 * ION.I_PUMP_0, rel=1e-15)
    assert J_K_0 == pytest.approx(2.0 * ION.BETA * ION.I_PUMP_0, rel=1e-15)
    # not a function of f' -- f' only enters q_ion
    for fp in (0.5, 1.0, 2.0):
        assert ION.background_fluxes() == (J_Na_0, J_K_0)
        assert ION.q_ion_from_fprime(fp) == pytest.approx(J_Na_0 * fp / ION.R0_HZ, rel=1e-15)


def test_resting_state_is_an_exact_fixed_point():
    """CORE regression (spec §4.1/§4.2 deviation form): with no spikes, (Na,K_o)=(18,4)
    gives both derivatives EXACTLY zero -- structurally, not by cancellation of two constants."""
    K = np.full((4, 4), 4.0)
    n = np.full((4, 4), 40)
    r = np.zeros((4, 4))
    Ip = np.full((4, 4), ION.I_PUMP_0)
    assert ION.dNa_dt(18.0, 4.0, 0.0, q_ion=0.01) == pytest.approx(0.0, abs=1e-18)
    dK = ION.dKo_dt(K, r, Ip, n, q_ion=0.01, dx_mm=0.625)
    assert np.max(np.abs(dK)) < 1e-18


def test_reverse_regression_a_dropping_background_flux_breaks_the_K_fixed_point():
    """rev2's form (no background K flux) must return dK_o/dt = -2*beta*I_pump_0."""
    K = np.full((4, 4), 4.0)
    n = np.full((4, 4), 40)
    dK = ION.dKo_dt(K, np.zeros((4, 4)), np.full((4, 4), ION.I_PUMP_0), n,
                    q_ion=0.01, dx_mm=0.625, _broken_no_background=True)
    assert np.max(np.abs(dK + 2.0 * ION.BETA * ION.I_PUMP_0)) < 1e-12


def test_reverse_regression_b_rev3_constant_form_breaks_the_EMPTY_voxel():
    """rev3's constant form (keep J_K_0, zero the pump term on empty voxels) makes an empty
    voxel accumulate K at +0.28221 mM/s -- Gate H would fail by construction (spec §4.2 rev4 P0-1)."""
    K = np.full((3, 3), 4.0)
    n = np.zeros((3, 3), int)
    dK = ION.dKo_dt(K, np.zeros((3, 3)), np.full((3, 3), ION.I_PUMP_0), n,
                    q_ion=0.01, dx_mm=0.625, _broken_empty_voxel_no_tissue=True)
    assert np.max(np.abs(dK - 2.0 * ION.BETA * ION.I_PUMP_0)) < 1e-12


def test_empty_voxel_is_an_exact_resting_fixed_point():
    """spec §4.2 empty-voxel contract: n_g == 0 is a SAMPLING GAP, not a tissue-free region.
    Only the spike excess is zeroed; the pump term takes the unresolved tissue's resting value."""
    K = np.full((5, 5), 4.0)
    n = np.full((5, 5), 40)
    n[2, 2] = 0
    r = np.zeros((5, 5))
    Ip = np.full((5, 5), ION.I_PUMP_0)
    Ip[2, 2] = np.nan                       # a truly empty voxel has NO cells to average
    dK = ION.dKo_dt(K, r, Ip, n, q_ion=0.01, dx_mm=0.625)
    assert np.isfinite(dK).all()
    assert abs(dK[2, 2]) < 1e-18
    assert np.max(np.abs(dK)) < 1e-18


def test_fprime_domain_raises_on_nonpositive_and_allows_above_one():
    for bad in (0.0, -0.5):
        with pytest.raises(ValueError):
            ION.q_ion_from_fprime(bad)
    assert ION.q_ion_from_fprime(2.0) > ION.q_ion_from_fprime(1.0)     # f' > 1 is legal


def test_interictal_steady_state_reproduces_the_spec_table():
    q = ION.q_ion_from_fprime(1.0)
    Na, Ko = ION.interictal_steady_state(q, ION.R0_HZ)
    assert Na == pytest.approx(20.07, abs=5e-3)
    assert Ko == pytest.approx(4.11, abs=5e-3)
    _, Ko50 = ION.interictal_steady_state(q, 50.0)
    assert Ko50 == pytest.approx(5.28, abs=5e-3)


def test_ms_to_s_dimensional_contract():
    """spec §4.2b: engine dt is ms, fluxes are mM/s.  Doubling dt_ion doubles the CONTINUOUS
    increment but leaves the per-spike increment untouched (guards the 1000x error)."""
    a = ION.ion_increment_terms(Na_i=19.0, K_o=4.3, spike_count=3, dt_ion_ms=0.5, q_ion=0.01)
    b = ION.ion_increment_terms(Na_i=19.0, K_o=4.3, spike_count=3, dt_ion_ms=1.0, q_ion=0.01)
    assert b["continuous"] == pytest.approx(2.0 * a["continuous"], rel=1e-12)
    assert b["spike"] == pytest.approx(a["spike"], rel=1e-15)
    assert a["spike"] == pytest.approx(3 * 0.01, rel=1e-15)


# =====================================================================================
#  T3 (7c) -- heterogeneous analytic pre-equilibrium (spec §4.2c)
# =====================================================================================
def _synthetic_rate_field(seed=0, n_grid=5, with_empty=True):
    """E/I- and space-inhomogeneous rate field on an n_grid x n_grid sheet."""
    rng = np.random.default_rng(seed)
    n_cells = n_grid * n_grid * 40
    NE = int(0.8 * n_cells)
    voxel = np.repeat(np.arange(n_grid * n_grid), 40)
    rng.shuffle(voxel)
    gain = 1.0 + 1.5 * (voxel % n_grid) / max(1, n_grid - 1)          # spatial gradient
    rate = np.where(np.arange(n_cells) < NE, 4.0, 9.0) * gain          # E vs I baseline differs
    rate = rate * rng.uniform(0.7, 1.3, n_cells)                       # per-cell scatter
    if with_empty:
        keep = voxel != 0                                              # voxel 0 becomes empty
        rate, voxel = rate[keep], voxel[keep]
        NE = int((np.arange(n_cells)[keep] < NE).sum())
    return rate[:NE], rate[NE:], voxel[:NE], voxel[NE:], n_grid


def test_heterogeneous_steady_state_residuals_are_machine_zero():
    rE, rI, vE, vI, ng = _synthetic_rate_field()
    q = ION.q_ion_from_fprime(1.0)
    out = ION.heterogeneous_steady_state(rE, rI, vE, vI, n_grid=ng, q_ion=q, dx_mm=0.625)
    assert out["max_abs_dNa_dt"] < 1e-8
    assert out["max_abs_dKo_dt"] < 1e-8
    assert out["n_empty_voxels"] == 1
    assert np.all(out["Na_star"] > 0) and np.all(out["K_o_star"] > 0)


def test_reverse_regression_single_global_scalar_init_leaves_a_large_residual():
    """The heterogeneous initializer must be DISTINGUISHABLE from the scalar one
    (spec §4.2c: a single global r0 leaves a slow spatial re-arrangement 11 s cannot expose)."""
    rE, rI, vE, vI, ng = _synthetic_rate_field()
    q = ION.q_ion_from_fprime(1.0)
    het = ION.heterogeneous_steady_state(rE, rI, vE, vI, n_grid=ng, q_ion=q, dx_mm=0.625)
    sca = ION.scalar_steady_state_init(rE, rI, vE, vI, n_grid=ng, q_ion=q, dx_mm=0.625)
    assert sca["q99_abs_dNa_dt"] > 1e4 * max(het["q99_abs_dNa_dt"], 1e-12)
    assert sca["q99_abs_dNa_dt"] > 1e-3


def test_finite_volume_K_budget_closes():
    """Gate H item 2: sources - (pump recovery + clearance + glia) - diffusion net flux
    matches the change in total extracellular K to < 1e-10 relative."""
    rE, rI, vE, vI, ng = _synthetic_rate_field(seed=3)
    q = ION.q_ion_from_fprime(1.0)
    rep = ION.k_budget_closure(rE, rI, vE, vI, n_grid=ng, q_ion=q, dx_mm=0.625,
                               dt_ion_ms=0.5, n_steps=400)
    assert rep["relative_error"] < 1e-10
    assert abs(rep["diffusion_net_flux"]) < 1e-12


def test_zero_flux_boundary_has_zero_net_diffusive_flux():
    rng = np.random.default_rng(1)
    K = 4.0 + rng.uniform(0, 2, (7, 7))
    lap = ION.diffusion_term(K, dx_mm=0.625)
    assert abs(float(lap.sum())) < 1e-12
    uniform = ION.diffusion_term(np.full((7, 7), 4.0), dx_mm=0.625)
    assert np.max(np.abs(uniform)) < 1e-18


def test_pump_3_to_2_stoichiometry_identity():
    """Gate H item 3: the SAME I_pump enters Na with coefficient 3 and K with 2*beta."""
    Na, Ko = 21.0, 4.6
    Ip = ION.pump_flux(Na, Ko)
    na_term = ION.dNa_dt(Na, Ko, 0.0, q_ion=0.0)
    ko_term = ION.dKo_dt(np.array([[Ko]]), np.zeros((1, 1)), np.array([[Ip]]),
                         np.array([[40]]), q_ion=0.0, dx_mm=0.625,
                         _pump_term_only=True)
    assert na_term == pytest.approx(-3.0 * (Ip - ION.I_PUMP_0), rel=1e-14)
    assert float(ko_term[0, 0]) == pytest.approx(-2.0 * ION.BETA * (Ip - ION.I_PUMP_0), rel=1e-14)


# =====================================================================================
#  T4 -- provenance table and analytic feasibility (plan §6)
# =====================================================================================
def test_every_inherited_param_has_a_single_source_label():
    for name, row in ION.PARAM_TABLE.items():
        assert row["kind"] in ("inherited", "derived", "effective"), name
        assert row["source"], name
        if row["kind"] == "inherited":
            assert row["source"] in ION.ALLOWED_SOURCES, f"{name}: {row['source']}"
            assert ";" not in row["source"] and " and " not in row["source"], name


def test_no_effective_param_is_labelled_inherited():
    """The model's own closure assumptions must not masquerade as literature values."""
    for name in ("q_K_per_spike", "g_K_ion", "f_prime"):
        assert ION.PARAM_TABLE[name]["kind"] == "effective", name
    for name in ("J_Na_0", "J_K_0", "q_ion", "I_pump_0", "E_K_0"):
        assert ION.PARAM_TABLE[name]["kind"] == "derived", name


def test_analytic_feasibility_reproduces_the_spec_table():
    rep = ION.analytic_feasibility()
    rows = {r["f_prime"]: r for r in rep["rows"]}
    assert set(rows) == {0.25, 0.5, 1.0, 2.0, 4.0}
    for fp in (0.5, 1.0, 2.0):
        assert rows[fp]["in_candidate_set"] is True
    for fp in (0.25, 4.0):
        assert rows[fp]["in_candidate_set"] is False
    assert rows[1.0]["q_ion"] == pytest.approx(0.01454, abs=5e-6)
    assert rows[1.0]["Na_star"] == pytest.approx(20.07, abs=5e-3)
    assert rows[1.0]["K_o_star"] == pytest.approx(4.11, abs=5e-3)
    assert rows[1.0]["dE_K_interictal_mV"] == pytest.approx(1.11, abs=5e-3)
    assert rows[1.0]["K_o_star_50hz"] == pytest.approx(5.28, abs=5e-3)
    assert rows[1.0]["dE_K_50hz_mV"] == pytest.approx(8.69, abs=5e-3)
    assert rows[0.5]["dE_K_50hz_mV"] == pytest.approx(5.07, abs=5e-3)
    assert rows[2.0]["dE_K_50hz_mV"] == pytest.approx(14.38, abs=5e-3)
    assert rows[4.0]["dE_K_50hz_pct_Vth"] == pytest.approx(126.0, abs=0.6)


def test_relaxation_times_differ_by_the_documented_83x():
    rep = ION.analytic_feasibility()
    assert rep["tau_Na_s"] == pytest.approx(54.42, abs=0.01)
    assert rep["tau_Ko_s"] == pytest.approx(0.655, abs=0.001)
    assert rep["tau_ratio"] == pytest.approx(83.0, abs=1.0)


def test_feasibility_gate_rejects_a_broken_rest_fixed_point():
    ok = ION.analytic_feasibility()
    assert ok["gates"]["rest_fixed_point"] is True
    assert ok["gates"]["empty_voxel_fixed_point"] is True
    assert ok["gates"]["J_Na_0_positive"] is True
    assert ok["gates"]["all_concentrations_positive"] is True
    assert ok["status"] == "PASS"
    bad = ION.analytic_feasibility(_break="no_background")
    assert bad["gates"]["rest_fixed_point"] is False
    assert bad["status"] == "FAIL"
    bad2 = ION.analytic_feasibility(_break="empty_voxel_no_tissue")
    assert bad2["gates"]["empty_voxel_fixed_point"] is False
    assert bad2["status"] == "FAIL"


# =====================================================================================
#  T1 -- engine voltage-unit audit (plan §3).  Dimension only; NOT the value of g_K_ion.
# =====================================================================================
def test_engine_voltage_scale_is_mV():
    a = ION.audit_voltage_units()
    assert a["V_th_mV"] == 18.0 and a["V_reset_mV"] == 11.0 and a["V_L_mV"] == 0.0
    assert a["params_unit_comments"]["V_th"] == "mV"
    assert a["params_unit_comments"]["V_reset"] == "mV"
    assert a["params_unit_comments"]["J_ext_E"] == "mV"


def test_conductance_membrane_drive_is_mV_dimensioned():
    """V_inf = (drive + g_rev)/(1 + g_rel); adding Delta to drive moves V_inf by Delta/(1+g_rel)."""
    drive, g_rel, g_rev = 12.0, 3.0, 25.0
    v0 = ION.v_inf(drive, g_rel, g_rev)
    assert v0 == pytest.approx((12.0 + 25.0) / 4.0)
    d = 1.7
    assert ION.v_inf(drive + d, g_rel, g_rev) - v0 == pytest.approx(d / (1.0 + g_rel))
    a = ION.audit_voltage_units()
    assert a["drive_and_g_rev_share_units"] is True


def test_delta_EK_injection_is_dimensionally_consistent():
    a = ION.audit_voltage_units()
    inj = ION.ion_membrane_current(K_o=5.0, Na_i=20.0, g_K_ion=1.0, eta_pump=0.0)
    expect = 1.0 * (ION.E_K(5.0, ION.K_i_from_Na_i(20.0)) - ION.E_K_0)
    assert inj == pytest.approx(expect, rel=1e-14)
    assert a["delta_E_K_injection_unit"] == "mV"
    # the audit fixes the DIMENSION only; g_K_ion = 1 is a declared normalization (spec §4.3)
    assert a["g_K_ion_is_a_unit_audit_conclusion"] is False
    assert a["g_K_ion_reference_value"] == 1.0
    assert a["g_K_ion_kind"] == "effective reference normalization"


def test_eta_pump_is_locked_to_zero_in_B0_B2():
    assert ION.ETA_PUMP_B0_B2 == 0.0
    with pytest.raises(ValueError):
        ION.ion_membrane_current(K_o=5.0, Na_i=20.0, g_K_ion=1.0, eta_pump=0.3)


def test_ion_layer_does_not_touch_existing_e_gaba_or_e_k():
    """The FCXR arm-C substrate sets e_gaba = e_k = 0 (leak reversal).  The ion layer injects
    Delta E_K as an ADDITIVE CURRENT and must leave both existing reversals untouched."""
    a = ION.audit_voltage_units()
    assert a["substrate_e_gaba"] == 0.0
    assert a["substrate_e_k"] == 0.0
    assert a["ion_layer_modifies_e_gaba_or_e_k"] is False
    assert a["status"] == "CONFIRMED"


# =====================================================================================
#  T2 -- initiation-site readout + power precondition (plan §4)
# =====================================================================================
CORE_A = np.array([5.0, 10.0])
CORE_B = np.array([15.0, 10.0])


def _synthetic_event(n_steps=40, n_cells=300, origin=CORE_A, seed=0, both_sides=False):
    """Cells on a line between the cores; recruitment sweeps outward from `origin`."""
    rng = np.random.default_rng(seed)
    x = np.linspace(2.0, 18.0, n_cells)
    pos = np.stack([x, np.full(n_cells, 10.0)], axis=1)
    d = np.abs(x - origin[0])
    if both_sides:
        keep = np.ones(n_cells, bool)
    else:
        keep = np.abs(x - origin[0]) < 6.0             # ONLY the origin core participates
    onset = np.clip((d / d.max() * (n_steps - 6)).astype(int) + 1, 1, n_steps - 2)
    spk = np.zeros((n_steps, n_cells), bool)
    for i in range(n_cells):
        if keep[i]:
            spk[onset[i], i] = True
            spk[min(onset[i] + 1, n_steps - 1), i] = True
    return spk, pos, [dict(t_on=0.0, t_off=n_steps * 0.05, dur_ms=n_steps * 0.05)]


def test_initiation_site_scores_every_event_not_only_two_sided_ones():
    """The rev1 readout needs BOTH cores to participate; on the accepted pump-off arm that left
    2 of 22 events scoreable.  The new readout must score a single-core event."""
    spk, pos, ev = _synthetic_event(origin=CORE_A, both_sides=False)
    legacy = ION.two_sided_forward_fraction(spk, pos, CORE_A, CORE_B, ev, dt=0.05, core_r=1.5)
    assert legacy["n_direction_events"] == 0
    assert np.isnan(legacy["forward_event_fraction"])
    new = ION.initiation_site_readout(spk, pos, CORE_A, CORE_B, ev, dt=0.05, core_r=1.5)
    assert new["n_scoreable"] == 1


def test_initiation_site_assigns_by_earliest_5pct_centroid():
    for origin, want in ((CORE_A, "A"), (CORE_B, "B")):
        spk, pos, ev = _synthetic_event(origin=origin, both_sides=True)
        out = ION.initiation_site_readout(spk, pos, CORE_A, CORE_B, ev, dt=0.05, core_r=1.5)
        assert out["n_scoreable"] == 1
        assert out["per_event"][0]["core"] == want


def test_ambiguous_when_centroid_is_equidistant():
    mid = 0.5 * (CORE_A + CORE_B)
    spk, pos, ev = _synthetic_event(origin=mid, both_sides=True)
    out = ION.initiation_site_readout(spk, pos, CORE_A, CORE_B, ev, dt=0.05, core_r=1.5)
    assert out["per_event"][0]["core"] == "ambiguous"
    assert out["n_scoreable"] == 0
    assert out["frac_ambiguous"] == pytest.approx(1.0)


# =====================================================================================
#  T6 / T7 -- Gate H and f' selection adjudication (plan §8, §9.3)
# =====================================================================================
def _gate_h_all_ok():
    return {name: dict(ok=True) for name, _ in ION._GATE_H_ORDER} | {
        "heterogeneous_init_residual": dict(
            ok=True, q95_abs_dNa_dt=1e-12, q99_abs_dNa_dt=2e-12, max_abs_dNa_dt=5e-12,
            q95_abs_dKo_dt=1e-13, q99_abs_dKo_dt=2e-13, max_abs_dKo_dt=4e-13)}


def test_gate_H_passes_only_when_every_item_passes():
    assert ION.adjudicate_gate_H(_gate_h_all_ok())["status"] == "PASS"


@pytest.mark.parametrize("item,code", [
    ("resting_fixed_point", "FAIL_EQUILIBRIUM"),
    ("empty_voxel_fixed_point", "FAIL_EMPTY_VOXEL"),
    ("heterogeneous_init_residual", "FAIL_INIT_RESIDUAL"),
    ("k_budget_closure", "FAIL_BUDGET"),
    ("pump_stoichiometry", "FAIL_STOICHIOMETRY"),
    ("ions_off_byte_parity", "FAIL_PARITY"),
    ("dt_ion_convergence", "FAIL_NUMERICAL"),
])
def test_gate_H_maps_each_failure_to_its_registered_code(item, code):
    checks = _gate_h_all_ok()
    checks[item] = dict(checks[item], ok=False)
    assert ION.adjudicate_gate_H(checks)["status"] == code


def test_gate_H_is_unresolved_when_only_the_population_mean_was_measured():
    """spec §4.2c: a flat population mean is NOT evidence -- the per-cell / per-voxel q95, q99 and
    max must all be present, or the gate is UNRESOLVED rather than PASS."""
    checks = _gate_h_all_ok()
    checks["heterogeneous_init_residual"] = dict(ok=True, mean_abs_dNa_dt=1e-14,
                                                 mean_abs_dKo_dt=1e-15)
    out = ION.adjudicate_gate_H(checks)
    assert out["status"] == "UNRESOLVED"
    assert "q95_abs_dNa_dt" in out["reason"]


def test_gate_H_is_unresolved_when_an_item_was_never_measured():
    checks = _gate_h_all_ok()
    del checks["checkpoint_restart_identity"]
    out = ION.adjudicate_gate_H(checks)
    assert out["status"] == "UNRESOLVED" and "checkpoint_restart_identity" in out["reason"]


def _f_meas(dK=0.3, sigma=0.01, ratio=2.6, decay=0.31, monotone=True, resid=0.4):
    return dict(dK_peak_single_mM=dK, sigma_rest_K_mM=sigma,
                integration_ratio_5th_over_1st=ratio, na_excess_decay_frac_20s=decay,
                na_excess_monotone_nonincreasing=monotone,
                k_returns_within_1sigma_3s=True, k_residual_after_3s_in_sigma=resid)


def test_f_prime_gates_pass_a_healthy_candidate():
    out = ION.evaluate_f_prime_gates(_f_meas())
    assert out["admissible"] is True
    assert all(v["ok"] for v in out["gates"].values())


def test_f_prime_measurable_gate_uses_an_absolute_floor_not_only_sigma():
    """The rev2 gate was relative to sigma alone, so a better initialization made it EASIER to
    pass -- it measured the initialization, not the potassium signal."""
    tiny = ION.evaluate_f_prime_gates(_f_meas(dK=0.02, sigma=1e-6))
    assert tiny["gates"]["measurable"]["ok"] is False
    assert tiny["gates"]["measurable"]["threshold"] == 0.15


def test_f_prime_safety_ceiling_is_two_sided():
    out = ION.evaluate_f_prime_gates(_f_meas(dK=1.2))
    assert out["gates"]["measurable"]["ok"] is True and out["gates"]["safe"]["ok"] is False
    assert out["admissible"] is False


def test_f_prime_na_recovery_band_rejects_both_too_fast_and_too_slow():
    assert ION.evaluate_f_prime_gates(_f_meas(decay=0.60))["gates"]["recovery_Na"]["ok"] is False
    assert ION.evaluate_f_prime_gates(_f_meas(decay=0.05))["gates"]["recovery_Na"]["ok"] is False
    assert ION.evaluate_f_prime_gates(_f_meas(decay=0.31))["gates"]["recovery_Na"]["ok"] is True
    assert ION.evaluate_f_prime_gates(
        _f_meas(decay=0.31, monotone=False))["gates"]["recovery_Na"]["ok"] is False


def test_f_prime_integration_gate_separates_passing_from_supralinear():
    """Passing 2.38 is NOT evidence of supralinear accumulation: pure linear superposition at
    200 ms spacing already predicts 2.97."""
    mid = ION.evaluate_f_prime_gates(_f_meas(ratio=2.6))["gates"]["integration"]
    assert mid["ok"] is True and mid["supralinear"] is False
    hi = ION.evaluate_f_prime_gates(_f_meas(ratio=3.4))["gates"]["integration"]
    assert hi["ok"] is True and hi["supralinear"] is True
    assert hi["ratio_vs_linear"] > 1.0


def test_f_prime_tie_break_is_closest_to_one_not_largest():
    rows = [dict(f_prime=0.5, admissible=True), dict(f_prime=1.0, admissible=True),
            dict(f_prime=2.0, admissible=True)]
    assert ION.select_f_prime(rows)["selected"] == 1.0
    rows2 = [dict(f_prime=0.5, admissible=True), dict(f_prime=1.0, admissible=False),
             dict(f_prime=2.0, admissible=True)]
    # 0.5 and 2.0 are equidistant in absolute terms; the lower (weaker feedback) wins
    assert ION.select_f_prime(rows2)["selected"] == 0.5


def test_f_prime_all_three_failing_is_a_bounded_negative():
    rows = [dict(f_prime=f, admissible=False) for f in (0.5, 1.0, 2.0)]
    out = ION.select_f_prime(rows)
    assert out["status"] == "NO_GO_ION_SCALE" and out["selected"] is None


def test_a_changed_protocol_may_not_inherit_the_canonical_verdict():
    """CLAUDE.md §5: the pre-registered tier is fixed at planning time. A result measured on a
    DIFFERENT object keeps its per-gate table but must not carry the contract's verdict label."""
    rows = [dict(f_prime=f, admissible=False) for f in (0.5, 1.0, 2.0)]
    contract = ION.select_f_prime(rows)
    out = ION.withhold_canonical_verdict(contract, protocol_deviation="ran on 40k, not the "
                                         "registered small network",
                                         blocking_gates_are_open_loop="integration is open-loop")
    assert out["status"] == "UNRESOLVED_T7_PROTOCOL"
    assert out["selected"] is None
    assert out["contract_verdict_if_protocol_had_matched"] == "NO_GO_ION_SCALE"
    assert "NOT a mechanism NO-GO" in out["semantics"]
    assert "not because a mechanism was refuted" in out["b2_entry"]
    # the contract implementation itself must be left untouched for a future faithful run
    assert ION.select_f_prime(rows)["status"] == "NO_GO_ION_SCALE"


def test_withholding_also_applies_when_the_contract_would_have_selected():
    """Withholding is about the OBJECT, not about the answer being unwelcome: a would-be SELECTED
    verdict from a deviating protocol is withheld just the same."""
    rows = [dict(f_prime=0.5, admissible=False), dict(f_prime=1.0, admissible=True),
            dict(f_prime=2.0, admissible=False)]
    contract = ION.select_f_prime(rows)
    assert contract["status"] == "SELECTED" and contract["selected"] == 1.0
    out = ION.withhold_canonical_verdict(contract, protocol_deviation="x",
                                         blocking_gates_are_open_loop="y")
    assert out["status"] == "UNRESOLVED_T7_PROTOCOL"
    assert out["selected"] is None
    assert out["contract_verdict_if_protocol_had_matched"] == "SELECTED"


# =====================================================================================
#  T9 -- Gate B adjudication (plan §11)
# =====================================================================================
def _run(n_sc=30, fa=0.6, fb=0.4, drift=1e-3, wave=0.02, sat=0.05, rate=4.16, tag="t"):
    return dict(
        job=dict(tag=tag, conn_seed=1, noise_seed=402),
        pooled=dict(n_scoreable=n_sc, frac_A=fa, frac_B=fb, mean_rate_hz=rate,
                    ied_rate_hz=2.2, iei_median_ms=321.0, iei_cv=0.39,
                    duration_median_ms=10.0),
        ion=dict(q99_abs_dNa_dt=drift, q99_abs_dKo_dt=drift, Na_mean_first=20.0,
                 Na_mean_last=20.0, k_wave_far_over_event=wave,
                 pump_saturation_frac_of_rho=sat))


_TOL = dict(mean_rate_hz=dict(off=4.15805625, margin=0.8431070173327828, underpowered=False),
            duration_median_ms=dict(off=10.0, margin=2.0736441353327724, underpowered=False),
            ied_rate_hz=dict(off=2.2, margin=1.6733200530681511, underpowered=True),
            iei_cv=dict(off=0.39150098010197826, margin=0.44864692901289116, underpowered=True))
_TPL = dict(stable_k=2, source="epilepsiae_1146 adaptive_cluster")


def test_gate_B_accepts_when_all_six_pass():
    out = ION.adjudicate_gate_B([_run(tag=f"t{i}") for i in range(6)], _TOL, template_layer=_TPL)
    assert out["status"] == "ACCEPTED"
    assert out["b_real"]["n_direction_passing"] == 6


def test_gate_B_needs_five_of_six_not_a_bare_majority():
    runs = [_run(tag=f"t{i}") for i in range(4)] + [_run(n_sc=8, tag="bad1"),
                                                    _run(fb=0.05, tag="bad2")]
    out = ION.adjudicate_gate_B(runs, _TOL, template_layer=_TPL)
    assert out["b_real"]["n_direction_passing"] == 4
    assert out["status"] == "REJECTED"
    runs[-1] = _run(tag="ok")
    assert ION.adjudicate_gate_B(runs, _TOL, template_layer=_TPL)["status"] == "ACCEPTED"


def test_gate_B_direction_rule_needs_both_count_and_balance():
    assert not ION.adjudicate_gate_B([_run(n_sc=19)], _TOL,
                                     template_layer=_TPL)["per_trajectory"][0]["direction"]["ok"]
    assert not ION.adjudicate_gate_B([_run(fa=0.9, fb=0.1)], _TOL,
                                     template_layer=_TPL)["per_trajectory"][0]["direction"]["ok"]
    assert ION.adjudicate_gate_B([_run(fa=0.85, fb=0.15)], _TOL,
                                 template_layer=_TPL)["per_trajectory"][0]["direction"]["ok"]


def test_gate_B_underpowered_metrics_are_not_equivalence_evidence():
    """An UNDERPOWERED metric sitting inside its tolerance must not be counted as a binding pass."""
    runs = [_run(tag=f"t{i}") for i in range(6)]
    for r in runs:
        r["pooled"]["ied_rate_hz"] = 3.5          # inside the wide UNDERPOWERED margin
    out = ION.adjudicate_gate_B(runs, _TOL, template_layer=_TPL)
    assert out["status"] == "ACCEPTED"
    assert "ied_rate_hz" not in out["b_model"]["binding_metrics_outside"]
    assert out["per_trajectory"][0]["tolerance"]["ied_rate_hz"]["underpowered"] is True


def test_gate_B_rejects_a_binding_metric_outside_tolerance():
    runs = [_run(tag=f"t{i}") for i in range(6)]
    for r in runs:
        r["pooled"]["duration_median_ms"] = 14.0     # the Gate I-a failure mode
    out = ION.adjudicate_gate_B(runs, _TOL, template_layer=_TPL)
    assert out["status"] == "REJECTED"
    assert "duration_median_ms" in out["b_model"]["binding_metrics_outside"]


def test_gate_B_rejects_slow_ion_countdown_and_whole_sheet_K_wave():
    assert ION.adjudicate_gate_B([_run(drift=0.2, tag=f"t{i}") for i in range(6)], _TOL,
                                 template_layer=_TPL)["status"] == "REJECTED"
    assert ION.adjudicate_gate_B([_run(wave=0.5, tag=f"t{i}") for i in range(6)], _TOL,
                                 template_layer=_TPL)["status"] == "REJECTED"
    assert ION.adjudicate_gate_B([_run(sat=0.8, tag=f"t{i}") for i in range(6)], _TOL,
                                 template_layer=_TPL)["status"] == "REJECTED"


def test_gate_B_statements_keep_the_two_layers_apart():
    """CLAUDE.md §6.3: the data supports 'two stable templates exist'; the model supports 'events
    initiate at both registered cores'.  Neither licenses 'bidirectional propagation'."""
    out = ION.adjudicate_gate_B([_run(tag=f"t{i}") for i in range(6)], _TOL, template_layer=_TPL)
    assert "both registered cores" in out["allowed_statement"]
    assert "bidirectional" not in out["allowed_statement"]
    assert any("bidirectional" in f for f in out["forbidden_statements"])
    assert any("eta_pump" in f for f in out["forbidden_statements"])
    rej = ION.adjudicate_gate_B([_run(n_sc=5, tag=f"t{i}") for i in range(6)], _TOL,
                                template_layer=_TPL)
    assert "did not recover" in rej["allowed_statement"]
    assert "refut" not in rej["allowed_statement"]


# =====================================================================================
#  T7.1 -- adjudication repair (user rulings 2026-07-28)
# =====================================================================================
def _m2(dK=0.30, net=0.5, slope=-1e-4, se=1e-5, ratio=1.78, valid=True):
    return dict(dK_peak_single_mM=dK, na_net_decay_frac=net, na_tail_slope=slope,
                na_tail_slope_se=se, na_tail_slope_t=slope / se,
                k_returns_within_1sigma_3s=True, k_residual_after_3s_in_sigma=0.01,
                numerically_valid=valid, integration_ratio_5th_over_1st=ratio,
                integration_linear_at_workpoint=2.795, sigma_rest_K_mM=0.109)


def test_v2_measurable_gate_drops_the_sigma_term_and_keeps_the_absolute_floor():
    """Ruling 1(a): sigma_rest is the background other real events leave, not instrument noise."""
    assert "measurable_sigma_mult" not in ION.F_GATES_V2
    assert ION.F_GATES_V2["measurable_abs_floor_mM"] == 0.15
    # 0.2867 failed the old 0.547 threshold; under the repaired gate it passes on the floor
    out = ION.evaluate_f_prime_gates_v2(_m2(dK=0.2867))
    assert out["gates"]["measurable"]["ok"] is True
    assert out["gates"]["measurable"]["threshold"] == 0.15
    assert ION.evaluate_f_prime_gates_v2(_m2(dK=0.10))["gates"]["measurable"]["ok"] is False


def test_v2_safe_ceiling_still_binds():
    assert ION.evaluate_f_prime_gates_v2(_m2(dK=1.15))["gates"]["safe"]["ok"] is False
    assert ION.evaluate_f_prime_gates_v2(_m2(dK=0.574))["admissible"] is True


def test_v2_integration_is_non_blocking():
    """Ruling 4: an open-loop measurement may not gate the phase that tests the closed loop."""
    out = ION.evaluate_f_prime_gates_v2(_m2(ratio=1.0))
    assert out["admissible"] is True                       # a terrible ratio does NOT block
    assert out["diagnostics"]["integration"]["blocking"] is False
    assert "integration" not in out["gates"]
    assert "risk" in out["diagnostics"]["integration"]["rule"].lower()


def test_v2_background_is_diagnostic_only():
    out = ION.evaluate_f_prime_gates_v2(_m2())
    assert out["diagnostics"]["background"]["blocking"] is False
    assert "background" not in out["gates"]


def test_v2_na_recovery_uses_net_decay_and_tail_trend_not_monotonicity():
    """Ruling 3: per-sample and smoothed-envelope monotonicity are both dropped."""
    ok = ION.evaluate_f_prime_gates_v2(_m2(net=0.66, slope=-1e-4))["diagnostics"]["na_recovery"]
    assert ok["ok"] is True
    assert "monoton" not in json.dumps(ok["rule"]).replace("monotonicity are both", "")
    # no clear net decay -> fails
    assert not ION.evaluate_f_prime_gates_v2(
        _m2(net=0.02))["diagnostics"]["na_recovery"]["ok"]
    # persistently rising tail -> fails
    assert not ION.evaluate_f_prime_gates_v2(
        _m2(net=0.66, slope=+1e-3, se=1e-5))["diagnostics"]["na_recovery"]["ok"]
    # a transient up-jump that leaves the SLOPE non-positive is tolerated
    assert ION.evaluate_f_prime_gates_v2(
        _m2(net=0.66, slope=-5e-5, se=1e-4))["diagnostics"]["na_recovery"]["ok"] is True


def test_v2_na_recovery_is_not_a_hard_gate_but_numerical_validity_is():
    assert "na_recovery" not in ION.evaluate_f_prime_gates_v2(_m2())["gates"]
    assert ION.evaluate_f_prime_gates_v2(_m2(valid=False))["admissible"] is False


def test_tail_slope_averages_over_transient_up_jumps():
    dt = 0.0025
    n = int(round(20.0 / dt))
    y = 0.02 * np.exp(-np.arange(n) * dt / 30.0)
    y[int(0.9 * n)] += 0.003                                # one big background up-jump
    slope, se, t = ION._tail_slope(y, dt, 5.0)
    assert slope < 0                                        # the trend still reads as decaying


def test_coupled_jacobian_predicts_faster_decay_than_the_frozen_K_reference():
    """Ruling 2: freezing K_o understates clearance, because I_pump rises with K_o too."""
    for fp in (0.5, 1.0, 2.0):
        J, Na, Ko = ION.coupled_working_point_jacobian(fp)
        assert J.shape == (2, 2)
        assert J[0, 0] < 0 and J[1, 1] < 0                  # both self-terms restoring
        assert J[0, 1] < 0                                  # higher K_o clears Na faster
        frozen = 1.0 - np.exp(20.0 * J[0, 0])
        coupled, *_ = ION.coupled_na_decay_prediction(fp, 0.02, 0.30)
        assert coupled > frozen                             # coupling speeds it up
        assert np.isfinite(coupled) and coupled > 0.0


def test_strong_co_elevated_K_can_predict_an_undershoot_and_is_not_clipped():
    """A real property of the linearised coupled system, not a bug: J[0,1] < 0, so a large K
    excursion drives Na down for as long as it lasts.  With the measured ratio (dK ~ 0.57 mM at
    the event voxel vs dNa ~ 0.022 mM per cell) the prediction can exceed 100%, i.e. the excess is
    driven BELOW baseline before recovering.  The predictor must report that, never clip it."""
    weak, *_ = ION.coupled_na_decay_prediction(1.0, 0.02, 0.0)
    strong, *_ = ION.coupled_na_decay_prediction(1.0, 0.02, 0.60)
    assert weak < strong
    assert strong > 1.0                                     # undershoot regime, reported not clipped
    assert ION.coupled_na_decay_prediction(1.0, 0.02, 0.60, t_s=20.0)[0] == strong


def test_v2_selection_returns_a_provisional_candidate_not_a_mechanism_verdict():
    rows = [dict(f_prime=0.5, admissible=True), dict(f_prime=1.0, admissible=True),
            dict(f_prime=2.0, admissible=False)]
    out = ION.select_f_prime_v2(rows)
    assert out["status"] == "PROVISIONAL_CANDIDATE" and out["selected"] == 1.0
    assert "NOT a claim that the closed loop" in out["semantics"]
    none = ION.select_f_prime_v2([dict(f_prime=f, admissible=False) for f in (0.5, 1.0, 2.0)])
    assert none["status"] == "NO_ADMISSIBLE_SCALE" and none["selected"] is None


def test_small_network_contract_records_what_survives_and_what_is_abolished():
    """Ruling 5: the numerical tests survive; dynamic f' selection from the small nets does not."""
    assert "Gate H numerical tests" in ION.SMALL_NET_CONTRACT_RETAINED
    assert "empty-voxel fixed point" in ION.SMALL_NET_CONTRACT_RETAINED
    assert any("dynamic f'" in s for s in ION.SMALL_NET_CONTRACT_ABOLISHED)
    assert any("faithful reproduction" in s for s in ION.SMALL_NET_CONTRACT_ABOLISHED)


# =====================================================================================
#  B2.1 -- calibration-instrument repair (spec 2026-07-28-topic4-fcxr-ion-B2_1-lock.md)
# =====================================================================================
def test_signed_slope_recovers_a_known_ramp():
    dt = 0.1
    t = np.arange(200) * dt
    y = np.stack([3.0 + 0.02 * t, 5.0 - 0.007 * t, np.full_like(t, 9.0)], axis=1)
    s = ION.signed_secular_slope(y, dt)
    assert s == pytest.approx([0.02, -0.007, 0.0], abs=1e-12)


def test_signed_slope_separates_a_stationary_but_EVENTFUL_series_from_a_drifting_one():
    """THE discriminating test, and the whole reason B2.1 exists.

    A statistically stationary trace that keeps getting event-driven excursions has NO secular
    trend, but its q99 of |first differences| is large. The old gate read that as drift.
    """
    rng = np.random.default_rng(0)
    dt, n = 0.1, 1000
    base = np.full(n, 20.0)
    for k in range(0, n, 40):                       # an event every 4 s, fully relaxing
        base[k:k + 8] += 0.35 * np.exp(-np.arange(min(8, n - k)) / 3.0)
    stationary = base[:, None]
    drifting = (base + 0.02 * np.arange(n) * dt)[:, None]

    s_stat = abs(float(ION.signed_secular_slope(stationary, dt)[0]))
    s_drift = float(ION.signed_secular_slope(drifting, dt)[0])
    fd_stat = float(np.quantile(np.abs(np.diff(stationary[:, 0])) / dt, 0.99))

    assert s_stat < 1e-3                            # no secular trend, correctly
    assert s_drift == pytest.approx(0.02, abs=1e-3)  # the real drift, recovered
    assert fd_stat > 100 * s_stat                   # the OLD statistic is huge on the SAME trace
    assert fd_stat > 0.5


def test_slope_stats_reports_q95_q99_and_the_signed_mean():
    rng = np.random.default_rng(1)
    slopes = rng.normal(0.0, 1e-3, 5000)
    st = ION.slope_stats(slopes)
    assert st["q99_abs"] > st["q95_abs"] > 0
    assert abs(st["mean_signed"]) < 1e-4
    assert st["max_abs"] >= st["q99_abs"]


def test_b2_1_slope_bounds_are_derived_from_the_interictal_excursion():
    """The bound is 10% of each variable's own interictal excursion over the 10 s window --
    derived, not hand-picked, and locked before any signed-slope measurement existed."""
    q = ION.q_ion_from_fprime(1.0)
    Na, Ko = ION.interictal_steady_state(q, ION.R0_HZ)
    assert ION.B2_1_SLOPE_BOUND_NA == pytest.approx(0.10 * (Na - ION.NA_I0) / 10.0, rel=1e-2)
    assert ION.B2_1_SLOPE_BOUND_K == pytest.approx(0.10 * (Ko - ION.K_O0) / 10.0, rel=1e-2)
    assert ION.B2_1_SLOPE_BOUND_NA > ION.B2_1_SLOPE_BOUND_K      # Na excursion is much larger


def test_rate_shrinkage_pulls_noisy_cells_toward_their_voxel_and_leaves_confident_ones():
    voxel = np.array([0, 0, 0, 1, 1, 1])
    counts = np.array([2.0, 40.0, 60.0, 2.0, 40.0, 60.0])       # 10 s window -> Hz = counts/10
    r = ION.shrink_rate_field(counts, voxel, window_s=10.0, n_voxels=2, n0=20.0)
    raw = counts / 10.0
    vox0 = raw[:3].mean()
    # the 2-spike cell is pulled most of the way to its voxel mean
    assert abs(r[0] - vox0) < abs(raw[0] - vox0) * 0.15
    # the 60-spike cell keeps most of its own rate
    assert abs(r[2] - raw[2]) < abs(raw[2] - vox0) * 0.3
    assert r.shape == counts.shape


def test_rate_shrinkage_treats_E_and_I_separately():
    """E and I baselines differ; pooling them would drag I cells toward the E mean."""
    voxel = np.array([0, 0, 0, 0])
    countsE = np.array([40.0, 40.0])
    countsI = np.array([100.0, 100.0])
    rE = ION.shrink_rate_field(countsE, voxel[:2], window_s=10.0, n_voxels=1, n0=20.0)
    rI = ION.shrink_rate_field(countsI, voxel[2:], window_s=10.0, n_voxels=1, n0=20.0)
    assert rE == pytest.approx([4.0, 4.0])
    assert rI == pytest.approx([10.0, 10.0])


def test_damped_update_moves_half_way_and_is_capped():
    cur = np.array([4.0, 8.0])
    meas = np.array([6.0, 4.0])
    nxt = ION.damped_rate_update(cur, meas, alpha=0.5)
    assert nxt == pytest.approx([5.0, 6.0])
    assert ION.B2_1_MAX_UPDATES == 3
    assert ION.B2_1_ALPHA == 0.5


def test_b2_1_adjudication_needs_both_rate_convergence_and_the_slope_bound():
    ok = dict(rate_rel_change=0.02, slope_q99_Na=1e-3, slope_q99_K=1e-4, n_updates=2,
              independent_window=True)
    assert ION.adjudicate_b2_1_selfconsistency(ok)["status"] == "CONVERGED"
    assert ION.adjudicate_b2_1_selfconsistency(
        dict(ok, rate_rel_change=0.2))["status"] == "NOT_CONVERGED"
    assert ION.adjudicate_b2_1_selfconsistency(
        dict(ok, slope_q99_Na=0.5))["status"] == "NOT_CONVERGED"
    # a result measured on the SAME window used to derive the field does not count
    assert ION.adjudicate_b2_1_selfconsistency(
        dict(ok, independent_window=False))["status"] == "NOT_CONVERGED"


def test_matched_control_validity_is_structural_and_the_FIRST_kick_is_the_sanity_check():
    """The arms are bit-identical up to the freeze block, so validity is STRUCTURAL. The first
    kick's response is the sanity check; later kicks are where the feedback is allowed to act."""
    out = ION.adjudicate_matched_control(
        closed=dict(spikes=[1000, 1020], participants=[500, 710], peaks=[0.68, 0.67]),
        open_=dict(spikes=[1005, 1002], participants=[502, 560], peaks=[0.65, 0.83]),
        structurally_identical_until_freeze=True)
    assert out["status"] == "COMPARABLE"
    assert out["ratio"]["closed_2nd_over_1st"] < out["ratio"]["open_2nd_over_1st"]
    # the LATE divergence is reported as the effect, not as a matching failure
    assert out["effect"]["participants_rel_diff_by_kick"][1] > 0.2
    assert out["sanity"]["kick1_participants_rel_diff"] < 0.15


def test_matched_control_a_late_divergence_must_NOT_void_the_comparison():
    """Regression on my own mis-specification: pooling both kicks into one max voided the control
    exactly when the feedback effect was largest -- a criterion that cannot tell 'control broken'
    from 'effect present' is not a validity criterion."""
    out = ION.adjudicate_matched_control(
        closed=dict(spikes=[1000, 1020], participants=[500, 1500], peaks=[0.7, 0.7]),
        open_=dict(spikes=[1005, 1002], participants=[502, 500], peaks=[0.7, 0.9]),
        structurally_identical_until_freeze=True)
    assert out["status"] == "COMPARABLE"
    assert "ratio" in out


def test_matched_control_is_void_when_the_FIRST_kick_already_disagrees():
    bad = ION.adjudicate_matched_control(
        closed=dict(spikes=[3000, 3100], participants=[1500, 1600], peaks=[1.9, 2.0]),
        open_=dict(spikes=[1005, 1002], participants=[502, 500], peaks=[0.7, 0.8]),
        structurally_identical_until_freeze=True)
    assert bad["status"] == "UNRESOLVED_MATCHED_CONTROL"
    assert "ratio" not in bad


def test_matched_control_is_void_without_the_structural_guarantee():
    bad = ION.adjudicate_matched_control(
        closed=dict(spikes=[1000, 1010], participants=[500, 505], peaks=[0.7, 0.9]),
        open_=dict(spikes=[1005, 1002], participants=[502, 500], peaks=[0.7, 0.8]),
        structurally_identical_until_freeze=False)
    assert bad["status"] == "UNRESOLVED_MATCHED_CONTROL"
    assert "ratio" not in bad


def test_power_precondition_is_a_hard_gate():
    """n_scoreable < 20 -> INSUFFICIENT_POWER; the function must NOT hand back a usable threshold."""
    bad = ION.direction_power_gate(dict(n_scoreable=7, frac_A=0.5, frac_B=0.5))
    assert bad["status"] == "INSUFFICIENT_POWER"
    assert "threshold" not in bad
    one_sided = ION.direction_power_gate(dict(n_scoreable=44, frac_A=1.0, frac_B=0.0))
    assert one_sided["status"] == "INSUFFICIENT_POWER"
    ok = ION.direction_power_gate(dict(n_scoreable=44, frac_A=0.7, frac_B=0.3))
    assert ok["status"] == "PASS"
