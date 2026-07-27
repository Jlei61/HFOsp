"""FCXR-ION B0: constitutive Na/K ion math, provenance table, voltage-unit audit,
initiation-site readout.

Every test below is one clause of
  docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md (rev4)
  docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md (rev3)
Tests are written BEFORE the implementation (plan §5/§6 "tests first").
"""
from __future__ import annotations

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


def test_power_precondition_is_a_hard_gate():
    """n_scoreable < 20 -> INSUFFICIENT_POWER; the function must NOT hand back a usable threshold."""
    bad = ION.direction_power_gate(dict(n_scoreable=7, frac_A=0.5, frac_B=0.5))
    assert bad["status"] == "INSUFFICIENT_POWER"
    assert "threshold" not in bad
    one_sided = ION.direction_power_gate(dict(n_scoreable=44, frac_A=1.0, frac_B=0.0))
    assert one_sided["status"] == "INSUFFICIENT_POWER"
    ok = ION.direction_power_gate(dict(n_scoreable=44, frac_A=0.7, frac_B=0.3))
    assert ok["status"] == "PASS"
