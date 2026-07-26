"""FCXR pump-lifecycle Task 1 TDD — dimensionless activity-dependent load + electrogenic pump.

Contract source: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md §2
(+ plan §3 test list). u_i is an ACTIVITY-DEPENDENT INTRACELLULAR LOAD (Na/pump-inspired), NOT a
sodium concentration, ATP model or ionic-homeostasis model. Each test = one contract clause; the
clauses that would silently corrupt the science if violated are:

  * spike jump is per-spike (NOT multiplied by dt) -- a dt-scaled jump makes a_load dt-dependent;
  * clearance IS multiplied by dt/tau_N -- the two sides of the mass balance have different dt laws;
  * the SAME phi(u) drives clearance and the membrane current -- two phis = free extra parameter;
  * excess current is Imax*(phi-p0) with NO positive part -- rectification injects a positive mean
    bias from baseline noise alone and destroys local smoothness for the response operator.
"""
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_pump as PUMP  # noqa: E402


# ============================== clause 1/2: phi(u) = u^h/(1+u^h) ==============================
def test_pump_activation_is_hill_h3_exact():
    u = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
    got = PUMP.pump_activation(u)
    assert np.allclose(got, u ** 3 / (1.0 + u ** 3))


def test_pump_activation_zero_load_gives_zero_activation():
    assert PUMP.pump_activation(np.array([0.0]))[0] == 0.0
    assert PUMP.pump_activation(0.0) == 0.0


def test_pump_activation_is_monotone_bounded_finite_and_smooth():
    u = np.linspace(0.0, 50.0, 5001)
    phi = PUMP.pump_activation(u)
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0) and np.all(phi <= 1.0)
    assert np.all(np.diff(phi) >= 0.0)                      # monotone non-decreasing
    # smooth: no kink -- second difference stays bounded and phi'(0)=0 for h=3 (unlike a rectifier)
    d2 = np.diff(phi, 2)
    assert np.max(np.abs(d2)) < 1e-3
    assert phi[1] / (u[1] ** 3) == pytest.approx(1.0, rel=1e-6)   # phi ~ u^3 near 0


def test_pump_activation_saturates_toward_one_but_never_reaches_it():
    assert PUMP.pump_activation(1.0) == pytest.approx(0.5)
    assert 0.99 < PUMP.pump_activation(100.0) < 1.0


# ============================== clause 3/4: dt laws differ per term ==============================
def test_spike_jump_is_per_spike_and_not_scaled_by_dt():
    """a_load is a per-spike jump. If it were multiplied by dt, halving dt would halve the jump."""
    u0 = np.zeros(3)
    spk = np.array([True, False, True])
    kw = dict(a_load=0.3, tau_N=1000.0, dt=0.05)
    u1 = PUMP.step_spike_load(u0, spk, **kw)
    kw2 = dict(kw, dt=0.025)
    u1_half = PUMP.step_spike_load(u0, spk, **kw2)
    assert u1[0] == pytest.approx(0.3) and u1[2] == pytest.approx(0.3)
    assert u1[1] == 0.0
    assert u1_half[0] == pytest.approx(u1[0])               # dt does NOT enter the jump


def test_clearance_is_scaled_by_dt_over_tau_N_using_pre_step_phi():
    u0 = np.array([2.0])
    no_spk = np.array([False])
    dt, tau = 0.05, 1000.0
    u1 = PUMP.step_spike_load(u0, no_spk, a_load=0.3, tau_N=tau, dt=dt)
    expected = 2.0 - (dt / tau) * PUMP.pump_activation(2.0)
    assert u1[0] == pytest.approx(expected, rel=1e-12)
    # halving dt halves the clearance increment
    u1h = PUMP.step_spike_load(u0, no_spk, a_load=0.3, tau_N=tau, dt=dt / 2)
    assert (u0[0] - u1h[0]) == pytest.approx((u0[0] - u1[0]) / 2, rel=1e-12)


def test_clearance_uses_pre_step_u_not_post_jump_u():
    """Causal order (spec §2.2): clearance is evaluated at u(t^-), the jump is added on top."""
    u0 = np.array([2.0])
    spk = np.array([True])
    dt, tau, a = 0.05, 1000.0, 0.4
    got = PUMP.step_spike_load(u0, spk, a_load=a, tau_N=tau, dt=dt)[0]
    assert got == pytest.approx(2.0 + a - (dt / tau) * PUMP.pump_activation(2.0), rel=1e-12)
    wrong = 2.0 + a - (dt / tau) * PUMP.pump_activation(2.0 + a)     # post-jump clearance
    assert abs(got - wrong) > 0.0


# ============================== clause 5/6: non-negativity + monotone clearing ==============================
def test_update_never_returns_negative_load():
    u0 = np.array([1e-9, 0.0])
    u1 = PUMP.step_spike_load(u0, np.array([False, False]), a_load=0.0, tau_N=1e-6, dt=1.0)
    assert np.all(u1 >= 0.0)


def test_zero_spike_state_clears_monotonically_toward_zero():
    u = np.array([3.0])
    prev = np.inf
    for _ in range(4000):
        u = PUMP.step_spike_load(u, np.array([False]), a_load=0.5, tau_N=50.0, dt=0.05)
        assert u[0] <= prev + 1e-15
        prev = u[0]
    assert u[0] < 3.0


def test_spike_counts_accumulate_linearly_in_the_jump():
    """N_i^spike may exceed 1 (integer count input): the jump must scale with the count."""
    u0 = np.zeros(2)
    counts = np.array([0, 3])
    u1 = PUMP.step_spike_load(u0, counts, a_load=0.2, tau_N=1000.0, dt=0.05)
    assert u1[1] == pytest.approx(0.6)
    assert u1[0] == 0.0


def test_non_finite_load_fails_fast():
    with pytest.raises(FloatingPointError):
        PUMP.step_spike_load(np.array([np.inf]), np.array([False]),
                             a_load=0.1, tau_N=1000.0, dt=0.05)


# ============================== clause 7: one phi for clearance AND membrane ==============================
def test_same_phi_drives_clearance_and_membrane_current():
    """The clearance term removed from u and the membrane pump activation must be the same phi(u).
    A second, differently-shaped activation would be a free unconstrained parameter."""
    u = np.array([0.3, 1.7, 4.0])
    dt, tau = 0.05, 800.0
    cleared = u - PUMP.step_spike_load(u, np.zeros(3, bool), a_load=0.0, tau_N=tau, dt=dt)
    phi_from_clearance = cleared * tau / dt
    phi_from_membrane = (PUMP.excess_pump_current(u, np.zeros(3), Imax=1.0)) / 1.0
    assert np.allclose(phi_from_clearance, phi_from_membrane, rtol=1e-12)


# ============================== clause 8/9/10: baseline-centered, NO positive part ==============================
def test_excess_current_is_imax_times_phi_minus_p0():
    u = np.array([0.5, 2.0])
    p0 = np.array([0.1, 0.2])
    got = PUMP.excess_pump_current(u, p0, Imax=3.0)
    assert np.allclose(got, 3.0 * (PUMP.pump_activation(u) - p0))


def test_excess_current_is_zero_when_phi_equals_p0():
    u = np.array([1.0])
    p0 = PUMP.pump_activation(u)
    assert PUMP.excess_pump_current(u, p0, Imax=5.0)[0] == 0.0


def test_excess_current_allows_negative_compensation_no_rectification():
    """REGRESSION for the rectified draft: phi<p0 MUST give a negative excess. A positive part would
    clamp it to 0, giving baseline noise a positive-only mean bias (spec §2.3)."""
    u = np.array([0.05])                                     # phi(u) ~ 1.2e-4
    p0 = np.array([0.4])
    got = PUMP.excess_pump_current(u, p0, Imax=2.0)[0]
    assert got < 0.0
    assert got == pytest.approx(2.0 * (PUMP.pump_activation(0.05) - 0.4))


def test_excess_current_mean_is_zero_on_a_symmetric_baseline_distribution():
    """The point of the p0 compensation: with p0 = E_baseline[phi(u)], the membrane sees zero MEAN
    extra current on baseline. Rectification would leave a strictly positive mean."""
    rng = np.random.default_rng(0)
    u = np.abs(rng.normal(1.0, 0.3, size=20000))
    p0 = float(PUMP.pump_activation(u).mean())
    ex = PUMP.excess_pump_current(u, np.full(u.shape, p0), Imax=1.0)
    assert abs(float(ex.mean())) < 1e-12
    rectified_mean = float(np.maximum(ex, 0.0).mean())
    assert rectified_mean > 1e-3                             # the forbidden variant is biased


# ============================== clause 11: h is fixed at 3 for the primary tier ==============================
def test_primary_tier_requires_h_equals_3():
    PUMP.require_primary_h(3)                                # no raise
    for bad in (2, 4, 1):
        with pytest.raises(ValueError):
            PUMP.require_primary_h(bad)


def test_h_is_still_a_parameter_for_the_deferred_sensitivity_tier():
    u = np.array([2.0])
    assert PUMP.pump_activation(u, h=2)[0] == pytest.approx(4.0 / 5.0)
    assert PUMP.pump_activation(u, h=3)[0] == pytest.approx(8.0 / 9.0)


# ============================== clause 12: primary load is spike-only ==============================
def test_primary_load_function_has_no_synaptic_conductance_input():
    """Spec §2.5: g_rec_raw has neither a driving force nor the applied tanh saturation, so it must
    not enter the PRIMARY load. This is a structural guard: no conductance/charge argument exists."""
    params = set(inspect.signature(PUMP.step_spike_load).parameters)
    assert params == {"u", "spikes", "a_load", "tau_N", "dt", "h"}
    banned = ("g_rec", "gerec", "conduct", "charge", "q_e", "influx", "i_e")
    src = inspect.getsource(PUMP.step_spike_load).lower()
    assert not any(b in src for b in banned)


# =====================================================================================
# Task 5 — virtual-SEEG component audit (Gate I-a readout identifiability, spec §I3).
# These are PROXY components, not a physical forward-voltage solution: the driving force is
# evaluated at the model's own force-match anchor v_match because the slow protocol never sees V.
# No sign is ever inferred from a magnitude.
# =====================================================================================
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from params import Params  # noqa: E402
from connectivity import place_neurons, build_connectivity  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

_SEED = 1
_SITES = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])


def _seeg_net(T=120.0, dt=0.1):
    p = Params(L=6.0, density=100.0, T=T, dt=dt, nu_ext_ratio=0.9, seed=_SEED)
    rng = np.random.default_rng(_SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    return p, net, NE, NI


def _seeg_cfg(**kw):
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0, e_gaba=0.0,
                e_k=0.0, ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
                gaba_gain=1.125, max_total_conductance=99.0)
    base.update(kw)
    return MZSlowVarsConfig(**base)


def _seeg_run(cfg, *, observe, p0=None):
    p, net, NE, NI = _seeg_net()
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=_SITES)
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    obs = PUMP.VirtualSeegComponentObserver(rec, cfg) if observe else None
    slow.seeg_observer = obs
    net["rng"] = np.random.default_rng(_SEED)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=slow, kick_center=np.array([3.0, 3.0]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth, lfp_recorder=rec)
    return res, slow, obs


def test_observer_off_and_on_give_identical_spikes():
    cfg = _seeg_cfg()
    res_off, _, _ = _seeg_run(cfg, observe=False)
    res_on, _, obs = _seeg_run(cfg, observe=True)
    assert np.array_equal(res_off["E_spk_bool"], res_on["E_spk_bool"])
    assert np.array_equal(res_off["lfp_trace"], res_on["lfp_trace"])
    assert res_on["E_spk_bool"].sum() > 0


def test_component_sum_identity_and_pump_separability():
    cfg = _seeg_cfg()
    _, _, obs = _seeg_run(cfg, observe=True)
    tr = obs.stack()
    assert np.allclose(tr["no_direct_pump"], tr["excitatory"] + tr["inhibitory"] + tr["adaptation"])
    assert np.allclose(tr["all_components"] - tr["no_direct_pump"], tr["pump"])


def test_no_direct_pump_excludes_the_pump_term_at_fixed_input():
    """Same synaptic input, different load -> the pump component moves, no_direct_pump does not."""
    cfg = _seeg_cfg(use_pump=True, pump_a_load=0.3, pump_tau_ms=500.0, pump_Imax=2.0)
    p, net, NE, NI = _seeg_net()
    N = NE + NI
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=_SITES)
    cfg.pump_p0_E = np.zeros(NE)
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    obs = PUMP.VirtualSeegComponentObserver(rec, cfg)
    slow.seeg_observer = obs
    I_E = np.linspace(1.0, 3.0, N); I_I = np.linspace(0.5, 1.5, N); I_E_rec = 0.6 * I_E
    slow.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)          # u = 0
    slow.u_pump_E[:] = 1.5
    slow.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)          # u = 1.5
    tr = obs.stack()
    assert np.allclose(tr["no_direct_pump"][0], tr["no_direct_pump"][1])
    assert not np.allclose(tr["pump"][0], tr["pump"][1])
    assert np.all(tr["pump"][1] < 0.0)                            # outward current, sign from the model
    assert np.all(tr["excitatory"] > 0.0) and np.all(tr["inhibitory"] < 0.0)


def test_legacy_abs_component_matches_the_blessed_lfp_recorder_exactly():
    cfg = _seeg_cfg()
    res, _, obs = _seeg_run(cfg, observe=True)
    tr = obs.stack()
    assert np.allclose(tr["legacy_abs"], res["lfp_trace"], rtol=0, atol=0)


def test_a_pure_slow_pump_sinusoid_cannot_make_no_direct_pump_broadband():
    """READOUT_CONTAMINATION regression: power injected only through the direct pump term must show
    up in all_components and NOT in no_direct_pump (spec §I3 / plan §7 Gate I-a readout)."""
    n, dt, ns = 4000, 1.0, 3
    rng = np.random.default_rng(0)
    nodp = rng.normal(0.0, 1e-3, size=(n, ns))
    t = np.arange(n) * dt / 1000.0
    pmp = np.stack([2.0 * np.sin(2 * np.pi * 3.0 * t) for _ in range(ns)], axis=1)
    traces = dict(excitatory=nodp, inhibitory=np.zeros_like(nodp), adaptation=np.zeros_like(nodp),
                  no_direct_pump=nodp, pump=pmp, all_components=nodp + pmp,
                  legacy_abs=np.abs(nodp))
    aud = PUMP.component_audit(traces, dt)
    assert aud["identity_all_minus_nodp_equals_pump_max_abs_err"] < 1e-12
    assert aud["identity_component_sum_max_abs_err"] < 1e-12
    assert aud["band_power_all_components"] > 100 * aud["band_power_no_direct_pump"]


def test_observer_trace_length_matches_the_early_stopped_run():
    cfg = _seeg_cfg()
    p, net, NE, NI = _seeg_net(T=200.0)
    N = NE + NI
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=_SITES)
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    obs = PUMP.VirtualSeegComponentObserver(rec, cfg)
    slow.seeg_observer = obs
    net["rng"] = np.random.default_rng(_SEED)
    res = simulate_kick(p, net, KICK_BOOST=40.0, slow=slow, kick_center=np.array([3.0, 3.0]),
                        r_kick=1.5, t_kick=20.0, V_th_per_neuron=np.full(N, 16.0),
                        lfp_recorder=rec, early_stop_runaway=True, es_thresh_hz=20.0, es_dur_ms=20.0)
    assert res["runaway_early_stop_ms"] is not None               # the early stop really fired
    assert obs.stack()["legacy_abs"].shape[0] == len(res["rate_E"])


# =====================================================================================
# Task 3 — p0 calibration / shrinkage / equivalence contracts
# =====================================================================================
def test_offline_raster_integration_matches_the_engine_step_by_step():
    """The sensor-only load is a pure function of the raster, so one simulation calibrates a whole
    candidate set offline. This pins the offline replay against the engine's own update."""
    rng = np.random.default_rng(2)
    raster = rng.random((60, 7)) < 0.25
    kw = dict(a_load=0.35, tau_N=400.0, dt=0.05)
    u_off, snaps, blk, spk, _ = PUMP.integrate_load_from_raster(
        raster, snapshot_steps=(9, 59), block_edges=((0, 30), (30, 60)), **kw)
    u = np.zeros(7)
    for t in range(60):
        u = PUMP.step_spike_load(u, raster[t], **kw)
    assert np.allclose(u_off, u)
    assert np.allclose(snaps[1], u)
    assert np.array_equal(spk.sum(axis=0), raster.sum(axis=0))
    assert blk.shape == (2, 7) and np.all(blk[1] >= blk[0])       # load builds up -> later block higher


def test_rate_decile_grouping_uses_rate_only_and_covers_all_cells():
    rng = np.random.default_rng(3)
    r = rng.gamma(2.0, 1.0, size=500)
    g = PUMP.rate_decile_groups(r)
    assert g.min() == 0 and g.max() == 9 and g.shape == r.shape
    # monotone: a higher rate never lands in a lower decile
    order = np.argsort(r)
    assert np.all(np.diff(g[order]) >= 0)


def test_shrinkage_weight_is_chosen_by_inner_block_cv_and_prefers_pooling_when_noisy():
    """Pure noise around a common group mean -> CV must pull the weight toward the group mean;
    a strong genuine per-cell signal -> CV must keep the per-cell estimate."""
    rng = np.random.default_rng(4)
    groups = np.repeat(np.arange(10), 20)
    truth_noisy = np.full(200, 0.2)
    noisy = np.stack([truth_noisy + rng.normal(0, 0.05, 200) for _ in range(6)])
    fit_noisy = PUMP.fit_p0_shrinkage(noisy, groups)
    truth_real = 0.05 + 0.02 * np.arange(200)
    real = np.stack([truth_real + rng.normal(0, 1e-4, 200) for _ in range(6)])
    fit_real = PUMP.fit_p0_shrinkage(real, groups)
    assert fit_noisy["weight"] > fit_real["weight"]
    assert fit_real["weight"] < 0.3
    assert fit_noisy["p0"].shape == (200,)


def test_shrinkage_requires_at_least_three_calibration_blocks():
    with pytest.raises(ValueError):
        PUMP.fit_p0_shrinkage(np.zeros((2, 10)), np.zeros(10, int))


def test_equivalence_margin_comes_from_block_variability_not_significance():
    blocks = [dict(rate=3.0), dict(rate=3.4), dict(rate=2.7), dict(rate=3.1), dict(rate=3.3)]
    m = PUMP.block_equivalence_margins(blocks, k=2.0)
    sd = float(np.std([3.0, 3.4, 2.7, 3.1, 3.3], ddof=1))
    assert m["rate"]["margin"] == pytest.approx(2.0 * sd)
    ok = PUMP.evaluate_baseline_equivalence(dict(rate=3.1), dict(rate=3.1 + 0.5 * m["rate"]["margin"]), m)
    bad = PUMP.evaluate_baseline_equivalence(dict(rate=3.1), dict(rate=3.1 + 1.5 * m["rate"]["margin"]), m)
    assert ok["all_within"] and not bad["all_within"] and bad["n_outside"] == 1


def test_missing_metric_fails_equivalence_instead_of_passing_silently():
    m = PUMP.block_equivalence_margins([dict(rate=3.0), dict(rate=3.4), dict(rate=2.7)])
    out = PUMP.evaluate_baseline_equivalence(dict(rate=3.0), {}, m)
    assert not out["all_within"] and out["per_metric"]["rate"]["status"] == "MISSING"


def test_required_ied_count_never_drops_below_the_prelocked_minimum():
    assert PUMP.required_ied_count([2, 3, 4])["n_ied_required"] == 20
    assert PUMP.required_ied_count([40, 44, 39])["n_ied_required"] == 40


def test_analytic_steady_load_solves_the_time_averaged_mass_balance():
    r = np.array([1.0, 3.84, 20.0])                              # Hz
    a_load, tau = 0.02, 2000.0
    u_star, frac_div = PUMP.analytic_steady_load(r, a_load=a_load, tau_N=tau)
    assert frac_div == 0.0
    # at u*, the per-step clearance exactly cancels the expected per-step jump
    assert np.allclose(PUMP.pump_activation(u_star), a_load * r * 1e-3 * tau)
    # replaying a Poisson raster from u* leaves the mean load put (no residual drift)
    rng = np.random.default_rng(11)
    dt, n = 0.05, 40000
    p_spk = r[1] * 1e-3 * dt
    raster = rng.random((n, 400)) < p_spk
    u0 = np.full(400, u_star[1])
    u_end = PUMP.integrate_load_from_raster(raster, a_load=a_load, tau_N=tau, dt=dt, u0=u0)[0]
    assert abs(float(u_end.mean()) - u_star[1]) < 0.1 * u_star[1]


def test_divergent_cells_are_reported_not_clamped_silently():
    """a_load*r*tau_N >= 1 means phi is pinned at 1 and the load has no steady state -- the fraction
    of such cells must be reported so the candidate can be ruled INADMISSIBLE."""
    r = np.array([1.0, 100.0])
    _, frac = PUMP.analytic_steady_load(r, a_load=0.02, tau_N=2000.0)
    assert frac == 0.5


# ---- A1 visibility (per-cell, event-locked, matched quiet control) ----
def test_event_locked_visibility_uses_participating_cells_not_the_population_mean():
    """REGRESSION for the first formulation: with only 4% of cells participating, a population-mean
    test dilutes the per-cell excursion ~25x and calls a clearly visible mechanism invisible."""
    NE, ne = 1000, 6
    rng = np.random.default_rng(5)
    part = np.zeros((ne, NE), bool)
    for k in range(ne):
        part[k, rng.choice(NE, 40, replace=False)] = True
    base = np.full((ne, NE), 0.15)
    jitter = rng.normal(0, 1e-4, size=(ne, NE))
    phi_on = base + jitter
    phi_off = phi_on + 0.013 * part                            # participating cells rise
    phi_qa = base + rng.normal(0, 1e-4, size=(ne, NE))
    phi_qb = phi_qa + rng.normal(0, 1e-4, size=(ne, NE))
    out = PUMP.event_locked_load_visibility(phi_on, phi_off, part, phi_qa, phi_qb)
    assert out["visible"] and out["ratio"] > 3.0 and out["n_events_scored"] == ne
    # the population-mean version of the same data is diluted below the bar
    pop_rise = float(np.median((phi_off - phi_on).mean(axis=1)))
    pop_quiet = float(np.std(phi_qa.mean(axis=1)))
    assert pop_rise < 3.0 * max(pop_quiet, 1e-12) or pop_rise < 0.05 * out["rise_median"]


def test_event_locked_visibility_fails_when_the_load_barely_moves():
    NE, ne = 500, 4
    part = np.zeros((ne, NE), bool); part[:, :20] = True
    phi_on = np.full((ne, NE), 0.1)
    phi_off = phi_on + 1e-6 * part
    rng = np.random.default_rng(6)
    phi_qa = np.full((ne, NE), 0.1)
    phi_qb = phi_qa + rng.normal(0, 1e-5, size=(ne, NE))
    out = PUMP.event_locked_load_visibility(phi_on, phi_off, part, phi_qa, phi_qb)
    assert not out["visible"]


def test_matched_quiet_intervals_never_pair_with_a_shorter_control():
    events = [(100, 200), (300, 340), (500, 900)]
    quiet = [(1000, 1150), (2000, 2060)]
    got = PUMP.matched_quiet_intervals(events, quiet)
    assert got[0] is not None and got[0][1] - got[0][0] == 100
    assert got[1] is not None and got[1][1] - got[1][0] == 40
    assert got[2] is None                                       # 400 steps: no quiet segment fits


def test_wide_margins_are_flagged_underpowered_not_silently_passed():
    """A margin wider than half the metric's own mean cannot separate "equivalent" from "cannot
    tell". The flag must survive into the verdict so a within-margin result is not oversold."""
    noisy = [dict(ied_rate_hz=v) for v in (1.0, 3.0, 0.5, 2.5, 1.5)]
    tight = [dict(mean_rate_hz=v) for v in (4.00, 4.02, 3.98, 4.01, 3.99)]
    mn = PUMP.block_equivalence_margins(noisy)
    mt = PUMP.block_equivalence_margins(tight)
    assert mn["ied_rate_hz"]["underpowered"] and not mt["mean_rate_hz"]["underpowered"]
    out = PUMP.evaluate_baseline_equivalence(dict(ied_rate_hz=1.7), dict(ied_rate_hz=1.8), mn)
    assert out["all_within"] and out["n_underpowered"] == 1
    assert out["underpowered_metrics"] == ["ied_rate_hz"]


# =====================================================================================
# Task 8/9 — frozen Z x P field construction and branch-conditioned slow flow (Gate T)
# =====================================================================================
def test_frozen_field_interpolates_between_baseline_and_high_and_stays_non_negative():
    u0 = np.array([0.5, 0.6, 0.7])
    uh = np.array([2.0, 0.1, 3.0])
    assert np.allclose(PUMP.frozen_load_field(u0, uh, 0.0), u0)
    assert np.allclose(PUMP.frozen_load_field(u0, uh, 1.0), uh)
    mid = PUMP.frozen_load_field(u0, uh, 0.5)
    assert np.allclose(mid, 0.5 * (u0 + uh)) and np.all(mid >= 0)


def test_controls_match_the_mean_EXCESS_ACTIVATION_not_raw_load():
    """Spec §T2: uniform / shuffle controls must match mean[phi(u)-p0]. Matching raw u instead would
    give a different point on the formal Z x P abscissa and make the comparison invalid."""
    rng = np.random.default_rng(9)
    u = np.abs(rng.lognormal(-0.3, 0.8, size=4000))
    p0 = np.full(4000, 0.12)
    P = PUMP.mean_excess_pump_activation(u, p0)
    uni = PUMP.matched_uniform_field(u, p0)
    shuf = PUMP.value_matched_shuffle_field(u, np.random.default_rng(7001))
    assert PUMP.mean_excess_pump_activation(uni, p0) == pytest.approx(P, abs=1e-9)
    assert PUMP.mean_excess_pump_activation(shuf, p0) == pytest.approx(P, abs=1e-12)
    assert np.allclose(uni, uni[0])                              # uniform in space
    assert not np.allclose(np.sort(shuf), shuf)                  # spatial order destroyed
    assert np.allclose(np.sort(shuf), np.sort(u))                # value multiset preserved
    assert abs(uni.mean() - u.mean()) > 1e-6                     # raw mean deliberately NOT matched


def test_shaped_and_matched_uniform_stay_distinguishable_at_equal_mean_excess():
    rng = np.random.default_rng(10)
    u = np.abs(rng.lognormal(-0.3, 0.8, size=2000))
    p0 = np.full(2000, 0.12)
    uni = PUMP.matched_uniform_field(u, p0)
    assert np.std(PUMP.pump_activation(u)) > 10 * np.std(PUMP.pump_activation(uni))


def test_slow_flow_signs_encode_load_build_up_and_inhibition_recovery():
    n = 500
    p0 = np.full(n, 0.1)
    z_dep = np.full(n, 0.4)
    u = np.full(n, 0.5)
    hi = PUMP.branch_slow_flow(np.full(n, 40.0), u, p0, z_dep, 0.9, a_load=0.02, tau_N=2000.0,
                               tau_z=5000.0)
    lo = PUMP.branch_slow_flow(np.full(n, 0.2), u, p0, z_dep, 0.9, a_load=0.02, tau_N=2000.0,
                               tau_z=5000.0)
    assert hi["dP_dt"] > 0 and lo["dP_dt"] < 0        # high branch loads up, quiet branch clears
    assert hi["dZ_dt"] > 0                            # inhibition recovering toward z_inf=1
    dep = PUMP.branch_slow_flow(np.full(n, 40.0), u, p0, np.full(n, 0.9), 0.1, a_load=0.02,
                                tau_N=2000.0, tau_z=5000.0)
    assert dep["dZ_dt"] < 0                           # sustained inhibition -> z depletes
    assert hi["Z"] == pytest.approx(0.4) and hi["P"] == pytest.approx(
        float(PUMP.pump_activation(0.5) - 0.1))


def test_observer_z_sensor_counts_cells_below_the_depletion_threshold():
    cfg = _seeg_cfg()
    p, net, NE, NI = _seeg_net()
    N = NE + NI
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=_SITES)
    obs = PUMP.VirtualSeegComponentObserver(rec, cfg, z_threshold=1.0)
    I_E = np.ones(N); I_I = np.zeros(N); I_I[:NE // 2] = 5.0      # half above threshold
    gE = np.zeros(NE); gI = np.zeros(NE); gM = np.zeros(NE)
    obs.sample(I_E, I_I, gE, gI, gM, None)
    assert obs.frac_z_inf_high() == pytest.approx(1.0 - (NE // 2) / NE)
    off = PUMP.VirtualSeegComponentObserver(rec, cfg)
    off.sample(I_E, I_I, gE, gI, gM, None)
    assert np.isnan(off.frac_z_inf_high())
