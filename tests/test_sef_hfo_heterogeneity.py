import numpy as np
import pytest
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E, mean_field, integrate_lif_field
from src.sef_hfo_field import _grid
from scripts.sef_hfo_step0d_anisotropy_control import principal_axis
from src.sef_hfo_heterogeneity import phi_eff_vth


def test_lif_rate_vth_default_matches_global():
    """Adding v_th param must not change existing behavior at the default."""
    r_default = lif_rate(5.0, 4.0, TAU_ME, TREF_E)
    r_explicit = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    assert r_explicit == r_default


def test_lif_rate_vth_lower_threshold_raises_rate():
    """Lowering threshold (easier to fire) raises rate at fixed mu,sigma —
    a monotone single-neuron sanity, NOT the closed-loop direction claim."""
    r_hi = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)
    r_lo = lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH - 2.0)
    assert r_lo > r_hi


def _offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    """Off-center seed matching the Step-0b validated chain."""
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)


def test_geometry_rho_controls_directionality():
    """ρ<1 (anisotropic) gives a stronger directional template than ρ=1 (isotropic)."""
    op = mean_field(1.0)
    def ratio_for(ell_par, ell_perp):
        out = integrate_lif_field(op, _offcenter_pulse(), theta_EE=0.0,
                                  ell_par=ell_par, ell_perp=ell_perp,
                                  return_peak_field=True, t_max=150.0)
        _angle, ratio = principal_axis(out[2] - op["nuE"])
        return ratio
    r_aniso = ratio_for(0.9, 0.45)   # rho = 0.5
    r_iso = ratio_for(0.6, 0.6)      # rho = 1.0
    assert r_iso < 1.5                       # isotropic: weak directional bias (off-center seed)
    assert r_aniso > r_iso + 0.3             # anisotropy creates a directional template


def test_phi_eff_zero_std_reduces_to_lif_rate():
    """vth_std=0 → no spread → exactly lif_rate at vth_mean."""
    r = phi_eff_vth(5.0, 4.0, TAU_ME, TREF_E, vth_mean=V_TH, vth_std=0.0)
    assert abs(r - lif_rate(5.0, 4.0, TAU_ME, TREF_E, v_th=V_TH)) < 1e-9


def test_lif_rate_raises_below_reset():
    """A threshold below reset is unphysical (the Siegert rate goes negative). lif_rate
    must RAISE rather than silently return a negative / clamped value — the 2026-06-07
    tripwire that the old unbounded-Gaussian + max(0,.) bug masked."""
    from src.sef_hfo_lif import V_RESET
    with pytest.raises(ValueError):
        lif_rate(8.0, 4.0, TAU_ME, TREF_E, v_th=V_RESET - 1.0)
    # exactly at reset is the legal zero-gap saturation (rate = 1/tau_ref), must NOT raise
    r_floor = lif_rate(8.0, 4.0, TAU_ME, TREF_E, v_th=V_RESET)
    assert abs(r_floor - 1.0 / TREF_E) < 1e-9


def test_phi_eff_matches_renormalized_truncated_dense_average():
    """The Gauss–Legendre quadrature must match a dense trapezoid of the RENORMALIZED
    TRUNCATED integrand (support [V_RESET, hi]) — legal v_th only.  Replaces the old
    test that compared against a dense average of the UNbounded Gaussian (which sampled
    v_th < V_RESET, i.e. validated the integration against the very illegal values the
    fix removes — circular)."""
    from src.sef_hfo_lif import V_RESET
    mu, sig, vm, vs = 8.0, 4.0, 19.0, 1.5
    hi = vm + 8.0 * vs
    grid = np.linspace(V_RESET, hi, 20001)
    w = np.exp(-0.5 * ((grid - vm) / vs) ** 2)
    r = np.array([lif_rate(mu, sig, TAU_ME, TREF_E, v_th=v) for v in grid])
    dense = float(np.trapz(w * r, grid) / np.trapz(w, grid))
    gl = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
    assert abs(gl - dense) < 1e-6


def test_phi_eff_strictly_decreasing_in_vth_mean():
    """PRIMARY SCIENTIFIC GATE (2026-06-07): a higher mean threshold must give a LOWER
    effective rate, for every vth_std actually used.  The unbounded-Gaussian + clamp bug
    violated this (rate rose then collapsed as vth_mean increased), which silently broke
    mean_match_vth's monotonicity premise and the whole forced chain."""
    mu, sig = 8.0, 4.0
    for vs in (0.5, 1.0, 1.5):
        vals = [phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
                for vm in np.linspace(14.0, 26.0, 25)]
        diffs = np.diff(vals)
        assert np.all(diffs < 0), (
            f"phi_eff not strictly decreasing in vth_mean at vth_std={vs}: {vals}")


def test_locked_std_params_stay_out_of_reset_knee():
    """Lock the GAP-LIMITED std choice (wide=1.5, narrow=0.5).  At the canonical
    operating point, mean-matched to nuE, the near-reset saturation knee [V_RESET, 13]
    must contribute < 5% of the effective rate — else Φ_eff measures reset-floor
    saturation of a near-spiking minority, not the collective gain the heterogeneity
    manipulation is meant to probe (advisor gate 2026-06-07).  The plan's original
    wide=4.0 fails this badly (~49%); these are the corrected, locked parameters."""
    from src.sef_hfo_lif import mean_field, V_RESET
    from src.sef_hfo_heterogeneity import mean_match_vth
    from scipy.integrate import quad
    op = mean_field(1.0)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    def knee_share(vs):
        vm = mean_match_vth(nuE, muE, sE, TAU_ME, TREF_E, vs)
        hi = vm + 8.0 * vs
        pdf = lambda v: np.exp(-0.5 * ((v - vm) / vs) ** 2)
        rate_w = lambda v: lif_rate(muE, sE, TAU_ME, TREF_E, v_th=v) * pdf(v)
        den = quad(rate_w, V_RESET, hi, limit=200)[0]
        knee = quad(rate_w, V_RESET, 13.0, limit=200)[0]
        return knee / den
    for vs in (1.5, 0.5):
        assert knee_share(vs) < 0.05, f"locked vth_std={vs}: reset-knee share {knee_share(vs):.3f} >= 5%"
    # NON-VACUOUSNESS: the plan's original wide=4.0 MUST be rejected by the same gate,
    # else the gate has no discriminating power (it would pass anything).
    assert knee_share(4.0) > 0.20, (
        f"gate is vacuous: plan's original vth_std=4.0 knee share {knee_share(4.0):.3f} "
        f"should be large (~0.49) and rejected")


def test_closed_loop_leading_canonical_op_stable():
    """Canonical white-noise operating point is robustly STABLE.
    Framework banner: max Re λ≈−0.05, k=0. Locks the promotion."""
    from src.sef_hfo_lif import mean_field, lif_gains, closed_loop_leading
    op = mean_field(1.0)
    g = lif_gains(op)
    res = closed_loop_leading(g["E"], g["I"])
    assert res["re_max"] < 0.0           # stable
    assert res["re_max"] > -0.15         # ~ -0.05, not wildly off
    assert res["k_star"] < 0.3           # dominant mode near k=0 (no finite-k Hopf)
    assert res["converged"] is True      # the reported re_max came from a real root
    assert res["n_converged"] == res["n_modes"]   # every k-mode resolved at this op


def test_closed_loop_leading_flags_failed_search():
    """If NO root is found in the fixed search box, re_max is the -inf sentinel and the
    'regime' string would spuriously read 'stable'. converged must be False so a Task-7
    stability gate does not read a failed search as 'very stable' (advisor 2026-06-07).
    Force failure with absurd gains that drive every root outside the box."""
    from src.sef_hfo_lif import closed_loop_leading
    res = closed_loop_leading(1e6, 1e6)
    if res["n_converged"] == 0:
        assert res["converged"] is False
        assert not np.isfinite(res["re_max"]) or res["re_max"] == -np.inf
        # the regime string itself must NOT read "stable" on a failed search
        assert res["regime"] == "unresolved"
    else:
        # if the box happened to still catch a root, converged must agree with finiteness
        assert res["converged"] == bool(np.isfinite(res["re_max"]))
        assert res["regime"] != "unresolved"


def test_eff_gain_matches_central_difference():
    """Reported slope/curvature must equal a finite-difference of phi_eff itself
    (consistency check — NOT a direction claim)."""
    from src.sef_hfo_heterogeneity import eff_gain_curvature, phi_eff_vth
    from src.sef_hfo_lif import TAU_ME, TREF_E, V_TH
    mu, sig, vm, vs = 8.0, 4.0, 19.0, 1.5   # clean (out-of-knee) regime; this is a
    g = eff_gain_curvature(mu, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs, h=1e-2)   # consistency check (FD vs FD)
    f = lambda m: phi_eff_vth(m, sig, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
    slope = (f(mu + 1e-2) - f(mu - 1e-2)) / (2e-2)
    curv = (f(mu + 1e-2) - 2 * f(mu) + f(mu - 1e-2)) / (1e-2 ** 2)
    assert abs(g["slope"] - slope) < 1e-6
    assert abs(g["curvature"] - curv) < 1e-4


def test_mean_match_restores_baseline_rate():
    """After mean-matching, Φ_eff at the op input equals the baseline rate,
    even though vth_std changed. This is Contract 2's whole point."""
    from src.sef_hfo_heterogeneity import mean_match_vth, phi_eff_vth
    from src.sef_hfo_lif import TAU_ME, TREF_E, V_TH
    mu, sig = 8.0, 4.0
    baseline = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=V_TH, vth_std=1.5)   # locked wide
    vm_matched = mean_match_vth(target_rate=baseline, mu=mu, sigma=sig,
                                tau_m=TAU_ME, tau_ref=TREF_E, vth_std=0.5)         # locked narrow
    got = phi_eff_vth(mu, sig, TAU_ME, TREF_E, vth_mean=vm_matched, vth_std=0.5)
    assert abs(got - baseline) < 1e-6


def test_optpoint_baseline_sits_at_self_consistent_rest():
    """Baseline and mean-matched layers must evaluate AT the self-consistent rest nuE —
    else slope/curvature are read off a non-rest input (review fix 2026-06-06)."""
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    res = analyze_optpoint(vth_std_wide=1.5, vth_std_narrow=0.5)
    nuE = res["operating_point"]["nuE"]
    assert abs(res["baseline"]["rate"] - nuE) < 1e-6
    assert abs(res["mean_matched"]["rate"] - nuE) < 1e-6


def test_optpoint_control_isolates_variance_effect():
    """raw narrowing moves both mean-rate and shape; mean-matched holds rate at nuE.
    Assert ONLY the control invariant (Contract 2), never the sign of the shape change
    (spec §7 non-circularity)."""
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    res = analyze_optpoint(vth_std_wide=1.5, vth_std_narrow=0.5)
    assert abs(res["mean_matched"]["rate"] - res["baseline"]["rate"]) < 1e-6
    for layer in ("baseline", "raw_narrow", "mean_matched"):
        assert np.isfinite(res[layer]["slope"]) and np.isfinite(res[layer]["curvature"])
        assert np.isfinite(res[layer]["closed_loop_re_max"])


def test_optpoint_default_op_closed_loop_converged():
    """[2026-06-07 hardening] At the DEFAULT op every layer's closed-loop search must
    fully resolve — converged True and n_converged == n_modes. This is the precondition
    for trusting closed_loop_re_max as a stability readout; the script REFUSES (raises)
    to emit a regime from a failed search rather than reporting a -inf as 'stable'."""
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    res = analyze_optpoint(vth_std_wide=1.5, vth_std_narrow=0.5)
    for layer in ("baseline", "raw_narrow", "mean_matched"):
        L = res[layer]
        assert L["closed_loop_converged"] is True
        assert L["closed_loop_n_converged"] == L["closed_loop_n_modes"]
        assert L["closed_loop_regime"] != "unresolved"


def test_optpoint_nondefault_op_knee_gate_raises():
    """[2026-06-07 hardening] The locked std are validated ONLY at the default op. At a
    non-default op where the std leave the reset-knee clean band, analyze_optpoint must
    RAISE (refuse to produce a stable/unstable interpretation from a saturation-dominated
    forced chain) rather than silently report. wide=4.0 is the saturation regime."""
    import pytest
    from scripts.run_sef_hfo_hetero_optpoint import analyze_optpoint
    with pytest.raises(RuntimeError):
        analyze_optpoint(w_ee_mult=1.2, vth_std_wide=4.0, vth_std_narrow=1.0)


def test_hetero_field_uniform_patch_matches_homogeneous():
    """With core==surround AND vth_std=0, integrate_hetero_field must reduce to the
    homogeneous integrate_lif_field exactly (same LUTs, same loop). vth_std=0 is
    required because integrate_lif_field uses bare lif_rate (no threshold spread)."""
    from src.sef_hfo_lif import mean_field, integrate_lif_field, V_TH
    from src.sef_hfo_heterogeneity import integrate_hetero_field
    from src.sef_hfo_field import _grid
    op = mean_field(1.0)
    X, Y = _grid(96, 12.0)
    pulse = lambda t: (8.0 * ((X ** 2 + Y ** 2) <= 1.5 ** 2) if t < 30.0 else 0.0)
    ext_h, _ = integrate_lif_field(op, pulse, t_max=80.0)
    ext_g, _ = integrate_hetero_field(op, pulse, x_patch=0.0, r_patch=2.0,
                                      vth_std_core=0.0, vth_std_surround=0.0,
                                      vth_mean_core=V_TH, vth_mean_surround=V_TH, t_max=80.0)
    assert np.max(np.abs(ext_h - ext_g)) < 5e-3


def test_patch_analysis_runs_both_layers():
    from scripts.run_sef_hfo_hetero_patch import analyze_patch
    res = analyze_patch(t_max=80.0, vth_std_wide=1.5, vth_std_narrow=0.5)
    assert set(res["layers"]) >= {"baseline", "raw_narrow", "mean_matched"}
    for layer in res["layers"].values():
        assert "label" in layer and "max_ext" in layer
    # Contract: surround must be defined (the call below must raise without it)
    import pytest
    from src.sef_hfo_heterogeneity import integrate_hetero_field
    from src.sef_hfo_lif import mean_field
    with pytest.raises(TypeError):
        integrate_hetero_field(mean_field(1.0), lambda t: 0.0,
                               x_patch=0.0, r_patch=2.0, vth_std_core=0.5,
                               vth_mean_core=18.0)  # missing surround kwargs → TypeError


def test_hetero_field_can_run_away_positive_control():
    """NON-VACUOUSNESS guard for the Task-9 null (2026-06-07). The first-round finding is
    that the locked heterogeneity manipulation (1.5→0.5, mean-matched) keeps every layer
    self_limited_propagation. That null is only meaningful if integrate_hetero_field CAN
    produce a non-self-limited response at all. Push a strongly excitable core (low
    vth_mean_core + strong pulse) and confirm it goes runaway — else a future refactor
    that accidentally made the integrator always self-limit would silently make the null
    vacuous."""
    from src.sef_hfo_lif import mean_field, classify_response, V_TH
    from src.sef_hfo_heterogeneity import integrate_hetero_field
    from src.sef_hfo_field import _grid
    op = mean_field(1.0)
    X, Y = _grid(96, 12.0)
    pulse = lambda t: (60.0 * (((X + 3) ** 2 + Y ** 2) <= 2.0 ** 2) if t < 30.0 else 0.0)
    ext, front = integrate_hetero_field(
        op, pulse, x_patch=-3.0, r_patch=2.0, vth_mean_core=12.5, vth_std_core=0.5,
        vth_mean_surround=V_TH, vth_std_surround=1.5, t_max=70.0)
    label, _ = classify_response(ext, front)
    assert label in ("runaway", "global_synchronous"), (
        f"integrator could not produce a non-self-limited response (got {label}); "
        f"the Task-9 'self-limit holds' null would be vacuous")


def test_analyze_margin_returns_expected_schema():
    """Structure/import guard for the finite-pulse margin sweep (Task 9b). Uses a tiny
    grid + short t_max purely to exercise the code path cheaply — it does NOT assert the
    science verdict. (A_runaway / margin_compressed depend on a long-enough t_max for the
    self-limited wave to actually RETURN; a short t_max mislabels a not-yet-returned wave
    as 'runaway' — a classify_response cutoff artifact, not real runaway.) The actual
    result — A_runaway unreachable, margin NOT compressed, at t_max=120 with the
    saturation-plateau grid — lives in committed margin.json."""
    from scripts.run_sef_hfo_hetero_patch import analyze_margin
    res = analyze_margin(A_grid=(8,), t_max=40.0)
    for tag in ("baseline_wide", "mean_matched_narrow"):
        L = res[tag]
        assert {"A_runaway", "saturated_at_high_A", "sweep"} <= set(L)
        assert len(L["sweep"]) == 1
        assert {"A", "label", "max_ext", "peak_rE_max"} <= set(L["sweep"][0])
    it = res["interpretation"]
    assert {"A_runaway_wide", "A_runaway_narrow", "margin_compressed",
            "runaway_unreached_in_saturated_grid"} <= set(it)


def test_phi_eff_param_vth_matches_phi_eff_vth():
    """phi_eff_param(param='v_th') must agree with the validated phi_eff_vth — proves the
    generalization + the _trunc_gl_avg refactor reproduce the V_th-only path."""
    from src.sef_hfo_heterogeneity import phi_eff_param, phi_eff_vth
    for vm, vs in [(18.0, 0.0), (18.98, 1.5), (18.11, 0.5), (19.0, 1.0)]:
        a = phi_eff_param(8.0, 4.0, TAU_ME, TREF_E, param="v_th", mean=vm, std=vs)
        b = phi_eff_vth(8.0, 4.0, TAU_ME, TREF_E, vth_mean=vm, vth_std=vs)
        assert abs(a - b) < 1e-9, f"param vs vth mismatch at vm={vm},vs={vs}: {a} vs {b}"


def test_phi_eff_param_zero_std_reduces_to_lif_rate():
    """std=0 → point mass → exactly lif_rate with that one argument substituted."""
    from src.sef_hfo_heterogeneity import phi_eff_param
    from src.sef_hfo_lif import lif_rate
    mu, sig = 8.0, 4.0
    assert abs(phi_eff_param(mu, sig, TAU_ME, TREF_E, param="tau_m", mean=25.0, std=0.0)
               - lif_rate(mu, sig, 25.0, TREF_E)) < 1e-12
    assert abs(phi_eff_param(mu, sig, TAU_ME, TREF_E, param="tau_ref", mean=3.0, std=0.0)
               - lif_rate(mu, sig, TAU_ME, 3.0)) < 1e-12
    assert abs(phi_eff_param(mu, sig, TAU_ME, TREF_E, param="sigma", mean=5.0, std=0.0)
               - lif_rate(mu, 5.0, TAU_ME, TREF_E)) < 1e-12


def test_phi_eff_param_never_samples_below_floor():
    """The truncated support must keep every lif_rate call inside the parameter's physical
    domain — e.g. sigma stays > 0 (a sigma<=0 would be unphysical / singular)."""
    import src.sef_hfo_lif as _lifmod
    from src.sef_hfo_heterogeneity import phi_eff_param
    seen = []
    orig = _lifmod.lif_rate
    def spy(mu, sigma, tau_m, tau_ref, v_th=_lifmod.V_TH):
        seen.append((sigma, tau_m, tau_ref, v_th))
        return orig(mu, sigma, tau_m, tau_ref, v_th=v_th)
    import src.sef_hfo_heterogeneity as _het
    _het.lif_rate = spy
    try:
        phi_eff_param(8.0, 4.0, TAU_ME, TREF_E, param="sigma", mean=4.0, std=1.5)
        assert all(s > 0 for (s, _tm, _tr, _v) in seen), "sigma sampled <= 0"
    finally:
        _het.lif_rate = orig


def test_mean_match_param_restores_rate_for_tau_m():
    """Per-parameter mean-matched control (Contract 2) generalizes beyond V_th: after
    matching tau_m's mean at a narrower std, phi_eff_param returns the baseline rate."""
    from src.sef_hfo_heterogeneity import phi_eff_param, mean_match_param
    mu, sig = 8.0, 4.0
    baseline = phi_eff_param(mu, sig, TAU_ME, TREF_E, param="tau_m", mean=TAU_ME, std=3.0)
    m = mean_match_param(baseline, mu, sig, TAU_ME, TREF_E, param="tau_m", std=1.0)
    got = phi_eff_param(mu, sig, TAU_ME, TREF_E, param="tau_m", mean=m, std=1.0)
    assert abs(got - baseline) < 1e-6


def test_phi_eff_multi_reduces_to_single_param():
    """phi_eff_multi with ONE active spec (others point masses) must match phi_eff_param —
    the joint tensor integral reduces to the 1-D truncated average."""
    from src.sef_hfo_heterogeneity import phi_eff_multi, phi_eff_param
    mu, sig = 8.0, 4.0
    a = phi_eff_multi(mu, sig, TAU_ME, TREF_E,
                      specs=[("v_th", 19.0, 1.5), ("sigma", sig, 0.0)], n_nodes=96)
    b = phi_eff_param(mu, sig, TAU_ME, TREF_E, param="v_th", mean=19.0, std=1.5)
    assert abs(a - b) < 1e-7, f"{a} vs {b}"


def test_phi_eff_multi_all_point_masses_is_lif_rate():
    """All specs std=0 → joint point mass → plain lif_rate at those values."""
    from src.sef_hfo_heterogeneity import phi_eff_multi
    from src.sef_hfo_lif import lif_rate
    got = phi_eff_multi(8.0, 4.0, TAU_ME, TREF_E,
                        specs=[("v_th", 18.0, 0.0), ("sigma", 5.0, 0.0)])
    assert abs(got - lif_rate(8.0, 5.0, TAU_ME, TREF_E, v_th=18.0)) < 1e-12


def test_screen_combo_schema():
    """Guard the combo screen's δ-match + schema (cheap: 1-param 'combo', coarse nodes)."""
    from scripts.run_sef_hfo_hetero_sensitivity import screen_combo
    from src.sef_hfo_lif import mean_field, lif_gains
    op = mean_field(1.0)
    gI = lif_gains(op)["I"]
    r = screen_combo(op, gI, ("v_th",), n_nodes=8)
    assert {"d_slope_pure_pct", "d_closed_loop_re_max_pure", "baseline_wide",
            "mean_matched_narrow"} <= set(r)
    assert np.isfinite(r["d_slope_pure_pct"]) and np.isfinite(r["d_closed_loop_re_max_pure"])


def test_step3b_combo_vth_sigma_numeric_lock():
    """Regression: recompute the V_th+sigma combo (drift gate) and lock its science —
    spread-narrowing steepens the curve by ~+16% but does NOT move the safety margin, and
    the closed-loop search converges in both layers. (n_nodes=24 for test speed; the
    committed combo.json uses 48 — the locked ranges are tolerant to that.)"""
    from scripts.run_sef_hfo_hetero_sensitivity import screen_combo
    from src.sef_hfo_lif import mean_field, lif_gains
    op = mean_field(1.0)
    gI = lif_gains(op)["I"]
    r = screen_combo(op, gI, ("v_th", "sigma"), n_nodes=24)
    assert 10.0 < r["d_slope_pure_pct"] < 22.0, r["d_slope_pure_pct"]      # curve steepens ~+16%
    assert abs(r["d_closed_loop_re_max_pure"]) < 0.005                     # margin does NOT move
    assert r["baseline_wide"]["closed_loop_converged"] is True
    assert r["mean_matched_narrow"]["closed_loop_converged"] is True


def test_step3b_committed_results_all_converged_and_in_range():
    """Regression: lock the committed Step3b artifacts — EVERY closed_loop_converged flag
    True (a failed search would read as spuriously 'stable'), and the key leverage numbers
    in their expected ranges (V_th/sigma move the curve, tau_m/tau_ref dead, no pack moves
    the margin)."""
    import json
    from pathlib import Path
    base = Path("results/topic4_sef_hfo/heterogeneity")
    sm = json.loads((base / "sensitivity_matrix.json").read_text())
    cb = json.loads((base / "combo.json").read_text())
    # every closed_loop_converged flag across both artifacts must be True
    flags = []
    for p, r in sm["params"].items():
        flags += [r["baseline_wide"]["closed_loop_converged"],
                  r["mean_matched_narrow"]["closed_loop_converged"]]
    flags += [sm["mu_shift_control"]["baseline"]["closed_loop_converged"],
              sm["mu_shift_control"]["shifted"]["closed_loop_converged"]]
    for r in cb["combos"].values():
        flags += [r["baseline_wide"]["closed_loop_converged"],
                  r["mean_matched_narrow"]["closed_loop_converged"]]
    assert all(flags), "a committed Step3b layer has closed_loop_converged=False"
    # leverage ranking + no-margin-move (science lock)
    P = sm["params"]
    assert 8.0 < P["v_th"]["d_slope_pure_pct"] < 16.0          # V_th moves the curve ~+12%
    assert 3.0 < P["sigma"]["d_slope_pure_pct"] < 9.0          # sigma ~+6%
    assert abs(P["tau_m"]["d_slope_pure_pct"]) < 1.0           # tau_m static-dead
    assert abs(P["tau_ref"]["d_slope_pure_pct"]) < 1.0         # tau_ref dead
    for p in ("v_th", "sigma", "tau_m", "tau_ref"):
        assert abs(P[p]["d_closed_loop_re_max_pure"]) < 0.005  # none move the margin
    cvs = cb["combos"]["v_th+sigma"]
    assert 10.0 < cvs["d_slope_pure_pct"] < 22.0
    assert abs(cvs["d_closed_loop_re_max_pure"]) < 0.005
    # the control DID move the margin (apparatus alive)
    assert abs(sm["mu_shift_control"]["d_closed_loop_re_max"]) > 0.005


def test_mean_match_raises_specific_not_bracketed_exception():
    """The not-bracketed case raises the SPECIFIC MeanMatchNotBracketed (a ValueError
    subclass) so a screen catches ONLY 'no leverage' and lets genuine numerical/domain
    ValueErrors propagate (engineering fix 2026-06-07)."""
    from src.sef_hfo_heterogeneity import mean_match_param, MeanMatchNotBracketed
    from src.sef_hfo_lif import mean_field, V_RESET
    op = mean_field(1.0)
    muE, sE, nuE = op["muE"], op["sE"], op["nuE"]
    assert issubclass(MeanMatchNotBracketed, ValueError)
    # v_th over a tiny low-threshold bracket: rate >> nuE at both ends => not bracketed
    with pytest.raises(MeanMatchNotBracketed):
        mean_match_param(nuE, muE, sE, TAU_ME, TREF_E, param="v_th", std=0.5,
                         bracket=(V_RESET + 0.5, V_RESET + 1.0))


# ---- SNN per-neuron threshold field (2026-06-08 spec; Task 1) ----
from src.sef_hfo_heterogeneity import sample_threshold_fields, local_vth_spread


def _toy_sheet(n=4000, L=3.0, fE=0.8, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, L, size=(n, 2))
    is_E = np.zeros(n, bool); is_E[: int(fE * n)] = True
    return pos, is_E, rng


def test_threshold_fields_share_surround_and_narrow_core():
    pos, is_E, rng = _toy_sheet()
    out = sample_threshold_fields(pos, is_E, patch_center=(1.5, 1.5),
                                  patch_radius=0.5, rng=rng,
                                  vth_mean=18.0, std_wide=1.5, std_narrow=0.5,
                                  v_reset=11.0, core_mean_shift=2.0)
    base, matched, unmatched = out["baseline"], out["matched"], out["unmatched"]
    core = out["core_mask"]
    surround = is_E & ~core
    # surround is BIT-IDENTICAL across the three (paired-seed contract)
    np.testing.assert_array_equal(base[surround], matched[surround])
    np.testing.assert_array_equal(base[surround], unmatched[surround])
    # localized design (2026-06-08b): surround is SCALAR-quiet (=vth_mean), only the
    # core carries heterogeneity; baseline core is WIDE, matched/unmatched NARROW
    assert np.allclose(base[surround], 18.0)          # quiet scalar surround
    assert base[core].std() > 1.0                     # baseline core heterogeneous (wide)
    assert matched[core].std() < base[core].std()
    assert unmatched[core].std() < base[core].std()
    # matched core mean ~ 18 (held), unmatched core mean ~ 16 (shifted down)
    assert abs(matched[core].mean() - 18.0) < 0.3
    assert abs(unmatched[core].mean() - 16.0) < 0.3
    # physical domain: nothing below reset
    assert (base[is_E] >= 11.0).all()
    assert (matched[is_E] >= 11.0).all() and (unmatched[is_E] >= 11.0).all()
    # I neurons keep scalar threshold
    assert np.allclose(base[~is_E], 18.0)


def test_local_vth_spread_shows_core_narrowing():
    # localized design: surround scalar (~0 spread); narrowing the core lowers its
    # local spread relative to the wide baseline core (the Rich pathology direction).
    pos, is_E, rng = _toy_sheet()
    out = sample_threshold_fields(pos, is_E, (1.5, 1.5), 0.5, rng)
    core = out["core_mask"]; surround = is_E & ~core
    sp_base = local_vth_spread(pos, out["baseline"], is_E, radius=0.3)
    sp_matched = local_vth_spread(pos, out["matched"], is_E, radius=0.3)
    assert np.nanmean(sp_matched[core]) < np.nanmean(sp_base[core])   # core narrows
    assert np.nanmean(sp_base[surround]) < 0.1                        # quiet surround
    assert np.isnan(sp_base[~is_E]).all()                            # I not colored
