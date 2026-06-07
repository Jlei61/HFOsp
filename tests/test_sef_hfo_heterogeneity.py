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
