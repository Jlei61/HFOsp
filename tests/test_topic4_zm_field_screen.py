# tests/test_topic4_zm_field_screen.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_screen import (elliptical_exp_kernel, gaussian_kernel, cell_mass_fraction,
                                        kernel_axis_and_ar)

def test_kernels_normalised_and_self_zero():
    K = elliptical_exp_kernel(32, 20.0, 0.537, 0.269, np.radians(30))
    assert abs(K.sum() - 1.0) < 1e-9 and K[0, 0] == 0.0
    assert abs(gaussian_kernel(32, 20.0, 2.0).sum() - 1.0) < 1e-9        # abs() -- a negative diff must fail

def test_kernel_axis_and_ar_recovered_at_several_rotations():
    """Covariance-eigen recovery (NOT row/col HWHM -- DX varies along axis 0, so row/col is easy to flip)."""
    for deg in (0.0, 30.0, 45.0, 75.0):
        K = elliptical_exp_kernel(64, 20.0, 0.537, 0.269, np.radians(deg))
        axis, ar = kernel_axis_and_ar(K, 20.0)
        d = np.degrees(axis) % 180.0
        assert min(abs(d - deg), 180 - abs(d - deg)) < 8.0, (deg, d)     # axis within 8 deg
        assert 1.5 < ar < 3.0, (deg, ar)                                  # AR near 2

def test_cell_mass_fraction_scales_with_resolution():
    q32 = cell_mass_fraction(20.0, 32); q64 = cell_mass_fraction(20.0, 64)
    assert 0.15 < q32 < 0.30 and 0.04 < q64 < 0.12 and q64 < q32          # finer cells hold less mass

# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import FieldParams, simulate_field, ARMS
from src.topic4_zm_field_meanfield import simulate_meanfield, MFParams

def _P(**kw):
    return FieldParams(W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0, **kw)

def test_w_frac_is_derived_not_half():
    p = _P(n=32)
    assert p.w_frac is None
    from src.topic4_zm_field_screen import resolve_w_frac
    assert abs(resolve_w_frac(p) - 0.226) < 0.03            # derived q_cell, NOT 0.5

def test_dual_arms_identical_on_the_uniform_manifold():
    p = _P(n=16)
    outs = {a: simulate_field(p, a, T=800., dt=0.25, r_init=np.full((16, 16), 0.15))["final_state"]["r"]
            for a in ("dual_global", "dual_local", "dual_mixed")}
    assert np.allclose(outs["dual_global"], outs["dual_local"], atol=1e-9)
    assert np.allclose(outs["dual_global"], outs["dual_mixed"], atol=1e-9)
    assert np.allclose(outs["dual_global"], outs["dual_global"].mean())    # stays uniform

def test_div_global_arm_forces_beta_zero():
    p = _P(n=16)
    a = simulate_field(p, "div_global", T=1500., dt=0.25, r_init=np.full((16, 16), 0.15))
    b = simulate_field(FieldParams(W0=2., alpha=2., beta=0.0, theta=.5, I0=1., n=16), "dual_global",
                       T=1500., dt=0.25, r_init=np.full((16, 16), 0.15))
    assert np.allclose(a["final_state"]["r"], b["final_state"]["r"], atol=1e-9)

def test_uniform_field_reduces_to_meanfield():
    p = _P(n=16)
    fr = simulate_field(p, "dual_global", T=1500., dt=0.25, r_init=np.full((16, 16), 0.15), record_stride=20)
    mf = simulate_meanfield(MFParams(2., 2., 4., .5, 1.), T=1500., dt=0.25, r0=0.15)
    got = fr["r_trace"].reshape(fr["r_trace"].shape[0], -1).mean(axis=1)
    assert np.allclose(got, mf[::20, 0][:len(got)], atol=1e-6)

# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import field_metrics

def _osc(n, nt, phases, amp=0.8, base=0.1, period=40):
    t = np.arange(nt); f = np.empty((nt, n, n))
    for i in range(n):
        for j in range(n):
            f[:, i, j] = base + amp * (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * t / period + phases[i, j])))
    return f.astype(np.float32)

def test_inphase_vs_desync_metrics():
    # nt=600 (not 400): settle=0.25 drops 25%, so 600 frames leave 450 = 11.25 cycles at period=40,
    # clearing the LOCKED ncyc>=10 oscillation gate. With nt=400 only 7.5 cycles survive, no cell counts
    # as oscillatory, and BOTH inputs fail-close to R_phase=1.0 -- the test could not discriminate at all.
    n, nt = 8, 600
    mi = field_metrics(_osc(n, nt, np.zeros((n, n))), 5.0)
    md = field_metrics(_osc(n, nt, np.random.default_rng(0).uniform(0, 2 * np.pi, (n, n))), 5.0)
    assert mi["median_R_phase"] > 0.8 and md["median_R_phase"] < 0.5
    assert mi["osc_frac"] > 0.9 and mi["active_area_frac"] > 0.9

def test_local_period_survives_a_flat_population_signal():
    """The IDEAL staggered state flattens P(t) -> population period is NaN, but each cell still cycles.
    The gate must use the LOCAL period, else the best result would fail on a NaN."""
    n, nt = 8, 600
    f = _osc(n, nt, np.random.default_rng(1).uniform(0, 2 * np.pi, (n, n)))
    m = field_metrics(f, 5.0)
    assert 100.0 < m["median_local_period_ms"] < 300.0      # ~40 bins * 5 ms = 200 ms
    assert m["osc_frac"] > 0.9

def test_plateau_and_tiny_active_set_loopholes():
    n, nt = 8, 400
    plateau = np.full((nt, n, n), 0.8, np.float32)
    assert field_metrics(plateau, 5.0)["osc_frac"] < 0.1
    tiny = np.full((nt, n, n), 0.8, np.float32)
    tiny[:, 0, 0] = _osc(1, nt, np.zeros((1, 1)))[:, 0, 0]
    assert field_metrics(tiny, 5.0)["active_area_frac"] < 0.1

def test_phase_coverage_reported_and_failclosed():
    n, nt = 8, 400
    m = field_metrics(np.full((nt, n, n), 0.8, np.float32), 5.0)   # no oscillating cells at all
    assert m["phase_coverage_frac"] == 0.0 and m["median_R_phase"] == 1.0   # fail closed

def test_osc_frac_denominator_is_all_cells_not_the_active_subset():
    """LOCKED contract: osc_frac's denominator is ALL cells, never the active subset. Here 4 of 64 cells
    oscillate (with nt=600, enough frames to clear the ncyc>=10 gate) while the other 60 are flat. The
    correct denominator gives 4/64=0.0625; an active-subset denominator would give 1.0 and would let a
    tiny-active-set state pass as fully oscillatory."""
    n, nt = 8, 600
    f = np.full((nt, n, n), 0.10, np.float32)
    f[:, :2, :2] = _osc(2, nt, np.zeros((2, 2)))      # 2x2 = 4 oscillating cells, same frame budget
    m = field_metrics(f, 5.0)
    assert abs(m["osc_frac"] - 4.0 / 64.0) < 1e-9
    assert m["active_area_frac"] < 0.1                 # only 4/64 cells are active at all

# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import (uniform_orbit, variational_jacobian, transverse_floquet, floquet_map)

def test_constant_orbit_recovers_the_jacobian_eigenvalue():
    """With a CONSTANT 'orbit', the monodromy is exp(J*T) -> lambda == max Re eig(J). Known answer."""
    p = _P(n=32)
    const = np.tile(np.array([[0.3, 0.2, 0.2]]), (400, 1))       # frozen (r,mu,S)
    lam = transverse_floquet(p, "dual_local", 2, 0, const, 0.25)
    J = variational_jacobian(p, "dual_local", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert abs(lam - float(np.max(np.linalg.eigvals(J).real))) < 5e-3

def _wk_kk(p, mx, my):
    from src.topic4_zm_field_screen import mode_responses
    return mode_responses(p, mx, my)

def test_global_arm_is_one_dimensional_off_dc():
    p = _P(n=32)
    J = variational_jacobian(p, "dual_global", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert J.shape == (1, 1)                                     # no spatial pool d.o.f. at k != 0
    Jl = variational_jacobian(p, "dual_local", *_wk_kk(p, 2, 0), 0.3, 0.2, 0.2)
    assert Jl.shape == (3, 3)

def test_dc_mode_rejected_and_excluded_from_the_map():
    p = _P(n=32)
    const = np.tile(np.array([[0.3, 0.2, 0.2]]), (200, 1))
    try:
        transverse_floquet(p, "dual_local", 0, 0, const, 0.25)
    except ValueError:
        pass
    else:
        raise AssertionError("DC mode must be rejected")
    fm = floquet_map(p, "dual_local", const, 0.25, m_max=3)
    assert (0, 0) not in [tuple(m) for m in fm["modes"]]

def test_dt_halving_sign_margin_on_a_real_orbit():
    p = _P(n=32)
    o1, _ = uniform_orbit(p, 0.25); o2, _ = uniform_orbit(p, 0.125)
    l1 = transverse_floquet(p, "dual_local", 2, 0, o1, 0.25)
    l2 = transverse_floquet(p, "dual_local", 2, 0, o2, 0.125)
    assert np.sign(l1) == np.sign(l2) and abs(l1 - l2) < 0.5 * max(abs(l1), abs(l2), 1e-6) + 1e-3


def _field_rhs_parts(p, arm, r, muL, SL, muG, SG):
    """One RHS evaluation replicating simulate_field's own expressions (for independent FD checking)."""
    import src.topic4_zm_field_screen as M
    n = p.n
    KE = np.fft.rfft2(M.elliptical_exp_kernel(n, p.L, p.l_par, p.l_perp, p.theta_EE))
    KS = np.fft.rfft2(M.gaussian_kernel(n, p.L, p.sigma_S))
    q = M.resolve_w_frac(p); w_rec, w_c = p.W0 * q, p.W0 * (1.0 - q)
    beta = M.arm_beta(p, arm)
    Se = M._S_eff(arm, SL, SG, p.eps_G)
    rec_E = w_rec * r + w_c * np.fft.irfft2(np.fft.rfft2(r) * KE, s=(n, n))
    u = p.I0 + rec_E / (1.0 + p.alpha * Se) - beta * Se - p.theta
    dr = (-r + M._Fsat(u, 0.5)) / p.tau_a
    z = M.psi_recruit(r, 0.0, p.r50, p.n_psi)
    conv = np.fft.irfft2(np.fft.rfft2(z ** p.p_pool) * KS, s=(n, n))
    A_L = np.maximum(conv, 0.0) ** (1.0 / p.p_pool)
    dmuL = (-muL + A_L) / p.tau_mu
    dSL = (-SL + p.S_max * muL) / p.tau_S
    return dr, dmuL, dSL


def _mode_field(n, mx, my):
    i = np.arange(n)[:, None]; j = np.arange(n)[None, :]
    return np.cos(2 * np.pi * (mx * i + my * j) / n)


def test_variational_jacobian_matches_finite_difference_of_the_field_rhs():
    """Independent mutation-catching net: the analytic Jacobian must equal a finite difference of the
    ACTUAL field RHS projected on the same Fourier mode. This fails if the base state uses Wk instead of
    W0, if psi_prime is evaluated at the wrong variable, if c_S is dropped for the mixed arm, or if the
    u0<=0 branch guard is removed (that last mutation flips the growth rate's sign)."""
    from src.topic4_zm_field_screen import mode_responses, variational_jacobian
    eps = 1e-6

    def coeff(fp, fm, C):
        d = (fp - fm) / (2 * eps)
        return float((d * C).sum() / (C * C).sum())

    for (r0, mu0, S0) in [(0.30, 0.20, 0.20), (0.05, 0.10, 0.50)]:   # 2nd base point has u0 < 0
        for arm in ("dual_local", "dual_mixed", "dual_global"):
            for (mx, my) in [(2, 0), (1, 3)]:                        # even and ODD modes
                p = FieldParams(W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0, n=16)
                C = _mode_field(p.n, mx, my)
                Wk, Kk = mode_responses(p, mx, my)
                J = variational_jacobian(p, arm, Wk, Kk, r0, mu0, S0)
                R = np.full((p.n, p.n), r0); MU = np.full((p.n, p.n), mu0); S = np.full((p.n, p.n), S0)
                fp = _field_rhs_parts(p, arm, R + eps * C, MU, S, mu0, S0)[0]
                fm = _field_rhs_parts(p, arm, R - eps * C, MU, S, mu0, S0)[0]
                assert abs(coeff(fp, fm, C) - J[0, 0]) < 1e-6, ("a_rr", arm, mx, my, r0)
                if J.shape[0] == 3:
                    sp = _field_rhs_parts(p, arm, R, MU, S + eps * C, mu0, S0)[0]
                    sm = _field_rhs_parts(p, arm, R, MU, S - eps * C, mu0, S0)[0]
                    assert abs(coeff(sp, sm, C) - J[0, 2]) < 1e-6, ("a_rS", arm, mx, my, r0)
                    mp = _field_rhs_parts(p, arm, R + eps * C, MU, S, mu0, S0)[1]
                    mm = _field_rhs_parts(p, arm, R - eps * C, MU, S, mu0, S0)[1]
                    assert abs(coeff(mp, mm, C) - J[1, 0]) < 1e-6, ("a_mr", arm, mx, my, r0)


def test_mode_responses_at_dc_returns_total_gain_and_unit_pool():
    """Kernel-layout check: at DC both normalised kernels contribute their full mass."""
    from src.topic4_zm_field_screen import mode_responses
    p = FieldParams(W0=2.0, alpha=2.0, beta=4.0, theta=0.5, I0=1.0, n=16)
    Wk, Kk = mode_responses(p, 0, 0)
    assert abs(Wk - p.W0) < 1e-9 and abs(Kk - 1.0) < 1e-9

# append to tests/test_topic4_zm_field_screen.py
from src.topic4_zm_field_screen import orbit_phasepoint_state, add_r_perturbation

def test_phasepoint_resets_every_field_uniformly():
    p = _P(n=8); orbit, _ = uniform_orbit(p, 0.25)
    st = orbit_phasepoint_state(p, orbit, len(orbit) // 3)
    for k in ("r", "muL", "SL"):
        assert np.allclose(st[k], st[k].flat[0])       # no leftover spatial S_L memory
    assert np.isscalar(st["muG"]) and np.isscalar(st["SG"])

def test_perturbation_is_zero_mean_and_r_only():
    p = _P(n=16); orbit, _ = uniform_orbit(p, 0.25)
    st0 = orbit_phasepoint_state(p, orbit, 5)
    st1 = add_r_perturbation({k: (v.copy() if hasattr(v, "copy") else v) for k, v in st0.items()}, 1e-4, 0, 16)
    assert abs(float((st1["r"] - st0["r"]).mean())) < 1e-12
    assert np.allclose(st1["SL"], st0["SL"]) and np.allclose(st1["muL"], st0["muL"])
