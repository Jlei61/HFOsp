# tests/test_sef_hfo_field.py
import numpy as np
from dataclasses import replace
from src.sef_hfo_field import SEFParams, build_kernels, make_Feff_lookup, integrate_field, anisotropic_gaussian, integrate_field_with_ic
from src.sef_hfo_stability import self_consistent_operating_point, build_dispersion_matrix, gaussian_hat

def test_params_scaffold_invariants():
    p = SEFParams()
    assert p.ell_perp < p.ell_par            # propagation axis exists
    assert p.sigma_I > p.ell_par             # wide inhibition
    assert p.tau_AMPA < p.tau_GABA           # fast excitation, slow inhibition
    assert p.b_a == 0.0                      # recovery OFF by default (switchable)
    assert p.erlang_n >= 1

def test_kernels_normalized():
    K = build_kernels(SEFParams(n=32, L=32.0))
    for name, k in K.items(): assert abs(k.sum() - 1.0) < 1e-9, name

def test_field_holds_at_fixed_point():
    p = SEFParams(n=32, L=32.0); op = self_consistent_operating_point(p, 0.4, 0.15)
    act = integrate_field(p, op, 0.4, 0.15, stim_fn=lambda t: 0.0, dt=0.05, t_max=5.0)
    assert np.max(np.abs(act[-1] - op["r_E0"])) < 1e-3

def test_recovery_lowers_steady_response_under_constant_drive():
    # WIRING test (not phenomenon): recovery's negative feedback lowers the uniform
    # steady response. Whether recovery enables self-limited propagation is 0b DATA.
    base = SEFParams(n=16, L=16.0)
    def steady(p):
        op = self_consistent_operating_point(p, 0.2, 0.1)
        return float(integrate_field(p, op, 0.2, 0.1, stim_fn=lambda t: 0.3,
                                     dt=0.05, t_max=120.0)[-1].mean())
    assert steady(replace(base, b_a=1.0, tau_a=10.0)) < steady(replace(base, b_a=0.0)) - 1e-4

def test_fft_kernel_matches_analytic_hat():
    # discrete FFT of the real-space kernel must match the analytic gaussian_hat at grid k
    p = SEFParams(n=64, L=64.0); g = anisotropic_gaussian(p.n, p.L, p.ell_par, p.ell_perp, 0.0)
    ghat = np.fft.fft2(np.fft.ifftshift(g)).real
    kx = 2*np.pi*np.fft.fftfreq(p.n, d=p.L/p.n)
    m = 3; analytic = gaussian_hat(kx[m], 0.0, p.ell_par, p.ell_perp)
    assert abs(ghat[m, 0] - analytic) < 1e-2

def test_field_linearization_matches_dispersion_eigenvalue():
    # DECISIVE consistency (advisor gate): seed a small mode-k perturbation, measure the
    # field's rE-mode decay, and confirm it matches the dispersion MATRIX's prediction for
    # the SAME observable (expm(M t) @ ic, ic = the rE-only initial perturbation).
    #
    # NOTE (deviation from plan verbatim): the plan compared rate_meas to the matrix's
    # max-Re eigenvalue. That is the wrong comparison here: an rE-only seed + rE-only FFT
    # tracking does NOT isolate the max-Re eigenmode -- at this operating point the max-Re
    # eigenvalue is a complex pair (-0.430+-0.224j) with tiny rE projection (0.053), while a
    # real -0.500 mode has 3x the rE projection, so the rE observable measures a mode
    # MIXTURE (~-0.63), not max-Re. The fix compares like-for-like: fit a rate to the field's
    # rE-mode AND to the matrix's predicted rE-mode over the same window. This is a STRONGER
    # check (verifies the full linearized operator on the identical observable, still via
    # build_dispersion_matrix) and is regime-robust. Verified |diff| 0.004-0.007 across
    # 3 operating points x 3 modes (tol 0.05). See step0_results writeup.
    from scipy.linalg import expm
    p = SEFParams(n=64, L=64.0); op = self_consistent_operating_point(p, 0.4, 0.15)
    n, L = p.n, p.L; x = (np.arange(n) - n//2) * (L/n)
    m = 2; kpar = 2*np.pi*m/L
    eps = 1e-4
    def stim_fn(t): return 0.0
    # perturb initial E by eps*cos(kpar x); measure growth of that Fourier component
    pert = eps*np.cos(kpar*x)[:, None]*np.ones((1, n))
    act = integrate_field_with_ic(p, op, 0.4, 0.15, stim_fn, dt=0.02, t_max=6.0, dE0=pert)
    amp = np.abs(np.fft.fft2(act - op["r_E0"])[:, m, 0])
    t = np.arange(len(amp))*0.02; lo, hi = len(amp)//4, len(amp)//2
    rate_meas = np.polyfit(t[lo:hi], np.log(amp[lo:hi] + 1e-30), 1)[0]
    # matrix prediction of the SAME observable: rE component of expm(M t) @ (rE-only IC)
    M = build_dispersion_matrix(p, op, kpar, 0.0)
    ic = np.zeros(M.shape[0]); ic[0] = float(amp[0])
    amp_pred = np.abs(np.array([(expm(M * tt) @ ic)[0] for tt in t]))
    rate_pred = np.polyfit(t[lo:hi], np.log(amp_pred[lo:hi] + 1e-30), 1)[0]
    assert abs(rate_meas - rate_pred) < 0.05
