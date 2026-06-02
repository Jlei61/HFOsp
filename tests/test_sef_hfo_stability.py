# tests/test_sef_hfo_stability.py
import numpy as np
from src.sef_hfo_stability import F_eff, gain, _f

def test_Feff_reduces_to_single_unit_when_sigma_zero():
    for h in [-1.0, 0.0, 1.5]:
        assert abs(F_eff(h, 0.0, 1e-6, 4.0) - _f(h, 4.0)) < 1e-4

def test_gain_sign_flips_between_steep_point_and_tail():
    g_steep_small = gain(0.0, 0.0, 0.3, 4.0); g_steep_large = gain(0.0, 0.0, 2.0, 4.0)
    assert g_steep_small > g_steep_large                 # steep point: lower sigma -> higher gain
    g_tail_small = gain(6.0, 0.0, 0.3, 4.0); g_tail_large = gain(6.0, 0.0, 2.0, 4.0)
    assert g_tail_small < g_tail_large                   # deep tail: sign flips

from src.sef_hfo_field import SEFParams
from src.sef_hfo_stability import self_consistent_operating_point

def test_operating_point_self_consistent_and_converged():
    p = SEFParams()
    op = self_consistent_operating_point(p, 0.4, 0.15)
    assert op["converged"] is True
    assert abs(F_eff(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta) - op["r_E0"]) < 1e-6
    assert abs(F_eff(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta) - op["r_I0"]) < 1e-6

def test_operating_point_flags_multiplicity():
    # strong recurrent excitation + low heterogeneity can be bistable -> >1 distinct root
    p = SEFParams(J_EE=4.0, sigma_phi=0.3, beta=8.0)
    op = self_consistent_operating_point(p, 0.5, 0.05)
    assert "n_distinct_roots" in op and op["n_distinct_roots"] >= 1
    if op["n_distinct_roots"] > 1:
        assert op["bistable"] is True

from src.sef_hfo_stability import (gaussian_hat, build_dispersion_matrix, eta_lin)

def test_matrix_reduces_to_rate_jacobian_without_kinetics():
    # erlang_n=0, tau_syn=0, b_a=0 => augmented matrix == bare 2x2 rate Jacobian
    p = SEFParams(erlang_n=0, tau_AMPA=0.0, tau_GABA=0.0, b_a=0.0)
    op = self_consistent_operating_point(p, 0.4, 0.15)
    M = build_dispersion_matrix(p, op, kpar=0.5, kperp=0.0)
    assert M.shape == (2, 2)
    G_E = gain(op["h_E0"], p.phi_bar, p.sigma_phi, p.beta)
    G_I = gain(op["h_I0"], p.phi_bar, p.sigma_phi, p.beta)
    WEE = p.J_EE * gaussian_hat(0.5, 0.0, p.ell_par, p.ell_perp)
    assert abs(M[0, 0] - (-1.0 + G_E * WEE) / p.tau_E) < 1e-12

def test_no_coupling_decoupled_eigenvalues():
    p = SEFParams(J_EE=0, J_EI=0, J_IE=0, J_II=0, erlang_n=0, tau_AMPA=0.0, tau_GABA=0.0)
    op = self_consistent_operating_point(p, 0.3, 0.1)
    ev = np.sort(np.linalg.eigvals(build_dispersion_matrix(p, op, 0.5, 0.0)).real)
    assert np.allclose(ev, np.sort([-1/p.tau_E, -1/p.tau_I]), atol=1e-9)

def test_stronger_EE_less_stable_at_fixed_operating_point():
    op = self_consistent_operating_point(SEFParams(), 0.4, 0.15)
    k = np.linspace(-2, 2, 25)
    assert eta_lin(SEFParams(J_EE=1.5), op, k) < eta_lin(SEFParams(J_EE=0.5), op, k)
