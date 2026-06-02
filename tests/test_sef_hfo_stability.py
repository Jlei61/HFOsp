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
