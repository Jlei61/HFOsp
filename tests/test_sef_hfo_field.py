# tests/test_sef_hfo_field.py
import numpy as np
from dataclasses import replace
from src.sef_hfo_field import SEFParams, build_kernels, make_Feff_lookup, integrate_field
from src.sef_hfo_stability import self_consistent_operating_point

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
