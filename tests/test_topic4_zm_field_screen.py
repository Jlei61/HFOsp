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
