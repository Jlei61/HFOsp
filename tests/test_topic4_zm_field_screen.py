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
