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
