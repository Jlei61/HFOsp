import numpy as np
import pytest
from src.sef_hfo_lif import lif_rate, V_TH, TAU_ME, TREF_E


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
