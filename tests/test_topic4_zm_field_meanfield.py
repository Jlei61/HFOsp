# tests/test_topic4_zm_field_meanfield.py
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_field_meanfield import (F, psi, psi_prime, MFParams, simulate_meanfield, detect_orbit)

def test_F_and_psi():
    assert F(-1.0) == 0.0 and abs(F(0.5) - 0.5) < 1e-9
    assert psi(0.0) == 0.0 and 0.0 < psi(0.4) < 1.0
    # psi_prime matches a numeric derivative
    h = 1e-6
    assert abs(psi_prime(0.5) - (psi(0.5 + h) - psi(0.5 - h)) / (2 * h)) < 1e-4

def test_dual_pool_oscillates():
    o = detect_orbit(simulate_meanfield(MFParams(W0=2., alpha=2., beta=4., theta=.5, I0=1.)), 0.25)
    assert o["oscillates"] and o["ncyc"] >= 6
    assert o["trough"] < 0.25 * o["peak"] and 100.0 < o["period_ms"] < 300.0

def test_divisive_only_beta0_has_no_orbit():
    """The CURRENT Z/M sg arm is beta_SG=0 -> purely divisive -> no synchronised orbit (Phase-0 finding)."""
    for alpha in (2.0, 8.0, 16.0):
        o = detect_orbit(simulate_meanfield(MFParams(W0=2., alpha=alpha, beta=0.0, theta=.5, I0=1.)), 0.25)
        assert not o["oscillates"]
