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

# append to tests/test_topic4_zm_field_meanfield.py
from src.topic4_zm_field_meanfield import contiguous_runs, meanfield_continuation

def test_contiguous_runs_splits_gaps():
    assert contiguous_runs([True, True, False, True, True, True]) == [(0, 2), (3, 6)]
    assert contiguous_runs([False, False]) == []

def test_continuation_minimal_intervention_prefers_smallest_beta():
    # beta=2 and beta=8 BOTH oscillate at (W0=2, alpha=2, I0~1.0-1.5) -> must pick beta=2 (least new
    # mechanism). Unconditional: if this grid yields no orbit the test FAILS (it must not pass vacuously).
    r = meanfield_continuation(grid=dict(W0=[2], alpha=[2], beta=[2, 8], theta=[0.5]),
                               I0s=np.arange(0.8, 1.81, 0.1), min_seg=3)
    assert r["has_orbit"], "expected an orbit for both beta=2 and beta=8 in this grid"
    assert r["operating_point"]["beta"] == 2

def test_continuation_reports_interior_levels_and_segment():
    r = meanfield_continuation(min_seg=5)
    assert r["has_orbit"], "Phase-0 must find an orbit for the field to be built"
    seg = r["segment"]
    assert len(seg["interior_I0s"]) >= 3
    assert seg["I0_lo"] < min(seg["interior_I0s"]) and max(seg["interior_I0s"]) < seg["I0_hi"]

def test_continuation_beta0_grid_has_no_orbit():
    r = meanfield_continuation(grid=dict(W0=[2, 4], alpha=[2, 8, 16], beta=[0.0], theta=[0.5]), min_seg=3)
    assert not r["has_orbit"] and r["operating_point"] is None
