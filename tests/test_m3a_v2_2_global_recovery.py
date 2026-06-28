"""TDD contracts for M3A-v2.2 global inhibitory recovery h_G.

Encodes the §B6 / design-spec contracts: h_G off-by-default byte-parity, M/B/Pi sensors +
chi_G smooth-AND, h_G ODE (k_G=0 still decays) + hG_script clamp/surrogate, E-only coupling,
proxy Y=P_global-beta_G*h_G (X invariant), sustained ramp+HOLD drive, and the pilot's
(time,neuron) readout + fail-closed segmentation + paired/order-invariant RNG.

Spec:  docs/snn_core_model_equations.md §B6
Plan:  docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2.2-global-recovery-plan.md

Run the fast set with `pytest tests/test_m3a_v2_2_global_recovery.py -m "not slow"`.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.topic4_m3a_v2_2_sensors import (  # noqa: E402
    hill, global_M, global_B, global_participation, chi_G)


def _tiny_field(use_hG=False, **cfgkw):
    """Small deterministic SpatialSlowField for the field-level h_G tests."""
    rng = np.random.default_rng(0)
    L, nE, nI = 10.0, 12, 6
    posE = rng.uniform(0, L, (nE, 2)); posI = rng.uniform(0, L, (nI, 2))
    cfg = SpatialSlowFieldConfig(n_grid=8, use_hG=use_hG, **cfgkw)
    return SpatialSlowField(nE + nI, 18.0, posE, posI, L, cfg=cfg), nE, nI


# ===========================================================================
# Task 2 -- config h_G fields + validate (off-by-default)
# ===========================================================================
def test_config_hG_off_by_default():
    c = SpatialSlowFieldConfig()
    assert c.use_hG is False and c.hG_init == 0.0 and c.k_G == 0.0
    c.validate()  # locked defaults are valid


def test_config_rejects_tau_s_nonpositive():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(tau_s=0.0).validate()


def test_config_rejects_negative_kG_and_lambdaG():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(k_G=-1.0).validate()
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(lambda_G=-0.1).validate()


def test_config_rejects_nonpositive_hill_thresholds():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(M50=0.0).validate()


# ===========================================================================
# Task 3 -- sensors (M/B/Pi + chi_G smooth-AND)
# ===========================================================================
def test_sensor_hill_half_at_z50():
    assert abs(hill(2.0, 2.0, 4.0) - 0.5) < 1e-9
    assert hill(0.0, 1.0, 4.0) == 0.0
    assert hill(1e6, 1.0, 4.0) > 0.999


def test_sensor_participation_single_hot_low_uniform_high():
    n = 8
    hot = np.zeros((n, n)); hot[0, 0] = 1.0
    uni = np.ones((n, n))
    assert global_participation(hot) < 0.05           # ~1/N_x
    assert global_participation(uni) > 0.99           # ~1


def test_sensor_B_soft_between_0_and_1():
    field = np.linspace(-2, 2, 16).reshape(4, 4)
    b = global_B(field, r_A=0.0, Delta_A=1.0)
    assert 0.0 < b < 1.0


def test_chi_low_for_local_axial_high_for_global():
    n = 16
    local = np.zeros((n, n)); local[7:9, :] = 1.0     # thin axial stripe
    Ml, Bl, Pl = global_M(local), global_B(local, 0.0, 0.3), global_participation(local)
    glob = np.full((n, n), 1.0)                        # broad high recruitment
    Mg, Bg, Pg = global_M(glob), global_B(glob, 0.0, 0.3), global_participation(glob)
    chi_local = chi_G(Ml, Bl, Pl, 0.5, 0.5, 0.45, 4, 4, 4)
    chi_glob = chi_G(Mg, Bg, Pg, 0.5, 0.5, 0.45, 4, 4, 4)
    assert chi_local < 0.1 < chi_glob
