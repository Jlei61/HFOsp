"""M3A-A2 builder: k_use derivation, NE propagation, mutual-exclusion guard."""
import sys, os
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_a2 import build_regional_resource, k_use_from_target, assert_a2_exclusive  # noqa: E402


def test_k_use_from_target():
    k = k_use_from_target(q_target=0.74, a_bar=0.05, tau_rec=5000.0)
    assert abs(k - (1.0 / 0.74 - 1.0) / (0.05 * 5000.0)) < 1e-12


def test_builder_derives_k_use_from_target():
    cm = np.zeros(12, bool); cm[:4] = True
    rr = build_regional_resource(12, 18.0, cm, NE=10, mode="core_only",
                                 q_target=0.74, a_bar=0.05, tau_rec=5000.0)
    assert abs(rr.cfg.k_use - k_use_from_target(0.74, 0.05, 5000.0)) < 1e-12


def test_builder_passes_NE():
    cm = np.zeros(12, bool); cm[:4] = True
    rr = build_regional_resource(12, 18.0, cm, NE=10, mode="core_only", k_use=0.0)
    assert rr.NE == 10                                    # [P0-1] NE must propagate (else I cells scaled)
    assert rr.is_E.sum() == 10


def test_mutual_exclusion_raises():
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="z", shunt_gaba=False, feedback_gain=0.0)
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="none", shunt_gaba=True, feedback_gain=0.0)
    with pytest.raises(ValueError):
        assert_a2_exclusive(slow_var="none", shunt_gaba=False, feedback_gain=8.0)
    assert_a2_exclusive(slow_var="none", shunt_gaba=False, feedback_gain=0.0)  # ok


def test_builder_requires_target_or_k_use():
    cm = np.zeros(12, bool); cm[:4] = True
    with pytest.raises(ValueError):
        build_regional_resource(12, 18.0, cm, NE=10, mode="core_only")  # neither q_target nor k_use
