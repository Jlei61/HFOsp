import numpy as np
import pytest

from scripts.run_topic4_zm_joint_morphology_canary import _paired_windows
from src.topic4_zm_ictal_transition import dose_local_connectivity_coefficients


def test_paired_windows_keeps_fixed_pre_and_delayed_post():
    values = np.arange(20, dtype=float)
    paired, half_ms = _paired_windows(
        values, 1.0, pre_window=(2.0, 7.0), post_window=(12.0, 17.0))
    assert half_ms == 5.0
    assert np.array_equal(paired[:5], np.arange(2.0, 7.0))
    assert np.array_equal(paired[5:], np.arange(12.0, 17.0))


def test_pathway_dose_default_is_exact_and_does_not_mutate_input():
    coefficients = np.arange(12, dtype=float).reshape(2, 6)
    original = coefficients.copy()
    dosed = dose_local_connectivity_coefficients(coefficients)
    assert np.array_equal(dosed, original)
    assert np.array_equal(coefficients, original)
    assert dosed is not coefficients


def test_pathway_dose_scales_only_requested_row():
    coefficients = np.ones((2, 6), dtype=float)
    dosed = dose_local_connectivity_coefficients(
        coefficients, ee_dose=1.0, etoi_dose=0.25)
    assert np.array_equal(dosed[0], np.ones(6))
    assert np.array_equal(dosed[1], np.full(6, 0.25))


def test_pathway_dose_rejects_negative_or_wrong_shape():
    with pytest.raises(ValueError, match="non-negative"):
        dose_local_connectivity_coefficients(np.ones((2, 6)), etoi_dose=-0.1)
    with pytest.raises(ValueError, match="shape"):
        dose_local_connectivity_coefficients(np.ones(6))
