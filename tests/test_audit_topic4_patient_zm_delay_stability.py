import pytest

from scripts.audit_topic4_patient_zm_delay_stability import (
    linear_dt_extrapolation,
)


def test_linear_dt_extrapolation_recovers_intercept():
    result = linear_dt_extrapolation(
        [0.5, 0.25, 0.1], [2.0, 1.5, 1.2])
    assert result["intercept_at_dt0"] == pytest.approx(1.0)
    assert result["slope_per_ms"] == pytest.approx(2.0)
    assert result["maximum_absolute_fit_residual"] < 1e-12


def test_linear_dt_extrapolation_rejects_single_point():
    with pytest.raises(ValueError, match="multi-point"):
        linear_dt_extrapolation([0.5], [1.0])
