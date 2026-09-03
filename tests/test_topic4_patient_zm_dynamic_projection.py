import numpy as np
import pytest

from scripts.audit_topic4_patient_zm_dynamic_projection import (
    first_downcrossing,
    smoothed_rate_at,
)


def test_first_downcrossing_interpolates_and_ignores_initial_below():
    time = np.asarray([0.0, 1.0, 2.0, 3.0])
    values = np.asarray([1.0, 0.9, 0.7, 0.6])
    assert first_downcrossing(time, values, 0.8) == pytest.approx(1.5)
    assert first_downcrossing(time, values, 0.5) is None


def test_smoothed_rate_preserves_constant_signal():
    time = np.arange(0.0, 100.0, 0.1)
    observed = smoothed_rate_at(
        time, np.full(time.size, 37.0), np.asarray([3.0, 50.0, 98.0]),
        window_ms=20.0)
    assert observed == pytest.approx([37.0, 37.0, 37.0])


def test_smoothed_rate_rejects_misaligned_input():
    with pytest.raises(ValueError, match="invalid"):
        smoothed_rate_at(np.arange(3.0), np.arange(4.0), np.arange(2.0))
