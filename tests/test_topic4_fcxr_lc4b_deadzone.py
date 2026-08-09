from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc4b_deadzone import (
    TARGET_CURRENT,
    build_locked_candidate,
    deadzone_activation,
)


def test_deadzone_is_exactly_zero_below_and_at_threshold():
    a = deadzone_activation([0.0, 2.0, 3.0], deadzone=3.0, excess_scale=2.0, n=4)
    assert np.array_equal(a, np.zeros(3))


def test_excess_half_scale_has_half_activation():
    a = deadzone_activation([5.0], deadzone=3.0, excess_scale=2.0, n=4)
    assert a[0] == pytest.approx(0.5)


def test_locked_rules_are_force_matched_and_do_not_search():
    quiet = np.asarray([0.0, 1.0, 2.0])
    ictal = np.asarray([6.0, 8.0, 10.0])
    c = build_locked_candidate(quiet, ictal)
    assert c["deadzone"] == 4.0
    assert c["K"] == 4.0
    a = deadzone_activation(ictal, deadzone=c["deadzone"], excess_scale=c["K"], n=c["n"])
    assert c["g_m_max"] * np.mean(a) == pytest.approx(TARGET_CURRENT)
    assert c["calibration"]["interictal_activation_max"] == 0.0


def test_overlapping_extremes_fail_before_simulation():
    with pytest.raises(ValueError, match="DEADZONE_NOT_IDENTIFIABLE"):
        build_locked_candidate([0.0, 4.0], [3.0, 5.0])


@pytest.mark.parametrize("kwargs", [
    dict(deadzone=-1.0, excess_scale=1.0, n=4),
    dict(deadzone=1.0, excess_scale=0.0, n=4),
    dict(deadzone=1.0, excess_scale=1.0, n=0),
])
def test_invalid_activation_parameters_fail_loudly(kwargs):
    with pytest.raises(ValueError):
        deadzone_activation([1.0], **kwargs)
