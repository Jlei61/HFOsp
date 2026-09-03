import numpy as np
import pytest

from scripts.audit_topic4_patient_zm_grid_convergence import (
    compare_branch_fields,
    remap_square_field,
)


def test_piecewise_constant_common_grid_remap_preserves_mean():
    source = np.arange(100, dtype=float)
    target = remap_square_field(source, source_grid=10, target_grid=60)
    assert target.shape == (3600,)
    assert np.mean(target) == pytest.approx(np.mean(source))
    assert np.unique(target, return_counts=True)[1] == pytest.approx(
        np.full(100, 36))


def test_branch_field_comparison_separates_scale_and_pattern():
    first = np.asarray([1.0, 2.0, 3.0, 4.0])
    second = 2.0 * first
    result = compare_branch_fields(first, second)
    assert result["centered_spatial_correlation"] == pytest.approx(1.0)
    assert result["relative_mean_rate_difference"] == pytest.approx(2.0 / 3.0)
    assert result["rms_field_difference_hz"] == pytest.approx(
        np.sqrt(np.mean(first ** 2)))


def test_common_grid_remap_rejects_non_divisible_target():
    with pytest.raises(ValueError, match="divisible"):
        remap_square_field(np.zeros(15 * 15), source_grid=15, target_grid=64)
