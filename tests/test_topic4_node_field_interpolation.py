import numpy as np
import pytest

from src.topic4_node_field_search import interpolate_spline_candidates


def _field(value):
    coefficients = np.full((4, 4), value, float)
    return {
        "field_type": "spline_continuous",
        "n_basis": 4,
        "degree": 3,
        "coefficients": coefficients.tolist(),
        "field_sha256": f"field-{value}",
    }


def test_interpolation_is_whole_sheet_and_preserves_endpoints():
    left, right = _field(0.0), _field(2.0)
    middle = interpolate_spline_candidates(
        left, right, weight=0.25, candidate_id="middle",
    )
    np.testing.assert_allclose(middle["coefficients"], 0.5)
    assert middle["component_count"] is None
    assert middle["peak_count_constraint"] is None
    assert middle["residual_coordinates"]["observation_coordinates_used"] is False


def test_interpolation_rejects_extrapolation():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        interpolate_spline_candidates(
            _field(0.0), _field(1.0), weight=1.1, candidate_id="bad",
        )
