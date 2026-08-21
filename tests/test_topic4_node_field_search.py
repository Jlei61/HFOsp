import numpy as np

from src.topic4_node_field_search import (
    coarse_residual_to_coefficients,
    residual_candidate,
    sobol_coarse_residuals,
)


def _anchor():
    values = np.zeros((18, 18), float)
    return {
        "field_type": "spline_continuous", "n_basis": 18, "degree": 3,
        "coefficients": values.tolist(), "field_sha256": "anchor",
    }


def test_coarse_residual_projection_is_whole_sheet_and_unit_rms():
    control = np.arange(16, dtype=float).reshape(4, 4)
    result = coarse_residual_to_coefficients(control)
    assert result["coefficients"].shape == (18, 18)
    assert np.isclose(result["surface_rms"], 1.0)
    assert result["projection_rmse"] < 1e-10


def test_sobol_residuals_are_deterministic_and_contact_free():
    first = sobol_coarse_residuals(n_residuals=5, seed=7)
    second = sobol_coarse_residuals(n_residuals=5, seed=7)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert all(np.isclose(np.mean(row), 0.0) for row in first)


def test_residual_candidate_changes_the_field_hash_without_components():
    projected = coarse_residual_to_coefficients(
        sobol_coarse_residuals(n_residuals=1, seed=4)[0],
    )["coefficients"]
    candidate = residual_candidate(
        _anchor(), projected, amplitude=0.5,
        candidate_id="candidate", residual_index=0,
    )
    assert candidate["field_sha256"] != "anchor"
    assert candidate["component_count"] is None
    assert candidate["residual_coordinates"]["observation_coordinates_used"] is False


def test_residual_candidate_records_stage_b_control_resolution():
    residual = np.ones((18, 18), dtype=float)
    candidate = residual_candidate(
        _anchor(), residual, amplitude=0.2, candidate_id="stage_b",
        residual_index=0, coarse_n_basis=6,
    )
    assert candidate["residual_coordinates"]["coarse_n_basis"] == 6
