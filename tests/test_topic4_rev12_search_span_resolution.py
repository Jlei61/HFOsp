import numpy as np

from scripts.audit_topic4_rev12_search_span_resolution import project_difference


def _field(coefficients):
    return {
        "coefficients": np.asarray(coefficients, float).tolist(),
        "n_basis": 18,
        "degree": 3,
    }


def test_projection_explains_low_frequency_capacity_difference():
    from src.topic4_node_field_search import cosine_sheet_residuals

    basis = cosine_sheet_residuals(
        maximum_frequency=3, target_n_basis=18,
        projection_grid_per_axis=41,
    )
    direction = np.asarray(basis["rows"][4]["coefficients"], float)
    result = project_difference(
        _field(np.zeros((18, 18))), _field(1.7 * direction),
        maximum_frequency=3, grid_per_axis=41,
        stored_n_basis=18, degree=3,
    )
    assert result["explained_fraction"] > 0.999999
    assert np.isclose(result["difference_surface_rms"], 1.7, rtol=2e-3)


def test_projection_reports_unresolved_high_frequency_component():
    from src.topic4_node_field_search import cosine_sheet_residuals

    basis = cosine_sheet_residuals(
        maximum_frequency=5, target_n_basis=18,
        projection_grid_per_axis=41,
    )
    direction = np.asarray(basis["rows"][-1]["coefficients"], float)
    result = project_difference(
        _field(np.zeros((18, 18))), _field(direction),
        maximum_frequency=3, grid_per_axis=41,
        stored_n_basis=18, degree=3,
    )
    assert result["explained_fraction"] < 0.05
