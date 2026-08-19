import numpy as np

from scripts.paper_figures.plot_fig5_data_driven_zm_main import (
    _accepted_display_xy,
    _fit_accepted_display,
)


def test_accepted_display_recovers_name_aligned_similarity_transform():
    current_names = np.asarray(["ICL1", "SCL6", "ICL2", "SCL7"])
    current = np.asarray([
        [-2.0, -1.0],
        [-1.0, 2.0],
        [2.0, 1.0],
        [1.0, -2.0],
    ])
    angle = np.deg2rad(31.0)
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    accepted_by_current_name = 0.87 * current @ rotation + np.asarray([10.0, 10.0])
    reference_names = np.asarray(["SCL7", "ICL2", "SCL6", "ICL1"])
    reference = np.asarray([
        accepted_by_current_name[np.flatnonzero(current_names == name)[0]]
        for name in reference_names
    ])

    display = _fit_accepted_display(
        current, current_names, reference, reference_names)
    transformed = _accepted_display_xy(current, display)
    expected = accepted_by_current_name - np.mean(reference, axis=0)

    assert np.allclose(transformed, expected, atol=1e-12, rtol=0.0)
    assert display["fit_rmse_mm"] < 1e-12
    assert display["fit_max_error_mm"] < 1e-12


def test_accepted_display_rejects_contact_identity_drift():
    with np.testing.assert_raises_regex(ValueError, "do not share all contacts"):
        _fit_accepted_display(
            np.zeros((2, 2)), ["ICL1", "SCL6"],
            np.zeros((2, 2)), ["ICL1", "SCL7"])
