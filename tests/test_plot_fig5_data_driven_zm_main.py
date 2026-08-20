import numpy as np

from scripts.paper_figures.plot_fig5_data_driven_zm_main import (
    _accepted_display_xy,
    _fit_accepted_display,
    _require_sustained_runaway,
    _runaway_ema,
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


def test_runaway_ema_matches_engine_recurrence():
    rate = np.asarray([0.0, 100.0, 50.0, 200.0])
    dt_ms = 0.1
    alpha = 1.0 - np.exp(-dt_ms / 20.0)
    expected = []
    state = 0.0
    for value in rate:
        state += alpha * (value - state)
        expected.append(state)
    assert np.array_equal(_runaway_ema(rate, dt_ms), np.asarray(expected))


def test_figure_rejects_rate_crossing_without_morphology_audit():
    with np.testing.assert_raises_regex(RuntimeError, "morphology audit"):
        _require_sustained_runaway({"model_ictal_onset_ms": 4115.0})


def test_figure_rejects_failed_morphology():
    payload = {"runaway_morphology": {"classification": {
        "all_checks_pass": False,
        "checks": {"global_recruitment": False, "frequency": True},
    }}}
    with np.testing.assert_raises_regex(RuntimeError, "global_recruitment"):
        _require_sustained_runaway(payload)


def test_figure_accepts_only_complete_morphology_pass():
    morphology = {"classification": {
        "all_checks_pass": True,
        "status": "SUSTAINED_HIGH_INTENSITY_OSCILLATION",
        "checks": {"global_recruitment": True, "frequency": True},
    }}
    assert _require_sustained_runaway(
        {"runaway_morphology": morphology}) is morphology


def test_figure_explicitly_accepts_author_selected_brief_dropout_workpoint():
    morphology = {
        "classification": {
            "all_checks_pass": False,
            "checks": {
                "majority_E_active_for_95pct_windows": False,
                "majority_sheet_recruited_for_95pct_windows": False,
                "population_frequency_increased": True,
            },
        },
        "full_field_recruitment": {
            "fraction_windows_majority_E_active": 0.92,
            "fraction_windows_majority_sheet_recruited": 0.92,
        },
    }
    payload = {"runaway_morphology": morphology}
    with np.testing.assert_raises_regex(RuntimeError, "majority_E_active"):
        _require_sustained_runaway(payload)
    assert _require_sustained_runaway(
        payload, allow_exploratory_workpoint=True) is morphology


def test_figure_exploratory_override_does_not_accept_frequency_failure():
    morphology = {
        "classification": {
            "all_checks_pass": False,
            "checks": {
                "majority_E_active_for_95pct_windows": False,
                "population_frequency_increased": False,
            },
        },
        "full_field_recruitment": {
            "fraction_windows_majority_E_active": 0.95,
            "fraction_windows_majority_sheet_recruited": 0.95,
        },
    }
    with np.testing.assert_raises_regex(RuntimeError, "population_frequency"):
        _require_sustained_runaway(
            {"runaway_morphology": morphology},
            allow_exploratory_workpoint=True,
        )
