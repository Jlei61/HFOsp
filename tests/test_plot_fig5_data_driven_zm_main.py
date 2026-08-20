import inspect

import numpy as np

from scripts.paper_figures.plot_fig5_data_driven_zm_main import (
    _require_sustained_runaway,
    _runaway_ema,
    _plot_response,
)


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


def test_response_panel_cannot_accept_slow_state_or_substrate_overlays():
    parameters = set(inspect.signature(_plot_response).parameters)
    assert "positions" not in parameters
    assert "disinhibition" not in parameters
    assert "adaptation" not in parameters


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
                "contact_frequency_increased": True,
                "population_frequency_increased": True,
                "population_rate_increased": True,
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
                "contact_frequency_increased": True,
                "population_frequency_increased": False,
                "population_rate_increased": True,
            },
        },
        "full_field_recruitment": {
            "fraction_windows_majority_E_active": 0.95,
            "fraction_windows_majority_sheet_recruited": 0.95,
        },
        "population_rate_frequency": {"spectral_centroid_shift_hz": 1.0},
    }
    with np.testing.assert_raises_regex(RuntimeError, "population_frequency"):
        _require_sustained_runaway(
            {"runaway_morphology": morphology},
            allow_exploratory_workpoint=True,
        )


def test_figure_override_uses_contact_frequency_with_population_shift_floor():
    morphology = {
        "classification": {
            "all_checks_pass": False,
            "checks": {
                "majority_E_active_for_95pct_windows": False,
                "majority_sheet_recruited_for_95pct_windows": False,
                "contact_frequency_increased": True,
                "population_frequency_increased": False,
                "population_rate_increased": True,
            },
        },
        "full_field_recruitment": {
            "fraction_windows_majority_E_active": 0.92,
            "fraction_windows_majority_sheet_recruited": 0.92,
        },
        "population_rate_frequency": {"spectral_centroid_shift_hz": 5.2},
    }
    assert _require_sustained_runaway(
        {"runaway_morphology": morphology},
        allow_exploratory_workpoint=True,
    ) is morphology
