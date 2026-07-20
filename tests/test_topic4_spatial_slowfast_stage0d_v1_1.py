from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from scripts.run_topic4_spatial_slowfast_stage0d_v1_1 import DEFAULT_CONFIG, _validate_config
from src.topic4_spatial_slowfast_stage0c_transfer import temporal_refinement_status
from src.topic4_spatial_slowfast_stage0d_v1_1 import (
    FIGURE_B_EMPTY_TEXT,
    centre_final_survivor_indices,
    compare_fork_outcomes,
    strict_temporal_amplitude_status,
)


def _candidate(*, mean: float = 6.0, frequency: float = 2.0, amplitude: float = 80.0) -> dict:
    return {
        "finite": True,
        "classification": "bounded_oscillatory_candidate",
        "tail_mean_hz": mean,
        "dominant_frequency_hz": frequency,
        "tail_peak_hz": amplitude + 0.4,
        "tail_trough_hz": 0.4,
        "support_violation_step_count": 0,
        "pool_bound_step_count": 0,
        "rate_bound_step_count": 0,
        "synapse_bound_step_count": 0,
        "negative_rate_step_count": 0,
        "over_100hz_tail_step_count": 0,
        "e_refractory_tail_occupancy_stepwise": 0.0,
        "i_refractory_tail_occupancy_stepwise": 0.0,
    }


def test_v1_1_config_preserves_every_v1_scientific_section() -> None:
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    old = _validate_config(cfg)
    for section in (
        "centre",
        "parameter_grid",
        "phase_source",
        "battery",
        "screen",
        "confirm",
        "dt_half",
        "classifier",
        "acceptance",
        "resource_contract",
        "scope",
    ):
        assert cfg[section] == old[section]
    assert cfg["result_root"] != old["result_root"]


def test_repair_rejects_pair_that_only_passes_legacy_frequency_gate() -> None:
    confirm = _candidate(frequency=2.0)
    refined = _candidate(frequency=2.30)
    assert temporal_refinement_status(confirm, refined, exact_error_pass=True) == "candidate_survives"
    status, audit = strict_temporal_amplitude_status(confirm, refined, exact_error_pass=True)
    assert status == "numerical_unresolved"
    assert audit["legacy_temporal_gate_candidate"]
    assert audit["frequency_difference_hz"] == pytest.approx(0.30)
    assert audit["frequency_limit_hz"] == pytest.approx(0.25)
    assert not audit["frequency_pass"]


def test_repair_accepts_written_frequency_boundary() -> None:
    status, audit = strict_temporal_amplitude_status(
        _candidate(frequency=2.0),
        _candidate(frequency=2.25),
        exact_error_pass=True,
    )
    assert status == "candidate_survives"
    assert audit["frequency_difference_hz"] == pytest.approx(0.25)
    assert audit["frequency_pass"]


def test_rate_and_amplitude_gates_remain_one_and_five_hz_floors() -> None:
    rate_status, rate_audit = strict_temporal_amplitude_status(
        _candidate(mean=6.0),
        _candidate(mean=7.1),
        exact_error_pass=True,
    )
    assert rate_status == "numerical_unresolved"
    assert rate_audit["rate_limit_hz"] == pytest.approx(1.0)
    amplitude_status, amplitude_audit = strict_temporal_amplitude_status(
        _candidate(amplitude=80.0),
        _candidate(amplitude=86.0),
        exact_error_pass=True,
    )
    assert amplitude_status == "candidate_survives"
    assert amplitude_audit["amplitude_limit_hz"] == pytest.approx(8.3)
    # The 10-percent arm dominates at this amplitude, as specified.
    assert amplitude_audit["amplitude_difference_hz"] == pytest.approx(6.0)
    assert amplitude_audit["amplitude_pass"]
    low_amplitude_status, low_amplitude_audit = strict_temporal_amplitude_status(
        _candidate(amplitude=20.0),
        _candidate(amplitude=26.0),
        exact_error_pass=True,
    )
    assert low_amplitude_status == "numerical_unresolved"
    assert low_amplitude_audit["amplitude_limit_hz"] == pytest.approx(5.0)
    assert not low_amplitude_audit["amplitude_pass"]


def _battery_rows() -> list[dict]:
    rows = []
    histories = ("phase_anchor", "fast_plus", "fast_minus", "pool_plus", "pool_minus")
    phases = ("phase_000", "phase_025", "phase_050", "phase_075")
    for z in (0.84, 0.85, 0.86):
        for alpha in (15.0, 16.0, 17.0):
            for phase in phases:
                for history in histories:
                    rows.append(
                        {
                            "z": z,
                            "alpha_G": alpha,
                            "phase_id": phase,
                            "history": history,
                            "final_status": "numerical_unresolved",
                            "dt_half_classification": None,
                        }
                    )
    assert len(rows) == 180
    return rows


def test_comparison_reports_exact_fork_status_changes() -> None:
    old = _battery_rows()
    new = deepcopy(old)
    new[17]["final_status"] = "candidate_survives"
    comparison = compare_fork_outcomes(old, new)
    assert comparison["n_aligned_histories"] == 180
    assert comparison["n_fork_status_changes"] == 1
    assert comparison["any_fork_status_changed"]


def test_figure_b_selection_is_final_centre_only() -> None:
    rows = _battery_rows()
    assert centre_final_survivor_indices(rows) == []
    centre = next(
        index
        for index, row in enumerate(rows)
        if row["z"] == 0.85 and row["alpha_G"] == 16.0
    )
    rows[centre]["final_status"] = "candidate_survives"
    rows[centre]["dt_half_classification"] = "bounded_oscillatory_candidate"
    off_centre = next(
        index
        for index, row in enumerate(rows)
        if row["z"] == 0.85 and row["alpha_G"] == 15.0
    )
    rows[off_centre]["final_status"] = "candidate_survives"
    rows[off_centre]["dt_half_classification"] = "bounded_oscillatory_candidate"
    assert centre_final_survivor_indices(rows) == [centre]
    assert FIGURE_B_EMPTY_TEXT == "none passed locked gate"
