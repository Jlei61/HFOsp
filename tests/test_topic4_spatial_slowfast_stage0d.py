from __future__ import annotations

import numpy as np

from scripts.run_topic4_spatial_slowfast_stage0d import DEFAULT_CONFIG, _validate_config
from src.topic4_spatial_slowfast_stage0c import PoolParameters, equilibrium_state
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    TransferResolution,
    TransferSupport,
    simulate_extended_forks,
)
from src.topic4_spatial_slowfast_stage0d import (
    MANHATTAN_NEIGHBOURS,
    build_local_battery,
    integrate_full_state_trace,
    point_metric_compatibility,
    select_phase_states,
    summarize_parameter_point,
    temporal_amplitude_status,
)


def test_locked_config_and_scope() -> None:
    import yaml

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    _validate_config(cfg)
    assert not any(cfg["scope"].values())
    assert cfg["resource_contract"] == {"blas_threads": 1, "max_memory_gib": 4.0}


def test_phase_selector_uses_last_complete_cycle() -> None:
    time = np.arange(0.0, 12005.0, 5.0)
    state = np.zeros((time.size, 9), dtype=float)
    state[:, 0] = 0.030 + 0.025 * np.cos(2.0 * np.pi * time / 600.0)
    state[:, 1:] = 0.01
    phases, rows = select_phase_states(
        time,
        state,
        tail_start_ms=7200.0,
        peak_height_hz=20.0,
        peak_prominence_hz=10.0,
        peak_min_distance_ms=300.0,
    )
    assert phases.shape == (4, 9)
    assert [row["phase_id"] for row in rows] == ["phase_000", "phase_025", "phase_050", "phase_075"]
    assert all(row["cycle_period_ms"] == 600.0 for row in rows)


def test_locked_battery_is_180_and_local_without_clipping() -> None:
    phases = np.vstack(
        [equilibrium_state((0.010 + index * 0.002, 0.020 + index * 0.003)) for index in range(4)]
    )
    points = tuple((z, alpha) for z in (0.84, 0.85, 0.86) for alpha in (15.0, 16.0, 17.0))
    metadata, states, params = build_local_battery(phases, points)
    assert len(metadata) == states.shape[0] == len(params) == 180
    centre_phase0 = [
        (row, states[index])
        for index, row in enumerate(metadata)
        if row["z"] == 0.85 and row["alpha_G"] == 16.0 and row["phase_id"] == "phase_000"
    ]
    lookup = {row["history"]: state for row, state in centre_phase0}
    assert np.allclose(lookup["fast_plus"][:6], phases[0, :6] * 1.03)
    assert np.allclose(lookup["fast_plus"][6:], phases[0, 6:])
    assert np.allclose(lookup["pool_minus"][:6], phases[0, :6])
    assert np.allclose(lookup["pool_minus"][6:], phases[0, 6:] * 0.97)
    assert np.all(states >= 0.0)


def test_full_state_phase_tracer_matches_authoritative_integrator() -> None:
    transfer = ExtendedSiegertTransfer.build(
        TransferSupport(-80.0, -20.0, 40.0, 2.0, 10.0),
        TransferResolution("test", 2.0, 1.0, 8),
    )
    state = equilibrium_state((0.001, 0.003))
    params = PoolParameters(0.9, 2.0)
    full = integrate_full_state_trace(state, params, transfer, dt_ms=0.25, duration_ms=20.0, save_stride=4)
    standard = simulate_extended_forks(
        state[None, :],
        [params],
        transfer,
        dt_ms=0.25,
        duration_ms=20.0,
        save_stride=4,
        audit_tail_fraction=0.4,
    )
    for key, column in (("rE_khz", 0), ("rI_khz", 1), ("rE_fast_khz", 6), ("mu_G", 7), ("S_G", 8)):
        assert np.allclose(full["state"][:, column], standard[key][:, 0], rtol=0.0, atol=1e-7)


def _candidate_row(amplitude: float = 80.0, frequency: float = 1.65, mean: float = 6.0) -> dict:
    return {
        "finite": True,
        "classification": "bounded_oscillatory_candidate",
        "tail_mean_hz": mean,
        "tail_peak_hz": amplitude + 0.4,
        "tail_trough_hz": 0.4,
        "dominant_frequency_hz": frequency,
        "support_violation_step_count": 0,
        "pool_bound_step_count": 0,
        "rate_bound_step_count": 0,
        "synapse_bound_step_count": 0,
        "negative_rate_step_count": 0,
        "over_100hz_tail_step_count": 0,
        "e_refractory_tail_occupancy_stepwise": 0.0,
        "i_refractory_tail_occupancy_stepwise": 0.0,
    }


def test_dt_half_gate_includes_amplitude() -> None:
    confirm = _candidate_row(amplitude=80.0)
    matched = _candidate_row(amplitude=82.0, frequency=1.66, mean=6.1)
    assert temporal_amplitude_status(confirm, matched, exact_error_pass=True) == "candidate_survives"
    wrong_amplitude = _candidate_row(amplitude=50.0, frequency=1.66, mean=6.1)
    assert temporal_amplitude_status(confirm, wrong_amplitude, exact_error_pass=True) == "numerical_unresolved"


def _final_row(family: str, phase: str, *, mean: float = 6.0, frequency: float = 1.65, amplitude: float = 80.0) -> dict:
    return {
        "z": 0.85,
        "alpha_G": 16.0,
        "final_status": "candidate_survives",
        "off_orbit": True,
        "perturbation_family": family,
        "phase_id": phase,
        "dt_half_tail_mean_hz": mean,
        "dt_half_frequency_hz": frequency,
        "dt_half_amplitude_hz": amplitude,
    }


def test_open_basin_requires_two_families_and_two_phases() -> None:
    rows = [_final_row("fast", "phase_000"), _final_row("pool", "phase_025")]
    point = summarize_parameter_point(rows, 0.85, 16.0)
    assert point["open_local_basin_support"]
    one_family = summarize_parameter_point([rows[0], {**rows[1], "perturbation_family": "fast"}], 0.85, 16.0)
    assert not one_family["open_local_basin_support"]
    one_phase = summarize_parameter_point([rows[0], {**rows[1], "phase_id": "phase_000"}], 0.85, 16.0)
    assert not one_phase["open_local_basin_support"]


def test_only_manhattan_points_can_replicate_and_metrics_must_match() -> None:
    assert (0.84, 16.0) in MANHATTAN_NEIGHBOURS
    assert (0.84, 15.0) not in MANHATTAN_NEIGHBOURS
    centre = {"mean_rate_hz": 6.0, "mean_frequency_hz": 1.65, "mean_amplitude_hz": 80.0}
    close = {"mean_rate_hz": 6.2, "mean_frequency_hz": 1.70, "mean_amplitude_hz": 82.0}
    far = {"mean_rate_hz": 9.0, "mean_frequency_hz": 2.5, "mean_amplitude_hz": 60.0}
    assert point_metric_compatibility(centre, close)
    assert not point_metric_compatibility(centre, far)
