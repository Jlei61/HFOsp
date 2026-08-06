"""Regression tests for the locked Stage-0E topology audit."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.run_topic4_spatial_slowfast_stage0e as runner
import src.topic4_spatial_slowfast_stage0e as stage0e
from src.topic4_spatial_slowfast_stage0c import PoolParameters


ROOT = Path(__file__).resolve().parents[1]


def _locked_config() -> dict:
    return yaml.safe_load(
        (ROOT / "config/topic4_spatial_slowfast_stage0e.yaml").read_text(encoding="utf-8")
    )


def test_locked_config_validates_and_rejects_point_scan() -> None:
    cfg = _locked_config()
    runner._validate_config(cfg)
    drifted = deepcopy(cfg)
    drifted["points"].append({"z": 0.85, "alpha_G": 17.0})
    with pytest.raises(ValueError, match="fixed points"):
        runner._validate_config(drifted)


def test_section_validation_is_exact_and_directed() -> None:
    stage0e.SectionDefinition().validate()
    with pytest.raises(ValueError, match="locked upward"):
        stage0e.SectionDefinition(direction="downward").validate()
    with pytest.raises(ValueError, match="level drifted"):
        stage0e.SectionDefinition(level=0.16).validate()


def test_upward_crossing_interpolates_time_and_all_state_coordinates() -> None:
    before = np.linspace(0.01, 0.09, 9)
    after = before + 0.18
    before[8] = 0.14
    after[8] = 0.17
    time_ms, crossing = stage0e.interpolate_upward_crossing(
        before, after, 10.0, 0.3, stage0e.SectionDefinition()
    )
    assert time_ms == pytest.approx(10.1)
    assert crossing[8] == pytest.approx(0.15)
    np.testing.assert_allclose(crossing[:8], before[:8] + (after[:8] - before[:8]) / 3.0)


def test_phase_resampling_removes_period_and_sampling_duration() -> None:
    phase_a = np.linspace(0.0, 1.0, 501)
    phase_b = np.linspace(0.0, 1.0, 911)

    def waveform(phase: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                0.2 + (index + 1) * 0.01 * np.sin(2.0 * np.pi * phase + index / 7.0)
                for index in range(9)
            ]
        )

    left = stage0e.phase_resample(600.0 * phase_a, waveform(phase_a), 0.0, 600.0, 256)
    right = stage0e.phase_resample(750.0 * phase_b, waveform(phase_b), 0.0, 750.0, 256)
    assert stage0e.aligned_waveform_residual(left, right, np.ones(9)) < 2e-6


def test_scaled_distance_uses_all_nine_coordinates() -> None:
    left = np.zeros(9)
    right = np.zeros(9)
    right[8] = 0.02
    scales = np.ones(9)
    scales[8] = 0.1
    assert stage0e.scaled_inf_distance(left, right, scales) == pytest.approx(0.2)


def test_central_jacobian_normalization_and_column_order_are_exact() -> None:
    rng = np.random.default_rng(7)
    expected = rng.normal(scale=0.15, size=(8, 8))
    scales = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    epsilon = 3e-4
    baseline = np.linspace(0.02, 0.18, 9)
    baseline[8] = 0.15
    returned = []
    for column in range(8):
        for sign in (1.0, -1.0):
            state = baseline.copy()
            state[:8] += sign * epsilon * scales[:8] * expected[:, column]
            returned.append(state)
    observed = stage0e.central_jacobian_from_returns(
        np.asarray(returned), scales, epsilon
    )
    np.testing.assert_allclose(observed, expected, atol=2e-13, rtol=2e-12)


def _floquet_row(matrix: np.ndarray, rho: float, epsilon: float = 1e-3) -> dict:
    return {
        "valid": True,
        "jacobian": np.asarray(matrix, dtype=float),
        "spectral_radius": float(rho),
        "epsilon_relative": float(epsilon),
    }


def test_epsilon_ladder_requires_jacobian_platform_not_only_rho() -> None:
    cfg = _locked_config()["floquet"]
    identity = np.eye(8) * 0.7
    stable = [
        _floquet_row(identity * factor, 0.7 * factor, epsilon)
        for factor, epsilon in zip((1.0, 1.001, 1.0012), (1e-3, 3e-4, 1e-4))
    ]
    assert stage0e.jacobian_ladder_summary(stable, cfg)["pass"]
    unstable_matrix = identity.copy()
    unstable_matrix[0, 1] = 0.8
    erratic = [
        _floquet_row(matrix, 0.7, epsilon)
        for matrix, epsilon in zip(
            (identity, identity, unstable_matrix), (1e-3, 3e-4, 1e-4)
        )
    ]
    assert not stage0e.jacobian_ladder_summary(erratic, cfg)["pass"]
    low_norm = [
        _floquet_row(np.eye(8) * factor, factor, epsilon)
        for factor, epsilon in zip((0.01, 0.019, 0.019), (1e-3, 3e-4, 1e-4))
    ]
    assert not stage0e.jacobian_ladder_summary(low_norm, cfg)["pass"]


def test_floquet_margin_is_fail_closed_near_unit_circle() -> None:
    cfg = _locked_config()["floquet"]
    ladders = {"pass": True}
    epsilons = (1e-3, 3e-4, 1e-4)
    base = [
        {"spectral_radius": value, "epsilon_relative": epsilon}
        for value, epsilon in zip((0.80, 0.81, 0.805), epsilons)
    ]
    half = [
        {"spectral_radius": value, "epsilon_relative": epsilon}
        for value, epsilon in zip((0.79, 0.80, 0.795), epsilons)
    ]
    assert stage0e.floquet_stability_summary(base, half, ladders, ladders, cfg)["pass"]
    near = [
        {"spectral_radius": value, "epsilon_relative": epsilon}
        for value, epsilon in zip((0.97, 0.98, 0.99), epsilons)
    ]
    summary = stage0e.floquet_stability_summary(near, near, ladders, ladders, cfg)
    assert not summary["pass"]
    assert summary["all_nontrivial_multipliers_inside_unit_circle"]
    assert not summary["robust_margin_pass"]


def test_floquet_report_preserves_all_eight_multipliers_including_one() -> None:
    multipliers = np.asarray([1.0, 0.8, 0.7j, -0.3, 0.2, 0.1, -0.1j, 0.0])
    report = stage0e.floquet_row_report(
        {
            "valid": True,
            "epsilon_relative": 1e-3,
            "jacobian": np.eye(8),
            "multipliers": multipliers,
            "spectral_radius": 1.0,
        }
    )
    assert len(report["multipliers"]) == 8
    assert report["multipliers"][0]["modulus"] == pytest.approx(1.0)


def test_return_battery_has_four_anchors_and_two_noncollinear_families_per_phase() -> None:
    cfg = _locked_config()["return_battery"]
    phase_states = np.tile(np.linspace(0.02, 0.18, 9), (4, 1))
    phase_states[:, 8] = np.asarray([0.15, 0.10, 0.05, 0.02])
    metadata, states = stage0e.build_return_battery(
        phase_states,
        phases=cfg["phases"],
        perturbation_fraction=cfg["perturbation_fraction"],
        fast_directions=cfg["fast_directions"],
        pool_directions=cfg["pool_directions"],
    )
    assert states.shape == (20, 9)
    assert sum(row["family"] == "anchor" for row in metadata) == 4
    assert sum(row["family"] == "fast" for row in metadata) == 8
    assert sum(row["family"] == "pool" for row in metadata) == 8
    drifted_fast = deepcopy(cfg["fast_directions"])
    drifted_fast[0][0] = -1
    with pytest.raises(ValueError, match="directions drifted"):
        stage0e.build_return_battery(
            phase_states,
            phases=cfg["phases"],
            perturbation_fraction=cfg["perturbation_fraction"],
            fast_directions=drifted_fast,
            pool_directions=cfg["pool_directions"],
        )


def test_return_battery_summary_requires_both_families() -> None:
    cfg = _locked_config()["return_battery"]
    fixed = np.linspace(0.02, 0.18, 9)
    fixed[8] = 0.15
    phases = np.tile(fixed, (4, 1))
    metadata, _ = stage0e.build_return_battery(
        phases,
        phases=cfg["phases"],
        perturbation_fraction=cfg["perturbation_fraction"],
        fast_directions=cfg["fast_directions"],
        pool_directions=cfg["pool_directions"],
    )
    n_returns, n_histories = 8, len(metadata)
    returned = np.tile(fixed, (n_returns, n_histories, 1))
    for history_index, row in enumerate(metadata):
        start = 2e-4 if row["family"] == "anchor" else 0.03
        stop = 1e-5 if row["family"] == "anchor" else 0.003
        returned[:, history_index, 0] += np.geomspace(start, stop, n_returns)
    audit = stage0e._empty_audit(n_histories)
    audit["n_euler_states"][:] = 100
    audit["peak_rE_hz"][:] = 30.0
    audit["moment_min"][:] = 0.0
    audit["moment_max"][:] = 1.0
    batch = {
        "return_state": returned,
        "return_time_ms": np.tile(np.arange(1, 9)[:, None] * 600.0, (1, n_histories)),
        "transversality_per_ms": np.ones((n_returns, n_histories)) * 0.005,
        "valid": np.ones(n_histories, dtype=bool),
        "audit": audit,
        "crossing_audit": [
            [{"clean": True} for _ in range(n_histories)] for _ in range(n_returns)
        ],
    }
    rows, summary = stage0e.summarize_return_battery(
        metadata, batch, fixed, np.ones(9), cfg
    )
    assert len(rows) == 20
    assert summary["pass"]
    broken = deepcopy(batch)
    broken["valid"] = broken["valid"].copy()
    pool_index = next(index for index, row in enumerate(metadata) if row["family"] == "pool")
    broken["valid"][pool_index] = False
    _, broken_summary = stage0e.summarize_return_battery(
        metadata, broken, fixed, np.ones(9), cfg
    )
    assert not broken_summary["families"]["pool"]["pass"]
    assert not broken_summary["pass"]


def test_dt_cycle_consistency_is_phase_aligned() -> None:
    cfg = _locked_config()["dt_half"]
    phase = np.linspace(0.0, 2.0 * np.pi, 256)
    waveform = np.column_stack([0.2 + 0.01 * np.sin(phase + index) for index in range(9)])
    base = {"valid": True, "period_ms": np.asarray([600.0, 600.1]), "waveform_first": waveform}
    half = {"valid": True, "period_ms": np.asarray([600.2, 600.0]), "waveform_first": waveform.copy()}
    summary = stage0e.dt_cycle_consistency_summary(base, half, np.ones(9), cfg)
    assert summary["pass"]
    half["period_ms"] = np.asarray([610.0, 610.0])
    assert not stage0e.dt_cycle_consistency_summary(base, half, np.ones(9), cfg)["pass"]


class _AlwaysSupportedTransfer:
    @staticmethod
    def support_mask(mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        return np.ones(np.broadcast(mu, sigma).shape, dtype=bool)


def _harmonic_rhs(state, prepared, transfer, *, mechanism, clamp_s, subtractive_beta_mv):
    del prepared, transfer, mechanism, clamp_s, subtractive_beta_mv
    omega = 2.0 * np.pi / 600.0
    rhs = np.zeros_like(state)
    rhs[:, 8] = omega * (state[:, 7] - 0.5)
    rhs[:, 7] = -omega * (state[:, 8] - 0.15)
    zeros = np.zeros(state.shape[0])
    ones = np.ones(state.shape[0])
    return rhs, (zeros, ones, zeros, ones, state[:, 8])


def test_interpolated_crossing_is_audited_before_acceptance(monkeypatch) -> None:
    monkeypatch.setattr(stage0e, "_rhs_and_moments", _harmonic_rhs)
    crossing = np.full(9, 0.02)
    crossing[0] = 0.100
    crossing[7] = 0.55
    crossing[8] = 0.15
    audit = stage0e.audit_crossing_state(
        crossing,
        PoolParameters(0.85, 16.0),
        _AlwaysSupportedTransfer(),
        stage0e.SectionDefinition(),
    )
    assert not audit["clean"]
    assert not audit["below_100hz"]


def test_multi_return_batch_equals_repeated_one_return_map(monkeypatch) -> None:
    monkeypatch.setattr(stage0e, "_rhs_and_moments", _harmonic_rhs)
    initial = np.full(9, 0.02)
    initial[0] = 0.01
    initial[7] = 0.55
    initial[8] = 0.15
    params = PoolParameters(0.85, 16.0)
    transfer = _AlwaysSupportedTransfer()
    section = stage0e.SectionDefinition()
    two = stage0e.integrate_to_returns_batch(
        initial[None, :], [params], transfer, dt_ms=1.0, n_returns=2, section=section
    )
    first = stage0e.integrate_to_returns_batch(
        initial[None, :], [params], transfer, dt_ms=1.0, n_returns=1, section=section
    )
    second = stage0e.integrate_to_returns_batch(
        first["return_state"][0], [params], transfer, dt_ms=1.0, n_returns=1, section=section
    )
    assert two["valid"][0] and first["valid"][0] and second["valid"][0]
    np.testing.assert_allclose(two["return_state"][0, 0], first["return_state"][0, 0])
    np.testing.assert_allclose(two["return_state"][1, 0], second["return_state"][0, 0])
    assert two["return_time_ms"][1, 0] == pytest.approx(
        first["return_time_ms"][0, 0] + second["return_time_ms"][0, 0]
    )


def test_overall_verdict_never_opens_stage1() -> None:
    points = [
        {"alpha_G": 15.0, "outcome": "no_periodic_orbit"},
        {"alpha_G": 16.0, "outcome": "stable_periodic_orbit"},
    ]
    assert runner._overall_verdict(points, True) == "STAGE0E_STABLE_PERIODIC_ORBIT_ALPHA16_ONLY"
    assert runner._overall_verdict(points, False) == "STAGE0E_ENGINEERING_OR_PROVENANCE_FAIL"


def test_floquet_failure_does_not_short_circuit_sibling_battery_or_physical_report(
    monkeypatch,
) -> None:
    cfg = _locked_config()
    scout = {
        "fatal": False,
        "audit": {"clean": True},
        "crossing_time_ms": np.arange(12, dtype=float) * 600.0,
        "crossing_state": np.tile(np.linspace(0.01, 0.15, 9), (12, 1)),
        "crossing_transversality_per_ms": np.ones(12) * 0.005,
        "period_ms": np.ones(11) * 600.0,
    }
    fixed = np.linspace(0.01, 0.15, 9)
    shooting = {"converged": True, "fixed_state": fixed}
    cycle = {"valid": True, "trace": {"audit": {"clean": True}}}
    called = {"physical": 0, "battery": 0}

    monkeypatch.setattr(stage0e, "integrate_full_trace", lambda *args, **kwargs: scout)
    monkeypatch.setattr(stage0e, "scout_scales", lambda *args, **kwargs: (np.ones(9), 0.0))
    monkeypatch.setattr(stage0e, "poincare_fixed_point_shooting", lambda *args, **kwargs: shooting)
    monkeypatch.setattr(stage0e, "shooting_cycle_validation", lambda *args, **kwargs: cycle)
    monkeypatch.setattr(stage0e, "shooting_gate_summary", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(stage0e, "dt_cycle_consistency_summary", lambda *args, **kwargs: {"pass": True})

    def physical(*args, **kwargs):
        called["physical"] += 1
        return {"valid": True, "above_100hz_occupancy": 0.0}

    monkeypatch.setattr(stage0e, "orbit_physical_summary", physical)
    monkeypatch.setattr(
        stage0e,
        "finite_difference_poincare_jacobian",
        lambda *args, epsilon_relative, **kwargs: {
            "valid": True,
            "epsilon_relative": epsilon_relative,
            "jacobian": np.eye(8),
            "multipliers": np.ones(8) * 0.5,
            "spectral_radius": 0.5,
        },
    )
    monkeypatch.setattr(stage0e, "jacobian_ladder_summary", lambda *args, **kwargs: {"pass": False})
    monkeypatch.setattr(
        stage0e,
        "floquet_stability_summary",
        lambda *args, **kwargs: {"pass": False, "reason": "synthetic_epsilon_failure"},
    )
    monkeypatch.setattr(stage0e, "interpolate_cycle_phases", lambda *args, **kwargs: np.tile(fixed, (4, 1)))
    monkeypatch.setattr(
        stage0e,
        "build_return_battery",
        lambda *args, **kwargs: ([{}] * 20, np.tile(fixed, (20, 1))),
    )

    def battery(*args, **kwargs):
        called["battery"] += 1
        return {"synthetic": True}

    monkeypatch.setattr(stage0e, "integrate_to_returns_batch", battery)
    monkeypatch.setattr(
        stage0e,
        "summarize_return_battery",
        lambda *args, **kwargs: ([], {"pass": True}),
    )
    result, _ = stage0e.audit_parameter_point(
        fixed, PoolParameters(0.85, 16.0), _AlwaysSupportedTransfer(), cfg
    )
    assert called == {"physical": 2, "battery": 1}
    assert result["return_battery"]["pass"]
    assert result["physical_acceptance_pass"]
    assert result["outcome"] == "periodic_orbit_numerically_unresolved"
    assert result["failed_gates"] == ["floquet_epsilon_dt_or_margin"]
