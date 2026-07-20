"""Tests for the Stage 0A canonical topology oracle."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from src.topic4_spatial_slowfast_topology import (
    NormalFormParameters,
    OrbitClassifierThresholds,
    SlowLoopParameters,
    analyze_closed_slow_loop,
    bracket_contains,
    classify_orbit,
    detect_entry_exit_boundaries,
    run_state_fork_map,
    simulate_closed_slow_loop,
    simulate_frozen,
)


def test_analytic_topology_has_distinct_entry_and_exit_boundaries():
    params = NormalFormParameters(beta=1.0, omega_hz=1.0)
    assert params.entry_mu == 0.0
    assert params.exit_mu == -0.25
    cycles = params.cycle_radii(-0.10)
    assert 0 < cycles["unstable"] < cycles["stable"]


def test_frozen_classifier_recovers_low_and_finite_cycle_state_fork():
    params = NormalFormParameters()
    time_s, states = simulate_frozen(
        [-0.10, -0.10],
        [[0.02, 0.0], [1.40, 0.0]],
        params=params,
        dt_s=0.01,
        duration_s=120.0,
        save_stride=5,
    )
    low = classify_orbit(time_s, states[:, 0])
    high = classify_orbit(time_s, states[:, 1])
    assert low["classification"] == "low_fixed_point"
    assert high["classification"] == "finite_limit_cycle"
    assert high["ceiling_occupancy"] == 0.0
    np.testing.assert_allclose(high["dominant_frequency_hz"], params.omega_hz, atol=0.03)


def test_classifier_rejects_ceiling_and_long_transient():
    thresholds = OrbitClassifierThresholds()
    time_s = np.arange(0.0, 80.0 + 0.05, 0.05)
    phase = 2.0 * np.pi * time_s
    ceiling = np.column_stack(
        (
            thresholds.ceiling_radius * np.cos(phase),
            thresholds.ceiling_radius * np.sin(phase),
        )
    )
    radius = np.linspace(1.0, 0.30, time_s.size)
    transient = np.column_stack((radius * np.cos(phase), radius * np.sin(phase)))
    assert classify_orbit(time_s, ceiling, thresholds)["classification"] == "saturation_or_ceiling"
    assert (
        classify_orbit(time_s, transient, thresholds)["classification"]
        == "indeterminate_long_transient"
    )


def test_initial_condition_map_recovers_numeric_entry_exit_brackets():
    mu_values = [-0.30, -0.275, -0.25, -0.10, 0.0, 0.025, 0.10]
    initial_radii = [0.02, 0.55, 1.40]
    rows, _ = run_state_fork_map(
        mu_values,
        initial_radii,
        duration_s=200.0,
        dt_s=0.01,
        save_stride=5,
    )
    boundaries = detect_entry_exit_boundaries(
        rows, low_initial_radius=0.02, high_initial_radius=1.40
    )
    assert bracket_contains(0.0, boundaries["entry_bracket_mu"])
    assert bracket_contains(-0.25, boundaries["exit_bracket_mu"])


def test_closed_slow_loop_enters_exits_returns_and_recovers_retriggerability():
    traces = simulate_closed_slow_loop(
        normal=NormalFormParameters(),
        slow=SlowLoopParameters(),
        dt_s=0.02,
        duration_s=135.0,
        save_stride=5,
    )
    result = analyze_closed_slow_loop(
        traces,
        normal=NormalFormParameters(),
        slow=SlowLoopParameters(),
        retrigger_duration_s=100.0,
    )
    assert result["pass"] is True
    assert result["manual_reset_used"] is False
    assert all(result["gates"].values())
    assert result["entry_crossing_s"] < result["episodes"][0]["onset_s"]
    assert result["exit_crossing_s"] < result["episodes"][0]["offset_s"]


def test_runner_requires_explicit_confirmation():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "run_topic4_spatial_slowfast_topology.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=root)
    assert proc.returncode == 2
    assert "pass --confirm-run" in proc.stderr
