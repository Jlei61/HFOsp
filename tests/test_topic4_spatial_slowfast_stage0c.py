"""Contract tests for the independent Stage-0C dynamic-pool screen."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from src.sef_hfo_lif import C_EE, TAU_ME, W_EE
from src.topic4_spatial_slowfast_stage0b import FastParameters, fast_rhs as stage0b_rhs
from src.topic4_spatial_slowfast_stage0c import (
    PoolParameters,
    build_state_forks,
    equilibrium_state,
    find_fixed_points,
    moments_from_state,
    numerical_jacobian,
    pool_rhs,
    recruitment_sensor,
    simulate_forks,
    summarize_stage0c,
)
from scripts.run_topic4_spatial_slowfast_stage0c import _alpha0_parity_audit


def test_recurrent_mean_and_variance_use_D_and_D_squared_only():
    state = equilibrium_state((0.020, 0.040))
    state[8] = 0.4
    base = PoolParameters(z=0.9, alpha_g=0.0)
    active = PoolParameters(z=0.9, alpha_g=4.0)
    mu0, sigma0, mui0, sigmai0, _ = moments_from_state(state, base)
    mu1, sigma1, mui1, sigmai1, _ = moments_from_state(state, active)
    divisor = 1.0 + active.alpha_g * state[8]
    rec_mean = TAU_ME * C_EE * (active.w_ee_mult * W_EE) * state[2]
    rec_var = TAU_ME * C_EE * (active.w_ee_mult * W_EE) ** 2 * state[2]
    np.testing.assert_allclose(mu0 - mu1, rec_mean * (1.0 - 1.0 / divisor), rtol=1e-12)
    np.testing.assert_allclose(sigma0**2 - sigma1**2, rec_var * (1.0 - 1.0 / divisor**2), rtol=1e-12)
    np.testing.assert_allclose([mui1, sigmai1], [mui0, sigmai0], rtol=0.0, atol=0.0)


def test_alpha_zero_reproduces_stage0b_first_six_dimensions_off_manifold():
    state6 = np.asarray([0.080, 0.150, 0.003, 0.008, 0.003, 0.008])
    state9 = np.r_[state6, 0.071, 0.42, 0.73]
    point = PoolParameters(z=0.87, alpha_g=0.0)
    observed = pool_rhs(state9, point)[:6]
    expected = stage0b_rhs(state6, FastParameters(1.1, 0.87, 1.0))
    np.testing.assert_array_equal(observed, expected)


def test_alpha0_runner_parity_helper_smoke():
    roots = find_fixed_points(PoolParameters(z=1.0, alpha_g=0.0))
    rows = [
        {
            "z": 1.0,
            "alpha_G": 0.0,
            "w_ee_mult": 1.1,
            "ratio": 1.0,
            "roots": roots,
        }
    ]
    audit = _alpha0_parity_audit(
        rows,
        [1.0],
        dt_ms=0.25,
        duration_ms=1.0,
        save_stride=2,
        audit_tail_fraction=0.40,
    )
    assert audit["pass"] is True


def test_pool_ode_is_not_numerically_clipped():
    state = equilibrium_state((0.001, 0.003))
    state[7] = 1.2
    state[8] = 1.3
    rhs = pool_rhs(state, PoolParameters(z=1.0, alpha_g=2.0))
    # A clipped implementation would silently project the state to one.  The ODE
    # instead returns a finite restoring derivative while preserving its input.
    assert state[7] == 1.2 and state[8] == 1.3
    assert rhs[7] < 0.0 and rhs[8] < 0.0


def test_fixed_points_use_full_nine_dimensional_stability():
    roots = find_fixed_points(PoolParameters(z=1.0, alpha_g=2.0))
    assert roots
    assert all(root["jacobian_dimension"] == 9 for root in roots)
    low = min(roots, key=lambda root: root["rE_hz"])
    state = equilibrium_state((low["rE_khz"], low["rI_khz"]))
    assert np.linalg.norm(pool_rhs(state, PoolParameters(z=1.0, alpha_g=2.0)), ord=np.inf) < 2e-8


def test_nine_dimensional_jacobian_matches_directional_rhs_difference():
    params = PoolParameters(z=0.9, alpha_g=8.0)
    state = equilibrium_state((0.020, 0.040))
    jac = numerical_jacobian(state, params)
    assert jac.shape == (9, 9)
    direction = np.asarray([0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.25, -0.15, 0.35])
    direction /= np.linalg.norm(direction)
    h = 2e-7
    observed = (pool_rhs(state + h * direction, params) - pool_rhs(state - h * direction, params)) / (2.0 * h)
    np.testing.assert_allclose(jac @ direction, observed, rtol=4e-3, atol=4e-6)


def test_forks_include_stage0b_and_pool_history_off_manifold_probes():
    point = {"z": 1.0, "alpha_G": 2.0, "w_ee_mult": 1.1, "ratio": 1.0, "roots": []}
    metadata, states, _ = build_state_forks([point])
    kinds = [row["initial_kind"] for row in metadata]
    assert kinds.count("on_manifold_probe") == 7
    assert kinds.count("stage0b_off_manifold_probe") == 4
    assert kinds.count("pool_off_manifold_probe") == 4
    assert states.shape == (15, 9)
    assert {row["initial_label"] for row in metadata if row["initial_kind"] == "stage0b_off_manifold_probe"} == {
        "e_synapse_loaded_i_low",
        "i_synapse_loaded_e_low",
        "rate_high_synapse_low",
        "rate_low_synapse_high",
    }


def test_sensor_locked_half_activation_contract():
    assert recruitment_sensor(0.004) == 0.0
    np.testing.assert_allclose(recruitment_sensor(0.005 + 0.015), 0.5, atol=1e-15)


def test_every_euler_step_audits_all_nine_state_dimensions_and_tail_ceiling():
    # Initial states deliberately violate one synaptic bound and the finite-high
    # 100-Hz tail ceiling.  save_stride skips intermediate states, but the stepwise
    # counters must still see every Euler state.
    state = equilibrium_state((0.120, 0.150))
    state[2] = 0.60
    simulation = simulate_forks(
        state[None, :],
        [PoolParameters(z=0.9, alpha_g=2.0)],
        dt_ms=0.25,
        duration_ms=2.0,
        save_stride=8,
    )
    assert int(simulation["audit_n_euler_states"]) == 9
    assert simulation["synapse_bound_step_count"][0] > 0
    assert simulation["over_100hz_tail_step_count"][0] > 0
    assert simulation["stepwise_tail_peak_rE_hz"][0] >= 100.0
    assert "negative_rate_tail_step_count" in simulation
    assert "e_refractory_tail_step_count" in simulation
    assert "i_refractory_tail_step_count" in simulation


def test_summary_clean_no_go_stays_closed():
    roots = [{"z": 1.0, "alpha_G": 0.0, "roots": [{"rE_hz": 1.0, "branch_class": "low_root", "stability": "stable", "lut_clip_at_root": False}]}]
    screen = [
        {"classification": "low_fixed_point", "initial_kind": "on_manifold_probe"},
        {"classification": "saturation_or_over_100hz", "initial_kind": "pool_off_manifold_probe"},
    ]
    summary = summarize_stage0c(roots, screen, [], alpha0_parity={"pass": True})
    assert summary["verdict"] == "CLEAN_NO_GO_DYNAMIC_POOL_LOW_OR_SATURATION_ONLY"
    assert summary["open_phi_or_spatial"] is False


def _candidate(z, alpha, label, rate=20.0):
    return {
        "z": z,
        "alpha_G": alpha,
        "initial_kind": "on_manifold_probe",
        "initial_label": label,
        "classification": "bounded_tonic_candidate",
        "tail_mean_hz": rate,
        "dominant_frequency_hz": 0.0,
    }


def _root(rate, branch, stability="stable"):
    return {
        "rE_hz": rate,
        "branch_class": branch,
        "stability": stability,
        "lut_clip_at_root": False,
    }


def test_go_requires_point_support_adjacency_alpha0_absence_and_z1_low():
    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(1.0, "low_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 2.0, "roots": [_root(21.0, "finite_high_root")]},
        {"z": 1.0, "alpha_G": 1.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 2.0, "roots": [_root(1.0, "low_root")]},
    ]
    confirm = [
        _candidate(0.9, 1.0, "probe_20hz", 20.0),
        _candidate(0.9, 1.0, "root_0_plus", 20.5),
        _candidate(0.9, 2.0, "probe_20hz", 21.0),
        _candidate(0.9, 2.0, "root_0_plus", 21.5),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["verdict"] == "GO_DYNAMIC_POOL_FINITE_FAST_OBJECT"
    assert summary["n_supported_parameter_points"] == 2
    assert summary["n_adjacent_support_pairs"] == 1


def test_single_fork_or_missing_z1_low_cannot_pass():
    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(1.0, "low_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 2.0, "roots": [_root(21.0, "finite_high_root")]},
    ]
    confirm = [
        _candidate(0.9, 1.0, "only_one", 20.0),
        _candidate(0.9, 2.0, "a", 21.0),
        _candidate(0.9, 2.0, "b", 21.5),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["stage0c_pass"] is False
    assert summary["n_adjacent_support_pairs"] == 0


def test_alpha0_same_object_blocks_counterfactual_gate():
    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 2.0, "roots": [_root(21.0, "finite_high_root")]},
        {"z": 1.0, "alpha_G": 1.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 2.0, "roots": [_root(1.0, "low_root")]},
    ]
    confirm = [
        _candidate(0.9, 1.0, "a", 20.0),
        _candidate(0.9, 1.0, "b", 20.5),
        _candidate(0.9, 2.0, "a", 21.0),
        _candidate(0.9, 2.0, "b", 21.5),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["stage0c_pass"] is False
    assert summary["n_adjacent_support_pairs"] == 0


def test_nonadjacent_supported_points_do_not_pass():
    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(1.0, "low_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 4.0, "roots": [_root(21.0, "finite_high_root")]},
        # alpha=2 exists on the axis, making alpha=1 and 4 non-adjacent.
        {"z": 0.8, "alpha_G": 2.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 1.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 4.0, "roots": [_root(1.0, "low_root")]},
    ]
    confirm = [
        _candidate(0.9, 1.0, "a", 20.0),
        _candidate(0.9, 1.0, "b", 20.5),
        _candidate(0.9, 4.0, "a", 21.0),
        _candidate(0.9, 4.0, "b", 21.5),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["stage0c_pass"] is False
    assert summary["n_adjacent_support_pairs"] == 0


def test_single_alpha0_oscillation_blocks_counterfactual():
    def oscillatory(z, alpha, label):
        row = _candidate(z, alpha, label, 20.0)
        row["classification"] = "bounded_oscillatory_candidate"
        row["dominant_frequency_hz"] = 5.0
        return row

    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(1.0, "low_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": []},
        {"z": 0.9, "alpha_G": 2.0, "roots": []},
        {"z": 1.0, "alpha_G": 1.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 2.0, "roots": [_root(1.0, "low_root")]},
    ]
    confirm = [
        oscillatory(0.9, 0.0, "one_control"),
        oscillatory(0.9, 1.0, "a"),
        oscillatory(0.9, 1.0, "b"),
        oscillatory(0.9, 2.0, "a"),
        oscillatory(0.9, 2.0, "b"),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["stage0c_pass"] is False
    assert summary["n_adjacent_support_pairs"] == 0


def test_confirm_row_with_stepwise_bound_violation_is_excluded():
    root_rows = [
        {"z": 0.9, "alpha_G": 0.0, "roots": [_root(1.0, "low_root")]},
        {"z": 0.9, "alpha_G": 1.0, "roots": [_root(20.0, "finite_high_root")]},
        {"z": 0.9, "alpha_G": 2.0, "roots": [_root(21.0, "finite_high_root")]},
        {"z": 1.0, "alpha_G": 1.0, "roots": [_root(1.0, "low_root")]},
        {"z": 1.0, "alpha_G": 2.0, "roots": [_root(1.0, "low_root")]},
    ]
    confirm = [
        _candidate(0.9, 1.0, "a", 20.0),
        {**_candidate(0.9, 1.0, "b", 20.5), "rate_bound_violation_any_step": True},
        _candidate(0.9, 2.0, "a", 21.0),
        _candidate(0.9, 2.0, "b", 21.5),
    ]
    summary = summarize_stage0c(root_rows, [], confirm, alpha0_parity={"pass": True})
    assert summary["stage0c_pass"] is False
    assert summary["n_supported_parameter_points"] == 1


def test_runner_requires_explicit_confirmation():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "run_topic4_spatial_slowfast_stage0c.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=root)
    assert proc.returncode == 2
    assert "pass --confirm-run" in proc.stderr
