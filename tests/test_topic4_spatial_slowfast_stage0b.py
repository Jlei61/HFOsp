"""Contract tests for the homogeneous Stage-0B E/I fast-topology screen."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from src.sef_hfo_lif import mean_field
from src.topic4_spatial_slowfast_stage0b import (
    FastParameters,
    classify_fork_batch,
    build_state_forks,
    exact_siegert_root_audit,
    lut_clip_audit,
    moments_from_state,
    classify_rate_trace,
    equilibrium_state,
    fast_rhs,
    find_fixed_points,
    numerical_jacobian,
    summarize_stage0b,
    summarize_exact_siegert_audit,
)


def test_self_consistent_root_and_rhs_share_one_equation():
    params = FastParameters(w_ee_mult=1.0, q=1.0, ratio=1.0)
    roots = find_fixed_points(params)
    assert roots
    low = min(roots, key=lambda row: row["rE_khz"])
    assert low["lut_clip_at_root"] is False
    assert all(key in low for key in ("muE_mV", "sigmaE_mV", "muI_mV", "sigmaI_mV"))
    state = equilibrium_state((low["rE_khz"], low["rI_khz"]))
    assert np.linalg.norm(fast_rhs(state, params), ord=np.inf) < 2e-7 / 10.0
    # The new root stays close to the canonical exact-LIF mean field despite using the
    # existing vectorized M3B transfer table for scan speed.
    canonical = mean_field(1.0, 1.0)
    assert abs(low["rE_khz"] - canonical["nuE"]) <= 0.2 * canonical["nuE"] + 2e-5


def test_full_sigma_jacobian_matches_directional_rhs_difference():
    params = FastParameters(w_ee_mult=1.2, q=0.94, ratio=1.0)
    state = equilibrium_state((0.025, 0.060))
    jac = numerical_jacobian(state, params)
    direction = np.asarray([0.3, -0.2, 0.4, -0.1, 0.2, -0.3])
    direction /= np.linalg.norm(direction)
    h = 2e-7
    observed = (fast_rhs(state + h * direction, params) - fast_rhs(state - h * direction, params)) / (2 * h)
    np.testing.assert_allclose(jac @ direction, observed, rtol=3e-3, atol=3e-6)


def test_off_manifold_probes_cover_synaptic_history_dimensions():
    point = {"w_ee_mult": 1.0, "q": 1.0, "ratio": 1.0, "roots": []}
    metadata, states, _ = build_state_forks([point])
    indices = [i for i, row in enumerate(metadata) if row["initial_kind"] == "off_manifold_probe"]
    assert len(indices) == 4
    assert {metadata[i]["initial_label"] for i in indices} == {
        "e_synapse_loaded_i_low",
        "i_synapse_loaded_e_low",
        "rate_high_synapse_low",
        "rate_low_synapse_high",
    }
    assert np.all(states[indices] >= 0)
    assert np.all(states[indices, 0] <= 0.5) and np.all(states[indices, 1] <= 1.0)
    assert all(
        not np.allclose(states[i, 2:], [states[i, 0], states[i, 1], states[i, 0], states[i, 1]])
        for i in indices
    )


def test_off_manifold_probes_start_inside_lut_support_across_locked_grid():
    for w_ee in np.arange(1.0, 1.51, 0.1):
        for q in np.arange(1.0, 0.79, -0.01):
            point = {"w_ee_mult": float(w_ee), "q": float(q), "ratio": 1.0, "roots": []}
            metadata, states, params = build_state_forks([point])
            indices = [
                i for i, row in enumerate(metadata) if row["initial_kind"] == "off_manifold_probe"
            ]
            moments = moments_from_state(states[indices], [params[i] for i in indices])
            repeated = [np.repeat(np.asarray(x)[None, :], 2, axis=0) for x in moments]
            audits = lut_clip_audit(*repeated)
            assert not any(audit["lut_clip_any_saved"] for audit in audits), (w_ee, q, audits)


def test_exact_siegert_locally_preserves_baseline_low_root():
    params = FastParameters(1.0, 1.0, 1.0)
    root = min(find_fixed_points(params), key=lambda row: row["rE_hz"])
    rows = exact_siegert_root_audit(
        [{"w_ee_mult": 1.0, "q": 1.0, "ratio": 1.0, "roots": [root]}]
    )
    assert rows[0]["exact_converged"] is True
    assert rows[0]["exact_rate_class"] == "low"
    assert rows[0]["exact_stability"] == "stable"


def test_exact_siegert_summary_is_fail_closed_on_unresolved_or_topology_flip():
    base = {
        "source_stability": "unstable",
        "source_rE_hz": 20.0,
        "exact_converged": True,
        "exact_stability": "unstable",
        "exact_rate_class": "finite_high",
    }
    assert summarize_exact_siegert_audit([base])["supports_lut_no_go"] is True
    unresolved = dict(base, exact_converged=False, exact_stability="unresolved")
    out = summarize_exact_siegert_audit([unresolved])
    assert out["supports_lut_no_go"] is False
    assert "not_all_source_roots_converged" in out["failure_reasons"]
    flipped = dict(base, exact_stability="stable")
    out = summarize_exact_siegert_audit([flipped])
    assert out["supports_lut_no_go"] is False
    assert "source_sub100_unstable_not_preserved" in out["failure_reasons"]


def test_classifier_rejects_over_100hz_ceiling_and_long_drift():
    time_ms = np.arange(0.0, 6000.0 + 5.0, 5.0)
    ceiling = 0.35 + 0.02 * np.sin(2 * np.pi * time_ms / 100.0)
    drift = (10.0 + 20.0 * time_ms / time_ms[-1]) / 1000.0
    assert classify_rate_trace(time_ms, ceiling)["classification"] == "saturation_or_over_100hz"
    assert classify_rate_trace(time_ms, drift)["classification"] == "indeterminate_long_transient"


def test_summary_fails_closed_without_long_confirm():
    roots = [{"w_ee_mult": 1.0, "q": 1.0, "roots": []}]
    screen = [{"classification": "bounded_tonic_candidate"}]
    summary = summarize_stage0b(roots, screen, [])
    assert summary["stage0b_pass"] is False
    assert summary["stage1_to_3_open"] is False
    assert summary["verdict"].startswith("INCONCLUSIVE")


def test_low_plus_saturation_triggers_clean_no_go_stop_rule():
    roots = [
        {
            "w_ee_mult": 1.0,
            "q": 1.0,
            "roots": [
                {"branch_class": "low_root", "stability": "stable"},
                {"branch_class": "saturation_cliff_root", "stability": "stable"},
            ],
        }
    ]
    screen = [
        {"classification": "low_fixed_point"},
        {"classification": "saturation_or_over_100hz"},
    ]
    summary = summarize_stage0b(roots, screen, [])
    assert summary["verdict"] == "CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY"
    assert summary["stop_rule_triggered"] is True
    assert summary["stage1_to_3_open"] is False


def test_unstable_exact_middle_root_cannot_block_clean_no_go():
    roots = [
        {
            "w_ee_mult": 1.3,
            "q": 0.9,
            "roots": [
                {"branch_class": "low_root", "stability": "stable"},
                {"branch_class": "finite_high_root", "stability": "unstable"},
                {"branch_class": "saturation_cliff_root", "stability": "stable"},
            ],
        }
    ]
    screen = [
        {"classification": "low_fixed_point", "initial_kind": "probe"},
        {"classification": "bounded_tonic_candidate", "initial_kind": "exact_root"},
        {"classification": "saturation_or_over_100hz", "initial_kind": "root_perturbation"},
    ]
    summary = summarize_stage0b(roots, screen, [])
    assert summary["verdict"] == "CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY"
    assert "bounded_tonic_candidate" not in summary["screen_classification_counts"]
    assert summary["all_forks_classification_counts"]["bounded_tonic_candidate"] == 1


def test_lut_clipped_finite_candidate_is_invalidated():
    time_ms = np.arange(0.0, 6000.0 + 5.0, 5.0)
    n = time_ms.size
    simulation = {
        "time_ms": time_ms,
        "rE_khz": np.full((n, 1), 0.020),
        "rI_khz": np.full((n, 1), 0.040),
        "muE_mV": np.full((n, 1), 130.0),  # above M3B LUT maximum 120 mV
        "sigmaE_mV": np.full((n, 1), 5.0),
        "muI_mV": np.full((n, 1), 20.0),
        "sigmaI_mV": np.full((n, 1), 5.0),
    }
    row = classify_fork_batch([{"initial_kind": "probe"}], simulation)[0]
    assert row["pre_lut_audit_classification"] == "bounded_tonic_candidate"
    assert row["classification"] == "lut_clipped_candidate_invalid"
    assert row["lut_clip_occupancy_saved"] == 1.0


def test_runner_requires_explicit_confirmation():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "scripts", "run_topic4_spatial_slowfast_stage0b.py")
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=root)
    assert proc.returncode == 2
    assert "pass --confirm-run" in proc.stderr
