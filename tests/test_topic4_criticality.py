"""TDD tests for topic4_criticality.load_crit_config (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - load_crit_config(path=None) loads config/topic4_criticality.yaml by default.
  - Locks units, verdict threshold-sweep, quality-gate floors, and the
    slow_to_ratefield terminology block (review additions #4/#8/#9/#15/P1-1).
"""
import pytest

from src.topic4_criticality import load_crit_config


# --- verbatim from task-0-brief.md Step 2 ---

def test_config_locks_units_verdicts_and_review_additions():
    c = load_crit_config()
    assert c["operator"]["alpha_units"] == "per_ms"
    assert c["verdict"]["alpha_near_zero_tol_per_ms"] == 0.002          # #4 per-ms, low default
    assert "threshold_sweep" in c["verdict"]                             # #4
    assert c["quality_gate"]["rate_scale_floor"] > 0                     # #8
    assert c["branching"]["branch_cluster_field_tol"] > 0                # #9
    assert set(c["slow_to_ratefield"]) == {"q_I", "g_K", "h_G"}          # P1-1
    assert c["finite_time_gain"]["mode"] == "directional_core"          # #15


# --- Task 2 review #1 gap-fill: _invert_phase_transform's only prior coverage was via
# build_conditional_atlas in test_topic4_crit_integration.py, which is @pytest.mark.integration
# + figdata-gated (silently skipped without the gitignored results/ tree). This is a fast,
# no-figdata characterization test of the algebraic inverse itself. ---

def test_invert_phase_transform_round_trips_apply_transform():
    from src.sef_hfo_m3_interface import _apply_transform
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    from src.topic4_criticality import _invert_phase_transform

    mapping, _ranges = default_precalib_mapping_and_ranges("m3a_v2_2_approach")
    recip_t = mapping["coordinates"]["phase_x_core"]["transform"]   # real coeffs: a=1/3, b=-1/3
    affine_t = {"type": "affine", "a": 2.0, "b": 0.1, "clip": [0.0, 1.0]}
    identity_t = {"type": "identity"}

    cases = [
        (identity_t, [0.0, 0.3, 0.7, 1.0]),
        (affine_t, [0.1, 0.25, 0.4]),        # a*x+b stays inside [0,1]: no clip distortion
        (recip_t, [0.3, 0.5, 0.75, 1.0]),    # q_core in [0.25,1] -> phase in [0,1]: no clip
    ]
    for transform, xs in cases:
        for x in xs:
            phase = _apply_transform(transform, x)
            assert _invert_phase_transform(transform, phase) == pytest.approx(x, abs=1e-9)
