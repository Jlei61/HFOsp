"""TDD tests for topic4_criticality.load_crit_config (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - load_crit_config(path=None) loads config/topic4_criticality.yaml by default.
  - Locks units, verdict threshold-sweep, quality-gate floors, and the
    slow_to_ratefield terminology block (review additions #4/#8/#9/#15/P1-1).
"""
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
