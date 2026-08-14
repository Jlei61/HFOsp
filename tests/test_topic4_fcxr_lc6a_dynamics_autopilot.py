from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_dynamics_autopilot.sh"


def test_autopilot_waits_for_graph_hard_gates_and_uses_dynamic_slot_refill():
    source = SCRIPT.read_text()
    assert "DONE_LC6A_GRAPH_FAMILY.json" in source
    assert "DONE_LC6A_TWO_HOP_AUDIT.json" in source
    assert "wait -n -p" not in source
    assert "wait_for_finished_pid" in source
    assert 'wait "$finished"' in source
    assert "MAX_SLOTS=4" in source
    assert "check_headroom" in source
    assert "STAGE_SWAP_BASELINE_MIB" in source
    assert "delta >= 256.0" in source
    assert "3.0*rss" in source
    assert "SUBMISSION_SLOT_CAP" in source


def test_autopilot_restart_skips_atomic_completed_arms():
    source = SCRIPT.read_text()
    assert "stage_done_path" in source
    assert "DONE_LC6A_FUNCTIONAL_${condition}.json" in source
    assert "DONE_${condition}.json" in source
    assert "DONE_LC6A_GAIN_${condition}.json" in source
    assert "skipped completed" in source


def test_autopilot_remeasures_natural_arm_rss_before_parallel_fill():
    source = SCRIPT.read_text()
    assert "remeasure_rss_budget" in source
    assert "LC6A_C0_CHUNK" in source
    c0 = source.index("run_pool natural C0")
    measured = source.index("RSS_BUDGET_GIB=$(remeasure_rss_budget)")
    parallel = source.index("run_pool natural C1 Q1 Q2 Q3")
    assert c0 < measured < parallel


def test_autopilot_preserves_c0_reference_order_and_fixed_five_arm_block():
    source = SCRIPT.read_text()
    assert "run_pool functional C0 C1 Q1 Q2 Q3" in source
    assert "run_pool natural C0" in source
    assert "run_pool natural C1 Q1 Q2 Q3" in source
    lock = "lock_topic4_fcxr_lc6a_local_classifier.py"
    assert lock in source
    assert source.index("run_pool natural C0") < source.index(lock)
    assert source.index(lock) < source.index("run_pool natural C1 Q1 Q2 Q3")
    assert source.index("run_pool natural C1 Q1 Q2 Q3") < source.index(
        "aggregate_topic4_fcxr_lc6a_phenotypes.py"
    )
    assert source.index("aggregate_topic4_fcxr_lc6a_phenotypes.py") < source.index(
        "run_topic4_fcxr_lc6a_gain_forks.py lock"
    )
    assert source.index("run_topic4_fcxr_lc6a_gain_forks.py finalize") < source.index(
        "run_topic4_fcxr_lc6a_confirmation.py build"
    )
    assert "--confirmation-lock" in source
    assert "run_topic4_fcxr_lc6a_confirmation.py finalize" in source
