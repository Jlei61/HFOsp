from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_topic4_fcxr_lc6a_dynamics_autopilot.sh"


def test_autopilot_waits_for_graph_hard_gates_and_uses_dynamic_slot_refill():
    source = SCRIPT.read_text()
    assert "DONE_LC6A_GRAPH_FAMILY.json" in source
    assert "DONE_LC6A_TWO_HOP_AUDIT.json" in source
    assert "wait -n -p finished" in source
    assert "MAX_SLOTS=4" in source
    assert "check_headroom" in source


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
