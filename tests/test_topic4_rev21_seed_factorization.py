from pathlib import Path

from scripts.run_topic4_rev12_node_worker import resolve_seed_pair


def test_omitted_dynamics_seed_preserves_historical_one_seed_semantics():
    assert resolve_seed_pair(2521, None) == (2521, 2521)


def test_dynamics_seed_can_vary_without_changing_topology_seed():
    assert resolve_seed_pair(2521, 2624) == (2521, 2624)


def test_zm_off_call_remains_separate_from_active_recording_path():
    source = Path("scripts/run_topic4_rev12_node_worker.py").read_text()
    assert "if slow is None:" in source
    assert "post_runaway_record_ms=float(simulation" in source
    assert "formal_interictal_stop_step(" in source
