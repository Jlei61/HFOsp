from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/lock_topic4_fcxr_lc6a_local_classifier.py"


def test_local_classifier_lock_is_c0_only_and_records_no_q_outcome_selection():
    source = SCRIPT.read_text()
    assert 'OUT / "trajectories/C0"' in source
    assert 'OUT / "trajectories/Q' not in source
    assert '"selection_used_Q_trajectory_outcomes": False' in source
    assert "rate_quantile=.995" in source
    assert "area_quantile=.99" in source
    assert "persistence_ms=500.0" in source


def test_local_classifier_lock_requires_exact_c0_control_parity_and_hashes():
    source = SCRIPT.read_text()
    assert 'get("control_parity", {}).get("spike_exact") is not True' in source
    assert "C0 spike stream hash mismatch" in source
    assert "existing local-classifier lock source hash mismatch" in source
