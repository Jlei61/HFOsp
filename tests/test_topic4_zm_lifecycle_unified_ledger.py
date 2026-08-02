import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "topic4_zm_lifecycle_unified_ledger",
    ROOT / "scripts/build_topic4_zm_lifecycle_unified_ledger.py",
)
U = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(U)


def test_successful_wave_overrides_earlier_not_run_row_with_same_config_id():
    early = {"rows": {"x": {
        "family": "m_response_panel", "adjudicated_status": "not_run_after_adaptive_stop",
    }}}
    later = {"rows": {"x": {
        "family": "m_response_panel", "adjudicated_status": "success",
        "artifact_path": "results/x/summary.json", "wall_s": 10.0,
    }}}
    rows = U.merge_adjudicated_ledgers([(Path("early.json"), early), (Path("later.json"), later)])
    assert rows["x"]["adjudicated_status"] == "success"
    assert rows["x"]["wall_s"] == 10.0
    assert rows["x"]["source_ledger_paths"] == ["early.json", "later.json"]


def test_merge_preserves_stage_and_unique_config_count(monkeypatch):
    monkeypatch.setattr(U, "_analysis_indexes", lambda: ({}, {}))
    fast = {"rows": {"a": {"family": "depression_only_lhs", "adjudicated_status": "success"}}}
    control = {"rows": {"b": {"family": "finite_control_dose", "adjudicated_status": "worker_failed"}}}
    payload = U.build_payload([(Path("fast.json"), fast), (Path("control.json"), control)])
    assert payload["n_unique_configs"] == 2
    assert payload["stage_counts"] == {"fast_phase_map": 1, "control_dose": 1}
    assert payload["status_counts"] == {"success": 1, "worker_failed": 1}
