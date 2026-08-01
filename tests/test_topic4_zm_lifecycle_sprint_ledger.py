import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/adjudicate_topic4_zm_lifecycle_sprint_ledger.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_sprint_ledger", SCRIPT)
L = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(L)


def test_adjudication_separates_adaptive_cancel_from_failure(monkeypatch):
    monkeypatch.setattr(
        L, "_artifact_from_log",
        lambda path: ROOT / "results/summary.json" if path == "ok" else None,
    )
    raw = {"rows": {
        "ok": {"status": "running", "log_path": "ok"},
        "cancel": {"status": "worker_failed", "log_path": "cancel"},
        "fail": {"status": "worker_failed", "log_path": "fail"},
        "pending": {"status": "pending"},
    }}
    decisions = {"decisions": [{"cancelled_config_ids": ["cancel"]}]}
    got = L.adjudicate(raw, decisions)
    assert got["rows"]["ok"]["adjudicated_status"] == "success"
    assert got["rows"]["cancel"]["adjudicated_status"] == "adaptively_cancelled"
    assert got["rows"]["fail"]["adjudicated_status"] == "worker_failed"
    assert got["rows"]["pending"]["adjudicated_status"] == "not_run_after_adaptive_stop"
