import importlib.util
import json
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


def test_adjudication_recovers_runtime_resource_and_version_receipt(tmp_path, monkeypatch):
    artifact = tmp_path / "results" / "summary.json"
    artifact.parent.mkdir()
    artifact.write_text(json.dumps({
        "wall_s": 12.5,
        "peak_rss_gb": 7.25,
        "runtime_git_sha": "abc123",
        "runaway_early_stop_ms": None,
        "finite_control": {"target": "all_E"},
        "observed_ms": 20000.0,
    }))
    monkeypatch.setattr(L, "_artifact_from_log", lambda path: artifact)
    monkeypatch.setattr(L, "ROOT", tmp_path)
    got = L.adjudicate({"rows": {"ok": {"status": "running", "log_path": "ok"}}}, {})
    row = got["rows"]["ok"]
    assert row["adjudicated_status"] == "success"
    assert row["wall_s"] == 12.5
    assert row["peak_rss_gb"] == 7.25
    assert row["runtime_git_sha"] == "abc123"
    assert row["observed_ms"] == 20000.0
