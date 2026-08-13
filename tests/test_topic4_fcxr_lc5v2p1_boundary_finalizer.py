import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_finalizer", ROOT / "scripts/finalize_topic4_fcxr_lc5v2p1_boundary_patch.py"
)
FINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINAL)


def test_waiter_prefers_done(monkeypatch, tmp_path):
    done, failed = tmp_path / "DONE.json", tmp_path / "FAILED.json"
    monkeypatch.setattr(FINAL, "BLOCK_DONE", done)
    monkeypatch.setattr(FINAL, "BLOCK_FAILED", failed)
    done.write_text("{}")
    assert FINAL.wait_for_block(.001) == "DONE"


def test_waiter_reports_failed(monkeypatch, tmp_path):
    done, failed = tmp_path / "DONE.json", tmp_path / "FAILED.json"
    monkeypatch.setattr(FINAL, "BLOCK_DONE", done)
    monkeypatch.setattr(FINAL, "BLOCK_FAILED", failed)
    failed.write_text("{}")
    assert FINAL.wait_for_block(.001) == "FAILED"
