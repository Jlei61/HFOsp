from pathlib import Path

import pytest

from scripts import monitor_topic4_fcxr_lc5v2p1_completion as monitor


def test_validate_dispatcher_accepts_only_the_locked_block_runner(monkeypatch):
    monkeypatch.setattr(
        monitor,
        "_cmdline",
        lambda pid: f"python scripts/{monitor.EXPECTED_DISPATCHER}",
    )
    assert monitor.EXPECTED_DISPATCHER in monitor.validate_dispatcher(123)


def test_validate_dispatcher_refuses_unrelated_pid(monkeypatch):
    monkeypatch.setattr(monitor, "_cmdline", lambda pid: "python unrelated.py")
    with pytest.raises(RuntimeError, match="refusing to signal"):
        monitor.validate_dispatcher(123)


def test_wait_for_either_prefers_done(tmp_path: Path):
    done = tmp_path / "DONE.json"
    failed = tmp_path / "FAILED.json"
    done.write_text("{}")
    assert monitor._wait_for_either(done, failed, 0.001) == ("DONE", done)
