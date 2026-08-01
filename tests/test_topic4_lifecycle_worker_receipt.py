import json
import time

import pytest

from src.topic4_lifecycle_worker_receipt import (
    WorkerReceipt, classify_stale_receipt,
)


def test_worker_receipt_records_success_and_context(tmp_path):
    path = tmp_path / "receipt.json"
    with WorkerReceipt(
        path, config_hash="cfg", git_sha="sha", command="demo", heartbeat_s=0.01
    ) as receipt:
        receipt.update_context(checkpoint_hash="ckpt")
        time.sleep(0.03)
        receipt.finish("success", artifact_path="result.json")
    got = json.loads(path.read_text())
    assert got["status"] == "success"
    assert got["checkpoint_hash"] == "ckpt"
    assert got["artifact_path"] == "result.json"
    assert got["peak_rss_gb"] >= 0


def test_worker_receipt_captures_python_exception(tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(RuntimeError, match="boom"):
        with WorkerReceipt(
            path, config_hash="cfg", git_sha="sha", command="demo",
            heartbeat_s=60,
        ):
            raise RuntimeError("boom")
    got = json.loads(path.read_text())
    assert got["status"] == "python_exception"
    assert got["exception_type"] == "RuntimeError"
    assert "boom" in got["traceback"]


def test_stale_running_receipt_is_resource_abort():
    payload = {
        "status": "running",
        "heartbeat_time_utc": "2026-08-01T00:00:00+00:00",
    }
    assert classify_stale_receipt(
        payload, stale_after_s=60, now_epoch=1785600000
    ) == "resource_abort"
