from __future__ import annotations

import hashlib
import json

from scripts.topic5_continuous_marked_state_h2b.run_v03_full_grid_state_queue import (
    COMPATIBLE_QUEUE_PRODUCER_SHA256S,
    GIB,
    _complete,
    _per_worker_memory_budget,
    _safe_worker_count,
    _set_embedding_batch_size,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_query_scaled_memory_budget_uses_all_large_patient_seeds_when_safe() -> None:
    workers, budget = _safe_worker_count(
        configured=8, pending_count=5, available=243 * GIB,
        max_query_rows=2768, cpu_count=64,
    )
    assert 14 * GIB < budget < 15 * GIB
    assert workers == 5


def test_query_scaled_memory_budget_uses_more_workers_when_safe() -> None:
    workers, budget = _safe_worker_count(
        configured=8, pending_count=20, available=243 * GIB,
        max_query_rows=912, cpu_count=64,
    )
    assert 7 * GIB < budget < 8 * GIB
    assert workers == 8


def test_embedding_batch_retry_only_changes_batch_argument() -> None:
    original = ["python", "extract.py", "--embedding-batch-size", "128", "--x", "y"]
    updated = _set_embedding_batch_size(original, 32)
    assert updated == ["python", "extract.py", "--embedding-batch-size", "32", "--x", "y"]
    assert original[3] == "128"


def test_complete_accepts_scheduler_only_v3_hash_but_rejects_unknown(tmp_path) -> None:
    output = tmp_path / "states.npz"
    query = tmp_path / "queries.csv"
    output.write_bytes(b"frozen-state-cache")
    query.write_text("anchor\n1\n", encoding="utf-8")
    manifest_path = output.with_suffix(".manifest.json")
    payload = {
        "status": "COMPLETE",
        "full_recorded_five_minute_grid": True,
        "cache_sha256": _sha256(output),
        "source_hashes": {"query_csv": _sha256(query)},
        "full_grid_queue_producer_sha256": next(
            iter(COMPATIBLE_QUEUE_PRODUCER_SHA256S)
        ),
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _complete(output, query) is True

    payload["full_grid_queue_producer_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    assert _complete(output, query) is False


def test_memory_budget_rejects_negative_query_count() -> None:
    try:
        _per_worker_memory_budget(-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("negative query counts must fail closed")
