from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.topic5_resource_guard import (
    FailureKind,
    ResourceAction,
    ResourceConfigurationError,
    ResourceSnapshot,
    ResourceThresholds,
    RetryAction,
    atomic_write_json,
    classify_failure,
    configure_torch_threads,
    decide_resource_retry,
    evaluate_resources,
    pin_thread_environment,
    read_resource_snapshot,
)


def test_pin_thread_environment_overwrites_unsafe_inherited_values():
    environment = {"OMP_NUM_THREADS": "40", "CUDA_VISIBLE_DEVICES": "0"}
    resolved = pin_thread_environment(1, environ=environment)
    assert all(
        environment[key] == "1"
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    )
    assert environment["CUDA_VISIBLE_DEVICES"] == ""
    assert environment["MALLOC_ARENA_MAX"] == "2"
    assert resolved["OMP_NUM_THREADS"] == "1"


class _FakeTorch:
    def __init__(self, interop=40, fail_interop=False):
        self.intra = 40
        self.interop = interop
        self.fail_interop = fail_interop

    def set_num_threads(self, value):
        self.intra = int(value)

    def set_num_interop_threads(self, value):
        if self.fail_interop:
            raise RuntimeError("parallel work already started")
        self.interop = int(value)

    def get_num_interop_threads(self):
        return self.interop


def test_configure_torch_threads_limits_both_pools():
    fake = _FakeTorch()
    configure_torch_threads(fake, 1)
    assert (fake.intra, fake.interop) == (1, 1)


def test_configure_torch_threads_fails_closed_on_late_mismatch():
    fake = _FakeTorch(interop=40, fail_interop=True)
    with pytest.raises(ResourceConfigurationError):
        configure_torch_threads(fake, 1)


def test_resource_thresholds_require_abort_below_pause():
    with pytest.raises(ResourceConfigurationError):
        ResourceThresholds(pause_available_gb=48, abort_available_gb=48)


@pytest.mark.parametrize(
    ("snapshot", "action", "reason"),
    [
        (ResourceSnapshot(100, 1), ResourceAction.CONTINUE, "resource_headroom_ok"),
        (
            ResourceSnapshot(60, 1),
            ResourceAction.PAUSE_NEW_WORK,
            "host_pause_memory_threshold",
        ),
        (
            ResourceSnapshot(40, 1),
            ResourceAction.ABORT_WORKER,
            "host_abort_memory_threshold",
        ),
        (
            ResourceSnapshot(100, 9),
            ResourceAction.ABORT_WORKER,
            "worker_rss_limit_exceeded",
        ),
        (
            ResourceSnapshot(None, 1),
            ResourceAction.ABORT_WORKER,
            "available_memory_unavailable",
        ),
    ],
)
def test_resource_decision_is_ordered_and_fail_closed(snapshot, action, reason):
    verdict = evaluate_resources(snapshot)
    assert verdict.action is action
    assert verdict.reason == reason


def test_proc_snapshot_parser_needs_no_psutil(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    status = tmp_path / "status"
    meminfo.write_text(
        "MemTotal:       262144000 kB\n"
        "MemAvailable:   131072000 kB\n"
        "SwapFree:         1048576 kB\n",
        encoding="utf-8",
    )
    status.write_text("Name:\tpython\nVmRSS:\t2097152 kB\n", encoding="utf-8")
    snapshot = read_resource_snapshot(
        meminfo_path=meminfo,
        status_path=status,
    )
    assert snapshot.available_gb == pytest.approx(125.0)
    assert snapshot.worker_rss_gb == pytest.approx(2.0)
    assert snapshot.swap_free_gb == pytest.approx(1.0)
    assert snapshot.source == "procfs"


def test_missing_proc_snapshot_fails_closed(tmp_path: Path):
    snapshot = read_resource_snapshot(
        meminfo_path=tmp_path / "missing_meminfo",
        status_path=tmp_path / "missing_status",
    )
    assert evaluate_resources(snapshot).action is ResourceAction.ABORT_WORKER


def test_exit_137_gets_exactly_one_reduced_resource_retry():
    first = decide_resource_retry(
        retries_used=0,
        workers=8,
        anchor_chunk=1024,
        null_chunk=4,
        returncode=137,
    )
    assert first.action is RetryAction.RETRY_REDUCED_RESOURCES
    assert first.failure_kind is FailureKind.EXIT_137
    assert (first.workers, first.anchor_chunk, first.null_chunk) == (4, 512, 2)
    assert first.retries_used == 1

    second = decide_resource_retry(
        retries_used=first.retries_used,
        workers=first.workers,
        anchor_chunk=first.anchor_chunk,
        null_chunk=first.null_chunk,
        returncode=137,
    )
    assert second.action is RetryAction.FAIL_CLOSED
    assert second.reason == "resource_retry_exhausted"


def test_memory_error_is_retryable_but_other_failures_are_not():
    assert classify_failure(exception=MemoryError()) is FailureKind.MEMORY_ERROR
    memory = decide_resource_retry(
        retries_used=0,
        workers=1,
        anchor_chunk=1,
        null_chunk=1,
        exception=MemoryError(),
    )
    assert memory.action is RetryAction.RETRY_REDUCED_RESOURCES
    assert (memory.workers, memory.anchor_chunk, memory.null_chunk) == (1, 1, 1)

    ordinary = decide_resource_retry(
        retries_used=0,
        workers=8,
        anchor_chunk=1024,
        null_chunk=4,
        returncode=2,
    )
    assert ordinary.action is RetryAction.FAIL_CLOSED
    assert ordinary.reason == "non_resource_failure"


def test_atomic_json_write_replaces_destination_without_leftover(tmp_path: Path):
    output = tmp_path / "state.json"
    atomic_write_json(output, {"version": 1})
    atomic_write_json(output, {"version": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"version": 2}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_json_failure_preserves_previous_destination(tmp_path: Path):
    output = tmp_path / "state.json"
    atomic_write_json(output, {"complete": True})
    previous = output.read_bytes()

    class NotJson:
        pass

    with pytest.raises(TypeError):
        atomic_write_json(output, {"bad": NotJson()})
    assert output.read_bytes() == previous
    assert list(tmp_path.glob("*.tmp")) == []

