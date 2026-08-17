from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import launch_topic5_stateful_event_rnn_v2_7_validation as launch
from src.topic5_resource_guard import ResourceSnapshot, ResourceThresholds


def _screen(path: Path, subject: str, architecture: float, refinement: float) -> None:
    path.write_text(
        json.dumps(
            {
                "subject": subject,
                "architecture_screen": [{"runtime_seconds": architecture}],
                "refinement_screen": [{"runtime_seconds": refinement}],
            }
        ),
        encoding="utf-8",
    )


def test_runtime_loader_uses_frozen_screen_and_orders_lpt(tmp_path: Path):
    _screen(tmp_path / "a.json", "a", 1, 2)
    _screen(tmp_path / "b.json", "b", 6, 5)
    _screen(tmp_path / "c.json", "c", 3, 4)
    tasks = launch.load_lpt_tasks(tmp_path, expected_count=3)
    assert [task.subject for task in tasks] == ["b", "c", "a"]
    assert [task.estimated_runtime_seconds for task in tasks] == [11, 7, 3]


def test_runtime_loader_fails_closed_on_missing_patient(tmp_path: Path):
    _screen(tmp_path / "a.json", "a", 1, 2)
    with pytest.raises(launch.ValidationLaunchError):
        launch.load_lpt_tasks(tmp_path, expected_count=34)


def test_worker_command_is_explicit_cuda_env_and_one_patient():
    command = launch.worker_command("subject_a")
    assert command[0] == str(launch.PYTHON)
    assert command[1] == str(launch.WORKER)
    assert command[-3:] == ["patients", "--subjects", "subject_a"]
    assert command.count("subject_a") == 1


def test_worker_environment_pins_every_native_pool_to_one():
    environment = launch.worker_environment(
        {"OMP_NUM_THREADS": "40", "CUDA_VISIBLE_DEVICES": "0"}
    )
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert environment[key] == "1"
    assert environment["CUDA_VISIBLE_DEVICES"] == ""


class _Process:
    def __init__(self, pid: int):
        self.pid = pid


def _running(subject: str, started: float, pid: int = 1):
    return launch.RunningPatient(
        task=launch.ValidationTask(subject, 1),
        process=_Process(pid),
        started_unix=started,
        stdout_path=Path("stdout"),
        stderr_path=Path("stderr"),
        stdout_handle=None,
        stderr_handle=None,
    )


def test_resource_audit_pauses_at_64_gib_without_killing_worker():
    running = {"a": _running("a", 1)}
    decision = launch.audit_running_workers(
        running,
        thresholds=ResourceThresholds(),
        host_snapshot=ResourceSnapshot(60, 0.1),
        worker_snapshots={"a": ResourceSnapshot(60, 1)},
    )
    assert decision.pause_new_work
    assert decision.abort_subject is None


def test_resource_audit_aborts_latest_worker_below_48_gib():
    running = {"a": _running("a", 1), "b": _running("b", 2)}
    decision = launch.audit_running_workers(
        running,
        thresholds=ResourceThresholds(),
        host_snapshot=ResourceSnapshot(40, 0.1),
        worker_snapshots={
            "a": ResourceSnapshot(40, 1),
            "b": ResourceSnapshot(40, 1),
        },
    )
    assert decision.abort_subject == "b"
    assert decision.reason == "host_abort_memory_threshold"


def test_resource_audit_aborts_worker_over_8_gib():
    running = {"a": _running("a", 1)}
    decision = launch.audit_running_workers(
        running,
        thresholds=ResourceThresholds(),
        host_snapshot=ResourceSnapshot(100, 0.1),
        worker_snapshots={"a": ResourceSnapshot(100, 8.01)},
    )
    assert decision.abort_subject == "a"
    assert decision.reason == "worker_rss_limit_exceeded"


def test_child_oom_log_is_retryable_but_ordinary_error_is_not(tmp_path: Path):
    stderr = tmp_path / "stderr.log"
    stderr.write_text("Traceback\nMemoryError: unable to allocate array\n")
    assert launch.resource_failure_reason(1, stderr) == "child_oom_log"
    stderr.write_text("Traceback\nValueError: malformed patient artifact\n")
    assert launch.resource_failure_reason(1, stderr) is None
    assert launch.resource_failure_reason(137, stderr) == "exit_137"


def _result(subject: str, rc: int = 0) -> launch.PatientResult:
    return launch.PatientResult(
        subject=subject,
        returncode=rc,
        retries_used=0,
        runtime_seconds=1,
        stdout_log=f"{subject}.stdout.log",
        stderr_log=f"{subject}.stderr.log",
        artifact=f"{subject}.json",
    )


def test_completion_marker_is_absent_for_incomplete_cohort(tmp_path: Path):
    output = tmp_path / "SCREEN_WORKERS_COMPLETE.json"
    tasks = [launch.ValidationTask("a", 1), launch.ValidationTask("b", 1)]
    with pytest.raises(launch.ValidationLaunchError):
        launch.write_completion_if_complete(
            output,
            tasks=tasks,
            results=[_result("a")],
            expected_count=2,
            initial_workers=8,
            final_workers=8,
            thresholds=ResourceThresholds(),
            started_unix=1,
        )
    assert not output.exists()


def test_completion_marker_is_atomic_and_requires_all_success(
    tmp_path: Path, monkeypatch
):
    worker = tmp_path / "worker.py"
    config = tmp_path / "config.yaml"
    parent = tmp_path / "parent"
    worker.write_text("worker", encoding="utf-8")
    config.write_text("config", encoding="utf-8")
    parent.mkdir()
    monkeypatch.setattr(launch, "WORKER", worker)
    monkeypatch.setattr(launch, "CONFIG", config)
    monkeypatch.setattr(launch, "PARENT_SCREEN", parent)
    monkeypatch.setattr(launch, "ROOT", tmp_path)

    output = tmp_path / "SCREEN_WORKERS_COMPLETE.json"
    tasks = [launch.ValidationTask("a", 2), launch.ValidationTask("b", 1)]
    payload = launch.write_completion_if_complete(
        output,
        tasks=tasks,
        results=[_result("a"), _result("b")],
        expected_count=2,
        initial_workers=8,
        final_workers=8,
        thresholds=ResourceThresholds(),
        started_unix=1,
    )
    assert payload["status"] == "SCREEN_WORKERS_COMPLETE"
    assert json.loads(output.read_text())["n_completed"] == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_cli_defaults_to_eight_workers_one_thread_and_30_second_audit():
    args = launch.build_parser().parse_args([])
    assert args.workers == 8
    assert args.threads_per_worker == 1
    assert args.monitor_interval_seconds == 30
    assert args.pause_available_gb == 64
    assert args.abort_available_gb == 48
    assert args.max_worker_rss_gb == 8


def test_host_memory_unavailable_without_running_worker_is_fatal():
    decision = launch.audit_running_workers(
        {},
        thresholds=ResourceThresholds(),
        host_snapshot=ResourceSnapshot(None, 0.1),
        worker_snapshots={},
    )
    assert decision.abort_subject is None
    assert decision.reason == "host_memory_unavailable"
