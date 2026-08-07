#!/usr/bin/env python3
"""Launch the 34-patient v2.7 validation screen with bounded resources.

Scheduling is patient-level and longest-processing-time first.  Historical
runtime estimates come only from the frozen v2.6 validation screen; no v2.7
validation score is read to determine execution order.  Each child process
handles exactly one patient.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_resource_guard import (  # noqa: E402
    FailureKind,
    ResourceAction,
    ResourceSnapshot,
    ResourceThresholds,
    RetryAction,
    atomic_write_json,
    classify_failure,
    decide_resource_retry,
    evaluate_resources,
    pin_thread_environment,
    read_resource_snapshot,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
WORKER = ROOT / "scripts/run_topic5_stateful_event_rnn_v2_7_cohort_worker.py"
CONFIG = ROOT / "config/topic5_stateful_event_rnn_v2_7.yaml"
PARENT_SCREEN = (
    ROOT
    / "results/topic5_stateful_event_sequence_rnn/v2_6/validation_screen/per_subject"
)
OUTPUT = ROOT / "results/topic5_stateful_event_sequence_rnn/v2_7/validation_screen"
COMPLETION = OUTPUT / "SCREEN_WORKERS_COMPLETE.json"
EXPECTED_SUBJECTS = 34


class ValidationLaunchError(RuntimeError):
    """Raised when the validation queue must fail closed."""


@dataclass(frozen=True)
class ValidationTask:
    subject: str
    estimated_runtime_seconds: float
    retries_used: int = 0


@dataclass
class RunningPatient:
    task: ValidationTask
    process: subprocess.Popen
    started_unix: float
    stdout_path: Path
    stderr_path: Path
    stdout_handle: Any
    stderr_handle: Any
    forced_resource_reason: Optional[str] = None

    def close_logs(self) -> None:
        self.stdout_handle.close()
        self.stderr_handle.close()


@dataclass(frozen=True)
class PatientResult:
    subject: str
    returncode: int
    retries_used: int
    runtime_seconds: float
    stdout_log: str
    stderr_log: str
    artifact: str


@dataclass(frozen=True)
class AuditDecision:
    pause_new_work: bool
    abort_subject: Optional[str]
    reason: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _screen_runtime(record: Mapping[str, Any]) -> float:
    rows = list(record.get("architecture_screen", [])) + list(
        record.get("refinement_screen", [])
    )
    if not rows:
        raise ValidationLaunchError("v2.6 screen record has no runtime rows")
    values = [float(row["runtime_seconds"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        raise ValidationLaunchError("v2.6 screen runtime is non-finite or negative")
    return float(sum(values))


def load_lpt_tasks(
    screen_root: Path = PARENT_SCREEN,
    *,
    expected_count: int = EXPECTED_SUBJECTS,
) -> list[ValidationTask]:
    """Read frozen v2.6 patient runtimes and return an LPT queue."""

    records = []
    for path in sorted(Path(screen_root).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        subject = str(record.get("subject", ""))
        if not subject or subject != path.stem:
            raise ValidationLaunchError(f"subject/path mismatch in {path}")
        records.append(
            ValidationTask(
                subject=subject,
                estimated_runtime_seconds=_screen_runtime(record),
            )
        )
    if len(records) != int(expected_count):
        raise ValidationLaunchError(
            f"expected {expected_count} v2.6 screen records, found {len(records)}"
        )
    subjects = [task.subject for task in records]
    if len(subjects) != len(set(subjects)):
        raise ValidationLaunchError("duplicate patient in v2.6 screen records")
    return sorted(
        records,
        key=lambda task: (-task.estimated_runtime_seconds, task.subject),
    )


def worker_command(subject: str) -> list[str]:
    """Return the explicit one-patient cuda_env worker command."""

    return [
        str(PYTHON),
        str(WORKER),
        "--config",
        str(CONFIG),
        "--phase",
        "patients",
        "--subjects",
        str(subject),
    ]


def worker_environment(base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    environment = dict(os.environ if base is None else base)
    pin_thread_environment(1, environ=environment, disable_cuda=True)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def audit_running_workers(
    running: Mapping[str, RunningPatient],
    *,
    thresholds: ResourceThresholds,
    host_snapshot: Optional[ResourceSnapshot] = None,
    worker_snapshots: Optional[Mapping[str, ResourceSnapshot]] = None,
) -> AuditDecision:
    """Apply host and per-worker thresholds without mutating processes."""

    host = read_resource_snapshot() if host_snapshot is None else host_snapshot
    if host.available_gb is None or not math.isfinite(float(host.available_gb)):
        return AuditDecision(False, _latest_subject(running), "host_memory_unavailable")
    available = float(host.available_gb)
    if available < float(thresholds.abort_available_gb):
        return AuditDecision(False, _latest_subject(running), "host_abort_memory_threshold")

    snapshots = worker_snapshots
    for subject, item in running.items():
        snapshot = (
            read_resource_snapshot(pid=item.process.pid)
            if snapshots is None
            else snapshots.get(subject)
        )
        poll = getattr(item.process, "poll", None)
        already_exited = callable(poll) and poll() is not None
        if snapshot is None and already_exited:
            continue
        if snapshot is None:
            return AuditDecision(False, subject, "worker_snapshot_missing")
        if snapshot.worker_rss_gb is None and already_exited:
            continue
        # Use the one host reading for all workers so a changing /proc sample
        # cannot give simultaneously running patients inconsistent host states.
        normalized = ResourceSnapshot(
            available_gb=available,
            worker_rss_gb=snapshot.worker_rss_gb,
            swap_free_gb=host.swap_free_gb,
            source=snapshot.source,
        )
        verdict = evaluate_resources(normalized, thresholds)
        if verdict.action is ResourceAction.ABORT_WORKER:
            return AuditDecision(False, subject, verdict.reason)

    if available < float(thresholds.pause_available_gb):
        return AuditDecision(True, None, "host_pause_memory_threshold")
    return AuditDecision(False, None, "resource_headroom_ok")


def _latest_subject(running: Mapping[str, RunningPatient]) -> Optional[str]:
    if not running:
        return None
    return max(running.values(), key=lambda item: item.started_unix).task.subject


def _patient_artifact(subject: str, output: Path = OUTPUT) -> Path:
    return Path(output) / "per_subject" / f"{subject}.json"


def validate_patient_artifact(subject: str, output: Path = OUTPUT) -> Path:
    path = _patient_artifact(subject, output)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationLaunchError(f"{subject}: missing or invalid screen artifact") from exc
    if record.get("subject") != subject:
        raise ValidationLaunchError(f"{subject}: artifact subject mismatch")
    if record.get("status") != "PATIENT_VALIDATION_PROFILE_SCREENED":
        raise ValidationLaunchError(f"{subject}: screen artifact status is incomplete")
    checks = record.get("contract_checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValidationLaunchError(f"{subject}: screen artifact contract failed")
    return path


def _open_patient(
    task: ValidationTask,
    *,
    log_root: Path,
    environment: Mapping[str, str],
) -> RunningPatient:
    log_root.mkdir(parents=True, exist_ok=True)
    suffix = f"attempt_{task.retries_used + 1}"
    stdout_path = log_root / f"{task.subject}.{suffix}.stdout.log"
    stderr_path = log_root / f"{task.subject}.{suffix}.stderr.log"
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    started = time.time()
    try:
        process = subprocess.Popen(
            worker_command(task.subject),
            cwd=ROOT,
            env=dict(environment),
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return RunningPatient(
        task=task,
        process=process,
        started_unix=started,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


def _terminate(item: RunningPatient, timeout_seconds: float = 10.0) -> None:
    if item.process.poll() is not None:
        return
    item.process.terminate()
    try:
        item.process.wait(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired:
        item.process.kill()
        item.process.wait()


def resource_failure_reason(
    returncode: int,
    stderr_path: Path,
    *,
    forced_reason: Optional[str] = None,
) -> Optional[str]:
    """Recognize explicit child OOM evidence without relabeling other errors."""

    if forced_reason is not None:
        return str(forced_reason)
    kind = classify_failure(returncode=int(returncode))
    if kind is not FailureKind.OTHER:
        return kind.value
    try:
        with Path(stderr_path).open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - (128 << 10)), os.SEEK_SET)
            tail = stream.read().decode("utf-8", errors="replace").lower()
    except OSError:
        return None
    tokens = (
        "memoryerror",
        "outofmemoryerror",
        "cuda out of memory",
        "cannot allocate memory",
        "std::bad_alloc",
    )
    return "child_oom_log" if any(token in tail for token in tokens) else None


def write_completion_if_complete(
    path: Path,
    *,
    tasks: list[ValidationTask],
    results: list[PatientResult],
    expected_count: int = EXPECTED_SUBJECTS,
    initial_workers: int,
    final_workers: int,
    thresholds: ResourceThresholds,
    started_unix: float,
    monitor_interval_seconds: float = 30.0,
) -> dict[str, Any]:
    """Publish the only completion marker, and only for a complete cohort."""

    expected = {task.subject for task in tasks}
    completed = {result.subject for result in results if result.returncode == 0}
    if len(tasks) != int(expected_count) or len(expected) != int(expected_count):
        raise ValidationLaunchError("completion blocked: task cohort is incomplete")
    if completed != expected or len(results) != int(expected_count):
        raise ValidationLaunchError("completion blocked: patient workers are incomplete")
    payload = {
        "contract": "topic5_stateful_event_sequence_rnn_v2_7_validation_workers",
        "status": "SCREEN_WORKERS_COMPLETE",
        "n_expected": int(expected_count),
        "n_completed": len(results),
        "initial_workers": int(initial_workers),
        "final_workers": int(final_workers),
        "threads_per_worker": 1,
        "monitor_interval_seconds": float(monitor_interval_seconds),
        "thresholds": asdict(thresholds),
        "started_unix": float(started_unix),
        "completed_unix": time.time(),
        "python": str(PYTHON),
        "worker": str(WORKER.relative_to(ROOT)),
        "config": str(CONFIG.relative_to(ROOT)),
        "worker_sha256": sha256(WORKER),
        "config_sha256": sha256(CONFIG),
        "runtime_source": str(PARENT_SCREEN.relative_to(ROOT)),
        "subjects_lpt": [task.subject for task in tasks],
        "results": [asdict(result) for result in results],
    }
    atomic_write_json(path, payload)
    return payload


def run_validation_queue(
    tasks: list[ValidationTask],
    *,
    workers: int = 8,
    monitor_interval_seconds: float = 30.0,
    thresholds: ResourceThresholds = ResourceThresholds(),
    output: Path = OUTPUT,
) -> tuple[list[PatientResult], int]:
    if not 1 <= int(workers) <= 16:
        raise ValidationLaunchError("workers must be in [1, 16]")
    if float(monitor_interval_seconds) <= 0:
        raise ValidationLaunchError("monitor interval must be > 0")
    pending = list(tasks)
    running: dict[str, RunningPatient] = {}
    results: list[PatientResult] = []
    environment = worker_environment()
    log_root = Path(output) / "launcher_logs"
    worker_limit = int(workers)
    resource_retries_used = 0
    pause_new_work = False
    next_audit = 0.0
    fatal_reason: Optional[str] = None

    while pending or running:
        now = time.time()
        for subject, item in list(running.items()):
            returncode = item.process.poll()
            if returncode is None:
                continue
            item.close_logs()
            del running[subject]
            runtime = time.time() - item.started_unix
            resource_reason = resource_failure_reason(
                int(returncode),
                item.stderr_path,
                forced_reason=item.forced_resource_reason,
            )
            resource_failure = resource_reason is not None
            if returncode == 0 and not resource_failure:
                try:
                    artifact = validate_patient_artifact(subject, output)
                except ValidationLaunchError as exc:
                    fatal_reason = str(exc)
                    break
                results.append(
                    PatientResult(
                        subject=subject,
                        returncode=0,
                        retries_used=item.task.retries_used,
                        runtime_seconds=runtime,
                        stdout_log=str(item.stdout_path.relative_to(ROOT)),
                        stderr_log=str(item.stderr_path.relative_to(ROOT)),
                        artifact=str(artifact.relative_to(ROOT)),
                    )
                )
                continue

            retry = decide_resource_retry(
                retries_used=resource_retries_used,
                workers=worker_limit,
                anchor_chunk=1,
                null_chunk=1,
                returncode=returncode if not resource_failure else None,
                exception=(
                    MemoryError(resource_reason)
                    if resource_failure
                    else None
                ),
            )
            if retry.action is RetryAction.RETRY_REDUCED_RESOURCES:
                resource_retries_used = retry.retries_used
                worker_limit = retry.workers
                pending.insert(
                    0,
                    ValidationTask(
                        item.task.subject,
                        item.task.estimated_runtime_seconds,
                        retries_used=retry.retries_used,
                    ),
                )
            else:
                fatal_reason = (
                    f"{subject}: worker rc={returncode}; {retry.reason}; "
                    f"resource_reason={resource_reason}"
                )
                break
        if fatal_reason is not None:
            break

        if now >= next_audit:
            decision = audit_running_workers(running, thresholds=thresholds)
            next_audit = now + float(monitor_interval_seconds)
            pause_new_work = decision.pause_new_work
            if decision.abort_subject is not None:
                item = running.get(decision.abort_subject)
                if item is None:
                    fatal_reason = decision.reason
                    break
                item.forced_resource_reason = decision.reason
                _terminate(item)
                next_audit = 0.0
                continue
            if decision.reason not in {
                "resource_headroom_ok",
                "host_pause_memory_threshold",
            }:
                fatal_reason = decision.reason
                break

        while (
            pending
            and not pause_new_work
            and len(running) < int(worker_limit)
        ):
            task = pending.pop(0)
            if task.subject in running:
                fatal_reason = f"duplicate active patient: {task.subject}"
                break
            running[task.subject] = _open_patient(
                task,
                log_root=log_root,
                environment=environment,
            )
        if fatal_reason is not None:
            break
        time.sleep(1.0)

    if fatal_reason is not None:
        for item in running.values():
            _terminate(item)
            item.close_logs()
        raise ValidationLaunchError(fatal_reason)
    return results, worker_limit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1, choices=(1,))
    parser.add_argument("--monitor-interval-seconds", type=float, default=30.0)
    parser.add_argument("--pause-available-gb", type=float, default=64.0)
    parser.add_argument("--abort-available-gb", type=float, default=48.0)
    parser.add_argument("--max-worker-rss-gb", type=float, default=8.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for required in (PYTHON, WORKER, CONFIG, PARENT_SCREEN):
        if not required.exists():
            raise SystemExit(f"required input missing: {required}")
    if COMPLETION.exists():
        raise SystemExit(f"completion marker already exists: {COMPLETION}")
    thresholds = ResourceThresholds(
        pause_available_gb=args.pause_available_gb,
        abort_available_gb=args.abort_available_gb,
        max_worker_rss_gb=args.max_worker_rss_gb,
    )
    tasks = load_lpt_tasks()
    started = time.time()
    try:
        results, final_workers = run_validation_queue(
            tasks,
            workers=args.workers,
            monitor_interval_seconds=args.monitor_interval_seconds,
            thresholds=thresholds,
        )
        payload = write_completion_if_complete(
            COMPLETION,
            tasks=tasks,
            results=results,
            initial_workers=args.workers,
            final_workers=final_workers,
            thresholds=thresholds,
            started_unix=started,
            monitor_interval_seconds=args.monitor_interval_seconds,
        )
    except (OSError, ValidationLaunchError) as exc:
        raise SystemExit(f"validation launch failed closed: {exc}") from exc
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
