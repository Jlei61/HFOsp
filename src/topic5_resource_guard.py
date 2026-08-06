"""Fail-closed resource helpers for Topic 5 cohort workers.

This module is intentionally independent of every scientific model and runner.
It centralizes four pieces of execution hygiene that should not be reimplemented
by individual analyses:

* pin native numerical libraries to a known number of threads;
* sample Linux memory without requiring :mod:`psutil`;
* turn memory observations and resource failures into explicit scheduler
  decisions; and
* publish JSON completion records atomically.

The helpers never change scientific settings such as seeds, observations,
anchors or model batches.  A resource retry may only reduce execution
parallelism and computational chunk sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import signal
import tempfile
from typing import Any, Callable, Mapping, MutableMapping, Optional


_GIB = float(1024**3)

THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class ResourceConfigurationError(ValueError):
    """Raised when a resource contract is internally inconsistent."""


def pin_thread_environment(
    threads_per_worker: int = 1,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    disable_cuda: bool = True,
    malloc_arena_max: int = 2,
) -> dict[str, str]:
    """Pin native-library threads before numerical libraries start work.

    ``environ`` is injectable for tests.  In a worker process this function
    should be called before importing numpy, scipy or torch.  Existing values
    are deliberately overwritten: inheriting ``OMP_NUM_THREADS=40`` would
    otherwise multiply an outer patient worker count by forty native threads.
    """

    threads = int(threads_per_worker)
    arenas = int(malloc_arena_max)
    if threads < 1:
        raise ResourceConfigurationError("threads_per_worker must be >= 1")
    if arenas < 1:
        raise ResourceConfigurationError("malloc_arena_max must be >= 1")
    target = os.environ if environ is None else environ
    for key in THREAD_ENVIRONMENT_KEYS:
        target[key] = str(threads)
    target["MALLOC_ARENA_MAX"] = str(arenas)
    target["TOKENIZERS_PARALLELISM"] = "false"
    if disable_cuda:
        target["CUDA_VISIBLE_DEVICES"] = ""
    return {key: target[key] for key in THREAD_ENVIRONMENT_KEYS + (
        "MALLOC_ARENA_MAX",
        "TOKENIZERS_PARALLELISM",
        *(('CUDA_VISIBLE_DEVICES',) if disable_cuda else ()),
    )}


def configure_torch_threads(torch_module: Any, threads_per_worker: int = 1) -> None:
    """Apply the same limit to torch intra-op and inter-op pools.

    PyTorch only permits changing the inter-op pool before parallel work has
    started.  If it is already fixed at the requested value, the worker is
    safe.  If it is fixed at another value, fail closed instead of silently
    accepting oversubscription.
    """

    threads = int(threads_per_worker)
    if threads < 1:
        raise ResourceConfigurationError("threads_per_worker must be >= 1")
    torch_module.set_num_threads(threads)
    try:
        torch_module.set_num_interop_threads(threads)
    except RuntimeError as exc:
        getter = getattr(torch_module, "get_num_interop_threads", None)
        current = int(getter()) if callable(getter) else None
        if current != threads:
            raise ResourceConfigurationError(
                "torch inter-op threads were initialized before resource pinning"
            ) from exc


@dataclass(frozen=True)
class ResourceThresholds:
    """Memory thresholds used by an outer scheduler or a worker guard."""

    pause_available_gb: float = 64.0
    abort_available_gb: float = 48.0
    max_worker_rss_gb: float = 8.0

    def __post_init__(self) -> None:
        values = (
            self.pause_available_gb,
            self.abort_available_gb,
            self.max_worker_rss_gb,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise ResourceConfigurationError("resource thresholds must be finite and > 0")
        if float(self.abort_available_gb) >= float(self.pause_available_gb):
            raise ResourceConfigurationError(
                "abort_available_gb must be lower than pause_available_gb"
            )


@dataclass(frozen=True)
class ResourceSnapshot:
    """A point-in-time host/worker memory observation, in GiB."""

    available_gb: Optional[float]
    worker_rss_gb: Optional[float]
    swap_free_gb: Optional[float] = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResourceAction(str, Enum):
    CONTINUE = "continue"
    PAUSE_NEW_WORK = "pause_new_work"
    ABORT_WORKER = "abort_worker"


@dataclass(frozen=True)
class ResourceVerdict:
    action: ResourceAction
    reason: str


def _finite_nonnegative(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) >= 0


def evaluate_resources(
    snapshot: ResourceSnapshot,
    thresholds: ResourceThresholds = ResourceThresholds(),
) -> ResourceVerdict:
    """Return a fail-closed resource decision.

    Missing/invalid available-memory measurements abort rather than guessing.
    A worker RSS violation also aborts.  Host memory between the abort and pause
    thresholds pauses new work while allowing active workers to finish.
    """

    if not _finite_nonnegative(snapshot.available_gb):
        return ResourceVerdict(
            ResourceAction.ABORT_WORKER, "available_memory_unavailable"
        )
    if not _finite_nonnegative(snapshot.worker_rss_gb):
        return ResourceVerdict(ResourceAction.ABORT_WORKER, "worker_rss_unavailable")
    if float(snapshot.worker_rss_gb) > float(thresholds.max_worker_rss_gb):
        return ResourceVerdict(ResourceAction.ABORT_WORKER, "worker_rss_limit_exceeded")
    if float(snapshot.available_gb) < float(thresholds.abort_available_gb):
        return ResourceVerdict(ResourceAction.ABORT_WORKER, "host_abort_memory_threshold")
    if float(snapshot.available_gb) < float(thresholds.pause_available_gb):
        return ResourceVerdict(ResourceAction.PAUSE_NEW_WORK, "host_pause_memory_threshold")
    return ResourceVerdict(ResourceAction.CONTINUE, "resource_headroom_ok")


def _parse_kib_table(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            value = float(fields[0])
        except ValueError:
            continue
        unit = fields[1].lower() if len(fields) > 1 else "kb"
        if unit not in {"kb", "kib"}:
            continue
        values[key] = value * 1024.0
    return values


def read_resource_snapshot(
    pid: Optional[int] = None,
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    status_path: Optional[Path] = None,
) -> ResourceSnapshot:
    """Read host available memory and one worker's RSS from Linux ``/proc``.

    The function has no psutil dependency.  Missing or malformed fields are
    represented as ``None`` and therefore produce a fail-closed verdict.
    ``status_path`` is injectable for deterministic unit tests.
    """

    worker_pid = os.getpid() if pid is None else int(pid)
    worker_status = (
        Path(f"/proc/{worker_pid}/status") if status_path is None else Path(status_path)
    )
    try:
        memory = _parse_kib_table(Path(meminfo_path))
    except (OSError, UnicodeError):
        memory = {}
    try:
        status = _parse_kib_table(worker_status)
    except (OSError, UnicodeError):
        status = {}

    def gib(values: Mapping[str, float], key: str) -> Optional[float]:
        value = values.get(key)
        return None if value is None else float(value) / _GIB

    return ResourceSnapshot(
        available_gb=gib(memory, "MemAvailable"),
        worker_rss_gb=gib(status, "VmRSS"),
        swap_free_gb=gib(memory, "SwapFree"),
        source="procfs",
    )


class FailureKind(str, Enum):
    MEMORY_ERROR = "memory_error"
    EXIT_137 = "exit_137"
    SIGKILL = "sigkill"
    OTHER = "other"


def classify_failure(
    *, returncode: Optional[int] = None, exception: Optional[BaseException] = None
) -> FailureKind:
    """Classify only explicit memory failures and kill exit codes as retryable."""

    if isinstance(exception, MemoryError):
        return FailureKind.MEMORY_ERROR
    if exception is not None:
        names = {cls.__name__ for cls in type(exception).__mro__}
        if "OutOfMemoryError" in names:
            return FailureKind.MEMORY_ERROR
    if returncode == 137:
        return FailureKind.EXIT_137
    if returncode == -int(signal.SIGKILL):
        return FailureKind.SIGKILL
    return FailureKind.OTHER


class RetryAction(str, Enum):
    RETRY_REDUCED_RESOURCES = "retry_reduced_resources"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True)
class RetryDecision:
    action: RetryAction
    failure_kind: FailureKind
    workers: int
    anchor_chunk: int
    null_chunk: int
    retries_used: int
    reason: str


def decide_resource_retry(
    *,
    retries_used: int,
    workers: int,
    anchor_chunk: int,
    null_chunk: int,
    returncode: Optional[int] = None,
    exception: Optional[BaseException] = None,
) -> RetryDecision:
    """Permit exactly one resource-only retry with reduced execution pressure.

    The caller owns scientific settings.  This function returns only reduced
    outer-worker and computational-chunk values; it cannot alter seeds,
    observations, anchors, null counts or model batches.
    """

    retries = int(retries_used)
    worker_count = int(workers)
    anchor_size = int(anchor_chunk)
    null_size = int(null_chunk)
    if retries < 0 or worker_count < 1 or anchor_size < 1 or null_size < 1:
        raise ResourceConfigurationError("retry inputs must be positive")
    kind = classify_failure(returncode=returncode, exception=exception)
    retryable = kind in {
        FailureKind.MEMORY_ERROR,
        FailureKind.EXIT_137,
        FailureKind.SIGKILL,
    }
    if retryable and retries == 0:
        return RetryDecision(
            action=RetryAction.RETRY_REDUCED_RESOURCES,
            failure_kind=kind,
            workers=max(1, worker_count // 2),
            anchor_chunk=max(1, anchor_size // 2),
            null_chunk=max(1, null_size // 2),
            retries_used=1,
            reason="single_resource_retry",
        )
    return RetryDecision(
        action=RetryAction.FAIL_CLOSED,
        failure_kind=kind,
        workers=worker_count,
        anchor_chunk=anchor_size,
        null_chunk=null_size,
        retries_used=retries,
        reason=(
            "resource_retry_exhausted" if retryable else "non_resource_failure"
        ),
    )


def atomic_write_json(
    path: Path,
    payload: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Write JSON through a same-directory tempfile and ``os.replace``.

    A unique tempfile avoids collisions between independent workers.  The file
    is flushed and fsynced before replacement; on any failure the previous
    destination remains intact and the temporary file is removed.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(
                payload,
                stream,
                indent=int(indent),
                sort_keys=bool(sort_keys),
                default=default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    except Exception:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise

