"""Job identity, atomic status files and resumable bookkeeping.

A job's identity is its full scientific and engineering context, not its output
filename.  Two runs that differ in any of goal / patient set / model family /
arm / seed / split / config / code revision / input hash are different jobs and
never share a status file.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Iterable

from .contracts import OUTPUT_ROOT, atomic_write_json, code_revision, jsonable, sha256_obj

STATES = ("PENDING", "RUNNING", "COMPLETE", "FAILED", "OOM", "NAN",
          "INVALID_INPUT", "SKIPPED_EXISTING")

JOBS_DIR = OUTPUT_ROOT / "jobs"
LOGS_DIR = OUTPUT_ROOT / "logs"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"


@dataclass(frozen=True)
class JobKey:
    goal: str
    family: str          # model family, e.g. "G2/R0/node_film"
    arm: str             # experimental arm within the family
    seed: int
    split: str           # "development" | "formal_test"
    cohort: str          # cohort tag, e.g. "all34" or "breadth8"
    config_hash: str
    input_hash: str
    code_revision: str

    @property
    def job_id(self) -> str:
        payload = asdict(self)
        digest = sha256_obj(payload)[:16]
        safe = f"{self.goal}__{self.family}__{self.arm}__s{self.seed}__{self.split}__{self.cohort}"
        safe = safe.replace("/", "-").replace(" ", "_")
        return f"{safe}__{digest}"


@dataclass
class JobRecord:
    key: JobKey
    state: str = "PENDING"
    started_at: float | None = None
    finished_at: float | None = None
    host: str = field(default_factory=socket.gethostname)
    pid: int | None = None
    failure_reason: str | None = None
    peak_rss_mib: float | None = None
    wall_seconds: float | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def path(self) -> Path:
        return JOBS_DIR / f"{self.key.job_id}.status.json"

    def write(self) -> Path:
        return atomic_write_json(self.path(), jsonable({
            "job_id": self.key.job_id, "key": asdict(self.key), "state": self.state,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "host": self.host, "pid": self.pid, "failure_reason": self.failure_reason,
            "peak_rss_mib": self.peak_rss_mib, "wall_seconds": self.wall_seconds,
            "outputs": self.outputs, "metrics": self.metrics,
        }))


def load_record(job_id: str) -> dict[str, Any] | None:
    path = JOBS_DIR / f"{job_id}.status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def is_complete(key: JobKey) -> bool:
    record = load_record(key.job_id)
    return bool(record and record.get("state") == "COMPLETE")


def peak_rss_mib() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:  # pragma: no cover
        return float("nan")


class JobRunner:
    """Context manager that records RUNNING / COMPLETE / failure states atomically."""

    def __init__(self, key: JobKey):
        self.record = JobRecord(key=key)

    def __enter__(self) -> JobRecord:
        self.record.state = "RUNNING"
        self.record.started_at = time.time()
        self.record.pid = os.getpid()
        self.record.write()
        return self.record

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.record.finished_at = time.time()
        self.record.wall_seconds = self.record.finished_at - (self.record.started_at or 0.0)
        self.record.peak_rss_mib = peak_rss_mib()
        if exc is None:
            if self.record.state == "RUNNING":
                self.record.state = "COMPLETE"
        else:
            text = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, MemoryError) or "out of memory" in str(exc).lower():
                self.record.state = "OOM"
            elif "nan" in str(exc).lower():
                self.record.state = "NAN"
            elif isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
                self.record.state = "INVALID_INPUT"
            else:
                self.record.state = "FAILED"
            self.record.failure_reason = text[:2000]
        self.record.write()
        return False


def collect_jobs() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(JOBS_DIR.glob("*.status.json")):
        try:
            rows.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue
    return rows


def write_job_manifest(planned: Iterable[JobKey]) -> Path:
    planned = list(planned)
    records = {r["job_id"]: r for r in collect_jobs()}
    rows = []
    for key in planned:
        record = records.get(key.job_id, {})
        rows.append({"job_id": key.job_id, "key": asdict(key),
                     "state": record.get("state", "PENDING"),
                     "wall_seconds": record.get("wall_seconds"),
                     "failure_reason": record.get("failure_reason")})
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"], 0) + 1
    return atomic_write_json(MANIFEST_DIR / "JOB_MANIFEST.json", {
        "contract": "topic5_epi_prssm_v0_1_job_manifest",
        "code_revision": code_revision(),
        "n_planned": len(rows), "state_counts": counts, "jobs": rows,
    })
