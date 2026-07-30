"""Process-specific resource checks for Topic-4 Z/M Phase C.

Shared-host swap can grow when the kernel reclaims unrelated resident pages.
The auditable execution evidence is therefore periodic per-worker VmSwap
samples plus a pre-publish child self snapshot.  It is deliberately not
described as an unobserved kernel peak.  Host-level swap remains a bounded
fail-safe diagnostic.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Iterable, MutableMapping, Optional


RESOURCE_RECEIPT_SCHEMA = "zm_phasec_worker_resource_receipt_v1_2026-07-29"
COORDINATOR_RUN_ENV = "TOPIC4_PHASEC_COORDINATOR_RUN_ID"
COORDINATOR_TOKEN_ENV = "TOPIC4_PHASEC_LAUNCH_TOKEN"


def parse_process_swap_kb(status_text: str) -> int:
    """Parse Linux ``/proc/<pid>/status`` VmSwap into KiB."""
    for line in str(status_text).splitlines():
        if line.startswith("VmSwap:"):
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1])
            break
    raise RuntimeError("process status lacks a readable VmSwap field")


def _process_status_is_terminal(status_text: str) -> bool:
    """Whether Linux still exposes an exited task without memory fields."""
    for line in str(status_text).splitlines():
        if line.startswith("State:"):
            fields = line.split()
            return len(fields) >= 2 and fields[1] in {"Z", "X", "x"}
    return False


def process_swap_kb(pid: int) -> Optional[int]:
    """Return one live process' VmSwap in KiB, or ``None`` once it exited.

    An exited process is *unavailable*, not a zero-valued live sample.
    ``/proc`` may disappear between opening and parsing the status file when a
    worker exits.  Retry that teardown race once; malformed status for a
    process whose status file still exists after the retry fails closed.
    """
    status = Path(f"/proc/{int(pid)}/status")
    last_error = None
    for _attempt in range(2):
        try:
            text = status.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        try:
            return parse_process_swap_kb(text)
        except RuntimeError as exc:
            last_error = exc
            if _process_status_is_terminal(text):
                return None
            if not status.exists():
                return None
    if not status.exists():
        return None
    raise RuntimeError(
        f"live process {pid} lacks a readable VmSwap field"
    ) from last_error


def worker_swap_snapshot(pids: Iterable[int]) -> dict:
    """Return an auditable per-worker swap snapshot."""
    by_pid = {}
    unavailable = []
    for pid in sorted({int(value) for value in pids}):
        value = process_swap_kb(pid)
        if value is None:
            unavailable.append(str(pid))
        else:
            by_pid[str(pid)] = int(value)
    values = list(by_pid.values())
    return {
        "worker_swap_kb_by_pid": by_pid,
        "worker_swap_unavailable_pids": unavailable,
        "worker_swap_total_kb": int(sum(values)),
        "worker_swap_max_kb": int(max(values, default=0)),
    }


def worker_process_swap_snapshot(processes: Iterable[object]) -> dict:
    """Sample owned live workers without mistaking exit teardown for corruption.

    ``Popen.poll`` is the coordinator's authoritative liveness check.  A child
    can exit after the first poll but before ``/proc/<pid>/status`` is parsed;
    in that narrow window Linux may retain a status record without memory
    fields.  Re-poll before treating that record as malformed.  A process that
    remains live after the second poll still fails closed.
    """
    by_pid = {}
    unavailable = []
    unique = {int(process.pid): process for process in processes}
    for pid, process in sorted(unique.items()):
        if process.poll() is not None:
            unavailable.append(str(pid))
            continue
        try:
            value = process_swap_kb(pid)
        except RuntimeError:
            if process.poll() is not None:
                unavailable.append(str(pid))
                continue
            raise
        if value is None:
            unavailable.append(str(pid))
        else:
            by_pid[str(pid)] = int(value)
    values = list(by_pid.values())
    return {
        "worker_swap_kb_by_pid": by_pid,
        "worker_swap_unavailable_pids": unavailable,
        "worker_swap_total_kb": int(sum(values)),
        "worker_swap_max_kb": int(max(values, default=0)),
    }


def worker_swap_exceeded(pids: Iterable[int], allowed_bytes: int = 0) -> bool:
    """Whether any Phase-C worker exceeds its locked swap allowance."""
    allowed_kb = int(allowed_bytes) // 1024
    return worker_swap_snapshot(pids)["worker_swap_max_kb"] > allowed_kb


def update_worker_swap_audit(
    audit: MutableMapping[str, dict],
    snapshot: dict,
    *,
    sampled_at: float,
    audit_key_by_pid: Optional[dict] = None,
) -> None:
    """Accumulate sample coverage without conflating reused Linux PIDs.

    Production coordinators pass ``audit_key_by_pid`` with one immutable launch
    token per worker.  The PID-only fallback is retained for small standalone
    diagnostics, but must not be used to build production receipts.
    """
    for pid, value in snapshot["worker_swap_kb_by_pid"].items():
        key = (
            str(audit_key_by_pid[pid])
            if audit_key_by_pid is not None and pid in audit_key_by_pid
            else pid
        )
        row = audit.setdefault(key, {
            "n_samples": 0,
            "first_sample_at": float(sampled_at),
            "last_sample_at": float(sampled_at),
            "observed_max_kb": 0,
        })
        if audit_key_by_pid is not None:
            if str(row.get("pid")) != str(pid):
                raise RuntimeError(
                    "worker swap audit launch-token/PID mapping drift"
                )
        if int(row["n_samples"]) == 0:
            row["first_sample_at"] = float(sampled_at)
        row["n_samples"] += 1
        row["last_sample_at"] = float(sampled_at)
        row["observed_max_kb"] = max(
            int(row["observed_max_kb"]), int(value)
        )


def register_worker_swap_audit(
    audit: MutableMapping[str, dict],
    *,
    pid: int,
    task_key: str,
    run_id: str,
    launch_token: str,
    launched_at: float,
) -> None:
    """Register one launch before it can contribute any resource evidence."""
    key = str(launch_token)
    if not key or key in audit:
        raise RuntimeError("duplicate or empty worker launch token")
    audit[key] = {
        "pid": int(pid),
        "task_key": str(task_key),
        "coordinator_run_id": str(run_id),
        "coordinator_launch_token": key,
        "launched_at": float(launched_at),
        "n_samples": 0,
        "first_sample_at": None,
        "last_sample_at": None,
        "observed_max_kb": 0,
    }


def record_final_worker_swap(
    audit: MutableMapping[str, dict],
    *,
    pid: int,
    launch_token: Optional[str] = None,
    value_kb: int,
    sampled_at: float,
) -> None:
    """Attach the child's at-publish self snapshot to its PID audit row."""
    key = str(launch_token) if launch_token is not None else str(int(pid))
    row = audit.setdefault(key, {
        "pid": int(pid),
        "n_samples": 0,
        "first_sample_at": None,
        "last_sample_at": None,
        "observed_max_kb": 0,
    })
    if int(row.get("pid", pid)) != int(pid):
        raise RuntimeError("final worker snapshot PID/launch-token mismatch")
    row["final_publish_sample_at"] = float(sampled_at)
    row["final_publish_swap_kb"] = int(value_kb)
    row["observed_max_kb"] = max(
        int(row["observed_max_kb"]), int(value_kb)
    )


def coordinator_identity_from_env() -> dict:
    """Return the coordinator identity inherited by one production cell."""
    run_id = os.environ.get(COORDINATOR_RUN_ENV)
    token = os.environ.get(COORDINATOR_TOKEN_ENV)
    if bool(run_id) != bool(token):
        raise RuntimeError("incomplete Phase-C coordinator environment")
    return {
        "coordinator_run_id": run_id,
        "coordinator_launch_token": token,
    }


def resource_receipt_path(artifact_path) -> Path:
    """Canonical immutable resource-audit receipt beside one JSON part."""
    path = Path(artifact_path)
    return path.with_name(path.name + ".resource_audit.json")


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_sha(payload: dict) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_resource_receipt(
    *,
    artifact_path,
    artifact_root,
    artifact_sha256: str,
    manifest_sha256: str,
    task_key: str,
    run_id: str,
    launch_token: str,
    pid: int,
    audit_row: dict,
    sampled_allowed_bytes: int,
) -> dict:
    """Build a receipt proving coordinator sampling for one terminal part."""
    n_live = int(audit_row.get("n_samples", 0))
    final_swap = audit_row.get("final_publish_swap_kb")
    observed_max = int(audit_row.get("observed_max_kb", 0))
    if n_live < 1:
        raise RuntimeError("terminal worker lacks a live VmSwap sample")
    if (
        int(audit_row.get("pid", -1)) != int(pid)
        or audit_row.get("task_key") != str(task_key)
        or audit_row.get("coordinator_run_id") != str(run_id)
        or audit_row.get("coordinator_launch_token") != str(launch_token)
    ):
        raise RuntimeError("worker audit identity does not match receipt")
    if final_swap != 0:
        raise RuntimeError("terminal worker lacks a zero pre-publish self sample")
    if observed_max * 1024 > int(sampled_allowed_bytes):
        raise RuntimeError("terminal worker exceeded sampled VmSwap allowance")
    body = {
        "schema": RESOURCE_RECEIPT_SCHEMA,
        "artifact_path": os.path.relpath(
            os.path.abspath(artifact_path), os.path.abspath(artifact_root)
        ),
        "artifact_sha256": str(artifact_sha256),
        "manifest_sha256": str(manifest_sha256),
        "task_key": str(task_key),
        "coordinator_run_id": str(run_id),
        "coordinator_launch_token": str(launch_token),
        "pid": int(pid),
        "n_live_samples": n_live,
        "first_live_sample_at": float(audit_row["first_sample_at"]),
        "last_live_sample_at": float(audit_row["last_sample_at"]),
        "sampled_observed_max_kb": observed_max,
        "pre_publish_self_snapshot_kb": int(final_swap),
        "sampled_allowed_bytes": int(sampled_allowed_bytes),
        "evidence_scope": (
            "periodic live VmSwap samples plus pre-publish child self "
            "snapshot; not a kernel peak measurement"
        ),
    }
    return {**body, "receipt_sha256": _object_sha(body)}


def publish_resource_receipt_once(path, receipt: dict) -> None:
    """Atomically publish one immutable receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resource receipt: {path}")
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def validate_resource_receipt(
    receipt_path,
    *,
    artifact_path,
    artifact_root,
    manifest_sha256: str,
    task_key: str,
) -> tuple[bool, str, Optional[dict]]:
    """Fail-closed validation used before reusing a production part."""
    path = Path(receipt_path)
    if not path.is_file():
        return False, "missing_resource_audit_receipt", None
    try:
        with path.open(encoding="utf-8") as handle:
            receipt = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"invalid_resource_audit_receipt:{exc}", None
    if not isinstance(receipt, dict):
        return False, "invalid_resource_audit_receipt_root", None
    claimed = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items()
            if key != "receipt_sha256"}
    try:
        self_hash_matches = claimed == _object_sha(body)
    except (TypeError, ValueError):
        self_hash_matches = False
    if not self_hash_matches:
        return False, "resource_audit_receipt_self_hash_mismatch", receipt
    artifact_path = Path(artifact_path)
    expected_relative = os.path.relpath(
        artifact_path.resolve(), Path(artifact_root).resolve()
    )
    try:
        artifact_sha = _sha256_file(artifact_path)
        with artifact_path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    except OSError as exc:
        return False, f"resource_audit_artifact_unreadable:{exc}", receipt
    except (ValueError, json.JSONDecodeError) as exc:
        return False, f"resource_audit_artifact_invalid_json:{exc}", receipt
    runtime = (
        artifact.get("runtime_provenance")
        if isinstance(artifact, dict) else None
    )
    try:
        contract_matches = (
            receipt.get("schema") == RESOURCE_RECEIPT_SCHEMA
            and receipt.get("artifact_path") == expected_relative
            and receipt.get("artifact_sha256") == artifact_sha
            and receipt.get("manifest_sha256") == manifest_sha256
            and receipt.get("task_key") == task_key
            and bool(receipt.get("coordinator_run_id"))
            and bool(receipt.get("coordinator_launch_token"))
            and int(receipt.get("n_live_samples", 0)) >= 1
            and receipt.get("sampled_observed_max_kb") == 0
            and receipt.get("pre_publish_self_snapshot_kb") == 0
            and receipt.get("sampled_allowed_bytes") == 0
            and isinstance(runtime, dict)
            and runtime.get("coordinator_run_id")
            == receipt.get("coordinator_run_id")
            and runtime.get("coordinator_launch_token")
            == receipt.get("coordinator_launch_token")
            and runtime.get("self_vm_swap_kb_at_publish") == 0
            and runtime.get("self_pid_at_publish") == receipt.get("pid")
        )
    except (TypeError, ValueError):
        contract_matches = False
    if not contract_matches:
        return False, "resource_audit_receipt_contract_mismatch", receipt
    return True, "valid_resource_audit_receipt", receipt


def block_coordinator_termination_signals():
    """Block handled termination signals during child ownership registration."""
    if not hasattr(signal, "pthread_sigmask"):
        raise RuntimeError(
            "Phase-C production requires pthread_sigmask for atomic ownership"
        )
    return signal.pthread_sigmask(
        signal.SIG_BLOCK, {signal.SIGHUP, signal.SIGTERM}
    )


def restore_coordinator_signal_mask(previous_mask) -> None:
    signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)


def terminate_owned_workers(rows: Iterable[dict], *, timeout_s: float = 10.0) -> None:
    """Best-effort cleanup limited to subprocesses owned by one coordinator."""
    materialized = list(rows)
    for row in materialized:
        process = row.get("proc", row.get("process"))
        if process is not None and process.poll() is None:
            process.terminate()
    for row in materialized:
        process = row.get("proc", row.get("process"))
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=float(timeout_s))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        handle = row.get("handle")
        if handle is not None and not handle.closed:
            handle.close()


def install_coordinator_signal_handlers() -> dict:
    """Turn HUP/TERM into normal Python exits so atexit cleanup runs."""
    previous = {}

    def _exit_from_signal(signum, _frame):
        raise SystemExit(f"Phase-C coordinator received signal {signum}")

    for signum in (signal.SIGHUP, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, _exit_from_signal)
    return previous


def restore_signal_handlers(previous: dict) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)
