#!/usr/bin/env python3
"""Durable end-to-end H2b v0.2 development cohort queue."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import socket
import threading
import time

import torch as _torch  # noqa: F401; load compatible native runtime first
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    H2B_V0_2_REVISION,
    V0_2_RESULT_ROOT,
    atomic_csv,
    atomic_json,
    sha256_file,
    utc_now,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SOURCE = Path("/home/honglab/leijiaxin/HFOsp")
CANONICAL_RESULT_ROOT = SOURCE / (
    "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2"
)
R1_ROOT = SOURCE / "results/epi_prssm/continuous_marked_state/r1"
PRIMARY_ARMS = ("B_history", "B_observation", "B_state", "memoryless")
WRONG_ARMS = (*PRIMARY_ARMS, "wrong_time")
PROBE_PRODUCTS = (
    "per_seed_probe_metrics.csv",
    "patient_median_probe_metrics.csv",
    "lead_curve.csv",
    "time_label_permutation.json",
    "positive_synthetic.json",
    "risk_probe_machine_audit.json",
)
CLAIM_FILENAME = ".task_claim.json"
COMPLETION_FILENAME = ".task_complete.json"
HEARTBEAT_SECONDS = 60
HEARTBEAT_STALE_SECONDS = 900


def _source_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
            text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _task_output_dir(root: Path, label: str) -> Path:
    subject, analysis = label.split("/", maxsplit=1)
    return root / "fits/by_subject" / subject / analysis


def _claim_path(root: Path, label: str) -> Path:
    return _task_output_dir(root, label) / CLAIM_FILENAME


def _completion_path(root: Path, label: str) -> Path:
    return _task_output_dir(root, label) / COMPLETION_FILENAME


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write beside the target and rename, so a reader never sees a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _foreign_writer_pids(root: Path, label: str) -> list[int]:
    """Find live probe processes already writing this task's output directory.

    The v0.2 run left three probes running with no claim at all -- two orphaned
    from a SIGSTOPped orchestrator, one started by hand outside the queue.  A
    claim file cannot see them, so ownership also asks the process table whether
    somebody is already writing here before taking the task.
    """
    target = str(_task_output_dir(root, label).resolve())
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
            continue
        parts = [chunk.decode("utf-8", "replace") for chunk in argv if chunk]
        if not any(chunk.endswith("run_risk_probe.py") for chunk in parts):
            continue
        if "--output-dir" not in parts:
            continue
        value = parts[parts.index("--output-dir") + 1]
        try:
            resolved = str(Path(value).resolve())
        except OSError:
            continue
        if resolved == target:
            found.append(pid)
    return sorted(found)


def _claim_probe_task(
        root: Path, label: str, risk_table: Path, *, command: list[str],
        ) -> dict | None:
    """Take exclusive ownership of one patient/arm, or decline.

    The claim is created with ``O_CREAT | O_EXCL`` so two workers racing on the
    same task cannot both believe they won -- a check-then-write would let both
    through.  A claim whose holder is still alive is never preempted, however old
    its heartbeat: the queue's own children were SIGSTOPped once while they kept
    computing, and stealing their task would have produced two writers for one
    output directory.
    """
    subject, analysis = label.split("/", maxsplit=1)
    foreign = _foreign_writer_pids(root, label)
    if foreign:
        return None
    path = _claim_path(root, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(risk_table)
    now = time.time()
    claim = {
        "patient_id": subject,
        "analysis": analysis,
        "task": label,
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "hostname": socket.gethostname(),
        "source_commit": _source_commit(),
        "revision": H2B_V0_2_REVISION,
        "risk_table_path": str(Path(risk_table).resolve()),
        "risk_table_sha256": digest,
        "command": list(command),
        "claimed_at_utc": utc_now(),
        "heartbeat_utc": utc_now(),
        "heartbeat_epoch": now,
        "output_dir": str(_task_output_dir(root, label)),
    }
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            held = {}
        if _pid_alive(held.get("pid", -1)):
            return None
        if float(now - float(held.get("heartbeat_epoch", 0.0))) < HEARTBEAT_STALE_SECONDS:
            return None
        if str(held.get("risk_table_sha256")) != digest:
            raise ValueError(
                f"{label}: stale claim was taken on a different input "
                f"({held.get('risk_table_sha256')} != {digest}); refusing to "
                "reclaim it blindly"
            )
        claim["reclaimed_from"] = {
            "pid": held.get("pid"), "pgid": held.get("pgid"),
            "hostname": held.get("hostname"),
            "heartbeat_utc": held.get("heartbeat_utc"),
        }
        _atomic_write_json(path, claim)
        return claim
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(claim, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    return claim


def _write_heartbeat(root: Path, label: str) -> None:
    path = _claim_path(root, label)
    try:
        claim = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if int(claim.get("pid", -1)) != os.getpid():
        return
    claim["heartbeat_utc"] = utc_now()
    claim["heartbeat_epoch"] = time.time()
    _atomic_write_json(path, claim)


def _release_claim(root: Path, label: str) -> None:
    path = _claim_path(root, label)
    try:
        claim = json.loads(path.read_text(encoding="utf-8"))
        if int(claim.get("pid", -1)) not in (os.getpid(), -1):
            return
    except FileNotFoundError:
        return
    except Exception:
        pass
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _output_digests(root: Path, label: str) -> dict[str, str]:
    output = _task_output_dir(root, label)
    return {
        name: sha256_file(output / name)
        for name in PROBE_PRODUCTS if (output / name).is_file()
    }


def _record_task_completion(
        root: Path, label: str, risk_table: Path, *,
        adopted: bool = False,
        metadata_migration: dict | None = None,
        ) -> dict:
    """Bind the finished products to their digests so a resume can verify them."""
    output = _task_output_dir(root, label)
    audit = _json(output / "risk_probe_machine_audit.json")
    receipt = {
        "status": "COMPLETE",
        "task": label,
        "revision": H2B_V0_2_REVISION,
        "recorded_utc": utc_now(),
        "risk_table_path": str(Path(risk_table).resolve()),
        "risk_table_sha256": sha256_file(risk_table),
        "execution_status": audit.get("execution_status", audit.get("status")),
        "scientific_estimability": audit.get("scientific_estimability"),
        "source_commit": _source_commit(),
        "output_sha256": _output_digests(root, label),
        "hash_baseline_adopted_from_existing_output": bool(adopted),
        "metadata_migration": metadata_migration,
    }
    _atomic_write_json(_completion_path(root, label), receipt)
    return receipt


def _upgrade_permutation_denominator_metadata(
        root: Path, label: str, risk_table: Path,
        ) -> dict | None:
    """Add explicit attempted/finite counts to intact legacy null products.

    Thirteen v0.2 probes finished before these two denominator fields were added.
    Their complete finite ``null_values`` vectors are the source of truth, so
    re-fitting for hours would change no scientific quantity.  This idempotent
    migration verifies that the vector length equals the pre-registered request,
    writes only the two counts into the full and compact audit payloads, and then
    re-binds every product hash in the task receipt.
    """
    output = _task_output_dir(root, label)
    permutation_path = output / "time_label_permutation.json"
    audit_path = output / "risk_probe_machine_audit.json"
    if not permutation_path.is_file() or not audit_path.is_file():
        return None
    permutation = _json(permutation_path)
    if permutation.get("status") != "COMPLETE":
        return None
    requested = int(permutation.get("n_permutations", -1))
    null_values = permutation.get("null_values") or []
    if requested < 1 or len(null_values) != requested:
        raise ValueError(
            f"{label}: cannot migrate permutation denominators because "
            f"{len(null_values)} null draws do not match request {requested}"
        )
    if not all(math.isfinite(float(value)) for value in null_values):
        raise ValueError(f"{label}: cannot migrate non-finite permutation draws")
    audit = _json(audit_path)
    if audit.get("status") != "COMPLETE":
        return None
    if (audit.get("input") or {}).get("risk_table_sha256") != sha256_file(risk_table):
        return None
    compact = dict(audit.get("time_label_permutation") or {})
    if compact.get("status") not in (None, "COMPLETE"):
        raise ValueError(f"{label}: compact permutation status drift")
    desired = {
        "n_permutations_run": requested,
        "n_finite_permutations": requested,
    }
    if (all(int(permutation.get(key, -1)) == value for key, value in desired.items())
            and all(int(compact.get(key, -1)) == value for key, value in desired.items())):
        return None
    before = {
        "time_label_permutation_sha256": sha256_file(permutation_path),
        "risk_probe_machine_audit_sha256": sha256_file(audit_path),
    }
    permutation.update(desired)
    compact.update({"status": "COMPLETE", **desired})
    migration = {
        "kind": "complete_null_denominator_metadata_only",
        "source_of_counts": "verified finite null_values vector length",
        "n_null_values": requested,
        "scientific_values_changed": False,
        "previous_sha256": before,
        "migrated_utc": utc_now(),
    }
    audit["time_label_permutation"] = compact
    audit["permutation_denominator_schema_migration"] = migration
    _atomic_write_json(permutation_path, permutation)
    _atomic_write_json(audit_path, audit)
    _record_task_completion(
        root, label, risk_table, metadata_migration=migration,
    )
    return migration


def _environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(REPO),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
    })
    return value


def _available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def _run_logged(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=REPO, env=_environment(),
            stdin=subprocess.DEVNULL, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
    return int(process.returncode)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    if pd.api.types.is_numeric_dtype(values):
        unique = set(values.dropna().astype(int).tolist())
        if unique.issubset({0, 1}):
            return values.astype(int).astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    if set(lowered.unique()).issubset({"true", "false"}):
        return lowered.map({"true": True, "false": False}).astype(bool)
    raise ValueError(f"support census column {name!r} is not strict boolean")


def _status(root: Path, stage: str, *, rows: list[dict] | None = None,
            status: str = "RUNNING", extra: dict | None = None) -> None:
    payload = {
        "status": status,
        "revision": H2B_V0_2_REVISION,
        "stage": stage,
        "updated_utc": utc_now(),
        "rows": rows or [],
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "paper_ready_figures_modified": False,
        **(extra or {}),
    }
    atomic_json(root / "QUEUE_STATUS.json", payload)


def _sync_results(root: Path, canonical: Path) -> None:
    if root.resolve() == canonical.resolve():
        return
    canonical.mkdir(parents=True, exist_ok=True)
    for source in root.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(root)
        target = canonical / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".sync_tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, target)


def _run_parallel(
        tasks: list[tuple[str, list[str], Path]], *, workers: int,
        ) -> list[dict]:
    def execute(label: str, command: list[str], log: Path) -> dict:
        code = _run_logged(command, log)
        return {"task": label, "returncode": code, "log": str(log)}

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(execute, label, command, log): label
            for label, command, log in tasks
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({
                    "task": futures[future], "returncode": 1,
                    "error": repr(exc),
                })
    return sorted(rows, key=lambda row: row["task"])


def _completed_probe_task(
        root: Path, label: str, risk_table: Path,
        ) -> dict | None:
    """Return a resumable success row only for a complete, input-bound probe.

    Probe fitting is the slowest CPU stage, so a durable restart must not refit a
    finished patient -- but the mere presence of an output directory is not
    enough.  Every declared product must exist, the risk-table digest recorded in
    the audit must still match the file on disk, and each product must still hash
    to the value bound at completion.

    A structurally unestimable patient counts as finished: the v0.2 contract §4
    keeps "not estimable" in the support denominator rather than treating it as a
    failure to retry.
    """
    output = _task_output_dir(root, label)
    required = tuple(output / name for name in PROBE_PRODUCTS)
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    migration = _upgrade_permutation_denominator_metadata(
        root, label, risk_table,
    )
    try:
        audit = _json(output / "risk_probe_machine_audit.json")
        if audit.get("status") != "COMPLETE":
            return None
        if audit.get("execution_status", "COMPLETE") != "COMPLETE":
            return None
        if (audit.get("boundary") or {}).get("revision") != H2B_V0_2_REVISION:
            return None
        if (audit.get("input") or {}).get("risk_table_sha256") != sha256_file(risk_table):
            return None
        if (audit.get("positive_synthetic") or {}).get("status") != "PASS":
            return None
    except Exception:
        return None

    receipt_path = _completion_path(root, label)
    if receipt_path.is_file():
        try:
            receipt = _json(receipt_path)
        except Exception:
            return None
        if receipt.get("risk_table_sha256") != sha256_file(risk_table):
            return None
        recorded = receipt.get("output_sha256") or {}
        observed = _output_digests(root, label)
        if not recorded or recorded != observed:
            return None
    else:
        # The probes finished before this receipt existed.  Adopt their digests
        # once, flagged as adopted rather than verified from birth, and enforce
        # them from here on.  Re-running an intact task would only burn hours.
        receipt = _record_task_completion(root, label, risk_table, adopted=True)
    return {
        "task": label,
        "returncode": 0,
        "status": "SKIPPED_COMPLETE_INPUT_BOUND",
        "scientific_estimability": receipt.get("scientific_estimability")
            or audit.get("scientific_estimability"),
        "hash_baseline_adopted_from_existing_output": bool(
            receipt.get("hash_baseline_adopted_from_existing_output")
        ),
        "permutation_denominator_metadata_migrated": migration is not None,
        "audit": str(output / "risk_probe_machine_audit.json"),
    }


def _execute_probe_task(
        root: Path, label: str, command: list[str], log: Path,
        ) -> dict:
    """Run one claimed probe, keeping its heartbeat alive for the whole fit."""
    risk_table = Path(command[command.index("--risk-table") + 1])
    already = _completed_probe_task(root, label, risk_table)
    if already is not None:
        return already
    claim = _claim_probe_task(root, label, risk_table, command=command)
    if claim is None:
        return {
            "task": label,
            "returncode": 0,
            "status": "SKIPPED_OWNED_BY_ANOTHER_LIVE_WRITER",
            "foreign_writer_pids": _foreign_writer_pids(root, label),
        }
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(HEARTBEAT_SECONDS):
            _write_heartbeat(root, label)

    heart = threading.Thread(target=beat, daemon=True)
    heart.start()
    try:
        code = _run_logged(command, log)
        row = {"task": label, "returncode": code, "log": str(log)}
        if code == 0:
            receipt = _record_task_completion(root, label, risk_table)
            row["status"] = "COMPLETE"
            row["scientific_estimability"] = receipt.get("scientific_estimability")
        else:
            row["status"] = "FAILED"
        return row
    finally:
        stop.set()
        _release_claim(root, label)


def _run_probe_tasks(
        root: Path, tasks: list[tuple[str, list[str], Path]], *, workers: int,
        ) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_execute_probe_task, root, label, command, log): label
            for label, command, log in tasks
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append({
                    "task": futures[future], "returncode": 1,
                    "status": "FAILED", "error": repr(exc),
                })
    return sorted(rows, key=lambda row: row["task"])


def _upstream_paths(subject: str, root: Path) -> dict[str, Path] | None:
    candidates = [
        root / "upstream_r1_2",
        R1_ROOT / "r1_7b_cohort_extension/upstream_r1_2",
        R1_ROOT / "r1_7a/upstream_r1_2",
    ]
    for base in candidates:
        paths = {
            "design": base / "cache" / subject / "full_design.npz",
            "manifest": base / "cache" / subject / "manifest.json",
            "baseline": base / "baselines" / subject / "seed_0/models.pt",
            "scaler": base / "bridge_e1" / subject / "seed_0/result.json",
        }
        if all(path.is_file() for path in paths.values()):
            return paths
    return None


def _combine_tables(paths: list[Path], output: Path) -> Path | None:
    readable = [path for path in paths if path.is_file() and path.stat().st_size > 0]
    if not readable:
        return None
    frames = [pd.read_csv(path) for path in readable]
    frame = pd.concat(frames, ignore_index=True, sort=False)
    atomic_csv(
        output,
        frame.where(pd.notna(frame), None).to_dict(orient="records"),
        fieldnames=list(frame.columns),
    )
    atomic_json(output.with_suffix(".manifest.json"), {
        "status": "COMPLETE",
        "revision": "h2b_v0_2_combined_risk_table_v1",
        "created_utc": utc_now(),
        "inputs": {str(path): sha256_file(path) for path in readable},
        "n_subjects": int(frame["patient_id"].astype(str).nunique()),
        "n_rows": int(len(frame)),
        "n_risk_sets": int(frame["risk_set_id"].astype(str).nunique()),
        "output_sha256": sha256_file(output),
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    })
    return output


def _combine_probe_outputs(
        root: Path, subjects: list[str], *, analysis: str,
        ) -> dict | None:
    source_dirs = [
        root / "fits/by_subject" / subject / analysis for subject in subjects
    ]
    available = [
        directory for directory in source_dirs
        if (directory / "risk_probe_machine_audit.json").is_file()
    ]
    if not available:
        return None
    output = root / "fits" / analysis
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "per_seed": "per_seed_probe_metrics.csv",
        "patient": "patient_median_probe_metrics.csv",
        "lead": "lead_curve.csv",
    }
    outputs = {}
    for label, name in files.items():
        paths = [directory / name for directory in available if (directory / name).is_file()]
        frames = [pd.read_csv(path) for path in paths]
        frame = pd.concat(frames, ignore_index=True, sort=False)
        destination = output / name
        atomic_csv(
            destination,
            frame.where(pd.notna(frame), None).to_dict(orient="records"),
            fieldnames=list(frame.columns),
        )
        outputs[label] = {
            "path": str(destination), "sha256": sha256_file(destination),
            "n_rows": int(len(frame)),
        }
    audits = {
        directory.parent.name: {
            "path": str(directory / "risk_probe_machine_audit.json"),
            "sha256": sha256_file(directory / "risk_probe_machine_audit.json"),
        }
        for directory in available
    }
    payload = {
        "status": "COMPLETE",
        "revision": "h2b_v0_2_patient_separate_probe_aggregation_v1",
        "created_utc": utc_now(),
        "analysis": analysis,
        "n_patients": len(available),
        "patient_audits": audits,
        "outputs": outputs,
        "heterogeneous_patient_feature_dimensions_never_pooled": True,
        "seed_aggregation": "median_within_patient",
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(output / "cohort_probe_index.json", payload)
    return payload


def _combine_phenotype_outputs(
        root: Path, subjects: list[str],
        ) -> dict:
    available = [
        root / "fits/by_subject" / subject / "phenotype"
        for subject in subjects
        if (root / "fits/by_subject" / subject / "phenotype"
            / "phenotype_transfer_machine_audit.json").is_file()
    ]
    output = root / "fits/phenotype"
    output.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for label, name in (
        ("per_seed", "per_seed_phenotype_metrics.csv"),
        ("patient", "patient_median_phenotype_metrics.csv"),
    ):
        paths = [directory / name for directory in available if (directory / name).is_file()]
        frames = [pd.read_csv(path) for path in paths if path.stat().st_size > 1]
        destination = output / name
        if frames:
            frame = pd.concat(frames, ignore_index=True, sort=False)
            atomic_csv(
                destination,
                frame.where(pd.notna(frame), None).to_dict(orient="records"),
                fieldnames=list(frame.columns),
            )
            outputs[label] = {
                "path": str(destination), "sha256": sha256_file(destination),
                "n_rows": int(len(frame)),
            }
    payload = {
        "status": "COMPLETE" if available else "NOT_ESTIMABLE_NO_FROZEN_TARGET",
        "revision": "h2b_v0_2_patient_separate_phenotype_aggregation_v1",
        "created_utc": utc_now(),
        "n_patients_run": len(available),
        "patient_audits": {
            directory.parent.name: {
                "path": str(directory / "phenotype_transfer_machine_audit.json"),
                "sha256": sha256_file(
                    directory / "phenotype_transfer_machine_audit.json"
                ),
            }
            for directory in available
        },
        "outputs": outputs,
        "target_reclustered": False,
        "heterogeneous_patient_feature_dimensions_never_pooled": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    atomic_json(output / "cohort_phenotype_index.json", payload)
    return payload


def run(args: argparse.Namespace) -> dict:
    root = Path(args.result_root).resolve()
    canonical = Path(args.canonical_result_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "QUEUE.lock"
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("H2b v0.2 queue is already running") from exc
    _status(root, "PREFLIGHT")

    inventory_path = root / "manifests/r1_7_checkpoint_inventory.json"
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    inventory = _json(inventory_path)
    if inventory.get("status") != "COMPLETE":
        raise ValueError("R1.7 checkpoint inventory is not COMPLETE")
    inventory_hash = sha256_file(inventory_path)

    census_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/build_v02_support_census.py",
        "--inventory", str(inventory_path), "--output-root", str(root),
    ]
    if _run_logged(census_command, root / "logs/support_census.log") != 0:
        raise RuntimeError("support census failed")
    census = _json(root / "manifests/support_census.json")
    if not all(census.get("raw_mounts_present", {}).values()):
        _status(
            root, "WAITING_FOR_DATA_MOUNTS", status="WAITING",
            extra={"raw_mounts_present": census.get("raw_mounts_present")},
        )
        _sync_results(root, canonical)
        return {"status": "WAITING_FOR_DATA_MOUNTS"}
    raw_required = int(census.get("n_subjects_requiring_raw_for_primary_h2b", 0))
    raw_ready = int(census.get("n_required_subjects_with_raw_cache", 0))
    if raw_required < 1 or raw_ready != raw_required:
        _status(
            root, "WAITING_FOR_REQUIRED_RAW_CACHES", status="WAITING",
            extra={"n_required": raw_required, "n_ready": raw_ready},
        )
        _sync_results(root, canonical)
        return {"status": "WAITING_FOR_REQUIRED_RAW_CACHES"}

    patients = pd.read_csv(root / "manifests/patient_support_census.csv")
    for name in (
        "coverage_available", "upstream_design_available",
        "raw_inference_cache_available",
    ):
        patients[name] = _strict_bool(patients[name], name)
    candidates = patients[
        (patients["n_checkpoint_available_seeds"].astype(int) > 0)
        & (patients["n_seizures_in_frozen_inventory"].astype(int) > 0)
        & (patients["primary_complete_coverage_seizures"].astype(int) > 0)
        & patients["coverage_available"]
    ].copy()
    missing_upstream = candidates[
        ~candidates["upstream_design_available"]
        & (candidates["primary_complete_coverage_seizures"].astype(int) > 0)
    ]["subject"].astype(str).tolist()
    if missing_upstream:
        _status(root, "REBUILD_MISSING_UPSTREAM", extra={"subjects": missing_upstream})
        command = [
            str(PYTHON),
            "scripts/topic5_continuous_marked_state_h2b/prepare_v02_upstream.py",
            "--subjects", *missing_upstream,
            "--result-root", str(root), "--gpus", *map(str, args.gpus),
        ]
        if _run_logged(command, root / "logs/upstream_preparation_queue.log") != 0:
            raise RuntimeError("missing R1.7B upstream rebuild/equivalence failed")
        if _run_logged(census_command, root / "logs/support_census.log") != 0:
            raise RuntimeError("post-preparation support census failed")
        patients = pd.read_csv(root / "manifests/patient_support_census.csv")
        for name in (
            "coverage_available", "upstream_design_available",
            "raw_inference_cache_available",
        ):
            patients[name] = _strict_bool(patients[name], name)

    ready = patients[
        (patients["n_checkpoint_available_seeds"].astype(int) > 0)
        & (patients["n_seizures_in_frozen_inventory"].astype(int) > 0)
        & (patients["primary_complete_coverage_seizures"].astype(int) > 0)
        & patients["coverage_available"]
        & patients["upstream_design_available"]
        & patients["raw_inference_cache_available"]
    ].copy()
    if ready.empty:
        raise RuntimeError("no checkpoint-available seizure subject is runnable")

    _status(root, "PREPARE_SUBJECT_QUERIES", extra={
        "subjects": ready["subject"].astype(str).tolist(),
    })
    query_tasks = []
    for row in ready.itertuples(index=False):
        subject = str(row.subject)
        upstream = _upstream_paths(subject, root)
        if upstream is None:
            raise RuntimeError(f"{subject}: upstream disappeared after census")
        command = [
            str(PYTHON),
            "scripts/topic5_continuous_marked_state_h2b/prepare_v02_subject_inputs.py",
            "--subject", subject,
            "--seizure-crosswalk", str(root / "manifests/seizure_crosswalk.csv"),
            "--coverage", str(row.coverage_path),
            "--design", str(upstream["design"]),
            "--design-sha256", sha256_file(upstream["design"]),
            "--design-manifest", str(upstream["manifest"]),
            "--result-root", str(root),
        ]
        query_tasks.append((
            subject, command, root / "logs/query_inputs" / f"{subject}.log",
        ))
    query_rows = _run_parallel(
        query_tasks, workers=min(int(args.cpu_workers), len(query_tasks)),
    )
    _status(root, "PREPARE_SUBJECT_QUERIES", rows=query_rows)
    if any(row["returncode"] != 0 for row in query_rows):
        raise RuntimeError("one or more subject query preparations failed")

    input_manifests = {}
    for subject in ready["subject"].astype(str):
        path = root / "risk_sets" / subject / "input_manifest.json"
        value = _json(path)
        input_manifests[subject] = value
    extraction_subjects = [
        subject for subject, value in input_manifests.items()
        if int(value["n_primary_eligible_seizures"]) > 0
    ]
    entries = [
        row for row in inventory["entries"]
        if row["checkpoint_available"] and row["subject"] in extraction_subjects
    ]
    _status(root, "EXTRACT_FROZEN_STATES", extra={
        "n_subjects": len(extraction_subjects), "n_cells": len(entries),
    })
    extraction_tasks = []
    for entry in entries:
        subject = str(entry["subject"])
        seed = int(entry["seed"])
        upstream = _upstream_paths(subject, root)
        manifest = input_manifests[subject]
        output = root / "state_cache" / subject / f"seed_{seed}/states.npz"
        output_manifest = output.with_suffix(".manifest.json")
        if output.is_file() and output_manifest.is_file():
            try:
                cached = _json(output_manifest)
                if sha256_file(output) == cached.get("cache_sha256"):
                    continue
            except Exception:
                pass
        command = [
            str(PYTHON),
            "scripts/topic5_continuous_marked_state_h2b/extract_states.py",
            "--subject", subject, "--seed", str(seed),
            "--checkpoint", str(entry["checkpoint_path"]),
            "--checkpoint-sha256", str(entry["checkpoint_sha256"]),
            "--allow-unstable-complete",
            "--queries", str(manifest["query_path"]),
            "--global-exclusions", str(manifest["global_exclusion_path"]),
            "--source-repo-root", str(SOURCE),
            "--design-path", str(upstream["design"]),
            "--design-sha256", sha256_file(upstream["design"]),
            "--design-manifest", str(upstream["manifest"]),
            "--coverage-path", str(manifest["coverage_path"]),
            "--coverage-sha256", str(manifest["coverage_sha256"]),
            "--history-baseline-path", str(upstream["baseline"]),
            "--history-baseline-sha256", sha256_file(upstream["baseline"]),
            "--explicit-scaler-result", str(upstream["scaler"]),
            "--explicit-scaler-result-sha256", sha256_file(upstream["scaler"]),
            "--checkpoint-inventory", str(inventory_path),
            "--checkpoint-inventory-sha256", inventory_hash,
            "--h2b-revision", H2B_V0_2_REVISION,
            "--embedding-batch-size", str(args.embedding_batch_size),
            "--output", str(output),
        ]
        extraction_tasks.append((
            f"{subject}/seed_{seed}", command,
            root / "logs/state_extraction" / f"{subject}_seed_{seed}.log",
        ))
    memory_workers = max(1, min(
        int(args.cpu_workers),
        int(max(1.0, math.floor((_available_gib() - 32.0) / 16.0))),
    ))
    extraction_rows = _run_parallel(extraction_tasks, workers=memory_workers)
    _status(root, "EXTRACT_FROZEN_STATES", rows=extraction_rows, extra={
        "workers": memory_workers,
        "host_memory_reservation_gib_per_worker": 16,
    })
    failed = [row for row in extraction_rows if row["returncode"] != 0]
    if failed:
        # Retry failures serially with a smaller embedding batch.  Completed
        # caches are atomic and are never rerun.
        retry_tasks = []
        by_label = {label: (command, log) for label, command, log in extraction_tasks}
        for row in failed:
            command, log = by_label[row["task"]]
            command = list(command)
            index = command.index("--embedding-batch-size") + 1
            command[index] = "32"
            retry_tasks.append((row["task"], command, log))
        retry_rows = _run_parallel(retry_tasks, workers=1)
        extraction_rows.extend({**row, "retry": True} for row in retry_rows)
        if any(row["returncode"] != 0 for row in retry_rows):
            _status(root, "EXTRACT_FROZEN_STATES", rows=extraction_rows, status="FAIL")
            raise RuntimeError("frozen state extraction failed after serial retry")

    _status(root, "BUILD_RISK_TABLES")
    risk_tasks = []
    for subject in extraction_subjects:
        command = [
            str(PYTHON),
            "scripts/topic5_continuous_marked_state_h2b/build_v02_risk_tables.py",
            "--subject", subject, "--result-root", str(root),
            "--controls-per-case", str(args.controls_per_case),
        ]
        risk_tasks.append((
            subject, command, root / "logs/risk_tables" / f"{subject}.log",
        ))
    risk_rows = _run_parallel(risk_tasks, workers=min(4, len(risk_tasks)))
    _status(root, "BUILD_RISK_TABLES", rows=risk_rows)
    if any(row["returncode"] != 0 for row in risk_rows):
        raise RuntimeError("one or more primary risk tables failed")

    _status(root, "FIT_LOW_CAPACITY_PROBES")
    probe_subjects = [
        subject for subject in extraction_subjects
        if int(input_manifests[subject]["n_primary_eligible_seizures"]) >= 2
    ]
    probe_tasks = []
    for subject in probe_subjects:
        primary_path = root / "risk_sets" / subject / "primary_risk_sets.csv"
        probe_tasks.append((
            f"{subject}/primary",
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
             "--risk-table", str(primary_path),
             "--output-dir", str(root / "fits/by_subject" / subject / "primary"),
             "--arms", *PRIMARY_ARMS,
             "--h2b-revision", H2B_V0_2_REVISION,
             "--n-permutations", str(args.n_permutations), "--overwrite"],
            root / "logs/probes" / f"{subject}_primary.log",
        ))
        wrong_path = root / "risk_sets" / subject / "matched_wrong_time_risk_sets.csv"
        if wrong_path.is_file():
            probe_tasks.append((
                f"{subject}/matched_wrong_time",
                [str(PYTHON),
                 "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
                 "--risk-table", str(wrong_path),
                 "--output-dir", str(
                     root / "fits/by_subject" / subject / "matched_wrong_time"
                 ),
                 "--arms", *WRONG_ARMS,
                 "--h2b-revision", H2B_V0_2_REVISION,
                 "--n-permutations", str(args.n_permutations), "--overwrite"],
                root / "logs/probes" / f"{subject}_matched_wrong_time.log",
            ))
    pending_probe_tasks = []
    skipped_probe_rows = []
    for label, command, log in probe_tasks:
        risk_table = Path(command[command.index("--risk-table") + 1])
        completed = _completed_probe_task(root, label, risk_table)
        if completed is None:
            pending_probe_tasks.append((label, command, log))
        else:
            skipped_probe_rows.append(completed)
    probe_workers = min(int(args.cpu_workers), max(1, len(pending_probe_tasks)))
    probe_rows = sorted(
        skipped_probe_rows + _run_probe_tasks(
            root, pending_probe_tasks, workers=probe_workers,
        ),
        key=lambda row: row["task"],
    )
    _status(root, "FIT_LOW_CAPACITY_PROBES", rows=probe_rows, extra={
        "workers": int(probe_workers),
        "n_expected_probe_tasks": len(probe_tasks),
        "n_skipped_complete_input_bound": len(skipped_probe_rows),
        "n_submitted": len(pending_probe_tasks),
        "n_not_estimable": sum(
            1 for row in probe_rows
            if row.get("scientific_estimability") == "NOT_ESTIMABLE"
        ),
    })
    if any(row["returncode"] != 0 for row in probe_rows):
        raise RuntimeError("one or more cohort probes failed")
    unowned = [
        row["task"] for row in probe_rows
        if row.get("status") == "SKIPPED_OWNED_BY_ANOTHER_LIVE_WRITER"
    ]
    if unowned:
        raise RuntimeError(
            "another live writer still owns these probe tasks; rerun after it "
            f"finishes rather than producing a second writer: {unowned}"
        )
    primary_index = _combine_probe_outputs(
        root, probe_subjects, analysis="primary",
    )
    wrong_index = _combine_probe_outputs(
        root, probe_subjects, analysis="matched_wrong_time",
    )
    if primary_index is None:
        raise RuntimeError("no patient has at least two primary 30-min seizures")

    _status(root, "SECONDARY_FROZEN_PHENOTYPE")
    phenotype_build = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/build_v02_phenotype_targets.py",
        "--result-root", str(root),
    ]
    if _run_logged(
        phenotype_build, root / "logs/phenotype/build_targets.log"
    ) != 0:
        raise RuntimeError("frozen phenotype target join failed")
    phenotype_manifest = _json(root / "reports/phenotype_target_availability.json")
    phenotype_tasks = []
    phenotype_subjects = []
    for subject, table in phenotype_manifest.get("subject_tables", {}).items():
        if int(table.get("n_available_target_rows", 0)) < 1:
            continue
        phenotype_subjects.append(str(subject))
        phenotype_tasks.append((
            str(subject),
            [str(PYTHON),
             "scripts/topic5_continuous_marked_state_h2b/run_phenotype_transfer.py",
             "--input", str(table["path"]),
             "--output-dir", str(
                 root / "fits/by_subject" / str(subject) / "phenotype"
             ),
             "--h2b-revision", H2B_V0_2_REVISION,
             "--overwrite"],
            root / "logs/phenotype" / f"{subject}.log",
        ))
    phenotype_rows = _run_parallel(
        phenotype_tasks, workers=min(2, len(phenotype_tasks)),
    )
    if any(row["returncode"] != 0 for row in phenotype_rows):
        raise RuntimeError("one or more frozen phenotype probes failed")
    phenotype_index = _combine_phenotype_outputs(
        root, phenotype_subjects,
    )
    _status(root, "SECONDARY_FROZEN_PHENOTYPE", rows=phenotype_rows, extra={
        "phenotype_status": phenotype_index["status"],
    })

    _status(root, "AGGREGATE_PATIENT_FIRST")
    aggregate_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/aggregate_v02_results.py",
        "--result-root", str(root),
    ]
    if _run_logged(
        aggregate_command, root / "logs/aggregate_patient_first.log"
    ) != 0:
        raise RuntimeError("patient-first aggregation failed")

    _status(root, "AUDIT_R1_7B_SOURCE")
    upstream_audit_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/audit_r1_7b_release.py",
        "--result-root", str(root),
    ]
    if _run_logged(
        upstream_audit_command, root / "logs/r1_7b_consumer_audit.log"
    ) != 0:
        raise RuntimeError("R1.7B consumer-side source audit failed")
    upstream_audit = _json(
        root / "reports/r1_7b_consumer_acceptance_audit.json"
    )
    if upstream_audit.get("status") != "PASS_EXPLORATORY_DEVELOPMENT_SOURCE":
        raise RuntimeError("R1.7B was not accepted as an exploratory source")

    _status(root, "AUDIT_RESULTS")
    audit_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/audit_v02_results.py",
        "--result-root", str(root), "--allow-precompletion",
    ]
    if _run_logged(audit_command, root / "logs/machine_audit.log") != 0:
        raise RuntimeError("pre-completion H2b v0.2 machine audit failed")
    preaudit = _json(root / "reports/machine_audit.json")
    if preaudit.get("status") != "PASS_PRECOMPLETION":
        raise RuntimeError("pre-completion audit did not issue PASS_PRECOMPLETION")

    completed = {
        "status": "COMPLETE",
        "revision": H2B_V0_2_REVISION,
        "created_utc": utc_now(),
        "n_checkpoint_available_subjects_with_seizure_inventory": int(len(ready)),
        "n_state_extraction_subjects": len(extraction_subjects),
        "n_checkpoint_seed_caches": len(entries),
        "n_probe_subjects": len(probe_subjects),
        "primary_probe_index": str(root / "fits/primary/cohort_probe_index.json"),
        "primary_probe_index_sha256": sha256_file(
            root / "fits/primary/cohort_probe_index.json"
        ),
        "matched_wrong_time_probe_index": (
            str(root / "fits/matched_wrong_time/cohort_probe_index.json")
            if wrong_index is not None else None
        ),
        "h1_stability_used_as_gate": False,
        "state_and_observer_frozen": True,
        "seizure_loss_updates_state": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "h3_or_t2_run": False,
        "paper_ready_figures_modified": False,
    }
    atomic_json(root / "COHORT_RUN_COMPLETE.json", completed)
    _status(root, "COHORT_RUN_COMPLETE", status="COMPLETE", extra=completed)
    final_audit_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/audit_v02_results.py",
        "--result-root", str(root),
    ]
    if _run_logged(final_audit_command, root / "logs/machine_audit.log") != 0:
        raise RuntimeError("final H2b v0.2 machine audit failed")
    final_audit = _json(root / "reports/machine_audit.json")
    if final_audit.get("status") != "PASS_COMPLETE":
        raise RuntimeError("final audit did not issue PASS_COMPLETE")
    completed["machine_audit_path"] = str(root / "reports/machine_audit.json")
    completed["machine_audit_sha256"] = sha256_file(
        root / "reports/machine_audit.json"
    )
    report_command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_h2b/build_v02_reports.py",
        "--result-root", str(root),
    ]
    if _run_logged(report_command, root / "logs/final_reports.log") != 0:
        raise RuntimeError("H2b v0.2 plain/technical report generation failed")
    completed["reports"] = {
        name: {
            "path": str(path), "sha256": sha256_file(path),
        }
        for name, path in {
            "plain": root / "reports/h2b_cross_task_v0_2_plain.md",
            "technical": root / "reports/h2b_cross_task_v0_2_technical.md",
            "handoff": root / "CURRENT_HANDOFF.md",
        }.items()
    }
    atomic_json(root / "COHORT_RUN_COMPLETE.json", completed)
    _status(root, "COHORT_RUN_COMPLETE", status="COMPLETE", extra=completed)
    _sync_results(root, canonical)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=V0_2_RESULT_ROOT)
    parser.add_argument(
        "--canonical-result-root", type=Path, default=CANONICAL_RESULT_ROOT,
    )
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--gpus", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--controls-per-case", type=int, default=5)
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as exc:
        root = Path(args.result_root).resolve()
        _status(root, "QUEUE_FAILED", status="FAIL", extra={"error": repr(exc)})
        _sync_results(root, Path(args.canonical_result_root).resolve())
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
