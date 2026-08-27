#!/usr/bin/env python3
"""Read-only watcher for the R1.7 machine-acceptance gate.

The watcher never imports code from the R1.7 worktree and never writes there.
It records a snapshot under the isolated H2b result root and exits only when all
handoff gates are true (or when ``--once`` is requested).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (
    RESULT_ROOT,
    R1_7_MACHINE_AUDIT,
    R1_7_WORKTREE,
    atomic_json,
    sha256_file,
)


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(R1_7_WORKTREE), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def _path_hash_pairs(value: Any) -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        path = value.get("path") or value.get("checkpoint")
        digest = value.get("sha256") or value.get("checkpoint_sha256")
        if isinstance(path, str) and isinstance(digest, str) and len(digest) == 64:
            yield path, digest
        for child in value.values():
            yield from _path_hash_pairs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _path_hash_pairs(child)


def _resolve_readonly(path: str) -> Path:
    source = Path(path)
    candidates = [source] if source.is_absolute() else [
        R1_7_WORKTREE / source,
        R1_7_MACHINE_AUDIT.parent.parent / source,
    ]
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def snapshot() -> dict[str, Any]:
    status = _run_git("status", "--porcelain")
    head = _run_git("rev-parse", "HEAD")
    branch = _run_git("branch", "--show-current")
    upstream = _run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    ahead = None
    if upstream.returncode == 0:
        counts = _run_git("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
        if counts.returncode == 0:
            behind_text, ahead_text = counts.stdout.strip().split()
            ahead = int(ahead_text)
            behind = int(behind_text)
        else:
            behind = None
    else:
        behind = None

    fit_root = R1_7_MACHINE_AUDIT.parent.parent / "fits"
    fit_results = sorted(fit_root.rglob("result.json")) if fit_root.exists() else []
    payload: dict[str, Any] | None = None
    audit_error = None
    if R1_7_MACHINE_AUDIT.exists():
        try:
            payload = json.loads(R1_7_MACHINE_AUDIT.read_text())
        except Exception as exc:  # fail closed; preserve exact error in watcher output
            audit_error = f"{type(exc).__name__}: {exc}"

    pairs = list(dict.fromkeys(_path_hash_pairs(payload))) if payload else []
    result_payloads: list[dict[str, Any]] = []
    if payload:
        # The R1.7 audit authenticates each result.json; the actual model path and
        # digest live inside that result.  Only follow a result after its own
        # audit digest has been reproduced, then independently hash the model.
        for row in payload.get("r1_fits", []):
            if not isinstance(row, dict):
                continue
            raw_path = row.get("path")
            expected = row.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(expected, str):
                continue
            result_path = _resolve_readonly(raw_path)
            if not result_path.is_file() or sha256_file(result_path) != expected:
                continue
            try:
                result_payload = json.loads(result_path.read_text())
            except Exception:
                continue
            if isinstance(result_payload, dict):
                result_payloads.append(result_payload)
                pairs.extend(_path_hash_pairs(result_payload))
    pairs = list(dict.fromkeys(pairs))
    hash_rows = []
    for raw_path, expected in pairs:
        source = _resolve_readonly(raw_path)
        actual = sha256_file(source) if source.is_file() else None
        hash_rows.append({
            "path": str(source),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "match": actual == expected,
        })
    checkpoint_rows = [row for row in hash_rows if row["path"].endswith((".pt", ".pth"))]
    source_rows = [row for row in hash_rows if row not in checkpoint_rows]

    audit_complete = bool(payload and payload.get("status") == "COMPLETE")
    audit_boundary = (payload or {}).get("boundary") or {}
    formal_value = (payload or {}).get(
        "formal_test_partition_opened",
        audit_boundary.get("formal_test_partition_opened"),
    )
    sealed_value = (payload or {}).get(
        "sealed_opened", audit_boundary.get("sealed_opened"),
    )
    formal_false = bool(payload and formal_value is False)
    sealed_false = bool(payload and sealed_value is False)
    declared_source_payloads = (payload or {}).get("r1_source_payloads") or []
    result_source_payloads = [item.get("source_hashes") for item in result_payloads]
    source_payloads_consistent = bool(declared_source_payloads) and bool(result_source_payloads)
    source_payloads_consistent = source_payloads_consistent and all(
        item in declared_source_payloads for item in result_source_payloads
    )
    source_hashes_consistent = (
        bool(source_rows)
        and all(row["match"] for row in source_rows)
        and source_payloads_consistent
    )
    checkpoint_hashes_consistent = bool(checkpoint_rows) and all(
        row["match"] for row in checkpoint_rows
    )
    code_committed = status.returncode == 0 and status.stdout.strip() == ""
    code_pushed = upstream.returncode == 0 and ahead == 0
    gates = {
        "machine_audit_complete": audit_complete,
        "exactly_50_fits": len(fit_results) == 50,
        "formal_false": formal_false,
        "sealed_false": sealed_false,
        "source_hashes_present_and_consistent": source_hashes_consistent,
        "checkpoint_hashes_present_and_consistent": checkpoint_hashes_consistent,
        "code_committed": code_committed,
        "code_pushed": code_pushed,
    }
    return {
        "status": "READY" if all(gates.values()) else "WAITING",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "watcher_pid": os.getpid(),
        "r1_7_worktree": str(R1_7_WORKTREE),
        "machine_audit": str(R1_7_MACHINE_AUDIT),
        "machine_audit_exists": R1_7_MACHINE_AUDIT.exists(),
        "machine_audit_sha256": (
            sha256_file(R1_7_MACHINE_AUDIT) if R1_7_MACHINE_AUDIT.is_file() else None
        ),
        "machine_audit_error": audit_error,
        "fit_result_count": len(fit_results),
        "git": {
            "head": head.stdout.strip() if head.returncode == 0 else None,
            "branch": branch.stdout.strip() if branch.returncode == 0 else None,
            "upstream": upstream.stdout.strip() if upstream.returncode == 0 else None,
            "ahead": ahead,
            "behind": behind,
            "dirty_paths": status.stdout.splitlines() if status.returncode == 0 else [],
        },
        "gates": gates,
        "hash_audit": hash_rows,
        "source_payloads_consistent": source_payloads_consistent,
        "authenticated_result_payload_count": len(result_payloads),
        "r1_7_outputs_authorized_for_h2b": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "manifests/r1_7_watch.json"
    )
    args = parser.parse_args()
    if args.interval_seconds < 60 and not args.once:
        raise SystemExit("watch interval must be at least 60 seconds")
    while True:
        current = snapshot()
        atomic_json(args.output, current)
        print(json.dumps({
            "checked_utc": current["checked_utc"],
            "status": current["status"],
            "fit_result_count": current["fit_result_count"],
            "gates": current["gates"],
        }, sort_keys=True), flush=True)
        if args.once or current["status"] == "READY":
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
