#!/usr/bin/env python3
"""Freeze the read-only R1.7 handoff decision used by H2b v0.1."""
from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.topic5_continuous_marked_state_h2b.contract import (
    RESULT_ROOT, atomic_json, sha256_file, utc_now,
)


def main() -> None:
    watch_path = RESULT_ROOT / "manifests/r1_7_watch.json"
    watch = json.loads(watch_path.read_text(encoding="utf-8"))
    ready = watch.get("r1_7_outputs_authorized_for_h2b") is True
    gates = watch.get("gates") or {}
    failed = sorted(name for name, value in gates.items() if value is not True)
    payload = {
        "status": "READY_FOR_IMPORT" if ready else "UNAVAILABLE_NOT_USED",
        "created_utc": utc_now(),
        "watch_snapshot": str(watch_path),
        "watch_snapshot_sha256": sha256_file(watch_path),
        "fit_result_count_observed": watch.get("fit_result_count"),
        "machine_audit_exists": watch.get("machine_audit_exists"),
        "failed_release_gates": failed,
        "r1_7_outputs_referenced_by_h2b": False,
        "r1_7_uncommitted_code_loaded_or_used_by_h2b": False,
        "r1_7_worktree_modified_by_h2b": False,
        "reason": (
            "R1.7 has not passed every release gate; H2b v0.1 closes with the "
            "independent E384 instrument only and does not import partial fits."
            if not ready else
            "R1.7 release gates passed; a separate hash-bound import is required."
        ),
        "scientific_effect": (
            "No checkpoint-available cohort H2b estimate is made in v0.1."
            if not ready else
            "This availability record alone does not import or analyse R1.7."
        ),
    }
    atomic_json(RESULT_ROOT / "reports/r1_7_availability.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
