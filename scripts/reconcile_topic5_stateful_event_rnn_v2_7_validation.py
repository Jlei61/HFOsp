#!/usr/bin/env python3
"""Reconcile a complete v2.7 screen after an interrupted parent launcher.

The original launcher session ended after 31 valid artifacts.  Three explicit
single-patient workers were subsequently rerun.  This script validates all 34
artifacts and records that recovery without inventing parent-launcher runtimes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.launch_topic5_stateful_event_rnn_v2_7_validation import (  # noqa: E402
    COMPLETION,
    CONFIG,
    EXPECTED_SUBJECTS,
    OUTPUT,
    WORKER,
    load_lpt_tasks,
    sha256,
    validate_patient_artifact,
)
from src.topic5_resource_guard import atomic_write_json  # noqa: E402


RECOVERED_SUBJECTS = ("epilepsiae_1073", "epilepsiae_1096", "epilepsiae_958")


def reconcile(output: Path = OUTPUT) -> dict:
    tasks = load_lpt_tasks(expected_count=EXPECTED_SUBJECTS)
    artifacts = []
    for task in tasks:
        path = validate_patient_artifact(task.subject, output)
        artifacts.append(
            {
                "subject": task.subject,
                "artifact": str(path.relative_to(ROOT)),
                "artifact_sha256": sha256(path),
                "recovered_after_launcher_interruption": task.subject in RECOVERED_SUBJECTS,
            }
        )
    if len(artifacts) != EXPECTED_SUBJECTS:
        raise RuntimeError("reconciliation cohort is incomplete")
    state = {
        "contract": "topic5_stateful_event_sequence_rnn_v2_7_validation_reconciliation",
        "status": "VALIDATION_ARTIFACTS_RECONCILED_COMPLETE",
        "n_expected": EXPECTED_SUBJECTS,
        "n_valid": len(artifacts),
        "original_launcher_completion_marker_present": COMPLETION.exists(),
        "original_launcher_runtime_reconstructed": False,
        "recovered_subjects": list(RECOVERED_SUBJECTS),
        "threads_per_worker": 1,
        "cuda_disabled": True,
        "config": str(CONFIG.relative_to(ROOT)),
        "config_sha256": sha256(CONFIG),
        "worker": str(WORKER.relative_to(ROOT)),
        "worker_sha256": sha256(WORKER),
        "artifacts": artifacts,
    }
    atomic_write_json(
        Path(output) / "SCREEN_ARTIFACT_RECONCILIATION.json", state
    )
    return state


def main() -> None:
    state = reconcile()
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
