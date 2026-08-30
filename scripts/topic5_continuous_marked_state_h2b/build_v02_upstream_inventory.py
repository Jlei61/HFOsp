#!/usr/bin/env python3
"""Freeze the R1.7B checkpoint universe for H2b v0.2.

This producer is intentionally independent of seizure metadata.  It verifies
every R1.7B result and adjacent checkpoint before the H2b cohort is assembled,
and it keeps failed/non-checkpoint cells in the denominator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


DEFAULT_SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
RELATIVE_R17_ROOT = Path(
    "results/epi_prssm/continuous_marked_state/r1/r1_7b_cohort_extension"
)
RELATIVE_OUTPUT = Path(
    "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2/"
    "manifests/r1_7_checkpoint_inventory.json"
)
REVISION = "continuous_marked_state_h2b_cross_task_v0_2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build(source_repo: Path, output: Path) -> dict:
    source_repo = source_repo.resolve()
    root = source_repo / RELATIVE_R17_ROOT
    queue_path = root / "QUEUE_STATUS.json"
    cohort_path = root / "manifests/cohort_inventory.json"
    summary_path = root / "reports/r1_7a_summary.json"
    for path in (queue_path, cohort_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    queue = read_json(queue_path)
    cohort = read_json(cohort_path)
    summary = read_json(summary_path)
    if queue.get("status") != "COMPLETE" or queue.get("scheduled_cells") != 85:
        raise ValueError("R1.7B queue is not the frozen 17 x 5 COMPLETE release")
    if queue.get("formal_test_partition_opened") is not False:
        raise ValueError("R1.7B opened formal test data")
    if queue.get("sealed_opened") is not False:
        raise ValueError("R1.7B opened sealed data")
    subjects = list(map(str, cohort.get("selected_subjects") or []))
    if len(subjects) != 17 or len(set(subjects)) != 17:
        raise ValueError("R1.7B cohort is not the frozen 17-subject universe")
    stable_subjects = set(map(str, summary.get("stable_state_subjects") or []))

    entries = []
    for subject in subjects:
        subject_summary = (summary.get("by_subject") or {}).get(subject)
        if subject_summary is None:
            raise ValueError(f"R1.7B summary lacks {subject}")
        for seed in range(5):
            result_path = root / "fits" / subject / f"seed_{seed}" / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(result_path)
            result = read_json(result_path)
            if result.get("status") != "COMPLETE":
                raise ValueError(f"incomplete R1.7B result: {result_path}")
            if str(result.get("subject")) != subject or int(result.get("seed", -1)) != seed:
                raise ValueError(f"R1.7B result identity mismatch: {result_path}")
            if result.get("formal_test_partition_opened") is not False:
                raise ValueError(f"formal partition opened: {result_path}")
            if result.get("sealed_opened") is not False:
                raise ValueError(f"sealed partition opened: {result_path}")
            checkpoint = result_path.with_name("model.pt")
            checkpoint_available = checkpoint.is_file()
            expected_checkpoint_hash = result.get("checkpoint_sha256")
            observed_checkpoint_hash = (
                sha256_file(checkpoint) if checkpoint_available else None
            )
            if checkpoint_available and observed_checkpoint_hash != expected_checkpoint_hash:
                raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
            if not checkpoint_available and expected_checkpoint_hash is not None:
                raise ValueError(f"missing checkpoint despite declared hash: {result_path}")
            entries.append({
                "subject": subject,
                "dataset": subject.split("_", 1)[0],
                "seed": seed,
                "result_path": str(result_path),
                "result_sha256": sha256_file(result_path),
                "checkpoint_path": str(checkpoint) if checkpoint_available else None,
                "checkpoint_sha256": observed_checkpoint_hash,
                "checkpoint_available": checkpoint_available,
                "analysis_status": result.get("analysis_status", "SCORED"),
                "stable_checkpoint": result.get("stable_checkpoint") is True,
                "h1_stable_subject": subject in stable_subjects,
                "h1_is_stratification_not_h2b_gate": True,
                "source_revision": result.get("revision"),
                "state_source_task": "continuous_background_and_ied_timing_exact_mark",
                "state_source_uses_seizure_labels": False,
                "seizure_gradient_path": False,
                "formal_test_partition_opened": False,
                "sealed_opened": False,
            })

    checkpoint_entries = [row for row in entries if row["checkpoint_available"]]
    payload = {
        "status": "COMPLETE",
        "revision": REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "source_release": {
            "queue_status": str(queue_path),
            "queue_status_sha256": sha256_file(queue_path),
            "cohort_inventory": str(cohort_path),
            "cohort_inventory_sha256": sha256_file(cohort_path),
            "summary": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "n_subjects": len(subjects),
        "n_cells": len(entries),
        "n_checkpoint_available_cells": len(checkpoint_entries),
        "n_instrument_failure_without_checkpoint": len(entries) - len(checkpoint_entries),
        "n_h1_stable_subjects": len(stable_subjects),
        "subjects": subjects,
        "h1_stable_subjects": sorted(stable_subjects),
        "h1_is_stratification_not_h2b_gate": True,
        "entries": entries,
        "boundary": {
            "development_only": True,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "h3_or_t2_run": False,
            "paper_ready_figures_modified": False,
        },
    }
    atomic_json(output.resolve(), payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (Path(__file__).resolve().parents[2] / RELATIVE_OUTPUT)
    payload = build(args.source_repo, output)
    print(json.dumps({
        key: payload[key] for key in (
            "status", "n_subjects", "n_cells", "n_checkpoint_available_cells",
            "n_instrument_failure_without_checkpoint", "n_h1_stable_subjects",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
