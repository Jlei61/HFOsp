#!/usr/bin/env python3
"""Migrate the G0 duration audit after a time/neuron-axis logging bug."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_topic4_xy_research as base

FOLDER = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1/execution/training_24s"


def main():
    snapshot_path = FOLDER / "runtime_snapshot.json"
    archive_path = FOLDER / "runtime_snapshot_before_duration_axis_fix.json"
    migration_path = FOLDER / "runtime_migration.json"
    if migration_path.exists():
        print(json.dumps(base.read(migration_path), indent=2)); return
    if any("run_topic4_multidimensional_worker.py" in path.read_text(errors="ignore")
           for path in Path("/proc").glob("[0-9]*/cmdline")):
        raise RuntimeError("cannot migrate while a multidimensional worker is alive")
    shutil.copy2(snapshot_path, archive_path)
    old_snapshot_sha = base.sha(archive_path)
    snapshot = base.read(snapshot_path)
    changed = "scripts/run_topic4_multidimensional_worker.py"
    old_source_sha = snapshot["source_hashes"][changed]
    new_source_sha = base.sha(ROOT / changed)
    if old_source_sha == new_source_sha:
        raise RuntimeError("duration audit source did not change")
    snapshot["source_hashes"][changed] = new_source_sha
    snapshot["migration"] = {
        "reason": "actual_duration_ms used spikes.shape[1] (neurons) instead of shape[0] (time steps)",
        "scope": "audit metadata only; raw arrays, physics, observer, duration configuration and seeds unchanged",
        "old_snapshot_path": str(archive_path),
        "old_snapshot_sha256": old_snapshot_sha,
    }
    base.write(snapshot_path, snapshot)
    migrated = {}
    for worker_path in sorted((FOLDER / "workers").glob("*.json")):
        record = base.read(worker_path)
        original_sha = base.sha(worker_path)
        with np.load(record["arrays"]["path"]) as arrays:
            actual_duration_ms = float(
                len(arrays["active_fraction"]) * float(arrays["active_fraction_bin_ms"])
            )
            envelope_duration_ms = float(
                arrays["contact_envelope"].shape[1]
                * float(arrays["contact_envelope_dt_ms"])
            )
            movie_duration_ms = float(
                arrays["sheet_activity_counts"].shape[0]
                * float(arrays["sheet_activity_frame_ms"])
            )
        if not np.isclose(actual_duration_ms, envelope_duration_ms, atol=2.0):
            raise RuntimeError(f"activity/envelope duration mismatch: {worker_path}")
        if not np.isclose(actual_duration_ms, movie_duration_ms, atol=2.0):
            raise RuntimeError(f"activity/movie duration mismatch: {worker_path}")
        wrong_value = record["simulation"].get("actual_duration_ms")
        record["simulation"]["actual_duration_ms"] = actual_duration_ms
        record["simulation"]["actual_duration_audit_migration"] = {
            "old_incorrect_value_ms": wrong_value,
            "source": "len(active_fraction) * active_fraction_bin_ms",
            "envelope_crosscheck_ms": envelope_duration_ms,
            "native_movie_crosscheck_ms": movie_duration_ms,
            "raw_arrays_changed": False,
        }
        base.write(worker_path, record)
        migrated[worker_path.name] = {
            "old_worker_sha256": original_sha,
            "new_worker_sha256": base.sha(worker_path),
            "arrays_sha256": record["arrays"]["sha256"],
            "corrected_actual_duration_ms": actual_duration_ms,
        }
        resource_path = FOLDER / "resource_logs" / f"{worker_path.stem}_summary.json"
        if not resource_path.exists():
            samples = [json.loads(line) for line in (
                FOLDER / "resource_logs" / f"{worker_path.stem}.jsonl"
            ).read_text().splitlines()]
            base.write(resource_path, {
                "candidate_id": record["candidate_id"],
                "topology_seed": record["topology_seed"],
                "dynamics_seed": record["dynamics_seed"],
                "exit_code": 0,
                "peak_process_tree": {
                    key: max(row[key] for row in samples)
                    for key in ("rss_bytes", "pss_bytes", "vms_bytes")
                },
                "recovered_after_controller_dispatch_pause": True,
            })
    migration = {
        "status": "DURATION_AXIS_AUDIT_MIGRATED",
        "old_snapshot_path": str(archive_path),
        "old_snapshot_sha256": old_snapshot_sha,
        "new_snapshot_sha256": base.sha(snapshot_path),
        "changed_source": changed,
        "old_source_sha256": old_source_sha,
        "new_source_sha256": new_source_sha,
        "physics_or_observer_changed": False,
        "raw_arrays_changed": False,
        "migrated_completed_workers": migrated,
    }
    base.write(migration_path, migration)
    migration["new_snapshot_sha256"] = base.sha(snapshot_path)
    print(json.dumps(migration, indent=2))


if __name__ == "__main__": main()
