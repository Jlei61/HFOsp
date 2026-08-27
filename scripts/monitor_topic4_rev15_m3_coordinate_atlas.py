#!/usr/bin/env python3
"""Resource-bounded controller for the frozen rev15 M3 coordinate atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev15_m3_coordinate_atlas as freezer
from scripts import monitor_topic4_rev14_m3_canary as base
from scripts import run_topic4_rev15_m3_coordinate_atlas_worker as worker


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_UNIT_PREFIX = "codex-t4-r15-m3atlas"
CONTROLLER_SCHEMA = "topic4_rev15_m3_coordinate_atlas_controller_v1"


def configure_base() -> None:
    base.WORKER = ROOT / "scripts/run_topic4_rev15_m3_coordinate_atlas_worker.py"
    base.COMPLETE_STATUS = worker.WORKER_STATUS
    base.MANIFEST_STATUS = freezer.STATUS
    base.MANIFEST_SCHEMA = freezer.MANIFEST_SCHEMA
    base.CONTROLLER_SCHEMA = CONTROLLER_SCHEMA
    base.DEFAULT_UNIT_PREFIX = DEFAULT_UNIT_PREFIX
    base.QUEUE_EMERGENCY_STATUS = "REV15_M3_ATLAS_QUEUE_EMERGENCY_STOPPED"
    base.PREWARM_INCOMPLETE_STATUS = "REV15_M3_ATLAS_FULL_PREWARM_INCOMPLETE"
    base.QUEUE_COMPLETE_STATUS = "REV15_M3_ATLAS_QUEUE_COMPLETE"
    base.QUEUE_DRAINING_STATUS = "REV15_M3_ATLAS_QUEUE_DRAINING_AFTER_FAILURE"
    base.QUEUE_FAILED_STATUS = "REV15_M3_ATLAS_QUEUE_FAILED"
    base.PREWARM_WAIT_STATUS = "REV15_M3_ATLAS_FULL_PREWARM_RESOURCE_WAIT"
    base.PREWARM_RUNNING_STATUS = "REV15_M3_ATLAS_FULL_PREWARM_RUNNING"
    base.QUEUE_WAIT_STATUS = "REV15_M3_ATLAS_QUEUE_RESOURCE_WAIT"
    base.QUEUE_RUNNING_STATUS = "REV15_M3_ATLAS_QUEUE_RUNNING"


def _load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    resources = config["resources"]
    expected_resources = {
        "maximum_workers": 14,
        "recommended_workers_after_full_prewarm": 12,
        "stop_launching_below_available_memory_gib": 64,
        "emergency_stop_below_available_memory_gib": 48,
        "worker_rss_safety_multiplier": 1.2,
        "minimum_free_disk_gib": 40,
        "monitor_interval_seconds": 600,
    }
    for key, value in expected_resources.items():
        if resources.get(key) != value:
            raise RuntimeError(f"rev15 atlas resource contract changed: {key}")
    if resources.get("numerical_threads_per_worker") != 1:
        raise RuntimeError("rev15 atlas workers must use one numerical thread")
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("rev15 atlas manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev15 atlas manifest is not frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev15 atlas manifest schema changed")
    if manifest.get("config_sha256") != base._sha256(config_path):
        raise RuntimeError("rev15 atlas manifest/config mismatch")
    provenance = manifest.get("provenance", {})
    if not provenance.get("formal_ready"):
        raise RuntimeError("rev15 atlas manifest provenance is not formal-ready")
    if provenance.get("git_commit") != expected_commit:
        raise RuntimeError("rev15 atlas manifest belongs to another commit")
    candidates = manifest.get("candidates", [])
    if len(candidates) != int(config["m3_design"]["candidate_count"]):
        raise RuntimeError("rev15 atlas candidate count changed")
    identifiers = [row["candidate_id"] for row in candidates]
    if len(identifiers) != len(set(identifiers)) or identifiers.count("uniform_node") != 1:
        raise RuntimeError("rev15 atlas candidate identities are invalid")
    return config, manifest


def main(argv: list[str] | None = None) -> None:
    configure_base()
    base._load_contract = _load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=12)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = base.run_controller(
        config_path=args.config.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_commit=args.expected_commit,
        unit_prefix=args.unit_prefix,
        execute=bool(args.execute),
        worker_cap=int(args.worker_cap),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

