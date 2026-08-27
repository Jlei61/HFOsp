#!/usr/bin/env python3
"""Resource-bounded controller for rev15 response-combination replication."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev15_m3_response_combinations as freezer
from scripts import monitor_topic4_rev14_m3_replication as base
from scripts import run_topic4_rev15_m3_response_combination_worker as worker


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_UNIT_PREFIX = "codex-t4-r15-m3combo"
CONTROLLER_SCHEMA = "topic4_rev15_m3_response_combinations_controller_v1"


def _validate_unit_prefix(prefix: str) -> str:
    token = base.base._unit_token(prefix)
    if not token.startswith(DEFAULT_UNIT_PREFIX):
        raise ValueError(
            f"rev15 combination unit prefix must start with {DEFAULT_UNIT_PREFIX!r}"
        )
    if any(marker in token for marker in base.base.PROTECTED_UNIT_MARKERS):
        raise ValueError("rev15 combination prefix overlaps a protected service")
    return token


def configure_base() -> None:
    base.freezer = freezer
    base.WORKER = ROOT / "scripts/run_topic4_rev15_m3_response_combination_worker.py"
    base.WORKER_STATUS = worker.WORKER_STATUS
    base.DEFAULT_UNIT_PREFIX = DEFAULT_UNIT_PREFIX
    base.CONTROLLER_SCHEMA = CONTROLLER_SCHEMA
    base._validate_unit_prefix = _validate_unit_prefix
    base.QUEUE_EMERGENCY_STATUS = "REV15_M3_COMBINATION_EMERGENCY_STOPPED"
    base.QUEUE_COMPLETE_STATUS = "REV15_M3_COMBINATION_QUEUE_COMPLETE"
    base.QUEUE_DRAINING_STATUS = "REV15_M3_COMBINATION_DRAINING_AFTER_FAILURE"
    base.QUEUE_FAILED_STATUS = "REV15_M3_COMBINATION_QUEUE_FAILED"
    base.QUEUE_WAIT_STATUS = "REV15_M3_COMBINATION_RESOURCE_WAIT"
    base.QUEUE_RUNNING_STATUS = "REV15_M3_COMBINATION_QUEUE_RUNNING"


def _load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    resources = config["resources"]
    expected_resources = {
        "maximum_workers": 10,
        "recommended_workers": 9,
        "measured_atlas_peak_rss_kib": 13336624,
        "measured_canary_peak_rss_kib": 13336624,
        "worker_rss_safety_multiplier": 1.2,
        "stop_launching_below_available_memory_gib": 64,
        "emergency_stop_below_available_memory_gib": 48,
        "minimum_free_disk_gib": 40,
        "monitor_interval_seconds": 600,
        "numerical_threads_per_worker": 1,
    }
    for key, value in expected_resources.items():
        if resources.get(key) != value:
            raise RuntimeError(f"rev15 combination resource contract changed: {key}")
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("rev15 response-combination manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev15 response-combination manifest is not frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev15 response-combination manifest schema changed")
    if manifest.get("config_sha256") != base._sha256(config_path):
        raise RuntimeError("rev15 response-combination manifest/config mismatch")
    provenance = manifest.get("provenance", {})
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("expected_git_commit") != expected_commit
        or provenance.get("formal_ready") is not True
        or provenance.get("all_explicit_paths_clean") is not True
        or provenance.get("all_explicit_paths_match_expected_commit") is not True
    ):
        raise RuntimeError("rev15 response-combination provenance is not formal")
    candidates = manifest.get("candidates", [])
    if len(candidates) != 15 or sum(row["selection_eligible"] for row in candidates) != 14:
        raise RuntimeError("rev15 response-combination candidate inventory changed")
    if [int(seed) for seed in manifest["search"]["active_network_seeds"]] != [2332, 2333]:
        raise RuntimeError("rev15 response-combination seed pool changed")
    return config, manifest


def main(argv: list[str] | None = None) -> None:
    configure_base()
    base._load_contract = _load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=9)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    payload = base.run_controller(
        config_path=args.config.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_commit=args.expected_commit,
        unit_prefix=args.unit_prefix,
        execute=bool(args.execute),
        worker_cap=int(args.worker_cap),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
