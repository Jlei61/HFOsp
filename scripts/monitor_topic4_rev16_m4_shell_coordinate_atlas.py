#!/usr/bin/env python3
"""Resource-bounded controller for the rev16 M4-shell coordinate atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev16_m4_shell_coordinate_atlas as freezer
from scripts import monitor_topic4_rev14_m3_replication as base
from scripts import run_topic4_rev16_m4_shell_coordinate_worker as worker


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_UNIT_PREFIX = "codex-t4-r16-m4shell"
CONTROLLER_SCHEMA = "topic4_rev16_m4_shell_coordinate_controller_v1"


def _validate_unit_prefix(prefix: str) -> str:
    token = base.base._unit_token(prefix)
    if not token.startswith(DEFAULT_UNIT_PREFIX):
        raise ValueError(f"rev16 prefix must start with {DEFAULT_UNIT_PREFIX!r}")
    if any(marker in token for marker in base.base.PROTECTED_UNIT_MARKERS):
        raise ValueError("rev16 prefix overlaps a protected service")
    return token


def configure_base() -> None:
    base.freezer = freezer
    base.WORKER = ROOT / "scripts/run_topic4_rev16_m4_shell_coordinate_worker.py"
    base.WORKER_STATUS = worker.WORKER_STATUS
    base.DEFAULT_UNIT_PREFIX = DEFAULT_UNIT_PREFIX
    base.CONTROLLER_SCHEMA = CONTROLLER_SCHEMA
    base._validate_unit_prefix = _validate_unit_prefix
    base.QUEUE_EMERGENCY_STATUS = "REV16_M4_SHELL_EMERGENCY_STOPPED"
    base.QUEUE_COMPLETE_STATUS = "REV16_M4_SHELL_QUEUE_COMPLETE"
    base.QUEUE_DRAINING_STATUS = "REV16_M4_SHELL_DRAINING_AFTER_FAILURE"
    base.QUEUE_FAILED_STATUS = "REV16_M4_SHELL_QUEUE_FAILED"
    base.QUEUE_WAIT_STATUS = "REV16_M4_SHELL_RESOURCE_WAIT"
    base.QUEUE_RUNNING_STATUS = "REV16_M4_SHELL_QUEUE_RUNNING"


def _load_contract(
    config_path: Path, artifact_root: Path, expected_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    expected_resources = {
        "maximum_workers": 8,
        "recommended_workers": 6,
        "measured_canary_peak_rss_kib": 13356224,
        "worker_rss_safety_multiplier": 1.2,
        "stop_launching_below_available_memory_gib": 80,
        "emergency_stop_below_available_memory_gib": 64,
        "minimum_free_disk_gib": 40,
        "monitor_interval_seconds": 600,
        "numerical_threads_per_worker": 1,
    }
    for key, value in expected_resources.items():
        if config["resources"].get(key) != value:
            raise RuntimeError(f"rev16 M4-shell resource changed: {key}")
    manifest_path = artifact_root / str(config["candidate_manifest"])
    if not manifest_path.is_file():
        raise RuntimeError("rev16 M4-shell manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != freezer.STATUS:
        raise RuntimeError("rev16 M4-shell manifest is not frozen")
    if manifest.get("schema_id") != freezer.MANIFEST_SCHEMA:
        raise RuntimeError("rev16 M4-shell manifest schema changed")
    if manifest.get("config_sha256") != base._sha256(config_path):
        raise RuntimeError("rev16 M4-shell manifest/config mismatch")
    provenance = manifest.get("provenance", {})
    if (
        provenance.get("git_commit") != expected_commit
        or provenance.get("expected_git_commit") != expected_commit
        or provenance.get("formal_ready") is not True
        or provenance.get("all_explicit_paths_clean") is not True
        or provenance.get("all_explicit_paths_match_expected_commit") is not True
    ):
        raise RuntimeError("rev16 M4-shell provenance is not formal")
    candidates = manifest.get("candidates", [])
    if len(candidates) != 40 or not all(
        bool(row.get("selection_eligible")) for row in candidates
    ):
        raise RuntimeError("rev16 M4-shell candidate inventory changed")
    if manifest.get("search", {}).get("active_network_seeds") != [2331, 2332, 2333]:
        raise RuntimeError("rev16 M4-shell network pool changed")
    return config, manifest


def main(argv: list[str] | None = None) -> None:
    configure_base()
    base._load_contract = _load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--unit-prefix", default=DEFAULT_UNIT_PREFIX)
    parser.add_argument("--worker-cap", type=int, default=6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    payload = base.run_controller(
        config_path=args.config.resolve(),
        artifact_root=args.artifact_root.resolve(),
        expected_commit=args.expected_commit,
        unit_prefix=args.unit_prefix,
        execute=bool(args.execute), worker_cap=int(args.worker_cap),
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
