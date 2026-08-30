#!/usr/bin/env python3
"""Verify the complete unseen-network rev16 Node confirmation inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import freeze_topic4_rev16_node_confirmation as freezer  # noqa: E402
from scripts import run_topic4_rev16_node_confirmation_worker as worker  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev16_node_confirmation.json"
OUTPUT_SCHEMA = "topic4_rev16_node_confirmation_audit_v1"
COMPLETE_STATUS = "REV16_NODE_CONFIRMATION_INVENTORY_COMPLETE"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(
    config_path: Path = DEFAULT_CONFIG, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(); artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    freezer._validate_config(config)
    manifest_path = artifact_root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != freezer.STATUS
        or manifest.get("schema_id") != freezer.MANIFEST_SCHEMA
        or manifest.get("config_sha256") != _sha256(config_path)
    ):
        raise RuntimeError("rev16 confirmation manifest is not frozen")
    expected_commit = manifest["provenance"]["git_commit"]
    records, missing, invalid = [], [], []
    worker_root = artifact_root / config["output_root"] / "workers"
    for candidate in manifest["candidates"]:
        candidate_id = candidate["candidate_id"]
        for seed in config["search"]["active_network_seeds"]:
            stem = worker_root / f"{candidate_id}_seed_{seed}"
            json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
            key = f"{candidate_id}:{seed}"
            if not json_path.is_file() or not npz_path.is_file():
                missing.append(key); continue
            try:
                payload = json.loads(json_path.read_text())
                provenance = payload.get("provenance", {})
                if (
                    payload.get("status") != worker.WORKER_STATUS
                    or payload.get("candidate_id") != candidate_id
                    or int(payload.get("seed", -1)) != int(seed)
                    or float(payload.get("simulation", {}).get("duration_ms")) != 20000.0
                    or payload.get("simulation", {}).get("runaway_early_stop_ms") is not None
                    or provenance.get("git_commit") != expected_commit
                    or int(provenance.get("runtime_modules_dirty", 1)) != 0
                    or int(provenance.get("runtime_modules_match_expected_commit", 0)) != 1
                    or any(payload.get("mechanism_freeze", {}).get(key) != "off"
                           for key in ("EE", "E_to_I", "Z_M"))
                    or payload.get("arrays", {}).get("sha256") != _sha256(npz_path)
                ):
                    raise RuntimeError("worker contract changed")
                records.append({
                    "candidate_id": candidate_id, "seed": int(seed),
                    "json": {"path": str(json_path), "sha256": _sha256(json_path)},
                    "npz": {"path": str(npz_path), "sha256": _sha256(npz_path)},
                })
            except Exception as error:
                invalid.append(f"{key}:{error}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status_lines = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    complete = len(records) == 6 and not missing and not invalid
    status = COMPLETE_STATUS if complete and not status_lines else (
        "INVALID_PROVENANCE" if status_lines else "INCOMPLETE"
    )
    output_path = artifact_root / config["output_root"] / "analysis/confirmation_audit.json"
    payload = {
        "schema_id": OUTPUT_SCHEMA, "status": status,
        "candidate_id": config["selected_candidate"]["candidate_id"],
        "network_seeds": config["search"]["active_network_seeds"],
        "inventory": {
            "expected_runs": 6, "present_validated": len(records),
            "missing": missing, "invalid_artifact": invalid,
            "complete_cartesian_product": complete,
        },
        "workers": records,
        "selection_source": config["inputs"]["selection_aggregate"],
        "boundaries": {**config["boundaries"], "SNN_simulation_run": False},
        "provenance": {
            "worker_commit": expected_commit, "analysis_commit": head,
            "worktree_status": status_lines, "formal_ready": not status_lines,
        },
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_path)},
    }
    freezer._atomic_json(output_path, payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
