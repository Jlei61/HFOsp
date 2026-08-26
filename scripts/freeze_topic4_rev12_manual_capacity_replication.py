#!/usr/bin/env python3
"""Freeze the historical smooth dual-core field as a non-selectable control."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_candidate(source_manifest: dict, stage_ac_audit: dict,
                    contract: dict) -> tuple[list[dict], dict]:
    if stage_ac_audit.get("status") != contract["required_stage_ac_status"]:
        raise RuntimeError("Stage-AC does not justify a capacity replication")
    candidate_id = str(contract["candidate_id"])
    matches = [
        row for row in source_manifest["candidates"]
        if row["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError("manual capacity control is absent or duplicated")
    source = matches[0]
    if source.get("role") != contract["required_source_role"]:
        raise RuntimeError("manual capacity role changed")
    if bool(source.get("selection_eligible", True)) != bool(
            contract["required_selection_eligible"]):
        raise RuntimeError("manual capacity eligibility changed")
    if source["node_field"].get("field_type") != "spline_continuous":
        raise RuntimeError("manual capacity field is not continuous spline")
    candidate = copy.deepcopy(source)
    candidate["role"] = "replicated_rigid_capacity_control_not_selectable"
    candidate["selection_eligible"] = False
    return [candidate], {
        "candidate_id": candidate_id,
        "field_sha256": candidate["node_field"]["field_sha256"],
        "source_role": source["role"],
        "selection_eligible": False,
        "historical_geometry_used": True,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
    }


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("capacity freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/finish_topic4_rev12_manual_capacity_replication.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("capacity runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"capacity runtime path drifted: {relative}")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"capacity input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidate(
        loaded["stage_t_manifest"], loaded["stage_ac_paired_audit"],
        config["capacity_control"],
    )
    if len(candidates) != int(config["capacity_control"]["expected_candidate_count"]):
        raise RuntimeError("capacity candidate count changed")
    payload = {
        "schema_id": "topic4_rev12_nd_manual_capacity_replication_manifest_v1",
        "status": "REV12ND_MANUAL_CAPACITY_REPLICATION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "capacity_control_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "soft_objective": config["soft_objective"],
        "pareto_selection": config["pareto_selection"],
        "inputs": input_audit,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "candidate": audit["candidate_id"],
        "n_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
