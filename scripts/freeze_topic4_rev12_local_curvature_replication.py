#!/usr/bin/env python3
"""Freeze a paired fit-network replication of the Stage-W near miss."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))


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


def build_candidates(stage_w_manifest: dict, stage_w_result: dict,
                     replication: dict) -> tuple[list[dict], dict]:
    if stage_w_result.get("status") != (
            "REV12ND_LOCAL_CURVATURE_DIRECT_ANALYSIS_COMPLETE"):
        raise RuntimeError("Stage-W direct analysis is incomplete")
    if stage_w_result.get("decision") != "STOP_NO_LOCAL_CANARY_JOINTLY_IMPROVED":
        raise RuntimeError("Stage-W decision does not require uncertainty replication")
    source_id = str(replication["candidate_id"])
    result_rows = {
        row["candidate_id"]: row for row in stage_w_result["candidate_results"]
    }
    if source_id not in result_rows:
        raise RuntimeError("replication candidate is absent from Stage-W results")
    near_miss = result_rows[source_id]
    delta = near_miss["aggregate_delta_from_anchor"]
    required_positive = tuple(replication["near_miss_positive_endpoints"])
    if any(float(delta[endpoint]) <= 0.0 for endpoint in required_positive):
        raise RuntimeError("replication candidate is not the declared near miss")
    if near_miss["fit_advancement_eligible"]:
        raise RuntimeError("an already eligible candidate does not need fit replication")
    source_rows = {
        row["candidate_id"]: row for row in stage_w_manifest["candidates"]
    }
    if set(("stage_w_anchor", source_id)) - source_rows.keys():
        raise RuntimeError("Stage-W manifest lacks the paired fields")
    candidates = []
    for old_id, new_id, role in (
        ("stage_w_anchor", "stage_x_anchor", "paired_fit_replication_anchor"),
        (source_id, "stage_x_f14p03", "paired_fit_replication_candidate"),
    ):
        source = source_rows[old_id]
        field = copy.deepcopy(source["node_field"])
        field["candidate_id"] = new_id
        field["role"] = role
        candidates.append({
            "candidate_id": new_id, "role": role,
            "selection_eligible": False,
            "source_candidate_ids": [old_id],
            "node_field": field,
        })
    if candidates[0]["node_field"]["field_sha256"] == (
            candidates[1]["node_field"]["field_sha256"]):
        raise RuntimeError("paired replication fields are identical")
    audit = {
        "source_candidate_id": source_id,
        "stage_w_aggregate_delta": delta,
        "stage_w_failed_aggregate_endpoints": near_miss[
            "failed_aggregate_endpoints"
        ],
        "stage_w_failed_network_support_endpoints": near_miss[
            "failed_network_support_endpoints"
        ],
        "reason": (
            "aggregate objective, both patient modes and direction improved, "
            "but three-network sign support and KMeans remained unresolved"
        ),
        "patient_heldout_used_for_replication_choice": False,
    }
    return candidates, audit


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("paired replication freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("paired replication runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"paired replication runtime path drifted: {relative}")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"paired replication input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_w_manifest"], loaded["stage_w_result"],
        config["paired_replication"],
    )
    payload = {
        "schema_id": "topic4_rev12_local_curvature_replication_manifest_v1",
        "status": "REV12ND_LOCAL_CURVATURE_REPLICATION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "replication_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "inputs": input_audit,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "n_fit_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
