#!/usr/bin/env python3
"""Freeze the bounded Stage-AH signed-depth mapping audit."""
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


def _mapping_hash(field_sha256: str, rho: float, formula: str) -> str:
    payload = json.dumps({
        "field_sha256": field_sha256,
        "signed_depth_shrinkage": float(rho),
        "formula": formula,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_candidates(source_manifest: dict, stage_ag_audit: dict,
                     contract: dict) -> tuple[list[dict], dict]:
    required = (
        "BROAD_FIELD_SEARCH_PATIENT_DIRECTION_TRADEOFF_PERSISTS_"
        "NO_BALANCED_NODE_CANDIDATE"
    )
    if stage_ag_audit.get("status") != required:
        raise RuntimeError("Stage-AG result does not justify a mapping audit")
    source_ids = list(contract["source_candidate_ids"])
    if source_ids != ["stage_ag_anchor", stage_ag_audit["best_soft_objective_candidate"]]:
        raise RuntimeError("mapping source fields changed")
    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    if any(candidate_id not in source for candidate_id in source_ids):
        raise RuntimeError("mapping source field missing")
    formula = str(contract["formula"])
    candidates = []
    combinations = []
    for source_id in source_ids:
        for rho in [float(value) for value in contract["new_rho_values"]]:
            token = str(rho).replace(".", "p")
            candidate_id = f"stage_ah_{source_id.removeprefix('stage_ag_')}_rho{token}"
            candidate = copy.deepcopy(source[source_id])
            candidate.update({
                "candidate_id": candidate_id,
                "role": "signed_depth_mapping_audit_not_selectable",
                "selection_eligible": False,
                "source_candidate_ids": [source_id],
                "node_mapping": {
                    "signed_depth_shrinkage": rho,
                    "mapping_sha256": _mapping_hash(
                        candidate["node_field"]["field_sha256"], rho, formula,
                    ),
                },
            })
            candidates.append(candidate)
            combinations.append({
                "candidate_id": candidate_id,
                "source_candidate_id": source_id,
                "field_sha256": candidate["node_field"]["field_sha256"],
                "signed_depth_shrinkage": rho,
                "mapping_sha256": candidate["node_mapping"]["mapping_sha256"],
            })
    if len(candidates) != int(contract["expected_new_candidate_count"]):
        raise RuntimeError("mapping candidate count changed")
    keys = {
        (row["node_field"]["field_sha256"], row["node_mapping"]["signed_depth_shrinkage"])
        for row in candidates
    }
    if len(keys) != len(candidates):
        raise RuntimeError("field and mapping pairs are not unique")
    if any(
        row["node_field"]["field_sha256"]
        != source[row["source_candidate_ids"][0]]["node_field"]["field_sha256"]
        for row in candidates
    ):
        raise RuntimeError("mapping audit changed a field hash")
    return candidates, {
        "source_candidate_ids": source_ids,
        "reference_rho": float(contract["reference_rho"]),
        "new_rho_values": [float(value) for value in contract["new_rho_values"]],
        "formula": formula,
        "preserved_quantity": contract["preserved_quantity"],
        "combinations": combinations,
        "patient_heldout_used": False,
        "manual_field_used": False,
        "selection_eligible": False,
    }


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


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("mapping freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_core_field_rev9.py",
        "src/topic4_zm_ictal_transition.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("mapping runtime paths are dirty")
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
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"mapping input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_ag_manifest"], loaded["stage_ag_paired_audit"],
        config["signed_depth_mapping"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_signed_depth_mapping_manifest_v1",
        "status": "REV12ND_SIGNED_DEPTH_MAPPING_AUDIT_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "mapping_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "soft_objective": config["soft_objective"],
        "pareto_selection": config["pareto_selection"],
        "inputs": inputs,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "n_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
