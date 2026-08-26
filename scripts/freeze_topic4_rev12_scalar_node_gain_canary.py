#!/usr/bin/env python3
"""Freeze Stage-AJ scalar expression gains for two unchanged Node fields."""
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


def _mapping_hash(field_sha256: str, gain: float, formula: str) -> str:
    payload = json.dumps({
        "field_sha256": field_sha256, "node_gain": float(gain),
        "formula": formula,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_candidates(source_manifest: dict, stage_ag_audit: dict,
                     inventory: dict, contract: dict) -> tuple[list[dict], dict]:
    required_ag = (
        "BROAD_FIELD_SEARCH_PATIENT_DIRECTION_TRADEOFF_PERSISTS_"
        "NO_BALANCED_NODE_CANDIDATE"
    )
    if stage_ag_audit.get("status") != required_ag:
        raise RuntimeError("Stage-AG result does not justify a gain canary")
    if inventory.get("status") != (
        "SCALAR_NODE_GAIN_NOT_PREVIOUSLY_TESTED_CURRENT_CAUSAL_EVENT_UNIT"
    ):
        raise RuntimeError("Stage-AI inventory does not justify a gain canary")
    source_ids = list(contract["source_candidate_ids"])
    if source_ids != ["stage_ag_anchor", stage_ag_audit["best_soft_objective_candidate"]]:
        raise RuntimeError("gain source fields changed")
    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    formula = str(contract["formula"])
    combinations = []
    candidates = []

    def append(source_id: str, gain: float) -> None:
        token = str(gain).replace(".", "p")
        candidate_id = f"stage_aj_{source_id.removeprefix('stage_ag_')}_gain{token}"
        candidate = copy.deepcopy(source[source_id])
        mapping_hash = _mapping_hash(
            candidate["node_field"]["field_sha256"], gain, formula,
        )
        candidate.update({
            "candidate_id": candidate_id,
            "role": "scalar_node_gain_canary_not_selectable",
            "selection_eligible": False,
            "source_candidate_ids": [source_id],
            "node_mapping": {
                "signed_depth_shrinkage": 1.0,
                "node_gain": float(gain),
                "mapping_sha256": mapping_hash,
            },
        })
        candidates.append(candidate)
        combinations.append({
            "candidate_id": candidate_id, "source_candidate_id": source_id,
            "field_sha256": candidate["node_field"]["field_sha256"],
            "node_gain": float(gain), "mapping_sha256": mapping_hash,
        })

    append(
        str(contract["uniform_null_source_candidate_id"]),
        float(contract["uniform_null_gain"]),
    )
    for source_id in source_ids:
        for gain in contract["new_gains_each_field"]:
            append(source_id, float(gain))
    if len(candidates) != int(contract["expected_new_candidate_count"]):
        raise RuntimeError("gain candidate count changed")
    if len({(row["field_sha256"], row["node_gain"]) for row in combinations}) != len(candidates):
        raise RuntimeError("gain candidates are not unique")
    return candidates, {
        "source_candidate_ids": source_ids,
        "reference_gain": float(contract["reference_gain"]),
        "new_gains_each_field": [float(value) for value in contract["new_gains_each_field"]],
        "uniform_null_gain": float(contract["uniform_null_gain"]),
        "formula": formula,
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
        raise RuntimeError("gain freezer is not at expected commit")
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
        raise RuntimeError("gain runtime paths are dirty")
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
            raise RuntimeError(f"gain input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_ag_manifest"], loaded["stage_ag_paired_audit"],
        loaded["stage_ai_inventory"], config["scalar_node_gain"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_scalar_node_gain_manifest_v1",
        "status": "REV12ND_SCALAR_NODE_GAIN_CANARY_FROZEN",
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
