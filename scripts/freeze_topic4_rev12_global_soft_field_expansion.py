#!/usr/bin/env python3
"""Freeze Stage-Z Pareto nominees for nine fresh fit networks."""
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


def build_candidates(source_manifest: dict, source_summary: dict,
                     nomination: dict, expansion: dict) -> tuple[list[dict], dict]:
    if source_summary.get("status") != "REV12ND_GLOBAL_SOFT_FIELD_FIT_COMPLETE":
        raise RuntimeError("Stage-Z fit summary is not complete")
    if nomination.get("status") != (
            "GLOBAL_FIT_NOMINEES_FROZEN_FROM_PREDECLARED_PARETO_RULE"):
        raise RuntimeError("Stage-Z nominations are not frozen")
    rows = {row["candidate_id"]: row for row in source_summary["rows"]}
    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    nominated = list(nomination["candidate_ids"])
    if len(nominated) != int(expansion["expected_nominee_count"]):
        raise RuntimeError("Stage-Z nominee count changed")
    if len(nominated) != len(set(nominated)):
        raise RuntimeError("Stage-Z nominees contain duplicates")
    anchor = str(expansion["anchor_candidate_id"])
    selected = [anchor, *nominated]
    for candidate_id in selected:
        if candidate_id not in rows or candidate_id not in source:
            raise RuntimeError(f"missing expansion field: {candidate_id}")
        if not rows[candidate_id].get("fit_valid", False):
            raise RuntimeError(f"invalid Stage-Z nominee: {candidate_id}")
        if not rows[candidate_id].get("selection_evaluable", False):
            raise RuntimeError(f"non-evaluable Stage-Z nominee: {candidate_id}")
    if len(selected) != int(expansion["expected_candidate_count"]):
        raise RuntimeError("expansion candidate count changed")
    candidates = []
    metrics = {}
    for candidate_id in selected:
        candidate = copy.deepcopy(source[candidate_id])
        candidate["role"] = (
            "paired_anchor" if candidate_id == anchor
            else "stage_z_predeclared_pareto_nominee"
        )
        candidate["selection_eligible"] = candidate_id != anchor
        candidates.append(candidate)
        row = rows[candidate_id]
        metrics[candidate_id] = {
            "mean_soft_objective": row["mean_soft_objective"],
            "mean_soft_mode_0": row["mean_soft_mode_0"],
            "mean_soft_mode_1": row["mean_soft_mode_1"],
            "mean_soft_causal_direction": row["mean_soft_causal_direction"],
            "mean_soft_causal_monotonicity": row[
                "mean_soft_causal_monotonicity"
            ],
            "soft_topology_across_network": row[
                "soft_topology_across_network"
            ],
            "soft_topology_mode_separation": row[
                "soft_topology_mode_separation"
            ],
        }
    return candidates, {
        "source_candidate_ids": selected,
        "source_metrics": metrics,
        "field_hashes_unchanged": all(
            candidate["node_field"]["field_sha256"]
            == source[candidate["candidate_id"]]["node_field"]["field_sha256"]
            for candidate in candidates
        ),
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
        raise RuntimeError("expansion freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/finish_topic4_rev12_global_soft_field_expansion.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("expansion runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"expansion runtime path drifted: {relative}")
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
            raise RuntimeError(f"expansion input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_z_manifest"], loaded["stage_z_summary"],
        loaded["stage_z_nomination"], config["expansion"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_global_soft_field_expansion_manifest_v1",
        "status": "REV12ND_GLOBAL_SOFT_FIELD_EXPANSION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "expansion_audit": audit,
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
        "status": payload["status"], "n_candidates": len(candidates),
        "n_fit_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
