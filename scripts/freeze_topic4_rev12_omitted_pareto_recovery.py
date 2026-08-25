#!/usr/bin/env python3
"""Freeze one Stage-Z Pareto field omitted by the radius diversity quota."""
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


def build_candidate(source_manifest: dict, source_summary: dict,
                    nomination: dict, paired_audit: dict,
                    recovery: dict) -> tuple[list[dict], dict]:
    if source_summary.get("status") != "REV12ND_GLOBAL_SOFT_FIELD_FIT_COMPLETE":
        raise RuntimeError("Stage-Z summary is incomplete")
    if paired_audit.get("status") != (
            "FRESH_FIT_BALANCED_MEAN_CANDIDATE_MODE_STABILITY_UNRESOLVED"):
        raise RuntimeError("Stage-AA does not justify omitted-candidate recovery")
    candidate_id = str(recovery["candidate_id"])
    rows = {row["candidate_id"]: row for row in source_summary["rows"]}
    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    if candidate_id not in rows or candidate_id not in source:
        raise RuntimeError("recovery candidate is absent")
    row = rows[candidate_id]
    if not row.get("fit_valid", False) or not row.get("selection_evaluable", False):
        raise RuntimeError("recovery candidate is not Stage-Z evaluable")
    if bool(recovery["required_stage_z_pareto_member"]) and not row.get(
            "pareto_member", False):
        raise RuntimeError("recovery candidate was not Stage-Z Pareto-optimal")
    nominated = set(nomination["candidate_ids"])
    if bool(recovery["required_absence_from_stage_z_nomination"]) \
            and candidate_id in nominated:
        raise RuntimeError("recovery candidate was already expanded")
    radius = source[candidate_id]["node_field"][
        "residual_coordinates"
    ]["radius"]
    same_radius_nominees = [
        nominee for nominee in nomination["candidate_ids"]
        if source[nominee]["node_field"].get(
            "residual_coordinates", {}
        ).get("radius") == radius
    ]
    maximum_per_radius = int(nomination["maximum_per_radius"])
    if len(same_radius_nominees) != maximum_per_radius:
        raise RuntimeError("recovery candidate was not omitted by a filled radius quota")
    candidate = copy.deepcopy(source[candidate_id])
    candidate["role"] = "omitted_stage_z_pareto_recovery"
    candidate["selection_eligible"] = True
    return [candidate], {
        "candidate_id": candidate_id,
        "field_sha256": candidate["node_field"]["field_sha256"],
        "stage_z_metrics": {
            key: row[key] for key in (
                "mean_soft_objective", "mean_soft_mode_0", "mean_soft_mode_1",
                "mean_soft_causal_direction", "mean_soft_causal_monotonicity",
                "soft_topology_across_network", "soft_topology_mode_separation",
            )
        },
        "radius": radius,
        "same_radius_nominees": same_radius_nominees,
        "maximum_per_radius": maximum_per_radius,
        "omitted_by_filled_radius_quota": True,
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
        raise RuntimeError("recovery freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/finish_topic4_rev12_omitted_pareto_recovery.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("recovery runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"recovery runtime path drifted: {relative}")
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
            raise RuntimeError(f"recovery input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidate(
        loaded["stage_z_manifest"], loaded["stage_z_summary"],
        loaded["stage_z_nomination"], loaded["stage_aa_paired_audit"],
        config["recovery"],
    )
    if len(candidates) != int(config["recovery"]["expected_candidate_count"]):
        raise RuntimeError("recovery candidate count changed")
    payload = {
        "schema_id": "topic4_rev12_nd_omitted_pareto_recovery_manifest_v1",
        "status": "REV12ND_OMITTED_PARETO_RECOVERY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "recovery_audit": audit,
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
