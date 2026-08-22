#!/usr/bin/env python3
"""Freeze the predeclared Stage-A shortlist for fresh-network selection."""
from __future__ import annotations

import argparse
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


def _resolve(artifact_root: Path, relative_path: str) -> Path:
    local = ROOT / relative_path
    return local if local.exists() else artifact_root / relative_path


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
    paths = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("selection freezer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("selection freezer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"selection freezer path drifted: {path}")
    return {"git_commit": expected, "path_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"selection input changed: {record['path']}")
    provenance = _provenance(config_path, args.expected_commit)
    summary_key = (
        "source_summary" if "source_summary" in config["inputs"]
        else "stage_a_summary"
    )
    manifest_key = (
        "source_manifest" if "source_manifest" in config["inputs"]
        else "stage_a_manifest"
    )
    summary = json.loads(_resolve(
        artifact_root, config["inputs"][summary_key]["path"],
    ).read_text())
    source_manifest = json.loads(_resolve(
        artifact_root, config["inputs"][manifest_key]["path"],
    ).read_text())
    expected_networks = len(summary["requested_seeds"])
    complete = {
        row["candidate_id"]: row for row in summary["rows"]
        if int(row["n_networks"]) == expected_networks
    }
    source = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    selected_ids = list(config["field_search"]["selection_candidate_ids"])
    if len(selected_ids) != len(set(selected_ids)):
        raise RuntimeError("selection candidate ids contain duplicates")
    if any(candidate not in complete or candidate not in source for candidate in selected_ids):
        raise RuntimeError("selection candidate is absent or lacks complete fit coverage")
    candidates = []
    fit_metrics = {}
    roles = config["field_search"]["selection_roles"]
    for candidate_id in selected_ids:
        candidate = dict(source[candidate_id])
        candidate["selection_role"] = roles[candidate_id]
        candidates.append(candidate)
        score = complete[candidate_id]["score"]
        topology = complete[candidate_id]["source_topology"]
        fit_metrics[candidate_id] = {
            "patient_objective": score["mean_patient_objective"],
            "mode_losses": [score["mean_mode_0_loss"], score["mean_mode_1_loss"]],
            "ood_fraction": score["mean_ood_fraction"],
            "source_topology_separation": topology["equal_network_between_mode_distance"],
            "source_topology_reproducibility": topology["mean_across_network_template_cosine"],
        }
    payload = {
        "schema_id": "topic4_rev12_nd_node_selection_manifest_v1",
        "status": "REV12ND_NODE_SELECTION_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "fit_metrics": fit_metrics,
        "selection_contract": {
            "source_stage": config["field_search"].get(
                "source_stage", "stage_a",
            ),
            "patient_heldout_used": False,
            "fresh_network_seeds": config["search"]["selection_network_seeds"],
            "paired_reference_retained": "stage_a_base",
            "EE": "off",
            "E_to_I": "off",
            "Z_M": "off",
        },
        "provenance": provenance,
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "candidate_ids": selected_ids,
    }, indent=2))


if __name__ == "__main__":
    main()
