#!/usr/bin/env python3
"""Freeze the anchor and best distinct historical field for rev12-ND canary."""
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
LIBRARY_CONFIGS = {
    Path(path).stem: path for path in (
        "config/topic4_rev10_d6_1_natural_kmeans_closeout.json",
        "config/topic4_rev10_d6_2_joint_continuous_field_surface.json",
        "config/topic4_rev10_d6_3_joint_field_replication.json",
        "config/topic4_rev11_nlc_frozen_substrate_confirmation.json",
    )
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _candidate_from_record(record_id: str, artifact_root: Path) -> tuple[dict, dict]:
    library, candidate_id = record_id.split("::", 1)
    config_path = ROOT / LIBRARY_CONFIGS[library]
    config = json.loads(config_path.read_text())
    manifest_path = artifact_root / config["output_root"] / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matches = [
        row for row in manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"historical source candidate is not unique: {record_id}")
    return matches[0], {
        "record_id": record_id, "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "manifest": str(manifest_path.relative_to(artifact_root)),
        "manifest_sha256": _sha256(manifest_path),
    }


def _frozen_git_provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    paths = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=ROOT, text=True,
    ).strip()
    expected_hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        expected_hashes[path] = hashlib.sha256(content).hexdigest()
        if expected_hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"freezer path differs from expected commit: {path}")
    if current != expected or dirty:
        raise RuntimeError("freezer commit or paths are not frozen")
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "path_sha256": expected_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path,
                        default=ROOT / "config/topic4_rev12_nd_node_dualmode_refit.json")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    provenance = _frozen_git_provenance(config_path, args.expected_commit)
    rescore_path = artifact_root / config["inputs"]["historical_rescore"]["path"]
    if _sha256(rescore_path) != config["inputs"]["historical_rescore"]["sha256"]:
        raise RuntimeError("historical rescore changed after config freeze")
    rescore = json.loads(rescore_path.read_text())
    best_id = rescore["best_distinct_field_id"]
    fields = {row["field_id"]: row for row in rescore["field_rows"]}
    if best_id not in fields:
        raise RuntimeError("best distinct field is absent from historical rescore")
    best_field = fields[best_id]
    source_record = sorted(best_field["source_records"])[0]
    distinct, distinct_source = _candidate_from_record(source_record, artifact_root)

    current_distinct = [
        row for row in rescore["field_rows"]
        if row["substrate_stratum"] == "current_spatial_ou_node_only"
        and row["field_sha256"] != rescore["anchor_field_sha256"]
    ]
    geometry_field = max(
        current_distinct,
        key=lambda row: (
            float(row["model_prototype_r2_on_heldout"]),
            -float(row["mean_patient_objective"]),
        ),
    )
    geometry_record = sorted(geometry_field["source_records"])[0]
    geometry, geometry_source = _candidate_from_record(
        geometry_record, artifact_root,
    )

    anchor_record = (
        "topic4_rev11_nlc_frozen_substrate_confirmation::node_baseline"
    )
    anchor, anchor_source = _candidate_from_record(anchor_record, artifact_root)
    if anchor["node_field"]["field_sha256"] != rescore["anchor_field_sha256"]:
        raise RuntimeError("anchor field hash changed")
    if distinct["node_field"]["field_sha256"] != best_field["field_sha256"]:
        raise RuntimeError("distinct field hash changed")
    if geometry["node_field"]["field_sha256"] != geometry_field["field_sha256"]:
        raise RuntimeError("geometry field hash changed")
    if distinct["node_field"]["field_sha256"] == anchor["node_field"]["field_sha256"]:
        raise RuntimeError("best distinct field collapsed onto the anchor")

    candidates = [
        {
            "candidate_id": "anchor_field",
            "role": "current_frozen_node_reference",
            "node_field": anchor["node_field"],
            "source": anchor_source,
        },
        {
            "candidate_id": "historical_distinct_field",
            "role": "best_distinct_current_spatial_ou_historical_initialization",
            "node_field": distinct["node_field"],
            "source": distinct_source,
        },
    ]
    if geometry["node_field"]["field_sha256"] not in {
            row["node_field"]["field_sha256"] for row in candidates}:
        candidates.append({
            "candidate_id": "historical_geometry_field",
            "role": "best_distinct_historical_heldout_geometry_initialization",
            "node_field": geometry["node_field"],
            "source": geometry_source,
        })
    payload = {
        "schema_id": "topic4_rev12_nd_node_canary_manifest_v1",
        "status": "REV12ND_NODE_CANARY_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "seed_pools": config["search"],
        "selection_boundary": (
            "The second field is selected after pooling matching current-spatial-OU "
            "records by field hash and counting each network seed once."
        ),
        "inputs": {
            "historical_rescore": str(rescore_path.relative_to(artifact_root)),
            "historical_rescore_sha256": _sha256(rescore_path),
        },
        "provenance": provenance,
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "candidates": [row["candidate_id"] for row in candidates],
    }, indent=2))


if __name__ == "__main__":
    main()
