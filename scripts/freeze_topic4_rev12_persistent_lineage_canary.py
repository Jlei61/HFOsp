#!/usr/bin/env python3
"""Freeze the persistent directed-lineage event-identity canary."""
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


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else root / relative


def select_candidates(source: dict, requested: list[str]) -> list[dict]:
    rows = {row["candidate_id"]: row for row in source["candidates"]}
    if len(set(requested)) != len(requested) or any(item not in rows for item in requested):
        raise RuntimeError("persistent-lineage canary candidates are invalid")
    return [rows[item] for item in requested]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("persistent-lineage freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py", "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("persistent-lineage runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"persistent-lineage input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    invalidation = json.loads(Path(
        inputs["event_identity_invalidation"]["path"]
    ).read_text())
    if invalidation["status"] != "INVALIDATED_IMMEDIATE_FRAME_EVENT_IDENTITY":
        raise RuntimeError("immediate-frame fit was not invalidated")
    event_unit = config["event_unit"]
    if (event_unit["name"] != "persistent_directed_spatiotemporal_lineage"
            or event_unit["contact_geometry_used_for_boundary"]):
        raise RuntimeError("persistent-lineage event contract drifted")
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    candidates = select_candidates(source, list(config["field_search"]["candidate_ids"]))
    payload = {
        "schema_id": "topic4_rev12_persistent_lineage_canary_manifest_v1",
        "status": "REV12ND_PERSISTENT_LINEAGE_CANARY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "event_unit": event_unit,
        "contact_readout": config["search"]["contact_readout"],
        "inputs": inputs,
        "provenance": {"git_commit": expected},
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
