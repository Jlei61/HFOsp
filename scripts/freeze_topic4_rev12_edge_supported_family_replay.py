#!/usr/bin/env python3
"""Freeze a small exact replay under the edge-supported event contract."""
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
        raise RuntimeError("edge-supported freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("edge-supported replay runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"edge-supported replay input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    source = json.loads(Path(inputs["source_fit_manifest"]["path"]).read_text())
    by_id = {row["candidate_id"]: row for row in source["candidates"]}
    candidate_ids = list(config["field_replay"]["candidate_ids"])
    if any(candidate_id not in by_id for candidate_id in candidate_ids):
        raise RuntimeError("edge-supported replay candidate is absent from fit manifest")
    audit = json.loads(Path(
        inputs["edge_supported_event_identity_audit"]["path"]
    ).read_text())
    if audit.get("schema_id") != "topic4_rev12_edge_supported_event_identity_audit_v1":
        raise RuntimeError("edge-supported event audit schema changed")
    if audit["verdict"].get("primary_clean_to_compound") != 0:
        raise RuntimeError("edge-supported audit makes clean detector events ambiguous")
    candidates = [by_id[candidate_id] for candidate_id in candidate_ids]
    payload = {
        "schema_id": "topic4_rev12_edge_supported_family_replay_manifest_v1",
        "status": "REV12ND_EDGE_SUPPORTED_FAMILY_REPLAY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "inputs": inputs,
        "provenance": {"git_commit": expected},
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(candidates),
        "network_seeds": config["search"]["fit_network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
