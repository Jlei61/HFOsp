#!/usr/bin/env python3
"""Freeze the full Stage-C field library under the cascade-event contract."""
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


def build_manifest_payload(config: dict, source: dict, *, inputs: dict,
                           config_path: str, config_sha256: str,
                           git_commit: str) -> dict:
    if config.get("event_unit", {}).get("name") != "spatiotemporal_cascade":
        raise RuntimeError("historical library must use cascade events")
    if config["event_unit"].get("contact_geometry_used_for_boundary") is not False:
        raise RuntimeError("contact geometry cannot define cascade boundaries")
    if config["field_search"].get("new_field_parameters_released") is not False:
        raise RuntimeError("historical library cannot release new field parameters")
    candidates = list(source["candidates"])
    identifiers = [str(row["candidate_id"]) for row in candidates]
    if len(candidates) != 18 or len(set(identifiers)) != 18:
        raise RuntimeError("corrected historical library requires 18 unique fields")
    return {
        "schema_id": "topic4_rev12_cascade_historical_library_manifest_v1",
        "status": "REV12ND_CASCADE_HISTORICAL_LIBRARY_FROZEN",
        "config": config_path,
        "config_sha256": config_sha256,
        "candidates": [{
            **row,
            "cascade_library_role": "historical_field_under_corrected_objective",
        } for row in candidates],
        "event_unit": config["event_unit"],
        "cascade_objective": config["cascade_objective"],
        "inputs": inputs,
        "provenance": {"git_commit": git_commit},
        "new_field_parameters_released": False,
    }


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
        raise RuntimeError("cascade freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("cascade runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"cascade input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": _sha256(path)}
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    payload = build_manifest_payload(
        config, source, inputs=inputs,
        config_path=str(config_path.relative_to(ROOT)),
        config_sha256=_sha256(config_path), git_commit=expected,
    )
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "n_candidates": len(payload["candidates"]),
    }, indent=2))


if __name__ == "__main__":
    main()
