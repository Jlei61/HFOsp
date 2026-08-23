#!/usr/bin/env python3
"""Freeze two non-selective fields for the population-excursion canary."""
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


def build_manifest_payload(config: dict, source: dict, *, audit_status: str,
                           config_path: str, config_sha256: str,
                           inputs: dict, git_commit: str) -> dict:
    """Build the non-selective canary manifest after all hashes are checked."""
    if audit_status != "REV12ND_POPULATION_EXCURSION_AUDIT_COMPLETE":
        raise RuntimeError("population-excursion audit is incomplete")
    event_unit = config.get("event_unit", {})
    if event_unit.get("name") != "population_excursion":
        raise RuntimeError("canary must use the population-excursion event unit")
    if event_unit.get("contact_geometry_used_for_boundary") is not False:
        raise RuntimeError("contact geometry cannot define event boundaries")
    if config.get("field_search", {}).get("purpose") != (
            "event-unit canary only; no candidate selection"):
        raise RuntimeError("canary field selection boundary changed")
    selected = list(config["field_search"]["source_candidate_ids"])
    if len(selected) != 2 or len(set(selected)) != 2:
        raise RuntimeError("canary requires exactly two distinct legacy fields")
    by_id = {str(row["candidate_id"]): row for row in source["candidates"]}
    if any(candidate_id not in by_id for candidate_id in selected):
        raise RuntimeError("event-unit canary candidate is absent")
    candidates = [
        {
            **by_id[candidate_id],
            "event_unit_role": "nonselective_population_excursion_canary",
        }
        for candidate_id in selected
    ]
    return {
        "schema_id": "topic4_rev12_population_excursion_canary_manifest_v1",
        "status": "REV12ND_POPULATION_EXCURSION_CANARY_FROZEN",
        "config": config_path,
        "config_sha256": config_sha256,
        "candidates": candidates,
        "event_unit": event_unit,
        "inputs": inputs,
        "provenance": {"git_commit": git_commit},
        "selection_forbidden": True,
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
        raise RuntimeError("canary freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("population-excursion runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"canary input changed: {record['path']}")
        inputs[key] = path
    audit = json.loads(inputs["population_excursion_audit"].read_text())
    source = json.loads(inputs["source_manifest"].read_text())
    payload = build_manifest_payload(
        config, source, audit_status=audit["status"],
        config_path=str(config_path.relative_to(ROOT)),
        config_sha256=_sha256(config_path),
        inputs={
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        git_commit=expected,
    )
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "n_candidates": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
