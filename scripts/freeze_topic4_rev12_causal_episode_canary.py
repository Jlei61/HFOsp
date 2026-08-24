#!/usr/bin/env python3
"""Freeze a non-selective causal-population-episode canary manifest."""
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


def build_manifest(config: dict, source: dict, *, config_path: str,
                   config_sha256: str, inputs: dict, commit: str) -> dict:
    event_unit = config["event_unit"]
    if (event_unit.get("name") != "causal_population_excursion"
            or event_unit.get("contact_geometry_used_for_boundary") is not False
            or event_unit.get("root_role") != (
                "topology_annotation_not_event_splitting_or_exclusion")):
        raise RuntimeError("causal population episode contract changed")
    if config["field_search"]["purpose"] != (
            "event-unit canary only; no candidate selection"):
        raise RuntimeError("event canary unexpectedly permits field selection")
    selected = list(config["field_search"]["source_candidate_ids"])
    by_id = {str(row["candidate_id"]): row for row in source["candidates"]}
    if len(selected) != len(set(selected)) or any(key not in by_id for key in selected):
        raise RuntimeError("causal-episode candidate inventory is invalid")
    return {
        "schema_id": "topic4_rev12_causal_episode_canary_manifest_v1",
        "status": "REV12ND_CAUSAL_EPISODE_CANARY_FROZEN",
        "config": config_path,
        "config_sha256": config_sha256,
        "candidates": [
            {**by_id[key], "event_unit_role": "nonselective_causal_episode_canary"}
            for key in selected
        ],
        "event_unit": event_unit,
        "inputs": inputs,
        "selection_forbidden": True,
        "provenance": {"git_commit": commit},
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
        raise RuntimeError("causal-episode freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_dualmode.py",
        "scripts/audit_topic4_rev12_population_excursion_canary.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("causal-episode freezer paths are dirty")
    checked = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"causal-episode input changed: {record['path']}")
        checked[key] = {"path": str(path), "sha256": _sha256(path)}
    source = json.loads(_resolve(
        artifact_root, config["inputs"]["source_manifest"]["path"],
    ).read_text())
    payload = build_manifest(
        config, source, config_path=str(config_path.relative_to(ROOT)),
        config_sha256=_sha256(config_path), inputs=checked, commit=expected,
    )
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "n_candidates": len(payload["candidates"]),
    }, indent=2))


if __name__ == "__main__":
    main()
