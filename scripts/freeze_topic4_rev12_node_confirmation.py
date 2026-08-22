#!/usr/bin/env python3
"""Freeze the selected Node field and original Node baseline for confirmation."""
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


def _by_id(manifest: dict) -> dict[str, dict]:
    rows = {str(row["candidate_id"]): row for row in manifest["candidates"]}
    if len(rows) != len(manifest["candidates"]):
        raise RuntimeError("candidate manifest contains duplicate ids")
    return rows


def build_confirmation_candidates(
    baseline_manifest: dict,
    selected_manifest: dict,
    selection_decision: dict,
    baseline_id: str,
    selected_id: str,
) -> list[dict]:
    if selection_decision["status"] != "REV12ND_NODE_SELECTION_COMPLETE":
        raise RuntimeError("selection decision is not complete")
    if selection_decision["selected_candidate_id"] != selected_id:
        raise RuntimeError("configured selected candidate differs from frozen decision")
    baseline_rows = _by_id(baseline_manifest)
    selected_rows = _by_id(selected_manifest)
    if baseline_id not in baseline_rows or selected_id not in selected_rows:
        raise RuntimeError("confirmation candidate is absent from its source manifest")
    baseline = dict(baseline_rows[baseline_id])
    selected = dict(selected_rows[selected_id])
    baseline["confirmation_role"] = "frozen_fig4_node_baseline"
    selected["confirmation_role"] = "pareto_selected_node_field"
    return [baseline, selected]


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    paths = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if head != expected:
        raise RuntimeError("confirmation freezer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("confirmation freezer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"confirmation freezer path drifted: {path}")
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
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"confirmation input changed: {record['path']}")
        inputs[key] = path
    candidates = build_confirmation_candidates(
        json.loads(inputs["baseline_manifest"].read_text()),
        json.loads(inputs["selected_manifest"].read_text()),
        json.loads(inputs["selection_decision"].read_text()),
        config["confirmation"]["baseline_candidate_id"],
        config["confirmation"]["selected_candidate_id"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_node_confirmation_manifest_v1",
        "status": "REV12ND_NODE_CONFIRMATION_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "confirmation_contract": {
            "paired_fresh_network_seeds": config["search"][
                "confirmation_network_seeds"
            ],
            **config["confirmation"],
            "complete_returned_events_are_primary": True,
            "selection_cannot_be_reopened_from_confirmation": True,
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "provenance": _provenance(config_path, args.expected_commit),
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "output": str(output),
        "candidate_ids": [row["candidate_id"] for row in candidates],
    }, indent=2))


if __name__ == "__main__":
    main()
