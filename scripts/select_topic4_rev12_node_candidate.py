#!/usr/bin/env python3
"""Apply the frozen rev12-ND Pareto-knee rule to a completed selection pool."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from src.topic4_node_selection import select_pareto_knee  # noqa: E402


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


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    paths = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_selection.py",
    ]
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("selection producer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("selection producer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"selection producer path drifted: {path}")
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
    output_root = artifact_root / config["output_root"]
    summary_path = output_root / "aggregate" / "selection_summary.json"
    manifest_path = artifact_root / config["candidate_manifest"]
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    requested = [int(seed) for seed in config["search"]["selection_network_seeds"]]
    if summary["requested_seeds"] != requested:
        raise RuntimeError("selection seed contract changed")
    if {row["candidate_id"] for row in summary["rows"]} != {
            row["candidate_id"] for row in manifest["candidates"]}:
        raise RuntimeError("selection summary does not cover the frozen manifest")
    if any(int(row["n_networks"]) != len(requested) for row in summary["rows"]):
        raise RuntimeError("selection coverage is incomplete")
    decision = select_pareto_knee(summary["rows"])
    payload = {
        "schema_id": "topic4_rev12_nd_node_selection_decision_v1",
        "status": "REV12ND_NODE_SELECTION_COMPLETE",
        "scientific_role": config["scientific_role"],
        **decision,
        "inputs": {
            "selection_summary": str(summary_path),
            "selection_summary_sha256": _sha256(summary_path),
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": _sha256(manifest_path),
        },
        "provenance": _provenance(config_path, args.expected_commit),
    }
    output = output_root / "aggregate" / "selection_decision.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "selected_candidate_id": payload["selected_candidate_id"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
