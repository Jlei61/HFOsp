#!/usr/bin/env python3
"""Freeze the rev18 observation-invariant multi-coordinate dual-field screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev17_dual_field_candidates import build_candidates  # noqa: E402
from src.topic4_rev18_dual_field_search import global_blueprints  # noqa: E402


STATUS = "REV18_DUAL_FIELD_GLOBAL_SCREEN_FROZEN"


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


def _provenance(config_path: Path, expected_commit: str) -> dict:
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/finish_topic4_rev18_dual_field_global_screen.py",
        "scripts/aggregate_topic4_rev18_dual_field_global_screen.py",
        "src/topic4_rev18_dual_field_search.py",
        "src/topic4_rev17_dual_field_candidates.py",
        "src/topic4_node_field_search.py",
        "src/topic4_rev17_dual_field_residual.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
    ]
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if current != expected:
        raise RuntimeError("rev18 freezer is not at the expected commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("rev18 formal runtime paths are dirty")
    hashes = {}
    for relative in tracked:
        observed = _sha256(ROOT / relative)
        committed = hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )).hexdigest()
        if observed != committed:
            raise RuntimeError(f"rev18 tracked path differs from commit: {relative}")
        hashes[relative] = observed
    return {
        "git_commit": expected,
        "expected_git_commit": expected,
        "tracked_paths": tracked,
        "tracked_sha256": hashes,
        "dirty": False,
        "formal_ready": True,
    }


def build_manifest(config_path: Path, artifact_root: Path, provenance: dict) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev18_dual_field_global_screen_v1":
        raise RuntimeError("rev18 config schema changed")
    inputs = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"rev18 input changed: {name}")
        inputs[name] = {"path": str(path), "sha256": observed, "verified": True}
    source = json.loads(_resolve(
        artifact_root, config["inputs"]["source_atlas_manifest"]["path"],
    ).read_text())
    design = config["dual_field_global_screen"]
    blueprints = global_blueprints(design)
    candidates, audit = build_candidates(
        source, blueprints,
        maximum_frequency=int(design["maximum_frequency"]),
        target_n_basis=int(design["target_n_basis"]),
        degree=int(design["degree"]), sheet_mm=float(design["sheet_mm"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    if len(candidates) != int(design["expected_candidate_count_including_anchor"]):
        raise RuntimeError("rev18 candidate count changed")
    candidates[0]["role"] = "exact_dual_anchor_global_screen_reference"
    for candidate in candidates[1:]:
        candidate["role"] = "rev18_multicoordinate_continuous_dual_field"
    return {
        "schema_id": "topic4_rev18_dual_field_global_screen_manifest_v1",
        "status": STATUS,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "candidate_audit": audit,
        "blueprints": blueprints,
        "event_unit": config["event_unit"],
        "source_topology": config["source_topology"],
        "search": config["search"],
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off",
            "Z_M": "off",
        },
        "inputs": inputs,
        "provenance": provenance,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = build_manifest(
        config_path, args.artifact_root.resolve(),
        _provenance(config_path, args.expected_commit),
    )
    config = json.loads(config_path.read_text())
    destination = args.artifact_root.resolve() / config["candidate_manifest"]
    _atomic_json(destination, payload)
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": len(payload["candidates"]),
        "manifest": str(destination),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
