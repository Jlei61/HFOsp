#!/usr/bin/env python3
"""Freeze observation-free residuals around the accepted dual Node field."""
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

from src.topic4_rev17_dual_field_residual import build_residual_atlas  # noqa: E402


STATUS = "REV17_DUAL_FIELD_RESIDUAL_ATLAS_FROZEN"
PREPARE_STATUS = "REV17_DUAL_FIELD_RESIDUAL_ATLAS_PREPARED_NO_SNN"


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


def _provenance(config_path: Path, expected_commit: str | None,
                require_clean: bool) -> dict:
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_rev17_dual_field_residual.py",
        "src/topic4_node_field_search.py",
        "src/topic4_observation_invariant_spline.py",
        "src/topic4_core_field_rev9.py",
        "src/topic4_zm_ictal_transition.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
    ]
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    expected = None
    if expected_commit is not None:
        expected = subprocess.check_output(
            ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
        ).strip()
        if current != expected:
            raise RuntimeError("rev17 freezer is not at the expected commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip()
    if require_clean and dirty:
        raise RuntimeError("rev17 formal runtime paths are dirty")
    hashes = {relative: _sha256(ROOT / relative) for relative in tracked}
    if expected is not None:
        for relative, observed in hashes.items():
            content = subprocess.check_output(
                ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
            )
            if hashlib.sha256(content).hexdigest() != observed:
                raise RuntimeError(f"rev17 tracked path differs from commit: {relative}")
    return {
        "git_commit": current, "expected_git_commit": expected,
        "tracked_paths": tracked, "tracked_sha256": hashes,
        "dirty": bool(dirty), "formal_ready": bool(expected and not dirty),
    }


def build_manifest(config_path: Path, *, artifact_root: Path,
                   provenance: dict, status: str) -> dict:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_dual_field_residual_atlas_v1":
        raise RuntimeError("rev17 config schema changed")
    inputs = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"rev17 input changed: {name}")
        inputs[name] = {"path": str(path), "sha256": observed, "verified": True}
    source_path = _resolve(
        artifact_root, config["inputs"]["source_manifest"]["path"],
    )
    source = json.loads(source_path.read_text())
    exact = source.get("exact_off_reconstruction")
    if not isinstance(exact, dict):
        raise RuntimeError("rev17 source manifest lacks exact dual reconstruction")
    design = config["dual_field_residual"]
    candidates, audit = build_residual_atlas(
        exact, maximum_frequency=int(design["maximum_frequency"]),
        amplitude=float(design["amplitude"]),
        target_n_basis=int(design["target_n_basis"]),
        degree=int(design["degree"]), sheet_mm=float(design["sheet_mm"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("rev17 expected candidate count changed")
    return {
        "schema_id": "topic4_rev17_dual_field_residual_manifest_v1",
        "status": status,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "residual_atlas_audit": audit,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal freeze requires --expected-commit")
    config_path = args.config.resolve()
    provenance = _provenance(
        config_path, args.expected_commit, require_clean=not args.prepare_only,
    )
    payload = build_manifest(
        config_path, artifact_root=args.artifact_root.resolve(),
        provenance=provenance,
        status=PREPARE_STATUS if args.prepare_only else STATUS,
    )
    if not args.prepare_only:
        config = json.loads(config_path.read_text())
        _atomic_json(
            args.artifact_root.resolve() / config["candidate_manifest"], payload,
        )
    print(json.dumps({
        "status": payload["status"],
        "candidate_count": len(payload["candidates"]),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
