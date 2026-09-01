#!/usr/bin/env python3
"""Freeze the rev20-DC one-factor candidate atlas."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev20_dual_core_mechanism import (  # noqa: E402
    build_one_factor_candidates, dual_core_field_sha256,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if current_commit != expected_commit:
        raise RuntimeError("candidate atlas must be frozen at expected HEAD")
    config_at_commit = subprocess.check_output(
        ["git", "show", f"{expected_commit}:{config_path.relative_to(ROOT)}"],
        cwd=ROOT,
    )
    if hashlib.sha256(config_at_commit).hexdigest() != _sha256(config_path):
        raise RuntimeError("config differs from expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        "src/topic4_rev20_dual_core_mechanism.py",
        "src/topic4_manual_dual_core.py",
        "scripts/freeze_topic4_rev20_dc_dual_core_mechanism.py",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("freezer runtime paths are dirty")

    verified_inputs = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"input hash changed: {name}")
        verified_inputs[name] = {"path": str(path), "sha256": observed}

    anchor = config["dual_core_anchor"]
    descriptor_hash = dual_core_field_sha256(
        anchor["centers_mm"], anchor["target_count"],
    )
    if descriptor_hash != anchor["field_sha256"]:
        raise RuntimeError("dualcore_s39 descriptor hash changed")
    source_blob = subprocess.check_output([
        "git", "show",
        f"{anchor['source_commit']}:{anchor['source_path']}",
    ], cwd=ROOT)
    if hashlib.sha256(source_blob).hexdigest() != anchor["source_blob_sha256"]:
        raise RuntimeError("historical dual-core source blob changed")

    candidates = build_one_factor_candidates(config)
    manifest = {
        "schema_id": "topic4_rev20_dc_candidate_manifest_v1",
        "status": "REV20_DC_DUAL_CORE_MECHANISM_ATLAS_FROZEN",
        "scientific_role": config["scientific_role"],
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "git_commit": current_commit,
        "dual_core_anchor": anchor,
        "historical_source_verified": True,
        "verified_inputs": verified_inputs,
        "candidate_count": len(candidates),
        "candidate_ids_sha256": hashlib.sha256(
            "\n".join(row["candidate_id"] for row in candidates).encode("ascii")
        ).hexdigest(),
        "candidates": candidates,
        "selection_boundary": (
            "patient-training unconditional complete-event distribution only; "
            "held-out, KMeans and OOD remain closed until level selection"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "candidate_count": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
