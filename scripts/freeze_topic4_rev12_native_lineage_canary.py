#!/usr/bin/env python3
"""Freeze the native-worker causal-lineage parity canary."""
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


def build_manifest_payload(config: dict, source: dict, *, config_path: str,
                           config_sha256: str, inputs: dict,
                           git_commit: str) -> dict:
    if config["event_unit"]["name"] != "directed_spatiotemporal_lineage":
        raise RuntimeError("native parity canary requires directed lineages")
    if config["search"]["contact_readout"]["source"] != (
            "lineage_restricted_sheet_activity"):
        raise RuntimeError("native parity canary requires root-restricted readout")
    requested = list(config["field_search"]["candidate_ids"])
    rows = {
        str(row["candidate_id"]): row for row in source["candidates"]
    }
    if any(candidate_id not in rows for candidate_id in requested):
        raise RuntimeError("native parity field is absent from the source manifest")
    candidates = [rows[candidate_id] for candidate_id in requested]
    return {
        "schema_id": "topic4_rev12_nd_native_lineage_canary_manifest_v1",
        "status": "REV12ND_NATIVE_LINEAGE_CANARY_FROZEN",
        "config": config_path,
        "config_sha256": config_sha256,
        "candidates": candidates,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "inputs": inputs,
        "provenance": {"git_commit": git_commit},
        "simulation_rerun": True,
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
        raise RuntimeError("native parity freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py",
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("native parity runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"native parity input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    audit = json.loads(Path(inputs["source_audit"]["path"]).read_text())
    if audit["status"] != "REV12ND_LINEAGE_RESTRICTED_READOUT_AUDIT_COMPLETE":
        raise RuntimeError("source causal-lineage audit is not complete")
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    payload = build_manifest_payload(
        config, source, config_path=str(config_path.relative_to(ROOT)),
        config_sha256=_sha256(config_path), inputs=inputs, git_commit=expected,
    )
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(payload["candidates"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
