#!/usr/bin/env python3
"""Freeze exact plus one selected rev17 dual field for unseen confirmation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


STATUS = "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
PREPARE_STATUS = "REV17_NODE_CONFIRMATION_CANDIDATES_PREPARED_NO_SNN"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else root / relative


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
                require_clean: bool) -> dict[str, Any]:
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/prepare_topic4_rev17_node_confirmation.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "src/topic4_core_field_rev9.py", "src/topic4_zm_ictal_transition.py",
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
            raise RuntimeError("rev17 confirmation freezer is not at expected commit")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip()
    if require_clean and dirty:
        raise RuntimeError("rev17 confirmation runtime paths are dirty")
    hashes = {relative: _sha256(ROOT / relative) for relative in tracked}
    if expected is not None:
        for relative, observed in hashes.items():
            content = subprocess.check_output(
                ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
            )
            if hashlib.sha256(content).hexdigest() != observed:
                raise RuntimeError(f"rev17 confirmation path differs from commit: {relative}")
    return {
        "git_commit": current, "expected_git_commit": expected,
        "tracked_paths": tracked, "tracked_sha256": hashes,
        "dirty": bool(dirty), "formal_ready": bool(expected and not dirty),
    }


def build_manifest(config_path: Path, *, artifact_root: Path,
                   provenance: dict[str, Any], status: str) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_node_confirmation_v1":
        raise RuntimeError("rev17 confirmation schema changed")
    if config.get("pathways") != {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off", "Z_M": "off",
    }:
        raise RuntimeError("rev17 confirmation opened a forbidden pathway")
    inputs = {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev17 confirmation input changed: {name}")
        inputs[name] = {"path": str(path), "sha256": record["sha256"], "verified": True}
    source = json.loads(Path(inputs["selection_manifest"]["path"]).read_text())
    if source.get("status") != "REV17_DUAL_FIELD_SELECTION_CANDIDATES_FROZEN":
        raise RuntimeError("rev17 confirmation source manifest is not frozen")
    candidate_id = config["selected_candidate"]["candidate_id"]
    by_id = {row["candidate_id"]: row for row in source["candidates"]}
    if candidate_id not in by_id or "exact_dual_anchor" not in by_id:
        raise RuntimeError("rev17 confirmation source candidates changed")
    selected = copy.deepcopy(by_id[candidate_id])
    if selected["node_mapping"]["mapping_sha256"] != config[
        "selected_candidate"
    ]["mapping_sha256"]:
        raise RuntimeError("rev17 selected dual mapping changed")
    candidates = [copy.deepcopy(by_id["exact_dual_anchor"]), selected]
    return {
        "schema_id": "topic4_rev17_node_confirmation_manifest_v1",
        "status": status, "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path), "candidates": candidates,
        "direction_audit": {
            "candidate_ids": [row["candidate_id"] for row in candidates],
            "field_reranking_performed": False,
            "selection_aggregate_sha256": config["inputs"][
                "selection_aggregate"
            ]["sha256"],
        },
        "event_unit": config["event_unit"],
        "source_topology": config["source_topology"],
        "search": config["search"],
        "confirmation_acceptance": config["confirmation_acceptance"],
        "pathways": config["pathways"], "inputs": inputs,
        "provenance": provenance, "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit")
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only and args.expected_commit is None:
        parser.error("formal freeze requires --expected-commit")
    config_path = args.config.resolve()
    payload = build_manifest(
        config_path, artifact_root=args.artifact_root.resolve(),
        provenance=_provenance(
            config_path, args.expected_commit,
            require_clean=not args.prepare_only,
        ),
        status=PREPARE_STATUS if args.prepare_only else STATUS,
    )
    if not args.prepare_only:
        config = json.loads(config_path.read_text())
        _atomic_json(args.artifact_root.resolve() / config["candidate_manifest"], payload)
    print(json.dumps({
        "status": payload["status"], "candidate_count": 2,
        "confirmation_jobs": 6, "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
