#!/usr/bin/env python3
"""Freeze the persistent-root coactivity event canary."""
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


def validate_event_canary_contract(config: dict) -> None:
    """Reject any drift that turns the event canary into field selection."""
    event_unit = config["event_unit"]
    name = str(event_unit["name"])
    if (name not in {
            "persistent_root_coactivity_episode", "causal_root_observation"}
            or event_unit["contact_geometry_used_for_boundary"]):
        raise RuntimeError("root-coactivity event contract drifted")
    if name == "persistent_root_coactivity_episode":
        expected_inclusion = "all positive activity mass; no dominance exclusion"
        expected_purpose = "engine-derived event identity canary only; no field selection"
    else:
        expected_inclusion = (
            "one directed causal root per evaluable observation; compound detector "
            "fragments retained outside KMeans"
        )
        expected_purpose = "causal-root event identity canary only; no field selection"
    if event_unit["root_inclusion"] != expected_inclusion:
        raise RuntimeError("causal-root inclusion contract drifted")
    if event_unit.get("causal_memory_method") != "local_ee_psp_tail":
        raise RuntimeError("engine-derived causal-memory method drifted")
    primary = float(event_unit["psp_tail_fraction"])
    fractions = [
        float(value) for value in event_unit["sensitivity_psp_tail_fractions"]
    ]
    if (not 0.0 < primary < 1.0
            or primary not in fractions
            or any(not 0.0 < value < 1.0 for value in fractions)
            or len(set(fractions)) != len(fractions)
            or float(event_unit["local_ee_delay_quantile"]) != 1.0):
        raise RuntimeError("engine-derived causal-memory contract drifted")
    dominances = [
        float(value) for value in event_unit.get(
            "sensitivity_minimum_dominances", [event_unit["minimum_dominance"]],
        )
    ]
    if (float(event_unit["minimum_dominance"]) not in dominances
            or any(not 0.5 < value <= 1.0 for value in dominances)
            or len(set(dominances)) != len(dominances)):
        raise RuntimeError("causal-root purity sensitivity drifted")
    field_search = config["field_search"]
    if (bool(field_search["new_field_parameters_released"])
            or field_search["purpose"] != expected_purpose):
        raise RuntimeError("event canary cannot release or select a Node field")


def event_canary_identity(event_name: str) -> tuple[str, str]:
    if str(event_name) == "causal_root_observation":
        return (
            "topic4_rev12_causal_root_canary_manifest_v1",
            "REV12ND_CAUSAL_ROOT_CANARY_FROZEN",
        )
    return (
        "topic4_rev12_root_coactivity_canary_manifest_v1",
        "REV12ND_ROOT_COACTIVITY_CANARY_FROZEN",
    )


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
        raise RuntimeError("root-coactivity freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py", "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("root-coactivity runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"root-coactivity input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    audit = json.loads(Path(inputs["causal_episode_audit"]["path"]).read_text())
    if audit["status"] != "REV12ND_POPULATION_EXCURSION_CANARY_AUDIT_COMPLETE":
        raise RuntimeError("causal-population episode audit is incomplete")
    if "root_coactivity_audit" in inputs:
        root_audit = json.loads(Path(inputs["root_coactivity_audit"]["path"]).read_text())
        if root_audit["status"] != "REV12ND_ROOT_COACTIVITY_CANARY_AUDIT_COMPLETE":
            raise RuntimeError("predecessor root-coactivity audit is incomplete")
    validate_event_canary_contract(config)
    event_unit = config["event_unit"]
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    by_id = {row["candidate_id"]: row for row in source["candidates"]}
    requested = list(config["field_search"]["candidate_ids"])
    if len(set(requested)) != len(requested) or any(key not in by_id for key in requested):
        raise RuntimeError("root-coactivity canary candidates are invalid")
    schema_id, status = event_canary_identity(event_unit["name"])
    payload = {
        "schema_id": schema_id,
        "status": status,
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": [by_id[key] for key in requested],
        "event_unit": event_unit,
        "contact_readout": config["search"]["contact_readout"],
        "inputs": inputs,
        "selection_forbidden": True,
        "provenance": {"git_commit": expected},
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
