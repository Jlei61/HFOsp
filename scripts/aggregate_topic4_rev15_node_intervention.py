#!/usr/bin/env python3
"""Aggregate the three-network crossed hotspot intervention."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

from scripts.run_topic4_rev15_node_intervention_worker import WORKER_STATUS
from src.topic4_node_intervention import crossed_hotspot_selectivity


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_node_intervention.json"
EXPECTED_CONFIG_SCHEMA = "topic4_rev15_node_crossed_intervention_v1"
EXPECTED_WORKER_SCHEMA = "topic4_rev15_node_crossed_intervention_worker_v1"
OUTPUT_SCHEMA = "topic4_rev15_node_crossed_intervention_aggregate_v1"
FREEZE_SCHEMA = "topic4_rev15_frozen_node_field_v1"
FROZEN_STATUS = "REV15_NODE_FIELD_FROZEN"
NOT_SELECTIVE_STATUS = "REV15_NODE_INTERVENTION_NOT_SELECTIVE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _provenance() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()
    return {"analysis_commit": head, "worktree_status": status,
            "formal_ready": not status, "SNN_simulation_run": False}


def _validate_worker(path: Path, *, seed: int, candidate_id: str,
                     config_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(f"intervention worker is missing: seed {seed}")
    payload = json.loads(path.read_text())
    if payload.get("schema_id") != EXPECTED_WORKER_SCHEMA:
        raise RuntimeError(f"intervention worker schema changed: seed {seed}")
    if payload.get("status") != WORKER_STATUS:
        raise RuntimeError(f"intervention worker is incomplete: seed {seed}")
    if int(payload.get("network_seed")) != int(seed):
        raise RuntimeError("intervention worker seed changed")
    if payload.get("candidate_id") != candidate_id:
        raise RuntimeError("intervention worker candidate changed")
    if payload.get("mechanism_freeze") != {
        "EE": "off", "E_to_I": "off", "Z_M": "off",
    }:
        raise RuntimeError("intervention worker activated another mechanism")
    if payload.get("inputs", {}).get("config", {}).get("sha256") != config_sha256:
        raise RuntimeError("intervention worker config changed")
    arrays = payload.get("arrays", {})
    arrays_path = Path(str(arrays.get("path", "")))
    if not arrays_path.is_file() or _sha256(arrays_path) != arrays.get("sha256"):
        raise RuntimeError("intervention worker arrays changed")
    provenance = payload.get("provenance", {})
    if provenance.get("formal_ready") is not True:
        raise RuntimeError("intervention worker provenance is invalid")
    parity = payload.get("projection_parity", {}).get("exact_array_parity", {})
    if set(parity) != {"h", "vtheta", "delta_vtheta"} or not all(parity.values()):
        raise RuntimeError("intervention worker field reconstruction lacks parity")
    native = payload.get("native_modes", {})
    for mode in (0, 1):
        branches = native.get(str(mode), {})
        required = {
            "sham", "mode0_hotspot", "mode0_matched_off_template",
            "mode1_hotspot", "mode1_matched_off_template",
        }
        if not required.issubset(branches):
            raise RuntimeError("intervention worker lacks a crossed branch")
        if not branches["sham"].get("event_occurred"):
            raise RuntimeError("intervention sham did not reproduce its event")
        for arm in required - {"sham"}:
            if branches[arm].get("pre_intervention_spike_parity") is not True:
                raise RuntimeError("intervention branch lacks pre-pulse parity")
    return payload, {
        "network_seed": int(seed), "json": {"path": str(path), "sha256": _sha256(path)},
        "npz": {"path": str(arrays_path), "sha256": _sha256(arrays_path)},
    }


def _effect_rows(selectivity: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, result in selectivity["modes"].items():
        for row in result["per_network"]:
            rows.append({"hotspot_mode": int(mode), **row})
    return rows


def aggregate(*, config_path: Path = DEFAULT_CONFIG,
              artifact_root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve()
    artifact_root = artifact_root.resolve()
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != EXPECTED_CONFIG_SCHEMA:
        raise RuntimeError("intervention config schema changed")
    output_root = artifact_root / config["output_root"]
    worker_root = output_root / "workers"
    payloads, inputs = [], []
    config_hash = _sha256(config_path)
    for seed in config["network_seeds"]:
        payload, record = _validate_worker(
            worker_root / f"intervention_seed_{int(seed)}.json",
            seed=int(seed), candidate_id=str(config["candidate_id"]),
            config_sha256=config_hash,
        )
        payloads.append(payload)
        inputs.append(record)
    selectivity = crossed_hotspot_selectivity(
        [{
            "network_seed": int(payload["network_seed"]),
            "native_modes": payload["native_modes"],
        } for payload in payloads],
        required_networks=int(config["decision"]["required_selective_networks"]),
    )
    provenance = _provenance()
    freeze = bool(selectivity["node_freeze_permitted"] and provenance["formal_ready"])
    status = (
        FROZEN_STATUS if freeze
        else "INVALID_PROVENANCE" if not provenance["formal_ready"]
        else NOT_SELECTIVE_STATUS
    )
    output_json = output_root / "analysis/node_intervention_aggregate.json"
    output_csv = output_root / "analysis/node_intervention_effects.csv"
    freeze_path = output_root / "analysis/node_freeze_manifest.json"
    result = {
        "schema_id": OUTPUT_SCHEMA,
        "status": status,
        "candidate_id": config["candidate_id"],
        "network_seeds": config["network_seeds"],
        "selectivity": selectivity,
        "node_freeze_permitted": freeze,
        "inputs": {
            "config": {"path": str(config_path), "sha256": config_hash},
            "workers": inputs,
            "hashed_inputs": config["inputs"],
        },
        "provenance": provenance,
        "mechanism_freeze": config["mechanism_freeze"],
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_json), "effects_csv": str(output_csv),
                    "freeze_manifest": str(freeze_path) if freeze else None},
    }
    _atomic_json(output_json, result)
    rows = _effect_rows(selectivity)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if freeze:
        _atomic_json(freeze_path, {
            "schema_id": FREEZE_SCHEMA,
            "status": "FROZEN",
            "candidate_id": config["candidate_id"],
            "network_seeds": config["network_seeds"],
            "intervention_aggregate": {
                "path": str(output_json), "sha256": _sha256(output_json),
            },
            "robust_config": config["inputs"]["robust_config"],
            "robust_manifest": config["inputs"]["robust_manifest"],
            "final_science_audit": config["inputs"]["final_science_audit"],
            "worker_inputs": inputs,
            "EE_EtoI_ZM": "off",
            "claim_boundary": config["claim_boundary"],
        })
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    payload = aggregate(config_path=args.config, artifact_root=args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "node_freeze_permitted": payload["node_freeze_permitted"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
