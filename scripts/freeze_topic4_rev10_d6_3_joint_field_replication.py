"""Freeze the warm-versus-joint-field D6.3 replication pair."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_3_joint_field_replication.json"
EXPECTED_ROLE = "development_only_continuous_field_joint_direction_replication"


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D6.3 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source = json.loads((ROOT / config["inputs"]["d6_2_manifest"]["path"]).read_text())
    verdict = json.loads((ROOT / config["inputs"]["d6_2_verdict"]["path"]).read_text())
    if source.get("status") != "REV10D6_2_JOINT_CONTINUOUS_FIELD_SURFACE_FROZEN":
        raise RuntimeError("D6.2 source library is not frozen")
    if verdict.get("status") != "REV10D6_2_JOINT_CONTINUOUS_FIELD_SIGNAL_NOT_OBSERVED":
        raise RuntimeError("D6.3 requires the frozen D6.2 near-boundary result")
    replication_id = config["field_replication"]["replication_candidate_id"]
    if verdict.get("diagnostic_display_candidate_id") != replication_id:
        raise RuntimeError("D6.3 candidate differs from the frozen D6.2 display rule")
    by_id = {
        row["candidate_id"]: row
        for row in source["candidate_set"]["candidates"]
    }
    candidate_ids = [
        config["field_replication"]["baseline_candidate_id"], replication_id,
    ]
    candidates = [deepcopy(by_id[candidate_id]) for candidate_id in candidate_ids]
    expected_coordinates = config["field_replication"]["replication_coordinates"]
    if candidates[1].get("d6_2_latent_coordinates") != expected_coordinates:
        raise RuntimeError("D6.3 latent coordinates changed")
    if len(candidates) != int(config["field_replication"]["candidate_count"]):
        raise RuntimeError("D6.3 candidate count changed")
    if any(row["node_field"].get("component_count") is not None for row in candidates):
        raise RuntimeError("D6.3 cannot introduce components")
    if any(np.any(np.asarray(row["coefficients"], float) != 0.0) for row in candidates):
        raise RuntimeError("D6.3 edge adapter must remain an exact no-op")
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (provenance["config_dirty"] or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("D6.3 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D6.3 workers exist before manifest freeze")
    return {
        "status": "REV10D6_3_JOINT_FIELD_REPLICATION_PAIR_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates), "candidates": candidates},
        "selection_freeze": {
            "paired_control_candidate_id": candidate_ids[0],
            "baseline_candidate_id": candidate_ids[0],
            "selected_nonzero_candidate_id": replication_id,
            "primary_candidate_id": replication_id,
            "selection_source": config["field_replication"]["selection_source"],
            "frozen_before_all_D6_3_networks": True,
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": source["direction_classifier_source"],
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "spatial_ou": config["fixed_spatial_ou"],
            "edge": "exact no-op", "beta": "closed",
        },
        "forbidden_builder_inputs": config["field_replication"]["forbidden_builder_inputs"],
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build_manifest(args.config, args.expected_commit)
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "candidate_ids": [row["candidate_id"] for row in payload["candidate_set"]["candidates"]],
        "network_seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
