#!/usr/bin/env python3
"""Freeze the independent rev11-NLC pathway-mechanism confirmation."""
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
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "config/topic4_rev11_nlc_pathway_mechanism_confirmation.json"
)
EXPECTED_ROLE = (
    "development_only_data_driven_node_local_connectivity_mechanism_confirmation"
)
EXPECTED_SOURCE_STATUS = (
    "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN"
)
OUTPUT_STATUS = "REV11NLC_PATHWAY_MECHANISM_CONFIRMATION_LIBRARY_FROZEN"


def mechanism_library(config, source_manifest):
    candidates = deepcopy(source_manifest["candidate_set"]["candidates"])
    expected_ids = config["candidate_library"]["arms"]
    if [row["candidate_id"] for row in candidates] != expected_ids:
        raise RuntimeError("frozen pathway arm order changed")
    if len(candidates) != int(config["candidate_library"]["candidate_count"]):
        raise RuntimeError("frozen pathway arm count changed")
    if len({row["node_field"]["field_sha256"] for row in candidates}) != 1:
        raise RuntimeError("pathway arms do not share one Node field")
    for candidate in candidates:
        coefficients = np.asarray(candidate["coefficients"], float)
        if array_sha256(coefficients) != candidate["coefficients_sha256"]:
            raise RuntimeError(
                f"coefficient hash changed for {candidate['candidate_id']}"
            )
    return candidates


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("pathway mechanism scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    source_manifest = json.loads(
        (ROOT / config["inputs"]["frozen_confirmation_manifest"]["path"])
        .read_text()
    )
    source_verdict = json.loads(
        (ROOT / config["inputs"]["frozen_confirmation_verdict"]["path"])
        .read_text()
    )
    if source_manifest.get("status") != EXPECTED_SOURCE_STATUS:
        raise RuntimeError("source frozen-substrate library is not accepted")
    if source_verdict.get("status") != (
            "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_PASS"):
        raise RuntimeError("source frozen-substrate verdict is not accepted")
    candidates = mechanism_library(config, source_manifest)

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
        raise RuntimeError("mechanism freezer runtime or config is not frozen")

    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("mechanism workers exist before manifest freeze")
    return {
        "status": OUTPUT_STATUS,
        "scientific_role": EXPECTED_ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
        "selection_freeze": {
            "paired_control_candidate_id": "node_baseline",
            "selected_joint_candidate_id": config[
                "candidate_library"
            ]["selected_candidate_id"],
            "all_candidates_copied_without_refit": True,
            "network_seed_is_independent_unit": True,
        },
        "direction_classifier": deepcopy(source_manifest["direction_classifier"]),
        "direction_classifier_source": deepcopy(
            source_manifest["direction_classifier_source"]
        ),
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "spatial_ou": config["fixed_spatial_ou"],
            "mechanism_readout": config["mechanism_readout"],
            "topology": "frozen",
            "delays": "frozen",
            "GABA": "frozen",
            "beta": "closed",
            "Z_M": "off",
        },
        "source_confirmation": {
            "manifest": config["inputs"]["frozen_confirmation_manifest"],
            "verdict": config["inputs"]["frozen_confirmation_verdict"],
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "candidate_ids": [
            row["candidate_id"]
            for row in payload["candidate_set"]["candidates"]
        ],
        "network_seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
