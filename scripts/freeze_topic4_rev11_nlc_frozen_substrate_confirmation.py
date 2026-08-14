#!/usr/bin/env python3
"""Freeze the final four-arm rev11-NLC substrate confirmation."""
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


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
EXPECTED_ROLE = (
    "development_only_data_driven_node_local_connectivity_frozen_confirmation"
)


def _ablation(selected, candidate_id, active_pathway):
    row = deepcopy(selected)
    coefficients = np.asarray(row["coefficients"], float)
    if active_pathway == "E_to_E":
        coefficients[1] = 0.0
        arm = "Node+EE"
    elif active_pathway == "E_to_I":
        coefficients[0] = 0.0
        arm = "Node+EtoI"
    else:
        raise ValueError(f"unsupported ablation pathway {active_pathway}")
    row.update({
        "candidate_id": candidate_id,
        "arm": arm,
        "coefficients": coefficients.tolist(),
        "coefficients_sha256": array_sha256(coefficients),
        "search_coordinates": {
            "control": True,
            "mechanism_ablation_of": selected["candidate_id"],
            "active_pathways": [active_pathway],
            "node_amplitude": 0.0,
            "edge_delta": np.zeros_like(coefficients).tolist(),
        },
    })
    return row


def confirmation_library(config, selection_manifest, selection_verdict):
    selected_id = config["candidate_library"]["selected_candidate_id"]
    if (selection_verdict["selected_candidate_id"] != selected_id
            or selection_verdict["winner_type"] != "NLC1_PARENT_CONNECTIVITY_CONTROL"):
        raise RuntimeError("NLC3 selected substrate changed")
    source = {
        row["candidate_id"]: row
        for row in selection_manifest["candidate_set"]["candidates"]
    }
    selected = deepcopy(source[selected_id])
    node = deepcopy(source["node_baseline"])
    rows = [
        node,
        _ablation(selected, "joint_04_ee_only", "E_to_E"),
        _ablation(selected, "joint_04_etoi_only", "E_to_I"),
        selected,
    ]
    expected_ids = config["candidate_library"]["arms"]
    if [row["candidate_id"] for row in rows] != expected_ids:
        raise RuntimeError("NLC3C arm order changed")
    if len(rows) != int(config["candidate_library"]["candidate_count"]):
        raise RuntimeError("NLC3C candidate count changed")
    node_hashes = {row["node_field"]["field_sha256"] for row in rows}
    if len(node_hashes) != 1:
        raise RuntimeError("NLC3C arms do not share one Node field")
    return rows


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("NLC3C scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    selection_manifest = json.loads(
        (ROOT / config["inputs"]["nlc3_selection_manifest"]["path"]).read_text()
    )
    selection_verdict = json.loads(
        (ROOT / config["inputs"]["nlc3_selection_verdict"]["path"]).read_text()
    )
    if selection_manifest.get("status") != (
        "REV11NLC_JOINT_NODE_CONNECTIVITY_SELECTION_LIBRARY_FROZEN"
    ):
        raise RuntimeError("NLC3 selection manifest is not frozen")
    if selection_verdict.get("status") != "REV11NLC_JOINT_SELECTION_COMPLETE":
        raise RuntimeError("NLC3 selection verdict is incomplete")
    candidates = confirmation_library(config, selection_manifest, selection_verdict)
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
        raise RuntimeError("NLC3C freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("NLC3C workers exist before manifest freeze")
    return {
        "status": "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN",
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
            "joint_candidate_copied_without_refit": True,
            "single_pathway_arms_are_deterministic_ablations": True,
            "network_seed_is_independent_unit": True,
        },
        "direction_classifier": deepcopy(selection_manifest["direction_classifier"]),
        "direction_classifier_source": deepcopy(
            selection_manifest["direction_classifier_source"]
        ),
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "spatial_ou": config["fixed_spatial_ou"],
            "topology": "frozen",
            "delays": "frozen",
            "GABA": "frozen",
            "beta": "closed",
            "Z_M": "off",
        },
        "source_selection": {
            "manifest": config["inputs"]["nlc3_selection_manifest"],
            "verdict": config["inputs"]["nlc3_selection_verdict"],
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
            row["candidate_id"] for row in payload["candidate_set"]["candidates"]
        ],
        "network_seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
