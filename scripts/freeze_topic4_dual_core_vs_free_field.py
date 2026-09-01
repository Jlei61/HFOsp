#!/usr/bin/env python3
"""Freeze the paired hand dual-core arm against the final free-field Node arm."""
from __future__ import annotations

import argparse
import hashlib
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


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_vs_free_field.json"
EXPECTED_ROLE = (
    "development_only_hand_dual_core_vs_continuous_field_distribution_comparison"
)


def _canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_manifest(config_path: Path, expected_commit: str) -> dict:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("dual-core comparison scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    source_path = ROOT / config["inputs"]["source_confirmation_manifest"]["path"]
    source = json.loads(source_path.read_text())
    if source.get("status") != "REV11NLC_FROZEN_SUBSTRATE_CONFIRMATION_LIBRARY_FROZEN":
        raise RuntimeError("source Figure 4 confirmation manifest is not frozen")
    source_node = next(
        deepcopy(row) for row in source["candidate_set"]["candidates"]
        if row["candidate_id"] == "node_baseline"
    )
    coefficients = np.asarray(source_node["coefficients"], float)
    if np.any(coefficients != 0.0):
        raise RuntimeError("source Node arm no longer has exact-zero edge coefficients")

    design = deepcopy(config["manual_dual_core"])
    node_field = {
        **design,
        "field_sha256": _canonical_sha256(design),
        "role": (
            "two frozen hand-placed centers; exact E-node budget; no patient "
            "event or contact information enters per-network allocation"
        ),
        "component_count": 2,
        "peak_count_constraint": 2,
    }
    candidate = deepcopy(source_node)
    candidate.update({
        "candidate_id": design["candidate_id"],
        "arm": "Hand dual core",
        "node_field": node_field,
        "coefficients": coefficients.tolist(),
        "coefficients_sha256": array_sha256(coefficients),
        "search_coordinates": {
            "control": True,
            "representation_only_comparison": True,
            "field_geometry": "two fixed centers",
            "field_budget": int(design["target_count"]),
            "edge_coefficients": "exact zero",
        },
    })

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("dual-core freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("dual-core workers exist before manifest freeze")
    return {
        "status": "REV11NLC_DUAL_CORE_COMPARISON_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {"n_candidates": 1, "candidates": [candidate]},
        "selection_freeze": {
            "selected_joint_candidate_id": design["candidate_id"],
            "manual_candidate_frozen_before_new_simulation": True,
            "free_field_candidate_id": "node_baseline",
            "free_field_workers_reused_without_resimulation": True,
            "network_seed_is_independent_unit": True,
        },
        "direction_classifier": deepcopy(source["direction_classifier"]),
        "direction_classifier_source": deepcopy(source["direction_classifier_source"]),
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "topology": "frozen",
            "delays": "frozen",
            "GABA": "frozen",
            "E_to_E_and_E_to_I_coefficients": "exact zero",
            "spatial_OU": source_node["spatial_ou"],
            "Z_M": "off",
            "node_budget": int(design["target_count"]),
        },
        "paired_source": {
            "config": config["inputs"]["source_confirmation_config"],
            "manifest": config["inputs"]["source_confirmation_manifest"],
            "candidate_id": "node_baseline",
        },
        "provenance": {**provenance, "config_dirty": config_dirty},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    payload = build_manifest(Path(args.config), args.expected_commit)
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "candidate": payload["candidate_set"]["candidates"][0]["candidate_id"],
        "seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
