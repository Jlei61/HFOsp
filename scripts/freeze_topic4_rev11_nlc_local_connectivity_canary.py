#!/usr/bin/env python3
"""Freeze the rev11-NLC local-connectivity capacity library."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (
    _runtime_provenance,
    _sha256,
)
from src.topic4_graph_edge_flow import array_sha256


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_local_connectivity_canary.json"


def _candidate(candidate_id, arm, coefficients, config):
    coefficients = np.asarray(coefficients, float)
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "coefficients": coefficients.tolist(),
        "coefficients_sha256": array_sha256(coefficients),
        "raw_logit_clip": float(config["local_connectivity_basis"]["raw_logit_clip_abs"]),
        "adaptation": {"mode": "off"},
        "inhibitory_resource": {"mode": "off"},
        "ee_std": {"mode": "off"},
        "spatial_ou": dict(config["fixed_spatial_ou"]),
        "mz": {"mode": "off"},
    }


def build_candidates(config):
    library = config["candidate_library"]
    bounds = np.asarray(config["local_connectivity_basis"]["coefficient_abs_bounds"], float)
    if bounds.shape != (6,) or np.any(bounds <= 0.0):
        raise ValueError("six positive coefficient bounds are required")
    zero = np.zeros((2, 6), float)
    candidates = [_candidate("node_baseline", "Node", zero, config)]

    ee_draws = qmc.Sobol(6, scramble=True, seed=int(library["seed"])).random_base2(2)[:3]
    ei_draws = qmc.Sobol(6, scramble=True, seed=int(library["seed"]) + 1).random_base2(2)[:3]
    joint_draws = qmc.Sobol(12, scramble=True, seed=int(library["seed"]) + 2).random_base2(3)[:6]
    for index, draw in enumerate(ee_draws):
        values = zero.copy(); values[0] = (2.0 * draw - 1.0) * bounds
        candidates.append(_candidate(f"ee_only_{index:02d}", "Node+EE", values, config))
    for index, draw in enumerate(ei_draws):
        values = zero.copy(); values[1] = (2.0 * draw - 1.0) * bounds
        candidates.append(_candidate(f"etoi_only_{index:02d}", "Node+EtoI", values, config))
    for index, draw in enumerate(joint_draws):
        values = (2.0 * draw.reshape(2, 6) - 1.0) * bounds[None, :]
        candidates.append(_candidate(f"joint_{index:02d}", "Node+EE+EtoI", values, config))
    if len(candidates) != int(library["candidate_count"]):
        raise RuntimeError("candidate count changed")
    if len({row["coefficients_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("duplicate local-connectivity candidate")
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(expected)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("freezer code/config must match the expected clean commit")

    direction_manifest_record = config["inputs"]["direction_classifier_manifest"]
    direction_manifest = json.loads((ROOT / direction_manifest_record["path"]).read_text())
    anchor_manifest_record = config["inputs"]["node_anchor_manifest"]
    anchor_manifest = json.loads((ROOT / anchor_manifest_record["path"]).read_text())
    anchor = [
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    ]
    if len(anchor) != 1 or anchor[0]["field_sha256"] != config["node_anchor"]["field_sha256"]:
        raise RuntimeError("Node anchor changed")
    candidates = build_candidates(config)
    payload = {
        "status": "REV11NLC_LOCAL_CONNECTIVITY_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "node_anchor": config["node_anchor"],
        "direction_classifier": direction_manifest["direction_classifier"],
        "selection_freeze": {"paired_control_candidate_id": "node_baseline"},
        "local_connectivity_basis": config["local_connectivity_basis"],
        "inputs": config["inputs"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "provenance": provenance,
        "claim_boundary": config["claim_boundary"],
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({
        "status": payload["status"], "candidate_count": len(candidates),
        "output": str(output), "expected_commit": expected,
    }, indent=2))


if __name__ == "__main__":
    main()
