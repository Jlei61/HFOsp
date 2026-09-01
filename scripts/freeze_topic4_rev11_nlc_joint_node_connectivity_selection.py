#!/usr/bin/env python3
"""Freeze the fresh-network NLC3 shortlist without changing candidates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_selection.json"
)
EXPECTED_ROLE = (
    "development_only_data_driven_node_local_connectivity_joint_selection"
)


def selection_library(config, fit_manifest, fit_verdict):
    fit_ids = list(config["candidate_library"]["fit_shortlist_ids"])
    if fit_ids != fit_verdict["fresh_selection_shortlist_ids"]:
        raise RuntimeError("NLC3 shortlist differs from the frozen NLC2 verdict")
    requested = fit_ids + list(
        config["candidate_library"]["additional_control_ids"]
    )
    requested = list(dict.fromkeys(requested))
    if len(requested) != int(config["candidate_library"]["candidate_count"]):
        raise RuntimeError("NLC3 candidate count changed")
    source = {
        row["candidate_id"]: row
        for row in fit_manifest["candidate_set"]["candidates"]
    }
    missing = sorted(set(requested) - set(source))
    if missing:
        raise RuntimeError(f"NLC3 candidates absent from NLC2: {missing}")
    return [deepcopy(source[candidate_id]) for candidate_id in requested]


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("NLC3 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    fit_manifest = json.loads(
        (ROOT / config["inputs"]["nlc2_fit_manifest"]["path"]).read_text()
    )
    fit_verdict = json.loads(
        (ROOT / config["inputs"]["nlc2_fit_verdict"]["path"]).read_text()
    )
    if fit_manifest.get("status") != (
        "REV11NLC_JOINT_NODE_CONNECTIVITY_FIT_LIBRARY_FROZEN"
    ):
        raise RuntimeError("NLC2 fit manifest is not frozen")
    if fit_verdict.get("status") != (
        "REV11NLC_JOINT_FIT_EXPLORATORY_CANDIDATE_FOUND"
    ):
        raise RuntimeError("NLC2 fit verdict is not selection-eligible")
    candidates = selection_library(config, fit_manifest, fit_verdict)
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
        raise RuntimeError("NLC3 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("NLC3 workers exist before manifest freeze")
    return {
        "status": "REV11NLC_JOINT_NODE_CONNECTIVITY_SELECTION_LIBRARY_FROZEN",
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
            "fit_shortlist_candidate_ids": fit_verdict[
                "fresh_selection_shortlist_ids"
            ],
            "additional_control_candidate_ids": config[
                "candidate_library"
            ]["additional_control_ids"],
            "candidate_parameters_copied_without_refit": True,
            "network_seed_is_independent_unit": True,
        },
        "direction_classifier": deepcopy(fit_manifest["direction_classifier"]),
        "direction_classifier_source": deepcopy(
            fit_manifest["direction_classifier_source"]
        ),
        "fixed_contract": {
            "network_seeds": config["search"]["selection_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "spatial_ou": config["fixed_spatial_ou"],
            "topology": "frozen",
            "delays": "frozen",
            "GABA": "frozen",
            "beta": "closed",
            "Z_M": "off",
        },
        "source_fit": {
            "manifest": config["inputs"]["nlc2_fit_manifest"],
            "verdict": config["inputs"]["nlc2_fit_verdict"],
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
        "n_candidates": payload["candidate_set"]["n_candidates"],
        "candidate_ids": [
            row["candidate_id"] for row in payload["candidate_set"]["candidates"]
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
