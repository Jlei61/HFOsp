"""Freeze five D6 continuous fields for a six-network natural KMeans closeout."""
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


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_1_natural_kmeans_closeout.json"
EXPECTED_ROLE = "development_only_continuous_field_natural_kmeans_fresh_closeout"


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D6.1 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source = json.loads((
        ROOT / config["inputs"]["d6_fit_manifest"]["path"]
    ).read_text())
    repaired = json.loads((
        ROOT / config["inputs"]["d6_repaired_selection"]["path"]
    ).read_text())
    if source.get("status") != "REV10D6_CONTINUOUS_FIELD_SENSITIVITY_LIBRARY_FROZEN":
        raise RuntimeError("D6 source library is not frozen")
    if repaired.get("status") != "REV10D6_NATURAL_KMEANS_RESCORING_COMPLETE":
        raise RuntimeError("D6 natural KMeans rescoring is incomplete")
    freeze = repaired["fresh_closeout_freeze"]
    source_rows = {
        row["candidate_id"]: row
        for row in source["candidate_set"]["candidates"]
    }
    candidates = []
    for candidate_id in freeze["candidate_ids"]:
        row = deepcopy(source_rows[candidate_id])
        row["fresh_closeout_selection_roles"] = freeze["selection_roles"][
            candidate_id
        ]
        candidates.append(row)
    if len(candidates) != int(config["field_search"]["candidate_count"]):
        raise RuntimeError("D6.1 candidate count changed")
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("D6.1 candidates are not unique")
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
        raise RuntimeError("D6.1 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D6.1 workers exist before manifest freeze")
    return {
        "status": "REV10D6_1_NATURAL_KMEANS_CLOSEOUT_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates), "candidates": candidates},
        "selection_freeze": {
            "selected_nonzero_candidate_id": freeze["primary_candidate_id"],
            "primary_candidate_id": freeze["primary_candidate_id"],
            "baseline_candidate_id": "edge_noop",
            "candidate_ids": freeze["candidate_ids"],
            "selection_roles": freeze["selection_roles"],
            "category_winners": freeze["category_winners"],
            "frozen_before_fresh_networks": True,
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": source["direction_classifier_source"],
        "fixed_contract": {
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "spatial_ou": config["fixed_spatial_ou"],
            "edge": "exact no-op", "beta": "closed",
        },
        "forbidden_builder_inputs": config["field_search"]["forbidden_builder_inputs"],
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
        "candidate_ids": payload["selection_freeze"]["candidate_ids"],
        "network_seeds": payload["fixed_contract"]["network_seeds"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
