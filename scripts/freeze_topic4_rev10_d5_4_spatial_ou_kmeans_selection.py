"""Freeze the D5.3 winner and matched controls on fresh selection networks."""
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

from scripts.run_topic4_rev9l_forced_source_worker import (
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_4_spatial_ou_kmeans_selection.json"
EXPECTED_ROLE = "development_only_translation_invariant_spatial_ou_kmeans_selection"


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D5.4 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    manifest_record = config["inputs"]["d5_3_manifest"]
    verdict_record = config["inputs"]["d5_3_verdict"]
    source = json.loads((ROOT / manifest_record["path"]).read_text())
    verdict = json.loads((ROOT / verdict_record["path"]).read_text())
    if source.get("status") != "REV10D5_3_SPATIAL_OU_KMEANS_GRID_FROZEN":
        raise RuntimeError("D5.3 source manifest is invalid")
    if verdict.get("status") != (
        "REV10D5_3_KMEANS_CANDIDATE_SELECTED_FOR_FRESH_CONFIRMATION"
    ):
        raise RuntimeError("D5.3 did not select a fresh-network candidate")
    by_id = {
        row["candidate_id"]: row for row in source["candidate_set"]["candidates"]
    }
    selected_id = verdict["selected_candidate_id"]
    selected = deepcopy(by_id[selected_id])
    if selected["spatial_ou"]["mode"] != "local":
        raise RuntimeError("D5.3 selected candidate is not local")
    permuted = deepcopy(selected)
    permuted["candidate_id"] = selected_id.replace("spou_local_", "spou_permuted_")
    permuted["spatial_ou"]["mode"] = "permuted"
    candidates = [deepcopy(by_id["edge_noop"]), selected, permuted]
    if any(np.any(row["coefficients"]) for row in candidates):
        raise RuntimeError("D5.4 contains a nonzero edge coefficient")

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    provenance["config_dirty"] = config_dirty
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("D5.4 freezer runtime is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D5.4 workers exist before manifest freeze")
    return {
        "status": "REV10D5_4_SPATIAL_OU_KMEANS_SELECTION_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)),
                   "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": 3, "candidates": candidates},
        "selection_freeze": {
            "selected_nonzero_candidate_id": selected_id,
            "matched_permuted_candidate_id": permuted["candidate_id"],
            "source_grid_verdict": {"path": verdict_record["path"],
                                    "sha256": verdict_record["sha256"]},
            "selection_networks_were_read": False,
        },
        "direction_classifier": source["direction_classifier"],
        "kmeans_selection_contract": config["search"]["kmeans_selection"],
        "static_edge_contract": "all 12 coefficients are exact zero",
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
        "selected": payload["selection_freeze"]["selected_nonzero_candidate_id"],
        "permuted": payload["selection_freeze"]["matched_permuted_candidate_id"],
    }, indent=2))


if __name__ == "__main__":
    main()
