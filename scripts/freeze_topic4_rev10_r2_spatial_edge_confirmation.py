"""Freeze the minimal rev10-R2.1 confirmation contrast before fresh networks."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    expected_role = (
        "development_only_observation_invariant_spatial_route_confirmation"
    )
    if config.get("scientific_role") != expected_role:
        raise RuntimeError("rev10-R2 confirmation scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    summary_path = ROOT / config["inputs"]["selection_summary"]["path"]
    manifest_path = ROOT / config["inputs"]["selection_candidate_manifest"]["path"]
    summary = json.loads(summary_path.read_text())
    source = json.loads(manifest_path.read_text())
    if summary.get("status") != "REV10R_RETURNED_ONLY_SELECTION_COMPLETE":
        raise RuntimeError("selection summary is incomplete")
    if source.get("status") != "REV10R2_SPATIAL_EDGE_SELECTION_LIBRARY_FROZEN":
        raise RuntimeError("selection candidate manifest is invalid")
    if summary["manifest"]["sha256"] != _sha256(manifest_path):
        raise RuntimeError("selection summary and candidate manifest differ")
    selected_id = summary["diagnostic_best_candidate_id"]
    by_id = {row["candidate_id"]: row for row in source["candidate_set"]["candidates"]}
    if selected_id == "edge_noop" or selected_id not in by_id:
        raise RuntimeError("selection did not nominate one nonzero edge candidate")
    selected_row = next(
        row for row in summary["candidate_rows"]
        if row["candidate_id"] == selected_id
    )
    if selected_row["n_runaway_networks"]:
        raise RuntimeError("selection best has runaway networks")

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
        raise RuntimeError("confirmation freezer runtime is not frozen")
    return {
        "status": "REV10R2_SPATIAL_EDGE_CONFIRMATION_LIBRARY_FROZEN",
        "scientific_role": expected_role,
        "candidate_set": {
            "candidates": [by_id["edge_noop"], by_id[selected_id]],
            "n_nonzero": 1,
            "n_exact_noop": 1,
        },
        "selection_freeze": {
            "rule": "equal-network returned-only selection diagnostic best plus exact no-op",
            "selected_nonzero_candidate_id": selected_id,
            "selection_row": selected_row,
            "source_selection_summary": {
                "path": str(summary_path.relative_to(ROOT)),
                "sha256": _sha256(summary_path),
            },
            "source_candidate_manifest": {
                "path": str(manifest_path.relative_to(ROOT)),
                "sha256": _sha256(manifest_path),
            },
            "confirmation_networks_were_read": False,
            "claim_role": (
                "negative-boundary confirmation because no selection network "
                "produced returned joint in-distribution mode A"
            ),
        },
        "direction_classifier": source["direction_classifier"],
        "fixed_contract": {
            **source["fixed_contract"],
            "confirmation_network_seeds": config["search"][
                "confirmation_network_seeds"
            ],
        },
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.out or ROOT / config["output_root"] / "candidate_manifest.json")
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "selected": payload["selection_freeze"]["selected_nonzero_candidate_id"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
