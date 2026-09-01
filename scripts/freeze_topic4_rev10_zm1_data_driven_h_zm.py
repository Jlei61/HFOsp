"""Freeze paired slow-off and transferred Z/M arms before fresh networks."""
from __future__ import annotations

import argparse
import copy
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
EXPECTED_ROLE = "development_only_data_driven_h_zm_consistency"
STATUS = "REV10ZM1_H_PLUS_ZM_LIBRARY_FROZEN"


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("ZM1 scientific role changed")
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    source_manifest_path = ROOT / config["inputs"]["source_d5_2_manifest"]["path"]
    source_verdict_path = ROOT / config["inputs"]["source_d5_2_verdict"]["path"]
    source = json.loads(source_manifest_path.read_text())
    verdict = json.loads(source_verdict_path.read_text())
    if source.get("status") != "REV10D5_2_SPATIAL_OU_CONFIRMATION_LIBRARY_FROZEN":
        raise RuntimeError("source D5.2 manifest is invalid")
    source_id = verdict.get("selected_local_candidate_id")
    source_rows = {
        row["candidate_id"]: row for row in source["candidate_set"]["candidates"]
    }
    if source_id not in source_rows:
        raise RuntimeError("D5.2 selected local candidate is absent")
    source_candidate = source_rows[source_id]
    if source_candidate.get("spatial_ou", {}).get("mode") != "local":
        raise RuntimeError("ZM1 source must retain the frozen local OU process")
    if not all(value == 0.0 for value in source_candidate["coefficients"]):
        raise RuntimeError("ZM1 requires exact no-op edge coefficients")

    slow_off = copy.deepcopy(source_candidate)
    slow_off["candidate_id"] = "h_spou_slow_off"
    slow_off["mz"] = {
        "mode": "off", "use_z": False, "use_m": False,
        "trace_stride_steps": int(config["mz_transfer"]["trace_stride_steps"]),
    }
    active = copy.deepcopy(source_candidate)
    active["candidate_id"] = "h_spou_zm_transfer"
    active["mz"] = {"mode": "z_plus_m", **{
        key: config["mz_transfer"][key]
        for key in (
            "use_z", "use_m", "I_th_EI", "tau_z", "tau_adp", "eta_m",
            "trace_stride_steps",
        )
    }}

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
        raise RuntimeError("ZM1 freezer runtime is not frozen")

    return {
        "status": STATUS,
        "scientific_role": EXPECTED_ROLE,
        "candidate_set": {
            "candidates": [slow_off, active],
            "paired_candidate_ids": [
                "h_spou_slow_off", "h_spou_zm_transfer",
            ],
        },
        "selection_freeze": {
            "selected_nonzero_candidate_id": "h_spou_zm_transfer",
            "paired_control_candidate_id": "h_spou_slow_off",
            "selected_before_fresh_networks": True,
            "fresh_networks_were_read": False,
        },
        "source_freeze": {
            "d5_2_candidate_id": source_id,
            "d5_2_manifest": config["inputs"]["source_d5_2_manifest"],
            "d5_2_verdict": config["inputs"]["source_d5_2_verdict"],
            "mz_transfer": config["mz_transfer"],
        },
        "direction_classifier": source["direction_classifier"],
        "fixed_contract": {
            "node_anchor": config["node_anchor"],
            "network_seeds": config["search"]["confirmation_network_seeds"],
            "simulation": config["search"]["simulation"],
            "detector": config["search"]["detector"],
            "same_network_and_dynamics_seeds_across_arms": True,
            "spatial_ou_is_identical_across_arms": True,
            "static_edge_coefficients": "all exact zero",
            "beta": "closed",
        },
        "claim_boundary": config["claim_boundary"],
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
    output = Path(
        args.out or ROOT / config["output_root"] / "candidate_manifest.json"
    )
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "paired_candidates": payload["candidate_set"]["paired_candidate_ids"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
