#!/usr/bin/env python3
"""Freeze the rev22 worker config and its config-bound candidate manifest.

This is a contract compiler, not a simulator. It carries the accepted rev20 event/readout
implementation forward, replaces only the scientific role, candidate design, seed pools and
output root, and binds the worker manifest to the byte content of the tracked config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json"
DEFAULT_RESPONSE = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                        "data_driven_dual_core_interictal_identifiability/response_design/"
                        "response_design_manifest.json")
DEFAULT_SEEDS = DEFAULT_RESPONSE.with_name("seed_manifest.json")
DEFAULT_EXECUTION_CONFIG = ROOT / "config/topic4_rev22_dci_response_execution.json"
DEFAULT_CANDIDATES = DEFAULT_RESPONSE.with_name("execution_candidate_manifest.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(path)


def _seeds(block: dict) -> list[int]:
    return [int(unit["topology_seed"]) for unit in block["units"]]


def build_execution_config(analysis: dict, rev20: dict, seed_manifest: dict,
                           candidate_manifest_path: str, frozen_contracts: dict) -> dict:
    fit = _seeds(seed_manifest["fit"])
    qualification = _seeds(seed_manifest["qualification"])
    confirmation = _seeds(seed_manifest["confirmation"])
    dynamics = sorted({
        int(unit["dynamics_seed"])
        for unit in seed_manifest["variance_decomposition_block"]["units"]
        if unit["dynamics_seed"] != unit["topology_seed"]
    })
    if len(fit) != 4 or len(qualification) != 6 or len(confirmation) != 12:
        raise ValueError("seed manifest does not implement the 4/6/12 rev22 contract")
    config = {
        "schema_id": "topic4_rev22_dci_response_execution_v1",
        "scientific_role": analysis["scientific_role"],
        "spec": analysis["spec"],
        "plan": analysis["plan"],
        "output_root": analysis["output_root"],
        "candidate_manifest": candidate_manifest_path,
        "network_cache": analysis["network_cache"],
        "inputs": dict(rev20["inputs"]),
        "frozen_contracts": dict(frozen_contracts),
        "dual_core_anchor": dict(analysis["dual_core_anchor"]),
        "reference": dict(analysis["reference"]),
        "search": {
            "canary_network_seeds": [fit[0]],
            "fit_network_seeds": fit,
            "selection_network_seeds": qualification,
            "confirmation_network_seeds": confirmation,
            "dynamics_seeds": dynamics,
            "simulation": dict(rev20["search"]["simulation"]),
            "contact_readout": dict(rev20["search"]["contact_readout"]),
        },
        "event_unit": dict(rev20["event_unit"]),
        "source_topology": dict(rev20["source_topology"]),
        "complete_distribution": dict(rev20["complete_distribution"]),
        "validation": dict(rev20["validation"]),
        "resources": dict(rev20["resources"]),
        "claim_boundary": analysis["claim_boundary"],
    }
    config["resources"]["maximum_workers"] = min(16, int(config["resources"]["maximum_workers"]) + 4)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--response-manifest", type=Path, default=DEFAULT_RESPONSE)
    parser.add_argument("--seed-manifest", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--execution-config-out", type=Path, default=DEFAULT_EXECUTION_CONFIG)
    parser.add_argument("--candidate-manifest-out", type=Path, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()

    analysis = json.loads(args.analysis_config.read_text())
    response = json.loads(args.response_manifest.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    rev20_path = ROOT / analysis["inputs"]["rev20_config"]["path"]
    if _sha256(rev20_path) != analysis["inputs"]["rev20_config"]["sha256"]:
        raise RuntimeError("rev20 execution config hash changed")
    if seeds.get("response_design_manifest_sha256") != _sha256(args.response_manifest):
        raise RuntimeError("seed manifest is not bound to the response design")
    if response.get("candidate_count") != len(response.get("candidates", [])):
        raise RuntimeError("response design candidate count is inconsistent")

    artifact_root = Path("/home/honglab/leijiaxin/HFOsp")
    try:
        candidate_relative = str(args.candidate_manifest_out.resolve().relative_to(artifact_root))
    except ValueError as exc:
        raise RuntimeError("formal candidate manifest must live under the artifact root") from exc
    contracts = {
        "analysis_config": {"path": str(args.analysis_config.resolve()), "sha256": _sha256(args.analysis_config)},
        "response_design": {"path": str(args.response_manifest.resolve()), "sha256": _sha256(args.response_manifest)},
        "seed_manifest": {"path": str(args.seed_manifest.resolve()), "sha256": _sha256(args.seed_manifest)},
        "geometry_domain": dict(response["domain_source"]),
    }
    execution = build_execution_config(
        analysis, json.loads(rev20_path.read_text()), seeds, candidate_relative, contracts,
    )
    config_sha = _atomic_json(args.execution_config_out, execution)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    candidate_manifest = {
        "schema_id": "topic4_rev22_dci_execution_candidate_manifest_v1",
        "config_sha256": config_sha,
        "response_design_manifest_sha256": _sha256(args.response_manifest),
        "seed_manifest_sha256": _sha256(args.seed_manifest),
        "git_commit": commit,
        "branch": response["branch"],
        "candidate_count": response["candidate_count"],
        "candidates": response["candidates"],
        "claim_boundary": "Execution binding only; parameter values are copied byte-for-byte from the frozen design.",
    }
    candidate_sha = _atomic_json(args.candidate_manifest_out, candidate_manifest)
    print(json.dumps({
        "status": "REV22_EXECUTION_CONTRACT_FROZEN",
        "execution_config": str(args.execution_config_out),
        "execution_config_sha256": config_sha,
        "candidate_manifest": str(args.candidate_manifest_out),
        "candidate_manifest_sha256": candidate_sha,
        "candidate_count": response["candidate_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
