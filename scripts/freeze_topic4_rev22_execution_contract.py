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
DEFAULT_TRANSITION = ROOT / "config/topic4_rev22_dci_transition_execution.json"
DEFAULT_CONNECTIVITY_AUDIT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
    "data_driven_dual_core_interictal_identifiability/connectivity_design_audit/"
    "connectivity_design_audit.json"
)


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
                           candidate_manifest_path: str, frozen_contracts: dict,
                           transition_input: dict | None = None) -> dict:
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
        # The simulation worker only needs the transition contract.  Patient training and
        # held-out artifacts belonged to the rev20 aggregate and must not even be hash-read
        # by a rev22 fit worker before candidate selection is frozen.
        "inputs": {"transition_config": dict(
            rev20["inputs"]["transition_config"] if transition_input is None else transition_input
        )},
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
    parser.add_argument("--transition-config", type=Path, default=DEFAULT_TRANSITION)
    parser.add_argument("--connectivity-design-audit", type=Path, default=DEFAULT_CONNECTIVITY_AUDIT)
    args = parser.parse_args()

    analysis = json.loads(args.analysis_config.read_text())
    response = json.loads(args.response_manifest.read_text())
    seeds = json.loads(args.seed_manifest.read_text())
    rev20_path = ROOT / analysis["inputs"]["rev20_config"]["path"]
    if _sha256(rev20_path) != analysis["inputs"]["rev20_config"]["sha256"]:
        raise RuntimeError("rev20 execution config hash changed")
    rev20 = json.loads(rev20_path.read_text())
    if seeds.get("response_design_manifest_sha256") != _sha256(args.response_manifest):
        raise RuntimeError("seed manifest is not bound to the response design")
    if response.get("candidate_count") != len(response.get("candidates", [])):
        raise RuntimeError("response design candidate count is inconsistent")
    transition = json.loads(args.transition_config.read_text())
    if transition.get("schema_id") != "topic4_rev22_dci_transition_execution_v1":
        raise RuntimeError("rev22 minimal transition contract is missing")
    if transition.get("source_transition_config_sha256") != rev20["inputs"]["transition_config"]["sha256"]:
        raise RuntimeError("minimal transition contract is not derived from the frozen source")
    audit = json.loads(args.connectivity_design_audit.read_text())
    if audit.get("status") != "CONNECTIVITY_DESIGN_ADMISSIBLE":
        raise RuntimeError("final ellipse-plus-learned connectivity design did not pass")
    if audit.get("response_design_manifest_sha256") != _sha256(args.response_manifest):
        raise RuntimeError("connectivity audit is not bound to the response design")

    artifact_root = Path("/home/honglab/leijiaxin/HFOsp")
    try:
        candidate_relative = str(args.candidate_manifest_out.resolve().relative_to(artifact_root))
    except ValueError as exc:
        raise RuntimeError("formal candidate manifest must live under the artifact root") from exc
    def record(path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": _sha256(path)}

    contracts = {
        "analysis_config": {"path": str(args.analysis_config.resolve()), "sha256": _sha256(args.analysis_config)},
        "response_design": {"path": str(args.response_manifest.resolve()), "sha256": _sha256(args.response_manifest)},
        "seed_manifest": {"path": str(args.seed_manifest.resolve()), "sha256": _sha256(args.seed_manifest)},
        "geometry_domain": dict(response["domain_source"]),
        "spec": record(ROOT / analysis["spec"]),
        "plan": record(ROOT / analysis["plan"]),
        "training_objective_module": record(ROOT / "src/topic4_rev22_interictal_objective.py"),
        "response_design_module": record(ROOT / "src/topic4_rev22_response_design.py"),
        "response_surface_module": record(ROOT / "src/topic4_rev22_response_surface.py"),
        "fit_aggregate_script": record(ROOT / "scripts/aggregate_topic4_rev22_fit.py"),
        "minimal_transition_config": record(args.transition_config),
        "connectivity_design_audit": record(args.connectivity_design_audit),
    }
    transition_record = {"path": str(args.transition_config.resolve().relative_to(ROOT.resolve())),
                         "sha256": _sha256(args.transition_config)}
    execution = build_execution_config(
        analysis, rev20, seeds, candidate_relative, contracts,
        transition_input=transition_record,
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
