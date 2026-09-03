#!/usr/bin/env python3
"""Freeze the minimal transition config visible to rev22 fit workers.

The historical transition config contains patient held-out and classifier inputs because it
served several later analyses.  A rev22 fit worker only needs the frozen substrate, Node,
contact, detector and placement contracts.  This compiler removes every validation-only
input before candidate selection and also freezes Z/M off.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json"
DEFAULT_OUT = ROOT / "config/topic4_rev22_dci_transition_execution.json"

REQUIRED_INPUTS = (
    "frozen_substrate_manifest",
    "rev9_base_config",
    "stage_config",
    "contact_contract",
    "common_detector_audit",
    "node_anchor_config",
    "placement_gradient_field",
    "placement_rank_displacement",
    "placement_geometry_t_a",
    "placement_geometry_t_b",
)
FORBIDDEN_INPUT_TOKENS = (
    "heldout", "direction_classifier", "shaft_aware_target", "shaft_aware_floor",
    "patient_support", "patient_ictal", "fig5",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sanitized_transition(source: dict, *, source_sha256: str) -> dict:
    inputs = source.get("inputs", {})
    missing = sorted(set(REQUIRED_INPUTS) - set(inputs))
    if missing:
        raise ValueError(f"source transition config misses required inputs: {missing}")
    selected = {key: dict(inputs[key]) for key in REQUIRED_INPUTS}
    lowered = " ".join(selected).lower()
    if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
        raise ValueError("validation-only input survived transition sanitization")
    required_sections = ("local_connectivity_basis", "engine_detector", "spatial_ou", "simulation")
    missing_sections = [key for key in required_sections if key not in source]
    if missing_sections:
        raise ValueError(f"source transition config misses execution sections: {missing_sections}")
    return {
        "schema_id": "topic4_rev22_dci_transition_execution_v1",
        "scientific_role": "rev22_interictal_substrate_execution_only",
        "source_transition_config_sha256": str(source_sha256),
        "inputs": selected,
        **{key: dict(source[key]) for key in required_sections},
        "zm": {
            "mode": "off",
            "use_z": False,
            "use_m": False,
            "role": "rev22 interictal fit contract; slow variables are forbidden",
        },
        "forbidden_before_candidate_freeze": [
            "patient held-out events", "patient KMeans labels", "OOD classifier",
            "patient ictal data", "Z/M", "Fig.5 endpoints",
        ],
    }


def atomic_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    source_hash = sha256(args.source)
    payload = sanitized_transition(json.loads(args.source.read_text()), source_sha256=source_hash)
    digest = atomic_json(args.out, payload)
    print(json.dumps({
        "status": "REV22_MINIMAL_TRANSITION_CONTRACT_FROZEN",
        "path": str(args.out),
        "sha256": digest,
        "source_sha256": source_hash,
        "declared_inputs": list(payload["inputs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
