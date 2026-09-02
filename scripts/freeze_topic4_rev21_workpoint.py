#!/usr/bin/env python3
"""Write the immutable rev21 work point before any patient ictal readout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output_root = args.artifact_root / config["output_root"]
    aggregate_path = output_root / "confirmation/aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    if aggregate.get("status") != (
            "CROSS_STATE_CONFIRMATION_ESTABLISHED_DEVELOPMENT_ONLY"):
        raise RuntimeError("fresh-seed cross-state confirmation is not established")
    if aggregate.get("patient_heldout_opened") or aggregate.get(
            "patient_ictal_inputs_read"):
        raise RuntimeError("confirmation opened forbidden patient inputs")
    manifest_path = output_root / "confirmation/candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    finalist = [row for row in manifest["candidates"]
                if row["candidate_id"] == "rev21_confirm_z_plus_m"]
    if len(finalist) != 1:
        raise RuntimeError("confirmed joint work point is missing or duplicated")

    provenance_paths = {
        "seed_factorization": output_root / "seed_audit/seed_factorization_audit.json",
        "coarse_aggregate": output_root / "coarse/aggregate.json",
        "timescale_aggregate": output_root / "timescale/aggregate.json",
        "confirmation_manifest": manifest_path,
        "confirmation_aggregate": aggregate_path,
    }
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "schema_id": "topic4_rev21_workpoint_frozen_v1",
        "status": "REV21_WORKPOINT_FROZEN_BEFORE_CLINICAL_ICTAL_READOUT",
        "git_commit": commit,
        "config_sha256": _sha256(config_path),
        "candidate": finalist[0],
        "source_finalist_candidate_id": manifest["source_finalist_candidate_id"],
        "provenance_sha256": {
            name: _sha256(path) for name, path in provenance_paths.items()
        },
        "patient_interictal_heldout_opened": False,
        "patient_ictal_inputs_read": False,
        "selection_boundary": (
            "model-internal ictal morphology plus patient-training interictal "
            "distribution, two-template alignment and OOD only"
        ),
        "claim_boundary": config["claim_boundary"],
    }
    output = output_root / "WORKPOINT_FROZEN.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "sha256": _sha256(output),
    }, indent=2))


if __name__ == "__main__":
    main()
