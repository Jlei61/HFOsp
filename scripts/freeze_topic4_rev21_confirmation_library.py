#!/usr/bin/env python3
"""Freeze the rev21 finalist and matched Z/M mechanism controls."""
from __future__ import annotations

import argparse
import copy
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


def build_confirmation_candidates(finalist: dict, off: dict) -> list[dict]:
    joint = copy.deepcopy(finalist)
    joint["source_candidate_id"] = joint["candidate_id"]
    joint["candidate_id"] = "rev21_confirm_z_plus_m"
    joint["family"] = "confirmation_joint"
    joint["is_reference"] = True

    z_only = copy.deepcopy(joint)
    z_only["candidate_id"] = "rev21_confirm_z_only"
    z_only["family"] = "confirmation_control"
    z_only["is_reference"] = False
    z_only["mechanisms"]["Z_M"] = "z_only"
    z_only["slow_variables"].update({
        "mode": "z_only", "use_z": True, "use_m": False,
    })

    m_only = copy.deepcopy(joint)
    m_only["candidate_id"] = "rev21_confirm_m_only"
    m_only["family"] = "confirmation_control"
    m_only["is_reference"] = False
    m_only["mechanisms"]["Z_M"] = "m_only"
    m_only["slow_variables"].update({
        "mode": "m_only", "use_z": False, "use_m": True,
    })

    slow_off = copy.deepcopy(off)
    slow_off["source_candidate_id"] = slow_off["candidate_id"]
    slow_off["candidate_id"] = "rev21_confirm_zm_off"
    slow_off["family"] = "confirmation_control"
    slow_off["is_reference"] = False
    return [joint, z_only, m_only, slow_off]


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
    aggregate_path = output_root / "timescale/aggregate.json"
    aggregate = json.loads(aggregate_path.read_text())
    finalist_id = aggregate.get("confirmation_finalist_candidate_id")
    if (aggregate.get("status") != "REV21_TIMESCALE_HAS_CONFIRMATION_FINALIST"
            or not finalist_id):
        raise RuntimeError("timescale screen did not freeze a confirmation finalist")
    if aggregate.get("patient_heldout_opened") or aggregate.get(
            "patient_ictal_inputs_read"):
        raise RuntimeError("timescale selection opened forbidden patient inputs")

    timescale_path = output_root / "timescale/candidate_manifest.json"
    timescale = json.loads(timescale_path.read_text())
    matches = [row for row in timescale["candidates"]
               if row["candidate_id"] == finalist_id]
    if len(matches) != 1:
        raise RuntimeError("timescale finalist is missing or duplicated")
    coarse_path = args.artifact_root / config["candidate_manifest"]
    coarse = json.loads(coarse_path.read_text())
    off = [row for row in coarse["candidates"]
           if row["candidate_id"] == "rev21_zm_off"]
    if len(off) != 1:
        raise RuntimeError("coarse Z/M-off reference is missing or duplicated")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "schema_id": "topic4_rev21_zm_confirmation_manifest_v1",
        "status": "REV21_ZM_CONFIRMATION_LIBRARY_FROZEN",
        "git_commit": commit,
        "config_sha256": _sha256(config_path),
        "timescale_aggregate_sha256": _sha256(aggregate_path),
        "timescale_manifest_sha256": _sha256(timescale_path),
        "coarse_manifest_sha256": _sha256(coarse_path),
        "source_finalist_candidate_id": finalist_id,
        "candidates": build_confirmation_candidates(matches[0], off[0]),
        "patient_heldout_opened": False,
        "patient_ictal_inputs_read": False,
    }
    output = output_root / "confirmation/candidate_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "candidate_count": len(payload["candidates"]),
    }, indent=2))


if __name__ == "__main__":
    main()
