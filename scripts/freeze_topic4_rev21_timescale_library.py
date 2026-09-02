#!/usr/bin/env python3
"""Freeze rev21 Z/M timescale candidates around the coarse internal winner."""
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


def _token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def build_timescale_candidates(config: dict, coarse_row: dict) -> list[dict]:
    slow = coarse_row["slow_variables"]
    if slow.get("mode") != "z_plus_m":
        raise RuntimeError("timescale refinement requires an active Z/M candidate")
    integrated = float(slow["eta_m"]) * float(slow["tau_adp"])
    rows = []
    for tau_z in config["timescale_grid"]["tau_z_ms"]:
        for tau_adp in config["timescale_grid"]["tau_adp_ms"]:
            row = copy.deepcopy(coarse_row)
            row["source_coarse_candidate_id"] = row["candidate_id"]
            row["candidate_id"] = (
                f"rev21_ts_tz{_token(tau_z)}_ta{_token(tau_adp)}"
            )
            row["family"] = "zm_timescale"
            row["level"] = {
                "I_th_EI_scale": float(slow["I_th_EI_scale"]),
                "integrated_M_scale": float(slow["integrated_M_scale"]),
                "tau_z_ms": float(tau_z),
                "tau_adp_ms": float(tau_adp),
            }
            row["is_reference"] = bool(
                float(tau_z) == float(slow["tau_z"])
                and float(tau_adp) == float(slow["tau_adp"])
            )
            row["slow_variables"].update({
                "tau_z": float(tau_z),
                "tau_adp": float(tau_adp),
                "eta_m": integrated / float(tau_adp),
            })
            rows.append(row)
    if len(rows) != 9 or len({row["candidate_id"] for row in rows}) != 9:
        raise RuntimeError("timescale library must contain nine unique candidates")
    if sum(row["is_reference"] for row in rows) != 1:
        raise RuntimeError("timescale grid must contain the coarse work point")
    products = {
        round(row["slow_variables"]["eta_m"]
              * row["slow_variables"]["tau_adp"], 12)
        for row in rows
    }
    if len(products) != 1:
        raise RuntimeError("eta_m*tau_adp drifted across the timescale grid")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        default=Path("/home/honglab/leijiaxin/HFOsp"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output_root = args.artifact_root / config["output_root"]
    coarse_path = output_root / "coarse/aggregate.json"
    coarse = json.loads(coarse_path.read_text())
    candidate_id = coarse.get("coarse_candidate_for_timescale_refinement")
    if coarse.get("status") != "REV21_COARSE_HAS_CROSS_STATE_CANDIDATE" or not candidate_id:
        raise RuntimeError("coarse screen did not freeze a cross-state candidate")
    if coarse.get("patient_heldout_opened") or coarse.get("patient_ictal_inputs_read"):
        raise RuntimeError("coarse selection opened forbidden patient inputs")
    source_path = args.artifact_root / config["candidate_manifest"]
    source = json.loads(source_path.read_text())
    matches = [row for row in source["candidates"]
               if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("coarse candidate is missing or duplicated")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "schema_id": "topic4_rev21_zm_timescale_manifest_v1",
        "status": "REV21_ZM_TIMESCALE_LIBRARY_FROZEN",
        "git_commit": commit,
        "config_sha256": _sha256(config_path),
        "coarse_aggregate_sha256": _sha256(coarse_path),
        "source_manifest_sha256": _sha256(source_path),
        "source_coarse_candidate_id": candidate_id,
        "candidates": build_timescale_candidates(config, matches[0]),
        "patient_heldout_opened": False,
        "patient_ictal_inputs_read": False,
    }
    output = output_root / "timescale/candidate_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "candidate_count": len(payload["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
