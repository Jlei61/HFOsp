"""Freeze the rev10-D local/global adaptation canary before SNN execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from itertools import product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d_local_adaptation_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _array_sha256(values):
    values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def candidate_library(config):
    count = int(config["spatial_edge_basis"]["coefficient_count"])
    zero = np.zeros(count, dtype=np.float64)
    zero_hash = _array_sha256(zero)
    candidates = [{
        "candidate_id": "edge_noop",
        "coefficients": zero.tolist(),
        "coefficients_sha256": zero_hash,
        "adaptation": {"mode": "off", "tau_ms": 0.0,
                       "increment_mV": 0.0,
                       "trace_dt_ms": config["adaptation_library"]["trace_dt_ms"]},
    }]
    for mode, tau_ms, increment_mV in product(
            config["adaptation_library"]["modes"],
            config["adaptation_library"]["tau_ms"],
            config["adaptation_library"]["increment_mV"]):
        candidate_id = (
            f"adapt_{mode}_t{int(tau_ms):04d}_q{int(round(1000 * increment_mV)):04d}"
        )
        candidates.append({
            "candidate_id": candidate_id,
            "coefficients": zero.tolist(),
            "coefficients_sha256": zero_hash,
            "adaptation": {
                "mode": mode,
                "tau_ms": float(tau_ms),
                "increment_mV": float(increment_mV),
                "trace_dt_ms": float(config["adaptation_library"]["trace_dt_ms"]),
            },
        })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_dynamic_accessibility_canary":
        raise RuntimeError("rev10-D scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source_record = config["inputs"]["frozen_direction_classifier_manifest"]
    source_path = ROOT / source_record["path"]
    source = json.loads(source_path.read_text())
    if source.get("status") != "REV10R2_SPATIAL_EDGE_LIBRARY_FROZEN":
        raise RuntimeError("frozen direction-classifier source is invalid")
    candidates = candidate_library(config)
    if len(candidates) != 19:
        raise RuntimeError("rev10-D candidate count changed")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "status": "REV10D_LOCAL_ADAPTATION_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
            "frozen_before_network_seeds": config["search"]["fit_network_seeds"],
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {
            "path": source_record["path"],
            "sha256": source_record["sha256"],
            "copied_without_refit": True,
        },
        "static_edge_contract": "all 12 coefficients are exact zero for every candidate",
        "forbidden_spatial_inputs": config["spatial_edge_basis"]["forbidden_inputs"],
        "git_commit_at_freeze": commit,
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
