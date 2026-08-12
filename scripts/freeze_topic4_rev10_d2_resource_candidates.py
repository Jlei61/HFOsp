"""Freeze q-only local/global inhibitory-resource candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402

DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d2_inhibitory_resource_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp"); os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def candidate_library(config):
    zero = np.zeros(int(config["spatial_edge_basis"]["coefficient_count"]), float)
    base = {"coefficients": zero.tolist(), "coefficients_sha256": array_sha256(zero)}
    trace = float(config["inhibitory_resource_library"]["trace_dt_ms"])
    rows = [{"candidate_id": "edge_noop", **base,
             "inhibitory_resource": {"mode": "off", "k_q_per_ms": 0.0,
                                     "trace_dt_ms": trace}}]
    library = config["inhibitory_resource_library"]
    shared = {key: library[key] for key in (
        "tau_q_ms", "q_min", "n_grid", "sigma_rate_mm", "tau_rate_ms",
        "sigma_q_mm", "eta_e", "eta_i", "a0", "a50", "trace_dt_ms",
        "update_interval_ms",
    )}
    for mode in library["modes"]:
        for k_q in library["k_q_per_ms"]:
            rows.append({
                "candidate_id": (
                    f"qresource_{mode}_k{int(round(1_000_000*k_q)):06d}"
                ),
                **base,
                "inhibitory_resource": {
                    "mode": mode, "k_q_per_ms": float(k_q), **shared,
                },
            })
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", default=str(DEFAULT_CONFIG)); args = parser.parse_args()
    config_path = Path(args.config).resolve(); config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_inhibitory_resource_accessibility_canary":
        raise RuntimeError("rev10-D2 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source_record = config["inputs"]["frozen_direction_classifier_manifest"]
    source = json.loads((ROOT / source_record["path"]).read_text())
    candidates = candidate_library(config)
    if len(candidates) != 7 or any(np.any(row["coefficients"]) for row in candidates):
        raise RuntimeError("rev10-D2 candidate library changed")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    payload = {
        "status": "REV10D2_INHIBITORY_RESOURCE_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates), "candidates": candidates,
                          "frozen_before_network_seeds": config["search"]["fit_network_seeds"]},
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {"path": source_record["path"], "sha256": source_record["sha256"], "copied_without_refit": True},
        "static_edge_contract": "all 12 coefficients are exact zero",
        "git_commit_at_freeze": commit,
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"; _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "n_candidates": len(candidates), "output": str(output)}, indent=2))


if __name__ == "__main__": main()
