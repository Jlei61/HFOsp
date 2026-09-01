"""Freeze translation-invariant spatial OU and exact-marginal controls."""
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

DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_spatial_ou_accessibility_canary.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def candidate_library(config):
    zero = np.zeros(int(config["spatial_edge_basis"]["coefficient_count"]), float)
    base = {"coefficients": zero.tolist(), "coefficients_sha256": array_sha256(zero)}
    off = {
        "mode": "off", "sigma_rate_per_ms": 0.0, "tau_ms": 0.0,
        "ell_mm": 0.0, "update_interval_ms": 0.0,
        "grid_spacing_mm": 0.0, "seed_offset": 0,
    }
    rows = [{"candidate_id": "edge_noop", **base, "spatial_ou": off}]
    library = config["spatial_ou_library"]
    for mode in library["modes"]:
        for sigma in library["sigma_rate_per_ms"]:
            for ell in library["ell_mm"]:
                rows.append({
                    "candidate_id": (
                        f"spou_{mode}_s{int(round(100 * sigma)):03d}"
                        f"_ell{int(round(100 * ell)):03d}"
                    ),
                    **base,
                    "spatial_ou": {
                        "mode": mode,
                        "sigma_rate_per_ms": float(sigma),
                        "tau_ms": float(library["tau_ms"]),
                        "ell_mm": float(ell),
                        "update_interval_ms": float(library["update_interval_ms"]),
                        "grid_spacing_mm": float(library["grid_spacing_mm"]),
                        "seed_offset": int(library["drive_seed_offset"]),
                    },
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
        "development_only_translation_invariant_spatial_ou_accessibility_canary"
    ):
        raise RuntimeError("rev10-D5 scientific role changed")
    for record in config["inputs"].values():
        if record["sha256"] == "TO_BE_FROZEN_AFTER_IMPLEMENTATION":
            raise RuntimeError(f"input hash is not frozen: {record['path']}")
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    timing = json.loads((ROOT / config["inputs"]["d4_1_timing_verdict"]["path"]).read_text())
    if timing["status"] != "REV10D4_1_FORMAL_GATE_FAIL_BUT_BROAD_ROUTE_TIMING_SUPPORTED":
        raise RuntimeError("D5 requires the bounded D4.1 route timing result")
    source_record = config["inputs"]["frozen_direction_classifier_manifest"]
    source = json.loads((ROOT / source_record["path"]).read_text())
    candidates = candidate_library(config)
    if len(candidates) != 9 or any(np.any(row["coefficients"]) for row in candidates):
        raise RuntimeError("rev10-D5 candidate library changed")
    payload = {
        "status": "REV10D5_SPATIAL_OU_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": _sha256(config_path)},
        "candidate_set": {
            "n_candidates": len(candidates), "candidates": candidates,
            "frozen_before_network_seeds": config["search"]["fit_network_seeds"],
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {
            "path": source_record["path"], "sha256": source_record["sha256"],
            "copied_without_refit": True,
        },
        "static_edge_contract": "all 12 coefficients are exact zero",
        "observation_invariance": {
            "forbidden_inputs": config["spatial_ou_library"]["forbidden_inputs"],
            "local_permuted_exact_value_multiset": True,
            "fixed_centers": False,
        },
        "git_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
