"""Freeze the D5.3 continuous spatial-OU KMeans grid."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d5_3_spatial_ou_kmeans_grid.json"
EXPECTED_ROLE = "development_only_translation_invariant_spatial_ou_kmeans_grid"


def _token(value):
    return f"{int(round(1000.0 * float(value))):04d}"


def candidate_library(config):
    count = int(config["spatial_edge_basis"]["coefficient_count"])
    zero = np.zeros(count, float)
    base = {"coefficients": zero.tolist(), "coefficients_sha256": array_sha256(zero)}
    rows = [{
        "candidate_id": "edge_noop", **base,
        "spatial_ou": {
            "mode": "off", "sigma_rate_per_ms": 0.0, "tau_ms": 0.0,
            "ell_mm": 0.0, "update_interval_ms": 0.0,
            "grid_spacing_mm": 0.0, "seed_offset": 0,
        },
    }]
    library = config["spatial_ou_library"]
    for sigma in library["sigma_rate_per_ms"]:
        for tau in library["tau_ms"]:
            rows.append({
                "candidate_id": f"spou_local_s{_token(sigma)}_tau{int(tau):03d}",
                **base,
                "spatial_ou": {
                    "mode": "local",
                    "sigma_rate_per_ms": float(sigma),
                    "tau_ms": float(tau),
                    "ell_mm": float(library["ell_mm"]),
                    "update_interval_ms": float(library["update_interval_ms"]),
                    "grid_spacing_mm": float(library["grid_spacing_mm"]),
                    "seed_offset": int(library["drive_seed_offset"]),
                },
            })
    return rows


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D5.3 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    verdict_record = config["inputs"]["d5_2_corrected_verdict"]
    verdict = json.loads((ROOT / verdict_record["path"]).read_text())
    if verdict.get("status") != (
        "REV10D5_2_DIRECTION_PROTOTYPES_RECOVERED_"
        "KMEANS_BELOW_PATIENT_BENCHMARK"
    ):
        raise RuntimeError("D5.3 requires the corrected D5.2 KMeans verdict")
    source_record = config["inputs"]["frozen_direction_classifier_manifest"]
    source = json.loads((ROOT / source_record["path"]).read_text())
    candidates = candidate_library(config)
    if len(candidates) != 13 or any(np.any(row["coefficients"]) for row in candidates):
        raise RuntimeError("D5.3 candidate library changed")
    if any(row["spatial_ou"]["mode"] not in {"off", "local"} for row in candidates):
        raise RuntimeError("D5.3 may not use observation-conditioned controls")

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
        raise RuntimeError("D5.3 freezer runtime is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D5.3 workers exist before the manifest freeze")
    return {
        "status": "REV10D5_3_SPATIAL_OU_KMEANS_GRID_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "n_exact_off": 1,
            "n_local": 12,
            "candidates": candidates,
            "frozen_before_network_seeds": config["search"]["fit_network_seeds"],
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {
            "path": source_record["path"], "sha256": source_record["sha256"],
            "copied_without_refit": True,
        },
        "kmeans_selection_contract": config["search"]["kmeans_selection"],
        "d5_2_anchor": {
            "candidate_id": verdict["selected_local_candidate_id"],
            "canonical_direction_purity": verdict["canonical_fig4c_kmeans"][
                "direction_purity"
            ],
            "patient_matched_q05": verdict[
                "patient_matched_kmeans_direction_purity"
            ]["q05"],
            "verdict_path": verdict_record["path"],
            "verdict_sha256": verdict_record["sha256"],
        },
        "static_edge_contract": "all 12 coefficients are exact zero",
        "forbidden_inputs": config["spatial_ou_library"]["forbidden_inputs"],
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build_manifest(args.config, args.expected_commit)
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": payload["candidate_set"]["n_candidates"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
