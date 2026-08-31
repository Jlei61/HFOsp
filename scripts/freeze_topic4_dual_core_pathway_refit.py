#!/usr/bin/env python3
"""Freeze the EE/E-to-I expression surface on the dual-core Node."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_pathway_refit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _candidate_id(g_ee: float, g_etoi: float) -> str:
    return f"gee{int(round(100 * g_ee)):03d}_getoi{int(round(100 * g_etoi)):03d}"


def freeze(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = ROOT / record["path"]
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {path}")

    dual_manifest = json.loads(
        (ROOT / config["inputs"]["dual_core_manifest"]["path"]).read_text()
    )
    dual_matches = [
        row for row in dual_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == "dualcore_s39"
    ]
    if len(dual_matches) != 1:
        raise RuntimeError("dualcore_s39 is not uniquely frozen")
    node_source = dual_matches[0]

    pathway_manifest = json.loads(
        (ROOT / config["inputs"]["source_pathway_manifest"]["path"]).read_text()
    )
    pathway_lookup = {
        row["candidate_id"]: row
        for row in pathway_manifest["candidate_set"]["candidates"]
    }
    refit = config["pathway_refit"]
    ee_source = np.asarray(
        pathway_lookup[refit["source_EE_candidate_id"]]["coefficients"], float,
    )
    etoi_source = np.asarray(
        pathway_lookup[refit["source_EtoI_candidate_id"]]["coefficients"], float,
    )
    if ee_source.shape != (2, 6) or etoi_source.shape != (2, 6):
        raise RuntimeError("source pathway rows must have shape (2, 6)")
    if not (np.all(ee_source[1] == 0.0) and np.all(etoi_source[0] == 0.0)):
        raise RuntimeError("source pathway manifests are not row-isolated")

    seeds = list(map(int, config["search"]["fit_network_seeds"]))
    duration_ms = float(config["search"]["duration_ms"]["fit"])
    candidates = []
    for g_ee in map(float, refit["g_EE"]):
        for g_etoi in map(float, refit["g_EtoI"]):
            coefficients = np.vstack([
                g_ee * ee_source[0],
                g_etoi * etoi_source[1],
            ]).astype(np.float64)
            candidate = copy.deepcopy(node_source)
            candidate.update({
                "candidate_id": _candidate_id(g_ee, g_etoi),
                "arm": "dual_core_pathway_expression_refit",
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": _array_sha256(coefficients),
                "raw_logit_clip": float(
                    config["local_connectivity_basis"]["raw_logit_clip_abs"]
                ),
                "search_coordinates": {
                    "g_EE": g_ee,
                    "g_EtoI": g_etoi,
                    "pathway_refit": True,
                    "source_EE_candidate_id": refit["source_EE_candidate_id"],
                    "source_EtoI_candidate_id": refit["source_EtoI_candidate_id"],
                },
                "allowed_network_seeds": seeds,
                "simulation_duration_ms": duration_ms,
                "save_activity_grid": False,
            })
            candidates.append(candidate)

    expected_count = len(refit["g_EE"]) * len(refit["g_EtoI"])
    if len(candidates) != expected_count or len({
        row["candidate_id"] for row in candidates
    }) != expected_count:
        raise RuntimeError("pathway surface candidate count is not exact")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    manifest = {
        "status": "REV16_DUAL_CORE_OOD_PHASE_FROZEN",
        "phase": "pathway_refit_screen",
        "scientific_role": config["scientific_role"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
        "direction_classifier": dual_manifest["direction_classifier"],
        "direction_classifier_source": dual_manifest[
            "direction_classifier_source"
        ],
        "fixed_contract": {
            "network_seeds": seeds,
            "duration_ms": duration_ms,
            "detector": config["search"]["detector"],
            "topology": "frozen",
            "delays": "frozen",
            "incoming_pathway_budgets": "conserved per postsynaptic target",
            "Node": "dualcore_s39 frozen",
            "spatial_OU": "frozen from dualcore_s39",
            "Z_M": "off",
        },
        "parent_inputs": config["inputs"],
        "provenance": {
            "commit": commit,
            "dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True,
            ).strip()),
        },
    }
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    manifest = freeze(args.config)
    print(json.dumps({
        "status": manifest["status"],
        "n_candidates": manifest["candidate_set"]["n_candidates"],
    }, indent=2))


if __name__ == "__main__":
    main()
