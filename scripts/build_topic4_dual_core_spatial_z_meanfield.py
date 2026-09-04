#!/usr/bin/env python3
"""Build a coarse deterministic model of the frozen rev21 dual-core SNN."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from src.topic4_dual_core_spatial_z import build_dual_core_z_map  # noqa: E402
from src.topic4_patient_zm_meanfield import (  # noqa: E402
    build_patient_coarse_model,
    save_patient_coarse_model,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json")
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/topic4_rev21_dual_core_zm_transition.json",
    )
    parser.add_argument(
        "--artifact-root", default="/home/honglab/leijiaxin/HFOsp",
    )
    parser.add_argument("--topology-seed", type=int, default=2542)
    parser.add_argument("--n-grid", type=int, default=10)
    parser.add_argument("--threshold-groups", type=int, default=8)
    parser.add_argument(
        "--out-root",
        default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                 "data_driven_dual_core_spatial_z/deterministic_meanfield"),
    )
    args = parser.parse_args()
    if args.n_grid < 1 or args.threshold_groups < 1:
        raise SystemExit("grid and threshold-group counts must be positive")

    started = time.time()
    artifact_root = Path(args.artifact_root).resolve()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    transition_path = (ROOT / config["inputs"]["transition_config"]["path"]).resolve()
    transition = load_round_config(transition_path)
    manifest_path = (
        artifact_root
        / "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
          "coarse_candidate_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    candidates = [
        row for row in manifest["candidates"]
        if row["candidate_id"] == "rev21_zm_off"
    ]
    if len(candidates) != 1:
        raise RuntimeError("rev21_zm_off must occur exactly once")
    candidate = candidates[0]
    mechanism = candidate["mechanisms"]
    node_mapping = candidate["node_mapping"]
    substrate = build_substrate(
        transition,
        config["reference"]["base_substrate_candidate_id"],
        int(args.topology_seed),
        cache_dir=str(artifact_root / config["network_cache"]),
        ee_dose=float(mechanism["g_EE"]),
        etoi_dose=float(mechanism["g_EtoI"]),
        node_candidate_override=candidate["node_field"],
        node_depth_shrinkage=float(node_mapping["signed_depth_shrinkage"]),
        node_gain=float(node_mapping["node_gain"]),
        ee_ellipse_angle_deg=float(mechanism["ellipse_angle_deg"]),
        ee_ellipse_aspect_ratio=float(mechanism["ellipse_aspect_ratio"]),
        artifact_root=artifact_root,
    )
    model = build_patient_coarse_model(
        substrate, n_grid=int(args.n_grid),
        threshold_groups=int(args.threshold_groups),
    )
    z_map = build_dual_core_z_map(
        model, substrate.positions_e, substrate.h_e,
        candidate["node_field"]["centers_mm"],
    )

    output_root = Path(args.out_root).resolve()
    stem = f"dualcore_topology_{args.topology_seed}_ngrid{args.n_grid}"
    model_path = output_root / f"{stem}.npz"
    model_record = save_patient_coarse_model(model_path, model)
    map_path = output_root / f"{stem}.zmap.npz"
    map_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=map_path.parent, prefix=map_path.stem + ".", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary,
            core_a_fraction_e=z_map.core_a_fraction_e,
            core_b_fraction_e=z_map.core_b_fraction_e,
            centers_mm=z_map.centers_mm,
            selected_count_per_core=z_map.selected_count_per_core,
        )
        os.replace(temporary, map_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    payload = {
        "status": "DUAL_CORE_SPATIAL_Z_MEANFIELD_BUILT",
        "scientific_role": (
            "deterministic frozen-spatial-Z reduction of the rev21 "
            "dualcore_s39 plus Joint=1.25 SNN substrate"
        ),
        "claim_boundary": (
            "A coarse graph/threshold reduction; not by itself a bifurcation, "
            "finite-SNN phase transition, anatomical-core or patient mechanism result."
        ),
        "topology_seed": int(args.topology_seed),
        "n_grid": int(args.n_grid),
        "cell_width_mm": float(model.sheet_l_mm / model.n_grid),
        "threshold_groups": int(args.threshold_groups),
        "source": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "transition_config": {
                "path": str(transition_path), "sha256": sha256(transition_path),
            },
            "candidate_manifest": {
                "path": str(manifest_path), "sha256": sha256(manifest_path),
            },
            "candidate_id": candidate["candidate_id"],
            "substrate_candidate_id": manifest["substrate_candidate_id"],
            "node_field": candidate["node_field"],
            "mechanisms": mechanism,
        },
        "network_cache": substrate.network_cache,
        "core_projection": {
            "centers_mm": z_map.centers_mm.tolist(),
            "selected_count_per_core": z_map.selected_count_per_core.tolist(),
            "selected_count_total": int(np.sum(z_map.selected_count_per_core)),
            "mixed_grid_cells": int(np.sum(
                (z_map.core_a_fraction_e + z_map.core_b_fraction_e > 0)
                & (z_map.core_a_fraction_e + z_map.core_b_fraction_e < 1)
            )),
        },
        "model_archive": model_record,
        "z_map_archive": {
            "path": str(map_path), "sha256": sha256(map_path),
            "bytes": map_path.stat().st_size,
        },
        "wall_seconds": float(time.time() - started),
    }
    json_path = output_root / f"{stem}.json"
    atomic_json(payload, json_path)
    print(json.dumps({
        "status": payload["status"], "model": model_record,
        "z_map": payload["z_map_archive"], "audit_json": str(json_path),
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
