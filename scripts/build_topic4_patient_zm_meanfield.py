#!/usr/bin/env python3
"""Build the patient-matched deterministic bridge used for Z/M continuation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))

from scripts.run_topic4_zm_phase_point import (  # noqa: E402
    _atomic_json,
    _resolved_path,
    _sha256,
    _working_directory,
    absolutize_round_inputs,
    validate_sources,
)
from src.topic4_patient_zm_meanfield import (  # noqa: E402
    build_patient_coarse_model,
    save_patient_coarse_model,
)
from src.topic4_zm_ictal_transition import (  # noqa: E402
    build_substrate,
    load_round_config,
)


def _matrix_audit(model):
    audit = {}
    for name in ("w_ee", "w_ei", "w_ie", "w_ii",
                 "v_ee", "v_ei", "v_ie", "v_ii"):
        matrix = np.asarray(getattr(model, name), float)
        rows = matrix.sum(axis=1)
        audit[name] = {
            "shape": list(matrix.shape),
            "nonzero": int(np.count_nonzero(matrix)),
            "row_sum_min": float(rows.min()),
            "row_sum_median": float(np.median(rows)),
            "row_sum_mean": float(rows.mean()),
            "row_sum_max": float(rows.max()),
        }
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/topic4_spatial_zm_phase_diagram_v1.json")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--threshold-groups", type=int, default=8)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.n_grid < 1 or args.threshold_groups < 1:
        raise SystemExit("--n-grid and --threshold-groups must be positive")

    config_path = _resolved_path(args.config)
    config = json.loads(config_path.read_text())
    source_audit = validate_sources(config)
    source = config["source"]
    trajectory_path = _resolved_path(source["trajectory_json"])
    trajectory = json.loads(trajectory_path.read_text())
    round_path = _resolved_path(source["round_config"])
    round_config = load_round_config(round_path)
    artifact_root = Path(source["repository_artifact_root"]).resolve()
    round_config = absolutize_round_inputs(round_config, artifact_root)

    output_root = Path(config["output_root"]).resolve() / "deterministic_meanfield"
    output_path = (Path(args.out).resolve() if args.out else output_root /
                   f"patient_coarse_ngrid{args.n_grid}.npz")
    started = time.time()
    with _working_directory(artifact_root):
        substrate = build_substrate(
            round_config, trajectory["candidate_id"], int(trajectory["seed"]),
            cache_dir=source["network_cache_dir"], ee_dose=1.0, etoi_dose=1.0)
    model = build_patient_coarse_model(
        substrate, n_grid=args.n_grid, threshold_groups=args.threshold_groups)
    model_record = save_patient_coarse_model(output_path, model)

    threshold_mean = np.sum(
        model.threshold_nodes_e * model.threshold_weights_e, axis=1)
    payload = {
        "status": "PATIENT_MATCHED_ZM_MEANFIELD_BUILT",
        "scientific_role": (
            "deterministic_fixed_point_bridge_for_empirical_spatial_zm_phase_screen"),
        "claim_boundary": (
            "The archive is a graph- and threshold-matched diffusion reduction. "
            "It is not by itself bifurcation or SNN bistability evidence."),
        "phase_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "source_audit": source_audit,
        "source_trajectory": {
            "path": str(trajectory_path),
            "candidate_id": trajectory["candidate_id"],
            "substrate_seed": int(trajectory["seed"]),
        },
        "network_cache": substrate.network_cache,
        "coarse_graining": {
            "n_grid": model.n_grid,
            "cell_width_mm": model.sheet_l_mm / model.n_grid,
            "threshold_groups": int(args.threshold_groups),
            "count_e_min_max": [int(model.count_e.min()), int(model.count_e.max())],
            "count_i_min_max": [int(model.count_i.min()), int(model.count_i.max())],
            "threshold_cell_mean_min_median_max_mv": [
                float(threshold_mean.min()), float(np.median(threshold_mean)),
                float(threshold_mean.max())],
        },
        "model_constants": {
            "tau_mem_e_ms": model.tau_mem_e_ms,
            "tau_mem_i_ms": model.tau_mem_i_ms,
            "tau_ref_e_ms": model.tau_ref_e_ms,
            "tau_ref_i_ms": model.tau_ref_i_ms,
            "tau_ampa_ms": model.tau_ampa_ms,
            "tau_gaba_ms": model.tau_gaba_ms,
            "v_reset_mv": model.v_reset_mv,
            "v_threshold_i_mv": model.v_threshold_i_mv,
            "j_ext_e_mv": model.j_ext_e_mv,
            "j_ext_i_mv": model.j_ext_i_mv,
            "nu_ext_per_ms": model.nu_ext_per_ms,
        },
        "pathway_audit": _matrix_audit(model),
        "model_archive": model_record,
        "wall_seconds": float(time.time() - started),
    }
    json_path = output_path.with_suffix(".json")
    _atomic_json(payload, json_path)
    print(json.dumps({
        "status": payload["status"], "model": model_record,
        "audit_json": str(json_path),
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
