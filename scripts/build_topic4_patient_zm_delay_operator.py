#!/usr/bin/env python3
"""Build a conservative coarse delay operator from the frozen patient SNN."""
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
from src.topic4_patient_zm_delay import (  # noqa: E402
    PATHWAYS,
    build_patient_coarse_delay_operator,
    delay_summary,
    pathway_variance_matrix,
    pathway_weight_matrix,
    save_patient_coarse_delay_operator,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402
from src.topic4_zm_ictal_transition import build_substrate, load_round_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="config/topic4_spatial_zm_phase_diagram_v1.json")
    parser.add_argument("--n-grid", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config_path = _resolved_path(args.config)
    config = json.loads(config_path.read_text())
    source_audit = validate_sources(config)
    source = config["source"]
    trajectory_path = _resolved_path(source["trajectory_json"])
    trajectory = json.loads(trajectory_path.read_text())
    round_path = _resolved_path(source["round_config"])
    round_config = absolutize_round_inputs(
        load_round_config(round_path),
        Path(source["repository_artifact_root"]).resolve())
    output_root = Path(config["output_root"]).resolve() / "deterministic_meanfield"
    output = (Path(args.out).resolve() if args.out else output_root /
              f"patient_coarse_delay_ngrid{args.n_grid}.npz")
    model_path = output_root / f"patient_coarse_ngrid{args.n_grid}.npz"
    model = load_patient_coarse_model(model_path)

    started = time.time()
    with _working_directory(Path(source["repository_artifact_root"]).resolve()):
        substrate = build_substrate(
            round_config, trajectory["candidate_id"], int(trajectory["seed"]),
            cache_dir=source["network_cache_dir"], ee_dose=1.0, etoi_dose=1.0)
    operator = build_patient_coarse_delay_operator(
        substrate, n_grid=int(args.n_grid))
    record = save_patient_coarse_delay_operator(output, operator)
    conservation = {}
    for name in PATHWAYS:
        delayed = pathway_weight_matrix(operator, name).toarray()
        expected = np.asarray(getattr(model, f"w_{name}"), float)
        delayed_variance = pathway_variance_matrix(operator, name).toarray()
        expected_variance = np.asarray(getattr(model, f"v_{name}"), float)
        error = np.max(np.abs(delayed - expected))
        variance_error = np.max(np.abs(delayed_variance - expected_variance))
        conservation[name] = {
            "maximum_absolute_weight_error": float(error),
            "maximum_absolute_variance_weight_error": float(variance_error),
            "first_moment_allclose_atol_1e-10": bool(np.allclose(
                delayed, expected, atol=1e-10, rtol=1e-12)),
            "second_moment_allclose_atol_1e-10": bool(np.allclose(
                delayed_variance, expected_variance,
                atol=1e-10, rtol=1e-12)),
        }
    payload = {
        "status": ("PATIENT_COARSE_DELAY_OPERATOR_BUILT"
                   if all(row["first_moment_allclose_atol_1e-10"]
                          and row["second_moment_allclose_atol_1e-10"]
                          for row in conservation.values())
                   else "DELAY_OPERATOR_CONSERVATION_FAILED"),
        "scientific_role": (
            "delay-aware linear stability of the patient-matched frozen-q "
            "fixed-point branches"),
        "claim_boundary": (
            "All realized recurrent edge delays are retained before optional "
            "history-grid rebinning. Both recurrent first and second edge-weight "
            "moments are conserved; the downstream stability runner declares "
            "whether variance feedback is active or frozen."),
        "phase_config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "source_audit": source_audit,
        "source_model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "network_cache": substrate.network_cache,
        "operator": record,
        "delay_summary": {
            name: delay_summary(operator, name) for name in PATHWAYS},
        "pathway_moment_conservation": conservation,
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(payload, output.with_suffix(".json"))
    print(json.dumps({
        "status": payload["status"], "operator": record,
        "delay_summary": payload["delay_summary"],
        "pathway_moment_conservation": conservation,
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
