#!/usr/bin/env python3
"""Continue one saved spatial-Z branch without repeating its expensive prefix."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    atomic_npz,
    load_z_map,
    point_arrays,
    sha256,
)
from src.topic4_dual_core_spatial_z import (  # noqa: E402
    SpatialZFixedPoint,
    path_state,
    pseudo_arclength_spatial_z,
    spatial_z_residual,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


def _restore_solution(model, z_map, rates, parameter) -> SpatialZFixedPoint:
    z_a, z_b, z_surround, field = path_state(z_map, float(parameter))
    second = z_map.z_second_moment_field(
        z_a=z_a, z_b=z_b, z_surround=z_surround)
    residual = spatial_z_residual(
        model, rates, z_field=field, z_second_moment=second)
    return SpatialZFixedPoint(
        rates=np.asarray(rates, float), parameter=float(parameter),
        z_a=z_a, z_b=z_b, z_surround=z_surround,
        eta_m=0.0, tau_m_slow_ms=500.0,
        converged=True, physical=True,
        residual_inf=float(np.max(np.abs(residual))),
        nfev=0, njev=0, message="restored from audited branch archive",
    )


def main() -> None:
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_dual_core_spatial_z/bifurcation/"
            "dualcore_spatial_z_bifurcation.json")
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", default=base)
    parser.add_argument("--arrays")
    parser.add_argument(
        "--prefix", choices=("entry_saddle", "recruited_arc", "extension"),
        required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--step-size", type=float, default=1e-4)
    parser.add_argument("--backtrack", type=int, default=0)
    parser.add_argument("--out-prefix", required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.step_size <= 0.0:
        raise SystemExit("steps and step size must be positive")

    started = time.time()
    result_path = Path(args.result).resolve()
    payload = json.loads(result_path.read_text())
    arrays_path = Path(args.arrays).resolve() if args.arrays else (
        result_path.with_suffix(".npz"))
    with np.load(arrays_path, allow_pickle=False) as archive:
        rates = np.asarray(archive[f"{args.prefix}__rates"], float)
        parameters = np.asarray(archive[f"{args.prefix}__s"], float)
    if rates.ndim != 2 or len(rates) < 2 or len(rates) != len(parameters):
        raise RuntimeError("saved branch does not contain aligned full states")
    model_path = Path(payload["substrate"]["model"]["path"])
    z_map_path = Path(payload["substrate"]["z_map"]["path"])
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    if not 0 <= args.backtrack <= len(rates) - 2:
        raise RuntimeError("backtrack is outside the saved branch")
    stop = len(rates) - int(args.backtrack)
    first = _restore_solution(
        model, z_map, rates[stop - 2], parameters[stop - 2])
    second = _restore_solution(
        model, z_map, rates[stop - 1], parameters[stop - 1])
    if max(first.residual_inf, second.residual_inf) >= 1e-8:
        raise RuntimeError("saved continuation endpoint no longer satisfies residual gate")

    points = pseudo_arclength_spatial_z(
        model, z_map, first, second, step_size=float(args.step_size),
        n_steps=int(args.steps), max_corrector_iterations=60)
    outputs = {}
    valid = point_arrays(
        model, z_map, points, "extension", outputs, save_rates=True)
    tangent = np.asarray([point.tangent_parameter for point in valid], float)
    turns = np.flatnonzero(tangent[:-1] * tangent[1:] <= 0.0)
    out_prefix = Path(args.out_prefix).resolve()
    atomic_npz(out_prefix.with_suffix(".npz"), **outputs)
    record = {
        "status": "SPATIAL_Z_BRANCH_EXTENSION_COMPLETE",
        "source": {
            "result": {"path": str(result_path), "sha256": sha256(result_path)},
            "arrays": {"path": str(arrays_path), "sha256": sha256(arrays_path)},
        },
        "branch_prefix": args.prefix,
        "requested_steps": int(args.steps),
        "returned_points": int(len(points)),
        "valid_points": int(len(valid)),
        "step_size": float(args.step_size),
        "backtrack": int(args.backtrack),
        "corrector_reached_step_limit": bool(len(points) == args.steps + 2),
        "tangent_crossing_indices": turns.astype(int).tolist(),
        "s_range": [float(outputs["extension__s"].min()),
                    float(outputs["extension__s"].max())],
        "mean_e_rate_hz_range": [
            float(outputs["extension__mean_e_hz"].min()),
            float(outputs["extension__mean_e_hz"].max())],
        "endpoint": {
            "s": float(outputs["extension__s"][-1]),
            "mean_e_rate_hz": float(outputs["extension__mean_e_hz"][-1]),
            "core_a_rate_hz": float(outputs["extension__core_a_hz"][-1]),
            "core_b_rate_hz": float(outputs["extension__core_b_hz"][-1]),
            "surround_rate_hz": float(outputs["extension__surround_hz"][-1]),
        },
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(record, out_prefix.with_suffix(".json"))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
