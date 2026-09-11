#!/usr/bin/env python3
"""Refine the localized-to-global fold on the dual-core spatial Z/M path.

This runner deliberately separates the low-state fold from the later fold at
which the stable spatially localized branch terminates.  The latter is the
only fixed-point saddle-node that can be proposed as a deterministic runaway
boundary, and it is exported as a candidate until native-delay trajectories
on both sides pass the bounded-versus-global state gate.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_dual_core_spatial_z_bifurcation import (  # noqa: E402
    atomic_json,
    atomic_npz,
    fold_evidence,
    load_z_map,
    sha256,
)
from src.topic4_dual_core_spatial_z import (  # noqa: E402
    closest_zero_eigenvalue,
    pseudo_arclength_spatial_z,
    regional_rates_hz,
    solve_spatial_z_fixed_point,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


DEFAULT_ROOT = Path(
    "/data/hfosp_topic4_fig45_artifacts/fig5/"
    "data_driven_dual_core_spatial_z")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coarse-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic4_dual_core_zm_regime_v1.json")
    parser.add_argument(
        "--out-prefix", type=Path,
        default=DEFAULT_ROOT / "runaway_boundary/localized_to_global_fold_m1")
    args = parser.parse_args()
    started = time.time()

    coarse_root = args.coarse_root.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dynamics = config["dynamics"]
    eta_m = float(dynamics["base_eta_m"])
    tau_m = float(dynamics["tau_m_ms"])
    model_path = coarse_root / (
        "deterministic_meanfield/dualcore_topology_2542_ngrid10.npz")
    z_map_path = model_path.with_suffix(".zmap.npz")
    atlas_path = coarse_root / (
        "bifurcation/branch_atlas/dualcore_spatial_z_branch_atlas.npz")
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    with np.load(atlas_path, allow_pickle=False) as atlas:
        s0 = np.asarray(atlas["core_a_entry__s"], float)
        rates0 = np.asarray(atlas["core_a_entry__rates"], float)
        folds0 = np.asarray(atlas["core_a_entry__fold_indices"], int)

    # The runaway candidate is selected structurally: it is the fold with the
    # largest s, not by a hand-entered array index or desired outcome.
    fold0_index = int(folds0[np.argmax(s0[folds0])])
    seed_indices = (fold0_index - 24, fold0_index - 12)
    adapted = []
    previous = None
    for index in seed_indices:
        initial = rates0[index] if previous is None else previous.rates
        solution = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(s0[index]), initial_rates=initial,
            eta_m=eta_m, tau_m_slow_ms=tau_m, maxfev=30000)
        if not solution.converged or not solution.physical:
            raise RuntimeError("failed to seed the adapted localized branch")
        adapted.append(solution)
        previous = solution

    points = pseudo_arclength_spatial_z(
        model, z_map, adapted[0], adapted[1], eta_m=eta_m,
        tau_m_slow_ms=tau_m, step_size=2e-6, n_steps=1200,
        max_corrector_iterations=60)
    valid = [point for point in points
             if point.solution.converged and point.solution.physical]
    tangent = np.asarray([point.tangent_parameter for point in valid], float)
    turns = np.flatnonzero(tangent[:-1] * tangent[1:] <= 0.0)
    if turns.size != 1:
        raise RuntimeError(
            f"expected one local fold, observed {turns.size}")
    turn = int(turns[0])
    local = valid[max(0, turn - 6):min(len(valid), turn + 8)]
    fold, zero_eigenvalues = fold_evidence(
        model, z_map, local, weights=(1.0, 1.0, 0.70),
        label="localized_to_global_candidate_fold")

    # Record the two physical branch sides next to the fold.  A saddle-node
    # requires the fixed-point residual Jacobian's real zero mode to change
    # sign together with the arclength tangent.
    side_records = []
    for offset, side in ((-4, "localized_branch"), (4, "saddle_branch")):
        point = valid[turn + offset]
        solution = point.solution
        eigenvalue = closest_zero_eigenvalue(model, solution, z_map)
        side_records.append({
            "side": side,
            "s": float(solution.parameter),
            "mean_e_rate_hz": float(solution.mean_rate_e_hz),
            "regional_e_rate_hz": regional_rates_hz(
                model, z_map, solution.rate_e),
            "fixed_point_zero_mode_real": float(np.real(eigenvalue)),
            "fixed_point_zero_mode_imag": float(np.imag(eigenvalue)),
            "tangent_s": float(point.tangent_parameter),
        })

    arrays = {
        "branch__s": np.asarray(
            [point.solution.parameter for point in valid], float),
        "branch__mean_e_hz": np.asarray(
            [point.solution.mean_rate_e_hz for point in valid], float),
        "branch__core_a_hz": np.asarray([
            regional_rates_hz(model, z_map, point.solution.rate_e)["core_a"]
            for point in valid], float),
        "branch__core_b_hz": np.asarray([
            regional_rates_hz(model, z_map, point.solution.rate_e)["core_b"]
            for point in valid], float),
        "branch__surround_hz": np.asarray([
            regional_rates_hz(model, z_map, point.solution.rate_e)["surround"]
            for point in valid], float),
        "branch__tangent_s": tangent,
        "branch__rates": np.asarray(
            [point.solution.rates for point in valid], float),
        "local__fixed_point_zero_mode_real": zero_eigenvalues,
    }
    out_prefix = args.out_prefix.resolve()
    atomic_npz(out_prefix.with_suffix(".npz"), **arrays)
    result = {
        "status": "LOCALIZED_TO_GLOBAL_FOLD_REFINED_DYNAMICAL_ESCAPE_PENDING",
        "substrate": "dualcore_s39 + Joint=1.25",
        "spatial_path": "Z_A=Z_B=1-s; Z_surround=1-0.70*s",
        "adaptation": {"eta_m": eta_m, "tau_m_ms": tau_m},
        "source_eta0_largest_s_fold": {
            "atlas_index": fold0_index,
            "s": float(s0[fold0_index]),
        },
        "localized_to_global_candidate_fold": fold,
        "adjacent_branch_sides": side_records,
        "selection_contract": (
            "largest-s fold on the independently continued core-A-entry "
            "family; selected before the native-delay outcome is read"),
        "claim_boundary": (
            "This establishes a saddle-node of the adapted frozen-spatial-Z "
            "fixed-point subsystem. It becomes a runaway boundary only if "
            "native-delay trajectories remain bounded below it and converge "
            "to a globally recruited state above it."),
        "sources": {
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": sha256(z_map_path)},
            "atlas": {"path": str(atlas_path), "sha256": sha256(atlas_path)},
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
        },
        "arrays": {"path": str(out_prefix.with_suffix('.npz'))},
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(result, out_prefix.with_suffix(".json"))
    print(json.dumps({
        "status": result["status"],
        "fold": fold,
        "output": str(out_prefix.with_suffix('.json')),
        "wall_seconds": result["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
