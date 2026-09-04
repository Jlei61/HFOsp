#!/usr/bin/env python3
"""Audit fold bifurcations after putting Z on the frozen dual-core field."""
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
from scipy.sparse.linalg import eigs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_dual_core_spatial_z import (  # noqa: E402
    DualCoreSpatialZMap,
    closest_zero_eigenvalue,
    path_state,
    pseudo_arclength_spatial_z,
    regional_rates_hz,
    solve_regional_z_fixed_point,
    solve_spatial_z_fixed_point,
    spatial_z_dynamic_jacobian,
    spatial_z_jacobian,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".json")
    os.close(descriptor)
    try:
        Path(temporary).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=path.stem + ".", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_z_map(path: Path) -> DualCoreSpatialZMap:
    with np.load(path, allow_pickle=False) as archive:
        result = DualCoreSpatialZMap(
            core_a_fraction_e=np.asarray(archive["core_a_fraction_e"], float),
            core_b_fraction_e=np.asarray(archive["core_b_fraction_e"], float),
            centers_mm=np.asarray(archive["centers_mm"], float),
            selected_count_per_core=np.asarray(
                archive["selected_count_per_core"], int),
        )
    return result


def point_arrays(model, z_map, points, prefix, arrays):
    valid = [point for point in points
             if point.solution.converged and point.solution.physical]
    arrays[f"{prefix}__s"] = np.asarray(
        [point.solution.parameter for point in valid], float)
    arrays[f"{prefix}__mean_e_hz"] = np.asarray(
        [point.solution.mean_rate_e_hz for point in valid], float)
    arrays[f"{prefix}__tangent_s"] = np.asarray(
        [point.tangent_parameter for point in valid], float)
    for region in ("core_a", "core_b", "surround"):
        arrays[f"{prefix}__{region}_hz"] = np.asarray([
            regional_rates_hz(model, z_map, point.solution.rate_e)[region]
            for point in valid
        ], float)
    return valid


def solution_arrays(model, z_map, solutions, prefix, arrays):
    valid = [solution for solution in solutions
             if solution.converged and solution.physical]
    arrays[f"{prefix}__s"] = np.asarray(
        [solution.parameter for solution in valid], float)
    arrays[f"{prefix}__mean_e_hz"] = np.asarray(
        [solution.mean_rate_e_hz for solution in valid], float)
    for region in ("core_a", "core_b", "surround"):
        arrays[f"{prefix}__{region}_hz"] = np.asarray([
            regional_rates_hz(model, z_map, solution.rate_e)[region]
            for solution in valid
        ], float)
    return valid


def fold_evidence(model, z_map, points, *, weights, label):
    valid = [point for point in points
             if point.solution.converged and point.solution.physical]
    tangent = np.asarray([point.tangent_parameter for point in valid], float)
    tangent_crossing = np.flatnonzero(tangent[:-1] * tangent[1:] <= 0.0)
    eigenvalues = np.asarray([
        float(np.real(closest_zero_eigenvalue(
            model, point.solution, z_map,
            core_a_weight=weights[0], core_b_weight=weights[1],
            surround_weight=weights[2])))
        for point in valid
    ])
    eigen_crossing = np.flatnonzero(eigenvalues[:-1] * eigenvalues[1:] <= 0.0)
    if not tangent_crossing.size or not eigen_crossing.size:
        raise RuntimeError(f"{label}: fold or zero-mode bracket missing")
    pair = min(
        ((int(i), int(j)) for i in tangent_crossing for j in eigen_crossing),
        key=lambda item: abs(item[0] - item[1]),
    )
    if abs(pair[0] - pair[1]) > 2:
        raise RuntimeError(f"{label}: fold and zero-mode brackets do not coincide")
    index = pair[1]
    left, right = valid[index].solution, valid[index + 1].solution
    left_eigen, right_eigen = eigenvalues[index:index + 2]
    fraction = float(left_eigen / (left_eigen - right_eigen))
    s_fold = float(
        left.parameter + fraction * (right.parameter - left.parameter))
    rate_fold = float(
        left.mean_rate_e_hz
        + fraction * (right.mean_rate_e_hz - left.mean_rate_e_hz))
    regional_left = regional_rates_hz(model, z_map, left.rate_e)
    regional_right = regional_rates_hz(model, z_map, right.rate_e)
    regional = {
        name: float(regional_left[name] + fraction * (
            regional_right[name] - regional_left[name]))
        for name in regional_left
    }
    z_a, z_b, z_surround, _ = path_state(
        z_map, s_fold, core_a_weight=weights[0],
        core_b_weight=weights[1], surround_weight=weights[2])
    return {
        "label": label,
        "s": s_fold,
        "regional_z": {
            "core_a": z_a, "core_b": z_b, "surround": z_surround},
        "mean_e_rate_hz": rate_fold,
        "regional_e_rate_hz": regional,
        "tangent_crossing_index": int(pair[0]),
        "zero_mode_crossing_index": int(pair[1]),
        "s_bracket": [float(left.parameter), float(right.parameter)],
        "tangent_s_bracket": [
            float(tangent[pair[0]]), float(tangent[pair[0] + 1])],
        "fixed_point_eigenvalue_real_bracket": [
            float(left_eigen), float(right_eigen)],
        "interpolation_fraction": fraction,
    }, eigenvalues


def distinct_partner_on_zero_mode(model, z_map, low, *, weights):
    _za, _zb, _zs, field = path_state(
        z_map, low.parameter, core_a_weight=weights[0],
        core_b_weight=weights[1], surround_weight=weights[2])
    field2 = z_map.z_second_moment_field(
        z_a=_za, z_b=_zb, z_surround=_zs)
    values, vectors = np.linalg.eig(spatial_z_jacobian(
        model, low.rates, z_field=field, z_second_moment=field2))
    vector = np.real(vectors[:, int(np.argmin(np.abs(values)))])
    vector /= np.max(np.abs(vector))
    for amplitude in (1e-4, 3e-4, 1e-3, 3e-3):
        for sign in (-1.0, 1.0):
            candidate = solve_spatial_z_fixed_point(
                model, z_map, parameter=low.parameter,
                initial_rates=low.rates + sign * amplitude * vector,
                core_a_weight=weights[0], core_b_weight=weights[1],
                surround_weight=weights[2], maxfev=20000)
            rms = float(np.sqrt(np.mean((candidate.rates - low.rates) ** 2)))
            if candidate.converged and candidate.physical and rms > 1e-7:
                return candidate, {
                    "zero_mode_at_low_root": {
                        "real": float(np.real(values[np.argmin(np.abs(values))])),
                        "imag": float(np.imag(values[np.argmin(np.abs(values))])),
                    },
                    "perturbation_amplitude": float(amplitude),
                    "perturbation_sign": float(sign),
                    "root_rms_separation_spikes_per_ms": rms,
                }
    raise RuntimeError("failed to recover the paired root along the zero mode")


def critical_mode_summary(model, z_map, points, eigenvalues, *, weights,
                          prefix, arrays):
    valid = [point for point in points
             if point.solution.converged and point.solution.physical]
    index = int(np.argmin(np.abs(np.asarray(eigenvalues, float))))
    point = valid[index]
    _za, _zb, _zs, field = path_state(
        z_map, point.solution.parameter, core_a_weight=weights[0],
        core_b_weight=weights[1], surround_weight=weights[2])
    field2 = z_map.z_second_moment_field(
        z_a=_za, z_b=_zb, z_surround=_zs)
    values, vectors = np.linalg.eig(spatial_z_jacobian(
        model, point.solution.rates, z_field=field,
        z_second_moment=field2))
    mode = int(np.argmin(np.abs(values)))
    eigenvalue, vector = values[mode], vectors[:, mode]
    mode_e = np.real(vector[:model.n_cells])
    mode_e /= np.max(np.abs(mode_e))
    arrays[f"{prefix}__critical_mode_e"] = mode_e.reshape(
        model.n_grid, model.n_grid)
    counts = np.asarray(model.count_e, float)
    memberships = {
        "core_a": z_map.core_a_fraction_e,
        "core_b": z_map.core_b_fraction_e,
        "surround": z_map.surround_fraction_e,
    }
    total_energy = float(np.sum(counts * mode_e ** 2))
    regional = {}
    for name, membership in memberships.items():
        regional_count = counts * membership
        regional[name] = {
            "rms": float(np.sqrt(np.sum(regional_count * mode_e ** 2)
                                 / np.sum(regional_count))),
            "fraction_of_e_mode_energy": float(
                np.sum(regional_count * mode_e ** 2) / total_energy),
        }
    peak = int(np.argmax(np.abs(mode_e)))
    x = (peak % model.n_grid + 0.5) * model.sheet_l_mm / model.n_grid
    y = (peak // model.n_grid + 0.5) * model.sheet_l_mm / model.n_grid
    return {
        "arc_index": int(index),
        "s": float(valid[index].solution.parameter),
        "eigenvalue": {"real": float(np.real(eigenvalue)),
                       "imag": float(np.imag(eigenvalue))},
        "peak_xy_mm": [float(x), float(y)],
        "regional_e_mode": regional,
    }


def empirical_ou_projection(worker_root: Path):
    records = []
    collection_lines = []
    for stage in ("coarse", "timescale"):
        for worker_json in sorted((worker_root / stage / "workers").glob("*.json")):
            payload = json.loads(worker_json.read_text())
            endpoint = payload.get("model_ictal_rev21")
            if endpoint is None:
                continue
            if not endpoint["clauses"]["operational_detector_reached"]:
                continue
            operational_onset = float(endpoint["landmarks"]["t_op_ms"])
            array_path = Path(payload["arrays"]["path"])
            with np.load(array_path, allow_pickle=False) as archive:
                time_ms = np.asarray(archive["slow_time_ms"], float)
                index = int(np.argmin(np.abs(time_ms - operational_onset)))
                records.append((
                    float(archive["slow_z_mean"][index]),
                    float(archive["slow_z_core_mean"][index]),
                    float(archive["slow_z_surround_mean"][index]),
                    abs(float(time_ms[index]) - operational_onset),
                ))
            collection_lines.append(
                f"{stage}/{worker_json.name}\t{sha256(worker_json)}\t"
                f"{payload['arrays']['sha256']}")
    if len(records) != 100:
        raise RuntimeError(
            f"expected 100 OU-on active runs across coarse/timescale, got {len(records)}")
    values = np.asarray(records, float)
    quantiles = np.quantile(values[:, :3], [0.1, 0.5, 0.9], axis=0)
    collection_hash = hashlib.sha256(
        ("\n".join(collection_lines) + "\n").encode()).hexdigest()
    return {
        "source": "rev21 coarse+timescale OU-on runs at operational onset",
        "worker_root": str(worker_root.resolve()),
        "worker_collection_sha256": collection_hash,
        "n_active_runs": int(values.shape[0]),
        "maximum_onset_to_saved_sample_error_ms": float(np.max(values[:, 3])),
        "z_mean_median": float(quantiles[1, 0]),
        "z_mean_q10_q90": [float(quantiles[0, 0]), float(quantiles[2, 0])],
        "z_core_median": float(quantiles[1, 1]),
        "z_core_q10_q90": [float(quantiles[0, 1]), float(quantiles[2, 1])],
        "z_surround_median": float(quantiles[1, 2]),
        "z_surround_q10_q90": [
            float(quantiles[0, 2]), float(quantiles[2, 2])],
    }


def leading_dynamic_eigenvalues(model, z_map, solution, *, k=8):
    field = z_map.z_field(
        z_a=solution.z_a, z_b=solution.z_b,
        z_surround=solution.z_surround)
    field2 = z_map.z_second_moment_field(
        z_a=solution.z_a, z_b=solution.z_b,
        z_surround=solution.z_surround)
    values = eigs(
        spatial_z_dynamic_jacobian(
            model, solution.rates, z_field=field,
            z_second_moment=field2, eta_m=solution.eta_m,
            tau_m_slow_ms=solution.tau_m_slow_ms),
        k=int(k), which="LR", return_eigenvectors=False,
        tol=1e-8, maxiter=30000)
    values = sorted(values, key=lambda value: value.real, reverse=True)
    return [{"real": float(value.real), "imag": float(value.imag)}
            for value in values]


def root_catalog(model, z_map, *, z_a_values, z_b_values, z_surround):
    shape = (len(z_b_values), len(z_a_values))
    root_count = np.zeros(shape, int)
    low_present = np.zeros(shape, bool)
    recruited_present = np.zeros(shape, bool)
    minimum_rate = np.full(shape, np.nan)
    maximum_rate = np.full(shape, np.nan)
    for row, z_b in enumerate(z_b_values):
        for column, z_a in enumerate(z_a_values):
            roots = []
            for initial_e in (1e-4, 0.08, 0.25, 0.40):
                initial = np.r_[
                    np.full(model.n_cells, initial_e),
                    np.full(model.n_cells, min(1.1 * initial_e, 0.49)),
                ]
                solution = solve_regional_z_fixed_point(
                    model, z_map, z_a=float(z_a), z_b=float(z_b),
                    z_surround=float(z_surround), initial_rates=initial,
                    maxfev=10000)
                if not solution.converged or not solution.physical:
                    continue
                if any(np.sqrt(np.mean((solution.rates - other.rates) ** 2))
                       < 1e-6 for other in roots):
                    continue
                roots.append(solution)
            rates = np.asarray([root.mean_rate_e_hz for root in roots], float)
            root_count[row, column] = len(roots)
            if not roots:
                continue
            minimum_rate[row, column] = float(np.min(rates))
            maximum_rate[row, column] = float(np.max(rates))
            low_present[row, column] = bool(np.any(rates < 5.0))
            for solution in roots:
                regional = regional_rates_hz(model, z_map, solution.rate_e)
                if (solution.mean_rate_e_hz >= 80.0
                        and min(regional.values()) >= 30.0):
                    recruited_present[row, column] = True
                    break
    state_code = low_present.astype(np.int8) + 2 * recruited_present.astype(np.int8)
    return {
        "root_count": root_count,
        "low_present": low_present,
        "recruited_present": recruited_present,
        "state_code": state_code,
        "minimum_rate_hz": minimum_rate,
        "maximum_rate_hz": maximum_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
                    "data_driven_dual_core_spatial_z")
    parser.add_argument(
        "--model", default=default_root
        + "/deterministic_meanfield/dualcore_topology_2542_ngrid10.npz")
    parser.add_argument(
        "--z-map", default=default_root
        + "/deterministic_meanfield/dualcore_topology_2542_ngrid10.zmap.npz")
    parser.add_argument(
        "--source-audit", default=default_root
        + "/deterministic_meanfield/dualcore_topology_2542_ngrid10.json")
    parser.add_argument(
        "--out", default=default_root
        + "/bifurcation/dualcore_spatial_z_bifurcation.json")
    parser.add_argument(
        "--ou-worker-root",
        default=("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                 "data_driven_dual_core_zm_transition"))
    args = parser.parse_args()
    started = time.time()
    model_path = Path(args.model).resolve()
    z_map_path = Path(args.z_map).resolve()
    source_audit_path = Path(args.source_audit).resolve()
    output = Path(args.out).resolve()
    model = load_patient_coarse_model(model_path)
    z_map = load_z_map(z_map_path)
    z_map.validate(model)
    weights = (1.0, 1.0, 0.70)
    arrays = {}

    # Outer recruited branch: continuation exposes the spatial fold chain.
    high_seed = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.15,
        initial_rates=np.r_[np.full(model.n_cells, 0.30),
                            np.full(model.n_cells, 0.33)], maxfev=20000)
    high_first = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.105, initial_rates=high_seed.rates,
        maxfev=20000)
    high_second = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.100, initial_rates=high_first.rates,
        maxfev=20000)
    if not all(item.converged for item in (high_seed, high_first, high_second)):
        raise RuntimeError("failed to seed the recruited spatial-Z branch")
    recruited_arc = pseudo_arclength_spatial_z(
        model, z_map, high_first, high_second, step_size=1e-4,
        n_steps=800, max_corrector_iterations=40)
    recruited_valid = point_arrays(
        model, z_map, recruited_arc, "recruited_arc", arrays)
    recruited_tangent = np.asarray(
        [item.tangent_parameter for item in recruited_valid])
    recruited_turns = np.flatnonzero(
        recruited_tangent[:-1] * recruited_tangent[1:] <= 0.0)
    if not recruited_turns.size:
        raise RuntimeError("recruited spatial branch has no fold")

    high_down = [high_seed]
    previous = high_seed
    for parameter in np.arange(0.149, 0.094, -0.001):
        previous = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter),
            initial_rates=previous.rates, maxfev=20000)
        if not previous.converged:
            raise RuntimeError(f"outer recruited branch failed at s={parameter}")
        high_down.append(previous)
    high_up = [high_seed]
    previous = high_seed
    for parameter in np.arange(0.151, 0.1971, 0.001):
        previous = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter),
            initial_rates=previous.rates, maxfev=20000)
        if not previous.converged:
            raise RuntimeError(f"inner recruited branch failed at s={parameter}")
        high_up.append(previous)
    upper_seed = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.200,
        initial_rates=np.r_[np.full(model.n_cells, 0.30),
                            np.full(model.n_cells, 0.33)], maxfev=20000)
    if not upper_seed.converged:
        raise RuntimeError("failed to seed upper recruited branch")
    high_upper = [upper_seed]
    previous = upper_seed
    for parameter in np.arange(0.205, 0.401, 0.005):
        previous = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter),
            initial_rates=previous.rates, maxfev=20000)
        if not previous.converged:
            raise RuntimeError(f"upper recruited branch failed at s={parameter}")
        high_upper.append(previous)
    outer_high = sorted(
        high_down + high_up[1:] + high_upper,
        key=lambda solution: solution.parameter)
    solution_arrays(model, z_map, outer_high, "outer_high", arrays)

    # Refine the first recovery fold on the recruited branch.
    index = int(recruited_turns[0])
    recovery_fine = pseudo_arclength_spatial_z(
        model, z_map, recruited_valid[index - 2].solution,
        recruited_valid[index - 1].solution, step_size=2e-6,
        n_steps=120, max_corrector_iterations=50)
    recovery_valid = point_arrays(
        model, z_map, recovery_fine, "recovery_fold", arrays)
    recovery_fold, recovery_eigenvalues = fold_evidence(
        model, z_map, recovery_valid, weights=weights,
        label="recruited_branch_recovery_fold")
    arrays["recovery_fold__eigen_real"] = recovery_eigenvalues
    recovery_mode = critical_mode_summary(
        model, z_map, recovery_valid, recovery_eigenvalues, weights=weights,
        prefix="recovery_fold", arrays=arrays)

    # Low branch and its paired unstable root at the runaway-entry fold.
    low_branch = []
    previous = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.0,
        initial_rates=np.r_[np.full(model.n_cells, 1e-4),
                            np.full(model.n_cells, 5e-4)], maxfev=20000)
    low_parameters = np.r_[
        np.linspace(0.0, 0.33, 67),
        [0.335, 0.337, 0.3374, 0.3375, 0.33755, 0.33758],
    ]
    for parameter in low_parameters:
        previous = solve_spatial_z_fixed_point(
            model, z_map, parameter=float(parameter),
            initial_rates=previous.rates, maxfev=20000)
        if not previous.converged:
            raise RuntimeError(f"low branch failed at s={parameter}")
        low_branch.append(previous)
    solution_arrays(model, z_map, low_branch, "low_branch", arrays)
    low = low_branch[-1]
    partner, partner_audit = distinct_partner_on_zero_mode(
        model, z_map, low, weights=weights)
    partner_first = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.337586,
        initial_rates=partner.rates, maxfev=20000)
    partner_second = solve_spatial_z_fixed_point(
        model, z_map, parameter=0.337590,
        initial_rates=partner_first.rates, maxfev=20000)
    if not partner_first.converged or not partner_second.converged:
        raise RuntimeError("failed to seed entry-fold paired branch")
    entry_fine = pseudo_arclength_spatial_z(
        model, z_map, partner_first, partner_second, step_size=2e-8,
        n_steps=160, max_corrector_iterations=60)
    entry_valid = point_arrays(
        model, z_map, entry_fine, "entry_fold", arrays)
    entry_fold, entry_eigenvalues = fold_evidence(
        model, z_map, entry_valid, weights=weights,
        label="low_branch_runaway_entry_fold")
    arrays["entry_fold__eigen_real"] = entry_eigenvalues
    entry_mode = critical_mode_summary(
        model, z_map, entry_valid, entry_eigenvalues, weights=weights,
        prefix="entry_fold", arrays=arrays)

    # A transparent, finite multi-start map: it is not advertised as exhaustive.
    z_a_values = np.linspace(0.55, 0.95, 11)
    z_b_values = np.linspace(0.55, 0.95, 11)
    phase = root_catalog(
        model, z_map, z_a_values=z_a_values, z_b_values=z_b_values,
        z_surround=0.80)
    arrays["phase__z_a"] = z_a_values
    arrays["phase__z_b"] = z_b_values
    arrays["phase__z_surround"] = np.asarray([0.80])
    for name, value in phase.items():
        arrays[f"phase__{name}"] = value

    empirical = empirical_ou_projection(Path(args.ou_worker_root).resolve())
    empirical["s_from_core_median"] = 1.0 - empirical["z_core_median"]
    empirical["predicted_z_surround_on_frozen_path"] = (
        1.0 - weights[2] * empirical["s_from_core_median"])
    empirical["surround_depletion_ratio_observed"] = (
        (1.0 - empirical["z_surround_median"])
        / (1.0 - empirical["z_core_median"]))
    empirical_s = float(empirical["s_from_core_median"])
    empirical_low = solve_spatial_z_fixed_point(
        model, z_map, parameter=empirical_s,
        initial_rates=low.rates, maxfev=20000)
    empirical_high = solve_spatial_z_fixed_point(
        model, z_map, parameter=empirical_s,
        initial_rates=np.r_[np.full(model.n_cells, 0.30),
                            np.full(model.n_cells, 0.33)], maxfev=20000)
    if not empirical_low.converged or not empirical_high.converged:
        raise RuntimeError("empirical-onset projection roots did not converge")
    empirical["deterministic_roots_on_frozen_path"] = {
        "low_mean_e_rate_hz": empirical_low.mean_rate_e_hz,
        "recruited_mean_e_rate_hz": empirical_high.mean_rate_e_hz,
        "low_regional_e_rate_hz": regional_rates_hz(
            model, z_map, empirical_low.rate_e),
        "recruited_regional_e_rate_hz": regional_rates_hz(
            model, z_map, empirical_high.rate_e),
    }
    low_dynamic = leading_dynamic_eigenvalues(
        model, z_map, empirical_low)
    high_dynamic = leading_dynamic_eigenvalues(
        model, z_map, empirical_high)
    empirical["zero_delay_frozen_variance_stability"] = {
        "low_root": {
            "maximum_real_part_per_ms": low_dynamic[0]["real"],
            "leading_eigenvalues_per_ms": low_dynamic},
        "recruited_root": {
            "maximum_real_part_per_ms": high_dynamic[0]["real"],
            "leading_eigenvalues_per_ms": high_dynamic},
        "interpretation": (
            "both representative roots are stable under this labelled "
            "zero-delay, operating-variance-frozen sensitivity"),
        "boundary": (
            "not a delay-aware stability theorem and does not include OU noise"),
    }

    array_path = output.with_suffix(".npz")
    atomic_npz(array_path, **arrays)
    source_audit = json.loads(source_audit_path.read_text())
    payload = {
        "status": "DUAL_CORE_SPATIAL_Z_FOLD_CHAIN_ESTABLISHED",
        "substrate": {
            "identity": "dualcore_s39 + Joint=1.25",
            "topology_seed": source_audit["topology_seed"],
            "centers_mm": source_audit["core_projection"]["centers_mm"],
            "selected_count_per_core": source_audit["core_projection"][
                "selected_count_per_core"],
            "n_grid": int(model.n_grid),
            "cell_width_mm": float(model.sheet_l_mm / model.n_grid),
            "model": {"path": str(model_path), "sha256": sha256(model_path)},
            "z_map": {"path": str(z_map_path), "sha256": sha256(z_map_path)},
            "source_audit": {
                "path": str(source_audit_path),
                "sha256": sha256(source_audit_path)},
        },
        "spatial_z_path": {
            "definition": "Z_A=1-s; Z_B=1-s; Z_surround=1-0.70*s",
            "weights": {"core_a": weights[0], "core_b": weights[1],
                        "surround": weights[2]},
            "mixed_cell_moment_closure": (
                "inhibitory mean uses within-cell E[Z]; inhibitory variance "
                "uses within-cell E[Z^2]"),
            "rationale": (
                "0.70 was frozen from the rev21 OU-on onset depletion ratio; "
                "the observed median ratio is reported separately, not refit."),
        },
        "folds": {
            "runaway_entry": entry_fold,
            "recruited_recovery": recovery_fold,
            "runaway_entry_critical_mode": entry_mode,
            "recruited_recovery_critical_mode": recovery_mode,
            "additional_recruited_branch_fold_count": int(
                max(0, recruited_turns.size - 1)),
            "all_recruited_arc_tangent_crossing_indices": (
                recruited_turns.astype(int).tolist()),
        },
        "paired_root_audit": partner_audit,
        "empirical_ou_on_projection": empirical,
        "phase_map": {
            "z_surround": 0.80,
            "z_a_range": [float(z_a_values[0]), float(z_a_values[-1])],
            "z_b_range": [float(z_b_values[0]), float(z_b_values[-1])],
            "n_per_axis": int(z_a_values.size),
            "initial_e_rates_spikes_per_ms": [1e-4, 0.08, 0.25, 0.40],
            "low_root_definition": "mean E rate < 5 Hz",
            "recruited_tonic_root_definition": (
                "mean E rate >=80 Hz and every regional E rate >=30 Hz"),
            "boundary": (
                "finite multi-start root catalog; absence of a found root is "
                "not proof of mathematical non-existence"),
        },
        "claim": (
            "The frozen 1-mm deterministic reduction of the data-driven dual-core "
            "substrate has a spatial saddle-node fold chain. The low-branch fold "
            "removes the low fixed point and therefore supplies a deterministic "
            "runaway boundary; the OU-on SNN crosses operationally earlier inside "
            "the coexistence region, so the fold organizes susceptibility but does "
            "not by itself time the stochastic transition."),
        "claim_boundary": (
            "This establishes folds in a coarse deterministic fast subsystem, not "
            "a thermodynamic phase transition, a delay-aware stability theorem, or "
            "a patient-specific inhibitory field. Core geometry is data-driven; Z "
            "values and the symmetric depletion path are model projections."),
        "implementation": {
            "runner": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve())},
            "spatial_z_module": {
                "path": str((ROOT / "src/topic4_dual_core_spatial_z.py").resolve()),
                "sha256": sha256(ROOT / "src/topic4_dual_core_spatial_z.py")},
            "meanfield_module": {
                "path": str((ROOT / "src/topic4_patient_zm_meanfield.py").resolve()),
                "sha256": sha256(ROOT / "src/topic4_patient_zm_meanfield.py")},
        },
        "arrays": {"path": str(array_path), "sha256": sha256(array_path)},
        "wall_seconds": float(time.time() - started),
    }
    atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "runaway_entry_fold": entry_fold,
        "recruited_recovery_fold": recovery_fold,
        "additional_recruited_folds": payload["folds"][
            "additional_recruited_branch_fold_count"],
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
