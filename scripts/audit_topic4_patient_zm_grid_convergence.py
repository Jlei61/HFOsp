#!/usr/bin/env python3
"""Audit branch identity and fold sensitivity across conservative Z/M grids.

The 2, 1.33 and 1 mm models are not three independently fitted models.  They
are conservative aggregations of the same realized patient SNN graph.  This
script follows the same high-rate branch from common low-q anchors, passes its
first fold by pseudo-arclength, and compares both branch fields and fold
locations without calling the finite SNN transition a thermodynamic phase
transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_zm_phase_point import _atomic_json, _atomic_npz  # noqa: E402
from src.topic4_patient_zm_meanfield import (  # noqa: E402
    fixed_point_jacobian,
    load_patient_coarse_model,
    pseudo_arclength_continue,
    solve_fixed_point,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remap_square_field(values, source_grid: int, target_grid: int = 60):
    """Piecewise-constant remap when target_grid is divisible by source_grid."""
    values = np.asarray(values, float)
    if values.shape != (int(source_grid) ** 2,):
        raise ValueError("field size does not match source grid")
    if int(target_grid) % int(source_grid):
        raise ValueError("target grid must be divisible by source grid")
    factor = int(target_grid) // int(source_grid)
    return np.repeat(
        np.repeat(values.reshape(source_grid, source_grid), factor, axis=0),
        factor, axis=1).ravel()


def compare_branch_fields(first, second):
    """Return scale and patient-field agreement for two common-grid branches."""
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("branch fields must be aligned vectors")
    difference = first - second
    centered_first = first - np.mean(first)
    centered_second = second - np.mean(second)
    denominator = np.linalg.norm(centered_first) * np.linalg.norm(centered_second)
    correlation = (float(np.dot(centered_first, centered_second) / denominator)
                   if denominator > 0.0 else float("nan"))
    pooled_mean = 0.5 * (abs(float(np.mean(first))) + abs(float(np.mean(second))))
    return {
        "mean_rate_difference_hz": float(abs(np.mean(first) - np.mean(second))),
        "relative_mean_rate_difference": float(
            abs(np.mean(first) - np.mean(second)) / max(pooled_mean, 1e-12)),
        "rms_field_difference_hz": float(np.sqrt(np.mean(difference ** 2))),
        "relative_rms_field_difference": float(
            np.sqrt(np.mean(difference ** 2)) / max(pooled_mean, 1e-12)),
        "centered_spatial_correlation": correlation,
    }


def _first_tangent_crossing(points):
    values = np.asarray([point.tangent_q for point in points], float)
    indices = np.flatnonzero((values[:-1] > 0.0) & (values[1:] < 0.0))
    if not indices.size:
        raise RuntimeError("continued branch has no positive-to-negative dq/ds turn")
    return int(indices[0])


def _valid(points):
    return [point for point in points if point.solution.converged]


def _interpolated_zero(x0, x1, y0, y1):
    fraction = float(y0 / (y0 - y1))
    return float(x0 + fraction * (x1 - x0)), fraction


def _follow_to_anchors(model, *, eta_m, tau_m_ms, anchors):
    rates = np.r_[np.full(model.n_cells, 0.30),
                  np.full(model.n_cells, 0.32)]
    previous_q = float(anchors[0])
    output = {}
    for target_q in anchors:
        target_q = float(target_q)
        count = max(1, int(np.ceil((target_q - previous_q) / 0.0025)))
        grid = ([target_q] if target_q == previous_q else
                np.linspace(previous_q, target_q, count + 1)[1:])
        solution = None
        for q in grid:
            solution = solve_fixed_point(
                model, q=float(q), eta_m=eta_m,
                tau_m_slow_ms=tau_m_ms, initial_rates=rates, maxfev=10000)
            if not solution.converged:
                raise RuntimeError(
                    f"fixed-q branch continuation failed at q={q:.6f}")
            rates = solution.rates
        output[target_q] = solution
        previous_q = target_q
    return output


def _fold_for_model(model, *, eta_m, tau_m_ms, anchors):
    fixed = _follow_to_anchors(
        model, eta_m=eta_m, tau_m_ms=tau_m_ms, anchors=anchors)
    first = fixed[float(anchors[-2])]
    second = fixed[float(anchors[-1])]
    coarse = None
    coarse_cross = None
    coarse_step = None
    # The 2-mm branch develops a sharper spatial turn and needs a smaller
    # arclength predictor.  Retrying from the identical anchors changes only
    # numerical resolution, not branch identity.
    for candidate_step, candidate_count in (
            (0.0025, 160), (0.001, 280), (0.0005, 520), (0.00025, 960)):
        candidate = _valid(pseudo_arclength_continue(
            model, first, second, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
            step_size=candidate_step, n_steps=candidate_count,
            max_corrector_iterations=36))
        try:
            candidate_cross = _first_tangent_crossing(candidate)
        except RuntimeError:
            continue
        coarse, coarse_cross, coarse_step = (
            candidate, candidate_cross, candidate_step)
        break
    if coarse is None or coarse_cross is None or coarse_step is None:
        raise RuntimeError("adaptive pseudo-arclength failed to pass first fold")
    fine_step = coarse_step / 10.0
    fine = _valid(pseudo_arclength_continue(
        model, coarse[max(0, coarse_cross - 2)].solution,
        coarse[max(1, coarse_cross - 1)].solution,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        step_size=fine_step, n_steps=32, max_corrector_iterations=36))
    fine_cross = _first_tangent_crossing(fine)
    micro_step = fine_step / 10.0
    micro = _valid(pseudo_arclength_continue(
        model, fine[max(0, fine_cross - 2)].solution,
        fine[max(1, fine_cross - 1)].solution,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        step_size=micro_step, n_steps=32, max_corrector_iterations=36))
    micro_cross = _first_tangent_crossing(micro)
    left, right = micro[micro_cross], micro[micro_cross + 1]
    q_tangent, tangent_fraction = _interpolated_zero(
        left.solution.q, right.solution.q, left.tangent_q, right.tangent_q)
    rate_tangent = (left.solution.mean_rate_e_hz + tangent_fraction
                    * (right.solution.mean_rate_e_hz
                       - left.solution.mean_rate_e_hz))

    eigen = []
    for point in micro:
        jacobian = fixed_point_jacobian(
            model, point.solution.rates, q=point.solution.q,
            eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
        value = eigs(
            sparse.csc_matrix(jacobian), k=1, sigma=0.0, which="LM",
            return_eigenvectors=False, tol=1e-10, maxiter=30000)[0]
        eigen.append(value)
    eigen = np.asarray(eigen)
    crossings = np.flatnonzero(
        (eigen[:-1].real > 0.0) & (eigen[1:].real < 0.0))
    if not crossings.size:
        raise RuntimeError("fixed-point zero eigenvalue does not cross at fold")
    eigen_cross = int(crossings[0])
    q_eigen, eigen_fraction = _interpolated_zero(
        micro[eigen_cross].solution.q,
        micro[eigen_cross + 1].solution.q,
        eigen[eigen_cross].real, eigen[eigen_cross + 1].real)
    rate_eigen = (micro[eigen_cross].solution.mean_rate_e_hz + eigen_fraction
                  * (micro[eigen_cross + 1].solution.mean_rate_e_hz
                     - micro[eigen_cross].solution.mean_rate_e_hz))
    return fixed, micro, eigen, {
        "q_from_tangent_zero": q_tangent,
        "q_from_eigenvalue_zero": q_eigen,
        "q_estimate_difference": abs(q_tangent - q_eigen),
        "mean_rate_e_hz_from_tangent_zero": rate_tangent,
        "mean_rate_e_hz_from_eigenvalue_zero": rate_eigen,
        "tangent_bracket": [float(left.tangent_q), float(right.tangent_q)],
        "eigenvalue_bracket": [
            float(eigen[eigen_cross].real),
            float(eigen[eigen_cross + 1].real)],
        "maximum_eigenvalue_imag_abs": float(np.max(np.abs(eigen.imag))),
        "maximum_micro_residual_inf": float(max(
            point.solution.residual_inf for point in micro)),
        "adaptive_arclength_steps": {
            "coarse": coarse_step, "fine": fine_step, "micro": micro_step},
    }


def _conservative_totals(model):
    values = {}
    for name in ("ee", "ei", "ie", "ii"):
        target_counts = (model.count_e if name in ("ee", "ei")
                         else model.count_i)
        values[name] = {
            "first_moment_population_weighted_total": float(
                np.sum(target_counts[:, None] * getattr(model, f"w_{name}"))),
            "second_moment_population_weighted_total": float(
                np.sum(target_counts[:, None] * getattr(model, f"v_{name}"))),
        }
    values["count_e_total"] = int(np.sum(model.count_e))
    values["count_i_total"] = int(np.sum(model.count_i))
    values["threshold_population_weighted_mean_mv"] = float(np.sum(
        model.count_e[:, None] * model.threshold_weights_e
        * model.threshold_nodes_e) / np.sum(model.count_e))
    return values


def main():
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_zm_phase_diagram/deterministic_meanfield")
    parser.add_argument("--base", default=base)
    parser.add_argument("--grids", type=int, nargs="+", default=[10, 15, 20])
    parser.add_argument("--eta-m", type=float, default=0.02)
    parser.add_argument("--tau-m-ms", type=float, default=12.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    started = time.time()
    base_path = Path(args.base).resolve()
    output = (Path(args.out).resolve() if args.out else base_path /
              "patient_zm_grid_convergence.json")
    grids = sorted(set(int(value) for value in args.grids))
    anchors = [0.775, 0.800, 0.840, 0.860, 0.865]
    models, records, arrays = {}, [], {}
    common_fields = {}
    for n_grid in grids:
        path = base_path / f"patient_coarse_ngrid{n_grid}.npz"
        model = load_patient_coarse_model(path)
        models[n_grid] = model
        fixed, micro, eigen, fold = _fold_for_model(
            model, eta_m=float(args.eta_m), tau_m_ms=float(args.tau_m_ms),
            anchors=anchors)
        anchor_rows = []
        common_fields[n_grid] = {}
        for q in anchors[:-1]:
            solution = fixed[float(q)]
            field_hz = 1000.0 * solution.rate_e
            common = remap_square_field(field_hz, n_grid, target_grid=60)
            common_fields[n_grid][q] = common
            key = f"ngrid{n_grid}_q{str(q).replace('.', 'p')}_rate_e_hz"
            arrays[key] = field_hz
            anchor_rows.append({
                "q": q,
                "mean_rate_e_hz": solution.mean_rate_e_hz,
                "spatial_sd_rate_e_hz": float(np.std(field_hz)),
                "residual_inf": float(solution.residual_inf),
                "common_60_grid_array_key": key,
            })
        arrays[f"ngrid{n_grid}_micro_q"] = np.asarray(
            [point.solution.q for point in micro])
        arrays[f"ngrid{n_grid}_micro_rate_e_hz"] = np.asarray(
            [point.solution.mean_rate_e_hz for point in micro])
        arrays[f"ngrid{n_grid}_micro_tangent_q"] = np.asarray(
            [point.tangent_q for point in micro])
        arrays[f"ngrid{n_grid}_micro_eigen_real"] = eigen.real
        records.append({
            "n_grid": n_grid,
            "cell_width_mm": model.sheet_l_mm / n_grid,
            "model": {"path": str(path), "sha256": _sha256(path)},
            "conservative_aggregation": _conservative_totals(model),
            "branch_anchors": anchor_rows,
            "fold": fold,
        })
        print(json.dumps({"n_grid": n_grid, "fold": fold}), flush=True)

    comparisons = []
    for index, first_grid in enumerate(grids):
        for second_grid in grids[index + 1:]:
            for q in anchors[:-1]:
                metrics = compare_branch_fields(
                    common_fields[first_grid][q], common_fields[second_grid][q])
                comparisons.append({
                    "first_n_grid": first_grid,
                    "second_n_grid": second_grid,
                    "q": q,
                    **metrics,
                })

    fold_q = {row["n_grid"]: row["fold"]["q_from_eigenvalue_zero"]
              for row in records}
    fine_pair = grids[-2:]
    fold_span = max(fold_q.values()) - min(fold_q.values())
    fine_pair_difference = abs(fold_q[fine_pair[0]] - fold_q[fine_pair[1]])
    fold_rate = {
        row["n_grid"]: row["fold"]["mean_rate_e_hz_from_eigenvalue_zero"]
        for row in records}
    fold_rate_span = max(fold_rate.values()) - min(fold_rate.values())
    fine_pair_rate_relative_difference = (
        abs(fold_rate[fine_pair[0]] - fold_rate[fine_pair[1]])
        / (0.5 * (fold_rate[fine_pair[0]] + fold_rate[fine_pair[1]])))
    conservative_rows = [row["conservative_aggregation"] for row in records]
    reference = conservative_rows[0]
    conservative_errors = []
    for row in conservative_rows[1:]:
        for name in ("ee", "ei", "ie", "ii"):
            for moment in ("first_moment_population_weighted_total",
                           "second_moment_population_weighted_total"):
                conservative_errors.append(
                    abs(row[name][moment] - reference[name][moment])
                    / max(abs(reference[name][moment]), 1e-15))
        conservative_errors.extend([
            abs(row["count_e_total"] - reference["count_e_total"])
            / reference["count_e_total"],
            abs(row["count_i_total"] - reference["count_i_total"])
            / reference["count_i_total"],
            abs(row["threshold_population_weighted_mean_mv"]
                - reference["threshold_population_weighted_mean_mv"])
            / reference["threshold_population_weighted_mean_mv"],
        ])
    maximum_relative_anchor_difference = max(
        row["relative_mean_rate_difference"] for row in comparisons)
    minimum_anchor_correlation = min(
        row["centered_spatial_correlation"] for row in comparisons)
    maximum_anchor_relative_rms_difference = max(
        row["relative_rms_field_difference"] for row in comparisons)
    finest_pair_correlations = [
        row["centered_spatial_correlation"] for row in comparisons
        if row["first_n_grid"] == fine_pair[0]
        and row["second_n_grid"] == fine_pair[1]]
    thresholds = {
        "fold_q_all_grid_span_max": 0.03,
        "fold_q_finest_pair_difference_max": 0.005,
        "fold_rate_finest_pair_relative_difference_max": 0.15,
        "anchor_relative_mean_rate_difference_max": 0.02,
        "anchor_relative_rms_field_difference_max": 0.02,
        "finest_pair_anchor_centered_spatial_correlation_min": 0.80,
        "conservative_total_relative_error_max": 1e-12,
        "tangent_vs_eigen_fold_q_difference_max": 1e-6,
        "fixed_point_residual_inf_max": 1e-8,
    }
    gates = {
        "conservative_aggregation_invariant": bool(
            max(conservative_errors, default=0.0)
            < thresholds["conservative_total_relative_error_max"]),
        "shared_branch_identity_at_common_anchors": bool(
            maximum_relative_anchor_difference
            < thresholds["anchor_relative_mean_rate_difference_max"]
            and maximum_anchor_relative_rms_difference
            < thresholds["anchor_relative_rms_field_difference_max"]),
        "finest_pair_spatial_pattern_agreement": bool(
            min(finest_pair_correlations)
            > thresholds["finest_pair_anchor_centered_spatial_correlation_min"]),
        "generic_fold_present_on_every_grid": bool(all(
            row["fold"]["q_estimate_difference"]
            < thresholds["tangent_vs_eigen_fold_q_difference_max"]
            and row["fold"]["maximum_eigenvalue_imag_abs"] < 1e-7
            and row["fold"]["maximum_micro_residual_inf"]
            < thresholds["fixed_point_residual_inf_max"]
            for row in records)),
        "fold_location_bounded_across_all_grids": bool(
            fold_span < thresholds["fold_q_all_grid_span_max"]),
        "fold_location_agrees_on_finest_pair": bool(
            fine_pair_difference
            < thresholds["fold_q_finest_pair_difference_max"]),
        "fold_rate_agrees_on_finest_pair": bool(
            fine_pair_rate_relative_difference
            < thresholds["fold_rate_finest_pair_relative_difference_max"]),
    }
    array_path = output.with_suffix(".npz")
    _atomic_npz(array_path, **arrays)
    payload = {
        "status": ("PATIENT_ZM_GRID_CONVERGENCE_AUDITED"
                   if all(gates.values()) else
                   "PATIENT_ZM_GRID_CONVERGENCE_HAS_FAILED_GATES"),
        "scientific_scope": (
            "same realized patient graph conservatively reduced to 2, 1.33 "
            "and 1 mm cells; deterministic frozen-uniform-q fixed points"),
        "claim_boundary": (
            "Shared reduced-model branch identity and fold-location sensitivity "
            "do not establish a thermodynamic or finite-SNN phase transition."),
        "parameters": {"eta_m": float(args.eta_m),
                       "tau_m_ms": float(args.tau_m_ms)},
        "common_branch_anchors_q": anchors[:-1],
        "models": records,
        "branch_identity_comparisons": comparisons,
        "summary": {
            "fold_q_span_all_grids": float(fold_span),
            "fold_q_difference_finest_pair": float(fine_pair_difference),
            "fold_rate_e_hz_span_all_grids": float(fold_rate_span),
            "fold_rate_e_relative_difference_finest_pair": float(
                fine_pair_rate_relative_difference),
            "maximum_anchor_relative_mean_rate_difference": float(
                maximum_relative_anchor_difference),
            "maximum_anchor_relative_rms_field_difference": float(
                maximum_anchor_relative_rms_difference),
            "minimum_anchor_centered_spatial_correlation": float(
                minimum_anchor_correlation),
            "minimum_finest_pair_anchor_centered_spatial_correlation": float(
                min(finest_pair_correlations)),
            "maximum_conservative_total_relative_error": float(
                max(conservative_errors, default=0.0)),
        },
        "thresholds": thresholds,
        "gates": gates,
        "arrays": {"path": str(array_path), "sha256": _sha256(array_path)},
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "summary": payload["summary"], "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
