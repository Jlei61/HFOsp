#!/usr/bin/env python3
"""Numerically validate the patient-matched frozen-q saddle-node.

This is deliberately narrower than the empirical OU-on phase screen.  It
audits the deterministic 1-mm coarse fixed-point system at eta_m=0.02 using
four independent ingredients of a generic saddle-node:

1. pseudo-arclength reverses in q;
2. two fold-participating roots coexist just below the turn;
3. one isolated real fixed-point Jacobian eigenvalue crosses zero; and
4. the transversality and quadratic normal-form coefficients are non-zero.
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
    fixed_point_q_derivative,
    fixed_point_residual,
    load_patient_coarse_model,
    pseudo_arclength_continue,
    solve_fixed_point,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _valid(points):
    return [point for point in points if point.solution.converged]


def _tangent_crossing(points):
    signs = np.asarray([point.tangent_q for point in points], float)
    indices = np.flatnonzero(signs[:-1] * signs[1:] <= 0.0)
    if not indices.size:
        raise RuntimeError("pseudo-arclength branch has no dq/ds reversal")
    return int(indices[0])


def _near_zero_modes(jacobian):
    matrix = sparse.csc_matrix(jacobian)
    right_values, right_vectors = eigs(
        matrix, k=4, sigma=0.0, which="LM", tol=1e-10, maxiter=30000)
    left_values, left_vectors = eigs(
        matrix.T, k=4, sigma=0.0, which="LM", tol=1e-10, maxiter=30000)
    right_order = np.argsort(np.abs(right_values))
    value = right_values[right_order[0]]
    right = right_vectors[:, right_order[0]]
    left_index = int(np.argmin(np.abs(left_values - value)))
    left = left_vectors[:, left_index]
    if abs(value.imag) > 1e-7:
        raise RuntimeError(f"nearest fold eigenvalue is not real: {value}")
    right = np.real_if_close(right, tol=1000).real
    left = np.real_if_close(left, tol=1000).real
    right /= np.linalg.norm(right)
    pairing = float(np.dot(left, right))
    if abs(pairing) < 1e-10:
        raise RuntimeError("left/right null vectors are numerically orthogonal")
    left /= pairing
    ordered_values = right_values[right_order]
    return value, right, left, ordered_values


def _interpolate_zero(x, y, values):
    left, right = float(values[x]), float(values[y])
    fraction = left / (left - right)
    return float(fraction)


def _solution_summary(solution):
    return {
        "q": float(solution.q),
        "mean_rate_e_hz": float(solution.mean_rate_e_hz),
        "mean_rate_i_hz": float(solution.mean_rate_i_hz),
        "residual_inf": float(solution.residual_inf),
        "converged": bool(solution.converged),
        "physical": bool(solution.physical),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                 "data_driven_zm_phase_diagram/deterministic_meanfield/"
                 "patient_coarse_ngrid20.npz"))
    parser.add_argument("--eta-m", type=float, default=0.02)
    parser.add_argument("--tau-m-ms", type=float, default=12.5)
    parser.add_argument("--probe-q", type=float, default=0.890700)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.time()
    model_path = Path(args.model).resolve()
    model = load_patient_coarse_model(model_path)
    output = (Path(args.out).resolve() if args.out else model_path.parent /
              f"patient_zm_saddle_node_validation_ngrid{model.n_grid}.json")
    eta_m = float(args.eta_m)
    tau_m_ms = float(args.tau_m_ms)

    initial = np.r_[np.full(model.n_cells, 0.30),
                    np.full(model.n_cells, 0.32)]
    first = solve_fixed_point(
        model, q=0.885, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        initial_rates=initial, maxfev=5000)
    second = solve_fixed_point(
        model, q=0.8875, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        initial_rates=first.rates, maxfev=5000)
    if not first.converged or not second.converged:
        raise RuntimeError("failed to seed the high-rate fixed-point branch")

    coarse = _valid(pseudo_arclength_continue(
        model, first, second, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        step_size=0.0025, n_steps=24, max_corrector_iterations=24))
    coarse_cross = _tangent_crossing(coarse)
    fine = _valid(pseudo_arclength_continue(
        model, coarse[max(0, coarse_cross - 2)].solution,
        coarse[max(1, coarse_cross - 1)].solution,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        step_size=0.00025, n_steps=24, max_corrector_iterations=24))
    fine_cross = _tangent_crossing(fine)
    micro = _valid(pseudo_arclength_continue(
        model, fine[max(0, fine_cross - 2)].solution,
        fine[max(1, fine_cross - 1)].solution,
        eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        step_size=0.000025, n_steps=24, max_corrector_iterations=24))
    micro_cross = _tangent_crossing(micro)

    eigenvalues = []
    for point in micro:
        jacobian = fixed_point_jacobian(
            model, point.solution.rates, q=point.solution.q,
            eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
        eigenvalues.append(eigs(
            sparse.csc_matrix(jacobian), k=1, sigma=0.0, which="LM",
            return_eigenvectors=False, tol=1e-10, maxiter=30000)[0])
    eigen_real = np.asarray([value.real for value in eigenvalues], float)
    eigen_imag = np.asarray([value.imag for value in eigenvalues], float)
    eigen_crossings = np.flatnonzero(eigen_real[:-1] * eigen_real[1:] <= 0.0)
    if not eigen_crossings.size:
        raise RuntimeError("fixed-point Jacobian has no zero-eigenvalue bracket")
    eigen_cross = int(eigen_crossings[0])

    tangent_fraction = _interpolate_zero(
        micro_cross, micro_cross + 1,
        [point.tangent_q for point in micro])
    eigen_fraction = _interpolate_zero(
        eigen_cross, eigen_cross + 1, eigen_real)
    tangent_left, tangent_right = micro[micro_cross], micro[micro_cross + 1]
    eigen_left, eigen_right = micro[eigen_cross], micro[eigen_cross + 1]
    q_tangent = tangent_left.solution.q + tangent_fraction * (
        tangent_right.solution.q - tangent_left.solution.q)
    q_eigen = eigen_left.solution.q + eigen_fraction * (
        eigen_right.solution.q - eigen_left.solution.q)
    rate_eigen = eigen_left.solution.mean_rate_e_hz + eigen_fraction * (
        eigen_right.solution.mean_rate_e_hz
        - eigen_left.solution.mean_rate_e_hz)

    closest_index = int(np.argmin(np.abs(eigen_real)))
    closest = micro[closest_index].solution
    jacobian = fixed_point_jacobian(
        model, closest.rates, q=closest.q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_ms)
    closest_value, right_null, left_null, near_values = _near_zero_modes(jacobian)
    q_derivative = fixed_point_q_derivative(
        model, closest.rates, q=closest.q, eta_m=eta_m,
        tau_m_slow_ms=tau_m_ms)
    transversality = float(np.dot(left_null, q_derivative))
    quadratic_by_step = []
    for step in (0.00025, 0.0005, 0.001, 0.002, 0.004):
        center = fixed_point_residual(
            model, closest.rates, q=closest.q, eta_m=eta_m,
            tau_m_slow_ms=tau_m_ms)
        plus = fixed_point_residual(
            model, closest.rates + step * right_null, q=closest.q,
            eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
        minus = fixed_point_residual(
            model, closest.rates - step * right_null, q=closest.q,
            eta_m=eta_m, tau_m_slow_ms=tau_m_ms)
        directional_second = (plus - 2.0 * center + minus) / step ** 2
        coefficient = 0.5 * float(np.dot(left_null, directional_second))
        quadratic_by_step.append({"step": step, "coefficient": coefficient})
    quadratic_values = np.asarray(
        [row["coefficient"] for row in quadratic_by_step], float)
    quadratic = float(np.median(quadratic_values))
    quadratic_relative_spread = float(
        np.ptp(quadratic_values) / max(abs(quadratic), 1e-15))

    # At a q just below the maximum, warm starts on opposite sides of the
    # arclength turn must converge to two distinct fold-participating roots.
    probe_q = float(args.probe_q)
    if not probe_q < min(q_tangent, q_eigen):
        raise ValueError("--probe-q must be just below the estimated fold")
    upper_seed = min(
        [point for point in coarse
         if point.solution.mean_rate_e_hz > rate_eigen],
        key=lambda point: abs(point.solution.q - probe_q))
    lower_seed = min(
        [point for point in coarse
         if point.solution.mean_rate_e_hz < rate_eigen],
        key=lambda point: abs(point.solution.q - probe_q))
    probe_upper = solve_fixed_point(
        model, q=probe_q, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        initial_rates=upper_seed.solution.rates, maxfev=10000)
    probe_lower = solve_fixed_point(
        model, q=probe_q, eta_m=eta_m, tau_m_slow_ms=tau_m_ms,
        initial_rates=lower_seed.solution.rates, maxfev=10000)
    if not probe_upper.converged or not probe_lower.converged:
        raise RuntimeError("failed to resolve both fold-participating probe roots")
    probe_separation_hz = abs(
        probe_upper.mean_rate_e_hz - probe_lower.mean_rate_e_hz)

    second_mode_magnitude = float(np.sort(np.abs(near_values))[1])
    closest_mode_magnitude = float(abs(closest_value))
    spectral_gap_ratio = second_mode_magnitude / max(
        closest_mode_magnitude, np.finfo(float).tiny)
    thresholds = {
        "residual_inf_max": 1e-9,
        "closest_eigenvalue_abs_max": 1e-3,
        "second_to_first_eigenvalue_ratio_min": 100.0,
        "normal_form_abs_min": 1e-6,
        "quadratic_relative_spread_max": 0.01,
        "probe_root_separation_hz_min": 1.0,
    }
    gates = {
        "pseudo_arclength_turn": bool(
            micro[micro_cross].tangent_q > 0.0
            and micro[micro_cross + 1].tangent_q < 0.0),
        "real_eigenvalue_crosses_zero": bool(
            eigen_real[eigen_cross] > 0.0
            and eigen_real[eigen_cross + 1] < 0.0
            and np.max(np.abs(eigen_imag)) < 1e-7),
        "fixed_point_residual": bool(
            closest.residual_inf < thresholds["residual_inf_max"]),
        "simple_isolated_zero_mode": bool(
            closest_mode_magnitude
            < thresholds["closest_eigenvalue_abs_max"]
            and spectral_gap_ratio
            > thresholds["second_to_first_eigenvalue_ratio_min"]),
        "transversality_nonzero": bool(
            abs(transversality) > thresholds["normal_form_abs_min"]),
        "quadratic_nonzero_and_converged": bool(
            abs(quadratic) > thresholds["normal_form_abs_min"]
            and quadratic_relative_spread
            < thresholds["quadratic_relative_spread_max"]),
        "two_fold_roots_below_turn": bool(
            probe_separation_hz
            > thresholds["probe_root_separation_hz_min"]),
    }

    arrays = {
        "micro_q": np.asarray([point.solution.q for point in micro]),
        "micro_rate_e_hz": np.asarray(
            [point.solution.mean_rate_e_hz for point in micro]),
        "micro_tangent_q": np.asarray([point.tangent_q for point in micro]),
        "micro_eigen_real": eigen_real,
        "micro_eigen_imag": eigen_imag,
        "closest_right_null": right_null,
        "closest_left_null": left_null,
        "probe_upper_rates": probe_upper.rates,
        "probe_lower_rates": probe_lower.rates,
    }
    npz_path = output.with_suffix(".npz")
    _atomic_npz(npz_path, **arrays)
    payload = {
        "status": ("GENERIC_SADDLE_NODE_NUMERICALLY_VALIDATED"
                   if all(gates.values()) else "SADDLE_NODE_AUDIT_FAILED"),
        "scientific_scope": (
            "patient-matched 1-mm coarse deterministic frozen-q fast subsystem; "
            "not a finite-size or thermodynamic phase-transition claim"),
        "model": {
            "path": str(model_path), "sha256": _sha256(model_path),
            "n_grid": int(model.n_grid),
            "cell_width_mm": float(model.sheet_l_mm / model.n_grid),
        },
        "parameters": {"eta_m": eta_m, "tau_m_ms": tau_m_ms},
        "fold": {
            "q_from_tangent_zero": float(q_tangent),
            "q_from_eigenvalue_zero": float(q_eigen),
            "q_estimate_difference": float(abs(q_tangent - q_eigen)),
            "mean_rate_e_hz_at_eigenvalue_zero": float(rate_eigen),
            "tangent_bracket": [
                float(micro[micro_cross].tangent_q),
                float(micro[micro_cross + 1].tangent_q)],
            "eigenvalue_bracket": [
                float(eigen_real[eigen_cross]),
                float(eigen_real[eigen_cross + 1])],
        },
        "closest_corrected_fixed_point": {
            **_solution_summary(closest),
            "nearest_eigenvalue": {
                "real": float(closest_value.real),
                "imag": float(closest_value.imag)},
            "second_mode_magnitude": second_mode_magnitude,
            "spectral_gap_ratio": spectral_gap_ratio,
        },
        "normal_form": {
            "left_right_pairing_after_normalization": float(
                np.dot(left_null, right_null)),
            "transversality_wT_Fq": transversality,
            "transversality_abs": float(abs(transversality)),
            "quadratic_half_wT_Fxx_vv": quadratic,
            "quadratic_abs": float(abs(quadratic)),
            "quadratic_by_directional_step": quadratic_by_step,
            "quadratic_relative_spread": quadratic_relative_spread,
        },
        "two_root_probe": {
            "q": probe_q,
            "upper_root": _solution_summary(probe_upper),
            "returned_root": _solution_summary(probe_lower),
            "rate_separation_hz": float(probe_separation_hz),
            "note": (
                "These are the two roots participating in the fold; a separate "
                "near-silent low root also exists and is not counted here."),
        },
        "numerical_thresholds": thresholds,
        "gates": gates,
        "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        "interpretation": (
            "The simultaneous arclength turn, two-root coalescence, simple real "
            "zero mode, non-zero wT Fq, and non-zero quadratic coefficient satisfy "
            "the numerical conditions for a generic saddle-node in this reduction."),
        "boundary": (
            "q is a frozen fast-subsystem coordinate, not q_min of the dynamic Z/M "
            "system. OU contributes its zero stationary mean here; stochastic SNN "
            "onset and grid convergence remain separate questions."),
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "fold": payload["fold"], "normal_form": payload["normal_form"],
        "two_root_probe": payload["two_root_probe"], "gates": gates,
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
