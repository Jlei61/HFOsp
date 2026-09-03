#!/usr/bin/env python3
"""Continue patient-matched Z/M mean-field branches and audit fold candidates."""
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
    dynamic_jacobian,
    fixed_point_jacobian,
    load_patient_coarse_model,
    pseudo_arclength_continue,
    solve_fixed_point,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _eta_key(value):
    return f"eta_{float(value):.5f}".replace("-", "m").replace(".", "p")


def _solution_row(solution, *, branch, tangent_q=None,
                  closest_zero_eigenvalue=None):
    rate_e = solution.rate_e
    return {
        "branch": branch,
        "q": float(solution.q),
        "eta_m": float(solution.eta_m),
        "converged": bool(solution.converged),
        "physical": bool(solution.physical),
        "residual_inf": float(solution.residual_inf),
        "mean_rate_e_hz": solution.mean_rate_e_hz,
        "mean_rate_i_hz": solution.mean_rate_i_hz,
        "spatial_sd_rate_e_hz": 1000.0 * float(np.std(rate_e)),
        "spatial_min_rate_e_hz": 1000.0 * float(np.min(rate_e)),
        "spatial_max_rate_e_hz": 1000.0 * float(np.max(rate_e)),
        "tangent_q": None if tangent_q is None else float(tangent_q),
        "fixed_point_eigenvalue_near_zero": (
            None if closest_zero_eigenvalue is None else {
                "real": float(np.real(closest_zero_eigenvalue)),
                "imag": float(np.imag(closest_zero_eigenvalue)),
            }),
    }


def _near_zero_eigenvalue(model, solution):
    jacobian = fixed_point_jacobian(
        model, solution.rates, q=solution.q, eta_m=solution.eta_m,
        tau_m_slow_ms=solution.tau_m_slow_ms)
    return eigs(
        sparse.csc_matrix(jacobian), k=1, sigma=0.0, which="LM",
        return_eigenvectors=False, tol=1e-9, maxiter=20000)[0]


def _leading_dynamic_eigenvalues(model, solution, *, k=6):
    jacobian = dynamic_jacobian(
        model, solution.rates, q=solution.q, eta_m=solution.eta_m,
        tau_m_slow_ms=solution.tau_m_slow_ms)
    values = eigs(
        jacobian, k=int(k), which="LR", return_eigenvectors=False,
        tol=1e-8, maxiter=30000)
    return sorted(values, key=lambda value: value.real, reverse=True)


def _interpolate_zero(left, right):
    lambda_left = left["fixed_point_eigenvalue_near_zero"]["real"]
    lambda_right = right["fixed_point_eigenvalue_near_zero"]["real"]
    fraction = lambda_left / (lambda_left - lambda_right)
    return {
        "q": float(left["q"] + fraction * (right["q"] - left["q"])),
        "mean_rate_e_hz": float(
            left["mean_rate_e_hz"] + fraction
            * (right["mean_rate_e_hz"] - left["mean_rate_e_hz"])),
        "interpolation_fraction": float(fraction),
        "eigenvalue_bracket": [float(lambda_left), float(lambda_right)],
        "q_bracket": [float(left["q"]), float(right["q"])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=("/data/hfosp_topic4_fig45_artifacts/fig5/"
                 "data_driven_zm_phase_diagram/deterministic_meanfield/"
                 "patient_coarse_ngrid20.npz"))
    parser.add_argument("--eta-m", type=float, nargs="+", default=[0.0, 0.02, 0.04, 0.08])
    parser.add_argument("--tau-m-ms", type=float, default=12.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if any(value < 0.0 for value in args.eta_m):
        raise SystemExit("--eta-m values must be non-negative")

    started = time.time()
    model_path = Path(args.model).resolve()
    model = load_patient_coarse_model(model_path)
    output = (Path(args.out).resolve() if args.out else model_path.parent /
              f"patient_zm_bifurcation_ngrid{model.n_grid}.json")
    arrays = {}
    families = []
    for eta_m in args.eta_m:
        eta_m = float(eta_m)
        key = _eta_key(eta_m)
        initial = np.r_[np.full(model.n_cells, 0.30),
                        np.full(model.n_cells, 0.32)]
        first = solve_fixed_point(
            model, q=0.885, eta_m=eta_m,
            tau_m_slow_ms=args.tau_m_ms, initial_rates=initial, maxfev=5000)
        second = solve_fixed_point(
            model, q=0.8875, eta_m=eta_m,
            tau_m_slow_ms=args.tau_m_ms, initial_rates=first.rates, maxfev=5000)
        if not first.converged or not second.converged:
            raise RuntimeError(f"failed to seed high branch for eta_m={eta_m}")

        regular = []
        previous = np.array(initial, copy=True)
        for q_regular in np.arange(0.775, 0.8876, 0.005):
            solution = solve_fixed_point(
                model, q=float(q_regular), eta_m=eta_m,
                tau_m_slow_ms=args.tau_m_ms, initial_rates=previous,
                maxfev=5000)
            if not solution.converged:
                raise RuntimeError(
                    f"regular high branch failed at eta_m={eta_m}, q={q_regular}")
            regular.append(solution)
            previous = solution.rates

        arc = pseudo_arclength_continue(
            model, first, second, eta_m=eta_m,
            tau_m_slow_ms=args.tau_m_ms, step_size=0.005,
            n_steps=42, max_corrector_iterations=20)
        valid_arc = [point for point in arc if point.solution.converged]
        signs = np.asarray([point.tangent_q for point in valid_arc])
        crossings = np.flatnonzero(signs[:-1] * signs[1:] <= 0.0)
        if not crossings.size:
            raise RuntimeError(f"no arclength fold bracket for eta_m={eta_m}")
        crossing = int(crossings[0] + 1)
        seed_left = valid_arc[max(0, crossing - 2)].solution
        seed_right = valid_arc[max(1, crossing - 1)].solution
        fine = pseudo_arclength_continue(
            model, seed_left, seed_right, eta_m=eta_m,
            tau_m_slow_ms=args.tau_m_ms, step_size=0.00025,
            n_steps=24, max_corrector_iterations=20)
        valid_fine = [point for point in fine if point.solution.converged]
        fine_rows = []
        for point in valid_fine:
            eigenvalue = _near_zero_eigenvalue(model, point.solution)
            fine_rows.append(_solution_row(
                point.solution, branch="fold_refinement",
                tangent_q=point.tangent_q,
                closest_zero_eigenvalue=eigenvalue))
        eigen_reals = np.asarray([
            row["fixed_point_eigenvalue_near_zero"]["real"]
            for row in fine_rows])
        eigen_crossings = np.flatnonzero(eigen_reals[:-1] * eigen_reals[1:] <= 0.0)
        if not eigen_crossings.size:
            raise RuntimeError(f"no Jacobian zero-mode bracket for eta_m={eta_m}")
        index = int(eigen_crossings[0])
        fold = _interpolate_zero(fine_rows[index], fine_rows[index + 1])

        # The low root is an anchor, not a claim that the OU-driven SNN has a
        # near-silent low state. OU is deliberately absent from fixed points.
        low = solve_fixed_point(
            model, q=fold["q"], eta_m=eta_m,
            tau_m_slow_ms=args.tau_m_ms,
            initial_rates=np.r_[np.full(model.n_cells, 1e-4),
                                np.full(model.n_cells, 5e-4)], maxfev=5000)
        if not low.converged:
            raise RuntimeError(f"low anchor failed for eta_m={eta_m}")

        arc_rows = [_solution_row(
            point.solution, branch="pseudo_arclength",
            tangent_q=point.tangent_q) for point in valid_arc]
        family = {
            "eta_m": eta_m,
            "tau_m_ms": float(args.tau_m_ms),
            "fold": fold,
            "low_anchor_at_fold_q": _solution_row(low, branch="low_anchor"),
            "regular_high_branch": [
                _solution_row(solution, branch="regular_high_branch")
                for solution in regular],
            "arc": arc_rows,
            "fold_refinement": fine_rows,
        }
        families.append(family)
        arrays[f"{key}__arc_q"] = np.asarray([row["q"] for row in arc_rows])
        arrays[f"{key}__arc_rate_e_hz"] = np.asarray(
            [row["mean_rate_e_hz"] for row in arc_rows])
        arrays[f"{key}__arc_spatial_sd_e_hz"] = np.asarray(
            [row["spatial_sd_rate_e_hz"] for row in arc_rows])
        arrays[f"{key}__regular_q"] = np.asarray(
            [solution.q for solution in regular])
        arrays[f"{key}__regular_rate_e_hz"] = np.asarray(
            [solution.mean_rate_e_hz for solution in regular])
        arrays[f"{key}__fold_q"] = np.asarray([row["q"] for row in fine_rows])
        arrays[f"{key}__fold_rate_e_hz"] = np.asarray(
            [row["mean_rate_e_hz"] for row in fine_rows])
        arrays[f"{key}__fold_eigen_real"] = eigen_reals

    # Stability is a separately labelled zero-delay/frozen-variance diagnostic.
    stability = []
    for eta_m in [value for value in args.eta_m if value in (0.0, 0.02)]:
        previous = np.r_[np.full(model.n_cells, 0.30),
                         np.full(model.n_cells, 0.32)]
        for q in (0.775, 0.800, 0.825, 0.835, 0.840, 0.845, 0.850):
            solution = solve_fixed_point(
                model, q=q, eta_m=eta_m, tau_m_slow_ms=args.tau_m_ms,
                initial_rates=previous, maxfev=5000)
            if not solution.converged:
                raise RuntimeError(f"stability branch solve failed at {eta_m=}, {q=}")
            previous = solution.rates
            values = _leading_dynamic_eigenvalues(model, solution)
            stability.append({
                "branch": "high", "eta_m": float(eta_m), "q": float(q),
                "mean_rate_e_hz": solution.mean_rate_e_hz,
                "leading_eigenvalues_per_ms": [
                    {"real": float(value.real), "imag": float(value.imag)}
                    for value in values],
                "maximum_real_part_per_ms": float(max(value.real for value in values)),
                "convention": "zero_delay_synaptic_dynamic_jacobian_with_operating_variance_frozen",
            })
            low = solve_fixed_point(
                model, q=q, eta_m=eta_m, tau_m_slow_ms=args.tau_m_ms,
                initial_rates=np.r_[np.full(model.n_cells, 1e-4),
                                    np.full(model.n_cells, 5e-4)], maxfev=5000)
            if not low.converged:
                raise RuntimeError(
                    f"low stability branch solve failed at {eta_m=}, {q=}")
            low_values = _leading_dynamic_eigenvalues(model, low)
            stability.append({
                "branch": "low", "eta_m": float(eta_m), "q": float(q),
                "mean_rate_e_hz": low.mean_rate_e_hz,
                "leading_eigenvalues_per_ms": [
                    {"real": float(value.real), "imag": float(value.imag)}
                    for value in low_values],
                "maximum_real_part_per_ms": float(
                    max(value.real for value in low_values)),
                "convention": "zero_delay_synaptic_dynamic_jacobian_with_operating_variance_frozen",
            })

    npz_path = output.with_suffix(".npz")
    _atomic_npz(npz_path, **arrays)
    payload = {
        "status": "PATIENT_MATCHED_ZM_BIFURCATION_CONTINUATION_COMPLETE",
        "scientific_role": "deterministic_skeleton_for_the_stationary_OU_driven_SNN_screen",
        "model_archive": {
            "path": str(model_path), "sha256": _sha256(model_path),
            "n_grid": model.n_grid,
            "cell_width_mm": model.sheet_l_mm / model.n_grid,
        },
        "controlled_parameter": (
            "q is frozen inhibitory efficacy in the fast subsystem; it is not q_min "
            "of the full dynamic slow system"),
        "ou_boundary": (
            "The fixed-point drift uses the stationary OU mean (zero). The actual SNN low "
            "state is noise-supported, so its 30-80 Hz rate must not be compared to the "
            "near-silent deterministic low root. OU-on evidence remains empirical."),
        "bifurcation_claim": (
            "A fold is assigned only when pseudo-arclength dq/ds reverses and the same "
            "fixed-point Jacobian has a real eigenvalue crossing zero."),
        "families": families,
        "stability_sensitivity": stability,
        "stability_boundary": (
            "The reported dynamic eigenvalues are zero-delay and freeze operating variance; "
            "they are a Hopf locator sensitivity, not a delay-aware stability theorem."),
        "arrays": {"path": str(npz_path), "sha256": _sha256(npz_path)},
        "wall_seconds": float(time.time() - started),
    }
    _atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "folds": [{"eta_m": row["eta_m"], **row["fold"]} for row in families],
        "wall_seconds": payload["wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
