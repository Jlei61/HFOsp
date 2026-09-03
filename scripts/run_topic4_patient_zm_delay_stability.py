#!/usr/bin/env python3
"""Evaluate delay-aware stability along patient-matched Z/M branches."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_zm_phase_point import _atomic_json  # noqa: E402
from src.topic4_patient_zm_delay import (  # noqa: E402
    delayed_discrete_linear_map,
    dominant_delay_modes,
    load_patient_coarse_delay_operator,
)
from src.topic4_patient_zm_meanfield import (  # noqa: E402
    load_patient_coarse_model,
    solve_fixed_point,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mode_row(multiplier, exponent):
    return {
        "multiplier_real": float(multiplier.real),
        "multiplier_imag": float(multiplier.imag),
        "exponent_real_per_ms": float(exponent.real),
        "exponent_imag_per_ms": float(exponent.imag),
        "frequency_hz": float(abs(exponent.imag) * 1000.0 / (2.0 * np.pi)),
    }


def main():
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_zm_phase_diagram/deterministic_meanfield")
    parser.add_argument("--model", default=f"{base}/patient_coarse_ngrid20.npz")
    parser.add_argument(
        "--delay-operator", default=f"{base}/patient_coarse_delay_ngrid20.npz")
    parser.add_argument("--history-dt-ms", type=float, default=0.5)
    parser.add_argument(
        "--variance-closure",
        choices=("self_consistent_variance", "frozen_variance"),
        default="self_consistent_variance")
    parser.add_argument("--eta-m", type=float, default=0.02)
    parser.add_argument("--tau-m-ms", type=float, default=12.5)
    parser.add_argument("--q", type=float, nargs="+",
                        default=[0.775, 0.800, 0.825, 0.840, 0.850,
                                 0.870, 0.885, 0.890])
    parser.add_argument("--branches", nargs="+", choices=("low", "high"),
                        default=["high", "low"])
    parser.add_argument("--k-eigen", type=int, default=8)
    parser.add_argument("--eigen-tolerance", type=float, default=2e-6)
    parser.add_argument("--continuation-max-step", type=float, default=0.005)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    started = time.time()
    model_path = Path(args.model).resolve()
    delay_path = Path(args.delay_operator).resolve()
    model = load_patient_coarse_model(model_path)
    operator = load_patient_coarse_delay_operator(delay_path)
    tag = str(args.history_dt_ms).replace(".", "p")
    closure_tag = ("selfvar" if args.variance_closure
                   == "self_consistent_variance" else "frozenvar")
    output = (Path(args.out).resolve() if args.out else model_path.parent /
              f"patient_zm_delay_stability_{closure_tag}_branchmatched_"
              f"dt{tag}_ngrid{model.n_grid}.json")
    signature = {
        "model_sha256": _sha256(model_path),
        "delay_operator_sha256": _sha256(delay_path),
        "history_dt_ms": float(args.history_dt_ms),
        "variance_closure": str(args.variance_closure),
        "eta_m": float(args.eta_m),
        "tau_m_ms": float(args.tau_m_ms),
        "k_eigen": int(args.k_eigen),
        "eigen_tolerance": float(args.eigen_tolerance),
        "continuation_max_step": float(args.continuation_max_step),
    }
    rows = []
    if output.exists():
        previous = json.loads(output.read_text())
        if previous.get("signature") != signature:
            raise RuntimeError("existing stability output has a different signature")
        rows = list(previous.get("points", []))
    completed = {(row["branch"], float(row["q"])) for row in rows}

    def checkpoint(status):
        payload = {
            "status": status,
            "scientific_role": (
                "delay-aware stability of patient-matched deterministic "
                "frozen-q fixed-point branches"),
            "signature": signature,
            "model": {"path": str(model_path), "sha256": _sha256(model_path)},
            "delay_operator": {"path": str(delay_path), "sha256": _sha256(delay_path)},
            "controlled_parameter": (
                "q is frozen inhibitory efficacy; M is dynamic in the linear map "
                "and self-consistent at each fixed point"),
            "points": sorted(rows, key=lambda row: (row["branch"], row["q"])),
            "boundary": (
                "Realized recurrent delays are retained after conservative coarse "
                "aggregation and history-grid rebinning. The declared variance "
                f"closure is {args.variance_closure}; stationary OU fluctuations "
                "are not part of this deterministic linearization."),
            "wall_seconds_this_invocation": float(time.time() - started),
        }
        _atomic_json(payload, output)

    for branch in args.branches:
        q_values = sorted(set(float(value) for value in args.q))
        previous_rates = (
            np.r_[np.full(model.n_cells, 0.30), np.full(model.n_cells, 0.32)]
            if branch == "high" else
            np.r_[np.full(model.n_cells, 1e-4), np.full(model.n_cells, 5e-4)])
        # Always enter the named high/low branch from the audited q=0.775
        # anchor; a direct solve near the fold can silently land on a different
        # spatial root.
        previous_q = 0.775
        if q_values[0] < previous_q:
            raise ValueError("all requested q values must be >= branch anchor 0.775")
        for q in q_values:
            count = max(1, int(np.ceil(
                (q - previous_q) / float(args.continuation_max_step))))
            solve_grid = ([q] if q == previous_q else
                          np.linspace(previous_q, q, count + 1)[1:])
            solution = None
            for q_solve in solve_grid:
                solution = solve_fixed_point(
                    model, q=float(q_solve), eta_m=float(args.eta_m),
                    tau_m_slow_ms=float(args.tau_m_ms),
                    initial_rates=previous_rates, maxfev=8000)
                if not solution.converged:
                    raise RuntimeError(
                        f"{branch} continuation failed at q={q_solve}")
                previous_rates = solution.rates
            previous_q = q
            assert solution is not None
            if (branch, q) in completed:
                continue
            matrix, matrix_metadata = delayed_discrete_linear_map(
                model, operator, solution.rates, q=q,
                eta_m=float(args.eta_m), tau_m_slow_ms=float(args.tau_m_ms),
                history_dt_ms=float(args.history_dt_ms),
                variance_closure=str(args.variance_closure))
            point_started = time.time()
            multipliers, exponents = dominant_delay_modes(
                matrix, history_dt_ms=float(args.history_dt_ms),
                k=int(args.k_eigen), tolerance=float(args.eigen_tolerance))
            modes = [_mode_row(multiplier, exponent)
                     for multiplier, exponent in zip(multipliers, exponents)]
            rows.append({
                "branch": branch,
                "q": q,
                "mean_rate_e_hz": float(solution.mean_rate_e_hz),
                "mean_rate_i_hz": float(solution.mean_rate_i_hz),
                "spatial_sd_rate_e_hz": float(1000.0 * np.std(solution.rate_e)),
                "fixed_point_residual_inf": float(solution.residual_inf),
                "maximum_real_exponent_per_ms": float(
                    max(row["exponent_real_per_ms"] for row in modes)),
                "leading_frequency_hz": float(modes[0]["frequency_hz"]),
                "linearly_stable": bool(
                    max(row["exponent_real_per_ms"] for row in modes) < 0.0),
                "modes": modes,
                "matrix": matrix_metadata,
                "eigensolve_wall_seconds": float(time.time() - point_started),
            })
            checkpoint("PATIENT_ZM_DELAY_STABILITY_PARTIAL")
            print(json.dumps({
                "branch": branch, "q": q,
                "rate_e_hz": solution.mean_rate_e_hz,
                "leading": modes[0],
            }), flush=True)
    checkpoint("PATIENT_ZM_DELAY_STABILITY_COMPLETE")
    print(json.dumps({"status": "PATIENT_ZM_DELAY_STABILITY_COMPLETE",
                      "output": str(output), "n_points": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
