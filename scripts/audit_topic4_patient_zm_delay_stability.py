#!/usr/bin/env python3
"""Close the delay-aware stability contract for the patient Z/M fold."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_zm_phase_point import _atomic_json  # noqa: E402
from src.topic4_patient_zm_delay import (  # noqa: E402
    delayed_discrete_linear_map,
    load_patient_coarse_delay_operator,
    stationary_delay_mode_vector,
)
from src.topic4_patient_zm_meanfield import load_patient_coarse_model  # noqa: E402


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows_by_key(payload):
    return {(row["branch"], float(row["q"])): row
            for row in payload["points"]}


def linear_dt_extrapolation(dt_ms, values):
    dt_ms = np.asarray(dt_ms, float)
    values = np.asarray(values, float)
    if dt_ms.shape != values.shape or dt_ms.size < 2:
        raise ValueError("dt extrapolation requires aligned multi-point vectors")
    slope, intercept = np.polyfit(dt_ms, values, 1)
    prediction = slope * dt_ms + intercept
    residual = values - prediction
    return {
        "intercept_at_dt0": float(intercept),
        "slope_per_ms": float(slope),
        "maximum_absolute_fit_residual": float(np.max(np.abs(residual))),
    }


def main():
    parser = argparse.ArgumentParser()
    base = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
            "data_driven_zm_phase_diagram/deterministic_meanfield")
    parser.add_argument("--base", default=base)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    base = Path(args.base).resolve()
    model_path = base / "patient_coarse_ngrid20.npz"
    operator_path = base / "patient_coarse_delay_ngrid20.npz"
    operator_json_path = operator_path.with_suffix(".json")
    saddle_path = base / "patient_zm_saddle_node_validation_ngrid20.json"
    saddle_npz_path = saddle_path.with_suffix(".npz")
    self_paths = {
        0.5: base / "patient_zm_delay_stability_selfvar_branchmatched_dt0p5_ngrid20.json",
        0.25: base / "patient_zm_delay_stability_selfvar_branchmatched_dt0p25_ngrid20.json",
        0.1: base / "patient_zm_delay_stability_selfvar_branchmatched_dt0p1_ngrid20.json",
    }
    frozen_path = base / "patient_zm_delay_stability_frozenvar_branchmatched_dt0p5_ngrid20.json"
    output = (Path(args.out).resolve() if args.out else base /
              "patient_zm_delay_stability_audit.json")
    model = load_patient_coarse_model(model_path)
    operator = load_patient_coarse_delay_operator(operator_path)
    operator_audit = json.loads(operator_json_path.read_text())
    saddle = json.loads(saddle_path.read_text())
    self_payload = {dt: json.loads(path.read_text())
                    for dt, path in self_paths.items()}
    frozen = json.loads(frozen_path.read_text())
    model_hash, operator_hash = _sha256(model_path), _sha256(operator_path)
    for dt, payload in self_payload.items():
        signature = payload["signature"]
        if signature["model_sha256"] != model_hash:
            raise RuntimeError(f"dt={dt} stability model hash is stale")
        if signature["delay_operator_sha256"] != operator_hash:
            raise RuntimeError(f"dt={dt} stability delay operator hash is stale")
        if signature["variance_closure"] != "self_consistent_variance":
            raise RuntimeError(f"dt={dt} is not the primary variance closure")
        if payload["status"] != "PATIENT_ZM_DELAY_STABILITY_COMPLETE":
            raise RuntimeError(f"dt={dt} stability artifact is incomplete")
    if (frozen["signature"]["delay_operator_sha256"] != operator_hash
            or frozen["signature"]["variance_closure"] != "frozen_variance"):
        raise RuntimeError("frozen-variance sensitivity artifact is stale")

    full = self_payload[0.5]["points"]
    high_full = [row for row in full if row["branch"] == "high"]
    low_full = [row for row in full if row["branch"] == "low"]
    common_q = [0.775, 0.800]
    convergence = []
    for q in common_q:
        rows = []
        for dt in sorted(self_payload, reverse=True):
            row = _rows_by_key(self_payload[dt])[("high", q)]
            rows.append({
                "history_dt_ms": dt,
                "maximum_real_exponent_per_ms": row[
                    "maximum_real_exponent_per_ms"],
                "leading_frequency_hz": row["leading_frequency_hz"],
            })
        growth_fit = linear_dt_extrapolation(
            [row["history_dt_ms"] for row in rows],
            [row["maximum_real_exponent_per_ms"] for row in rows])
        frequency_fit = linear_dt_extrapolation(
            [row["history_dt_ms"] for row in rows],
            [row["leading_frequency_hz"] for row in rows])
        convergence.append({
            "q": q,
            "points": rows,
            "growth_extrapolation": growth_fit,
            "frequency_extrapolation": frequency_fit,
        })

    with np.load(saddle_npz_path, allow_pickle=False) as archive:
        closest_rates = np.asarray(archive["closest_rates"], float)
        right_null = np.asarray(archive["closest_right_null"], float)
    closest = saddle["closest_corrected_fixed_point"]
    unit_mode = []
    for dt in (0.5, 0.25, 0.1):
        lifted = stationary_delay_mode_vector(
            model, operator, right_null,
            tau_m_slow_ms=float(saddle["parameters"]["tau_m_ms"]),
            history_dt_ms=dt)
        closure_rows = []
        for closure in ("self_consistent_variance", "frozen_variance"):
            matrix, metadata = delayed_discrete_linear_map(
                model, operator, closest_rates, q=float(closest["q"]),
                eta_m=float(saddle["parameters"]["eta_m"]),
                tau_m_slow_ms=float(saddle["parameters"]["tau_m_ms"]),
                history_dt_ms=dt, variance_closure=closure)
            residual = matrix @ lifted - lifted
            closure_rows.append({
                "variance_closure": closure,
                "relative_residual_inf": float(
                    np.linalg.norm(residual, ord=np.inf)
                    / np.linalg.norm(lifted, ord=np.inf)),
                "auxiliary_residual_inf": float(np.linalg.norm(
                    residual[2 * model.n_cells:], ord=np.inf)),
                "matrix": metadata,
            })
        unit_mode.append({"history_dt_ms": dt, "closures": closure_rows})

    self_unit_residual = max(
        row["relative_residual_inf"]
        for dt_row in unit_mode for row in dt_row["closures"]
        if row["variance_closure"] == "self_consistent_variance")
    frozen_unit_residual = min(
        row["relative_residual_inf"]
        for dt_row in unit_mode for row in dt_row["closures"]
        if row["variance_closure"] == "frozen_variance")
    thresholds = {
        "unit_mode_relative_residual_max": 1e-5,
        "unit_mode_auxiliary_residual_max": 1e-10,
        "minimum_positive_dt0_growth_extrapolation_per_ms": 0.0,
        "far_from_fold_frequency_min_hz": 20.0,
        "far_from_fold_frequency_max_hz": 35.0,
    }
    conservation = operator_audit["pathway_moment_conservation"]
    gates = {
        "delay_operator_conserves_first_and_second_moments": all(
            row["first_moment_allclose_atol_1e-10"]
            and row["second_moment_allclose_atol_1e-10"]
            for row in conservation.values()),
        "full_high_branch_is_delay_unstable": bool(
            len(high_full) == 8 and all(
                not row["linearly_stable"] for row in high_full)),
        "full_low_branch_is_delay_stable": bool(
            len(low_full) == 8 and all(
                row["linearly_stable"] for row in low_full)),
        "high_instability_sign_persists_to_native_delay_grid": all(
            all(point["maximum_real_exponent_per_ms"] > 0.0
                for point in row["points"])
            and row["growth_extrapolation"]["intercept_at_dt0"]
            > thresholds["minimum_positive_dt0_growth_extrapolation_per_ms"]
            for row in convergence),
        "far_from_fold_mode_is_beta_range": all(
            thresholds["far_from_fold_frequency_min_hz"]
            < point["leading_frequency_hz"]
            < thresholds["far_from_fold_frequency_max_hz"]
            for row in convergence for point in row["points"]),
        "saddle_null_lifts_to_delay_unit_mode": bool(
            self_unit_residual
            < thresholds["unit_mode_relative_residual_max"]
            and all(
                row["auxiliary_residual_inf"]
                < thresholds["unit_mode_auxiliary_residual_max"]
                for dt_row in unit_mode for row in dt_row["closures"]
                if row["variance_closure"] == "self_consistent_variance")),
        "frozen_variance_is_not_used_for_fold_zero_mode": bool(
            frozen_unit_residual > self_unit_residual),
    }
    payload = {
        "status": ("PATIENT_ZM_DELAY_STABILITY_AUDITED"
                   if all(gates.values()) else
                   "PATIENT_ZM_DELAY_STABILITY_HAS_FAILED_GATES"),
        "scientific_scope": (
            "linear stability of patient-matched 1-mm deterministic frozen-q "
            "fixed points with realized recurrent conduction delays"),
        "primary_closure": (
            "self_consistent stationary-diffusion variance response; its "
            "zero-frequency gain matches the saddle-node fixed-point Jacobian"),
        "sensitivity_closure": (
            "frozen operating variance, retained only to compare with the older "
            "M3B dynamic convention"),
        "source": {
            "model": {"path": str(model_path), "sha256": model_hash},
            "delay_operator": {"path": str(operator_path), "sha256": operator_hash},
            "delay_operator_audit": {
                "path": str(operator_json_path),
                "sha256": _sha256(operator_json_path)},
            "saddle_node": {"path": str(saddle_path),
                            "sha256": _sha256(saddle_path)},
            "stability": [
                {"history_dt_ms": dt, "path": str(path),
                 "sha256": _sha256(path)}
                for dt, path in self_paths.items()],
            "frozen_variance_sensitivity": {
                "path": str(frozen_path), "sha256": _sha256(frozen_path)},
        },
        "full_dt0p5_branch": {
            "n_high": len(high_full), "n_low": len(low_full),
            "high_q_range": [min(row["q"] for row in high_full),
                             max(row["q"] for row in high_full)],
            "low_q_range": [min(row["q"] for row in low_full),
                            max(row["q"] for row in low_full)],
            "high_growth_real_per_ms_range": [
                min(row["maximum_real_exponent_per_ms"] for row in high_full),
                max(row["maximum_real_exponent_per_ms"] for row in high_full)],
            "low_growth_real_per_ms_range": [
                min(row["maximum_real_exponent_per_ms"] for row in low_full),
                max(row["maximum_real_exponent_per_ms"] for row in low_full)],
        },
        "history_grid_convergence": convergence,
        "saddle_unit_mode": {
            "closest_fixed_point_q": closest["q"],
            "fixed_point_nearest_eigenvalue": closest["nearest_eigenvalue"],
            "by_history_grid": unit_mode,
            "maximum_self_consistent_relative_residual_inf": self_unit_residual,
            "minimum_frozen_variance_relative_residual_inf": frozen_unit_residual,
        },
        "thresholds": thresholds,
        "gates": gates,
        "interpretation": (
            "The generic saddle-node remains the stationary branch geometry and "
            "its null direction is a delay-map unit mode. Realized delays add a "
            "separate unstable complex mode to the high equilibrium (about 27 Hz "
            "far from the fold), while the near-silent branch remains stable. "
            "Therefore the SNN tonic plateau should not be described as a stable "
            "high fixed point; it may be a nonlinear oscillatory/fluctuating "
            "attractor around the high-rate skeleton, which is not proven here."),
        "claim_boundary": (
            "This deterministic linearization contains recurrent conduction "
            "delays but not the realized stationary OU trajectory, finite-size "
            "spiking noise or a nonlinear limit-cycle continuation. Its 27-Hz "
            "mode is not asserted to equal the 40-52 Hz virtual-contact ripple."),
    }
    _atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "full_dt0p5_branch": payload["full_dt0p5_branch"],
        "history_grid_convergence": convergence,
        "saddle_unit_mode": payload["saddle_unit_mode"],
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
