#!/usr/bin/env python3
"""Project three prospective OU-SNN trajectories onto the frozen-q Z/M fold.

The reduction eliminates M only at stationarity (M*=tau_M r_E).  We therefore
store two complementary projections for every seed: (q, r_E) and
(D=1-q, A=eta_M M).  q_core and q_mean are both retained because a scalar
uniform-q manifold cannot decide which spatial summary should trigger a finite
network transition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_zm_phase_point import _atomic_json, _atomic_npz  # noqa: E402
from src.topic4_patient_zm_meanfield import (  # noqa: E402
    load_patient_coarse_model,
    solve_fixed_point,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_downcrossing(time, values, threshold):
    """Linearly interpolate the first >threshold to <=threshold crossing."""
    time = np.asarray(time, float)
    values = np.asarray(values, float)
    if time.shape != values.shape or time.ndim != 1:
        raise ValueError("time and values must be aligned vectors")
    hits = np.flatnonzero((values[:-1] > threshold) & (values[1:] <= threshold))
    if not hits.size:
        return None
    index = int(hits[0])
    fraction = ((threshold - values[index])
                / (values[index + 1] - values[index]))
    return float(time[index] + fraction * (time[index + 1] - time[index]))


def smoothed_rate_at(time_ms, rate_hz, target_time_ms, *, window_ms=20.0):
    """Centered rectangular smoothing followed by interpolation."""
    time_ms = np.asarray(time_ms, float)
    rate_hz = np.asarray(rate_hz, float)
    if time_ms.shape != rate_hz.shape or time_ms.size < 2:
        raise ValueError("rate time series is invalid")
    dt = float(np.median(np.diff(time_ms)))
    width = max(1, int(round(float(window_ms) / dt)))
    smoothed = uniform_filter1d(rate_hz, size=width, mode="nearest")
    return np.interp(np.asarray(target_time_ms, float), time_ms, smoothed)


def _value_at(time, values, target):
    return float(np.interp(float(target), np.asarray(time, float),
                           np.asarray(values, float)))


def _manifold_arrays(bifurcation_npz, model, *, eta_m, tau_m_ms):
    with np.load(bifurcation_npz, allow_pickle=False) as archive:
        prefix = f"eta_{eta_m:.5f}".replace(".", "p")
        regular_q = np.asarray(archive[f"{prefix}__regular_q"], float)
        regular_rate = np.asarray(
            archive[f"{prefix}__regular_rate_e_hz"], float)
        arc_q = np.asarray(archive[f"{prefix}__arc_q"], float)
        arc_rate = np.asarray(archive[f"{prefix}__arc_rate_e_hz"], float)
    fold_index = int(np.argmax(arc_q))
    high_q = np.r_[regular_q, arc_q[:fold_index + 1]]
    high_rate = np.r_[regular_rate, arc_rate[:fold_index + 1]]
    order = np.argsort(high_q)
    high_q, high_rate = high_q[order], high_rate[order]
    returned_q = arc_q[fold_index:]
    returned_rate = arc_rate[fold_index:]
    returned_order = np.argsort(returned_q)
    returned_q = returned_q[returned_order]
    returned_rate = returned_rate[returned_order]

    low_q = np.linspace(0.775, 1.0, 24)
    rates = np.r_[np.full(model.n_cells, 1e-4),
                  np.full(model.n_cells, 5e-4)]
    low_rate = []
    for q in low_q:
        solution = solve_fixed_point(
            model, q=float(q), eta_m=eta_m,
            tau_m_slow_ms=tau_m_ms, initial_rates=rates, maxfev=8000)
        if not solution.converged:
            raise RuntimeError(f"low branch solve failed at q={q}")
        rates = solution.rates
        low_rate.append(solution.mean_rate_e_hz)
    return {
        "high_q": high_q,
        "high_rate_e_hz": high_rate,
        "returned_q": returned_q,
        "returned_rate_e_hz": returned_rate,
        "low_q": low_q,
        "low_rate_e_hz": np.asarray(low_rate),
    }


def main():
    parser = argparse.ArgumentParser()
    phase_root = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
                  "data_driven_zm_phase_diagram")
    tonic_root = ("/data/hfosp_topic4_fig45_artifacts/fig5/"
                  "data_driven_node_local_connectivity_plus_zm/spatial_zm_ou/"
                  "tonic_confirmation_v2")
    parser.add_argument("--phase-root", default=phase_root)
    parser.add_argument("--trajectory-root", default=tonic_root)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1841, 1842, 1843])
    parser.add_argument("--eta-m", type=float, default=0.02)
    parser.add_argument("--tau-m-ms", type=float, default=12.5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    phase_root = Path(args.phase_root).resolve()
    trajectory_root = Path(args.trajectory_root).resolve()
    deterministic = phase_root / "deterministic_meanfield"
    bifurcation_json = deterministic / "patient_zm_bifurcation_ngrid20.json"
    bifurcation_npz = bifurcation_json.with_suffix(".npz")
    saddle_json = deterministic / "patient_zm_saddle_node_validation_ngrid20.json"
    saddle = json.loads(saddle_json.read_text())
    q_fold = float(saddle["fold"]["q_from_eigenvalue_zero"])
    rate_fold = float(saddle["fold"]["mean_rate_e_hz_at_eigenvalue_zero"])
    model_path = deterministic / "patient_coarse_ngrid20.npz"
    model = load_patient_coarse_model(model_path)
    manifold = _manifold_arrays(
        bifurcation_npz, model, eta_m=float(args.eta_m),
        tau_m_ms=float(args.tau_m_ms))
    output = (Path(args.out).resolve() if args.out else phase_root /
              "dynamic_projection/patient_zm_snn_manifold_projection.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    array_path = output.with_suffix(".npz")
    arrays = {f"manifold_{key}": value for key, value in manifold.items()}
    runs = []
    high_rate_at_qmin = float(np.interp(
        0.775, manifold["high_q"], manifold["high_rate_e_hz"]))
    for seed in args.seeds:
        json_path = trajectory_root / f"tonic_b0_v2_s{seed}.json"
        npz_path = json_path.with_suffix(".npz")
        metadata = json.loads(json_path.read_text())
        if metadata["tonic_verdict"] != "TONIC_GLOBAL_RUNAWAY":
            raise RuntimeError(f"seed {seed} is not a confirmed tonic runaway")
        with np.load(npz_path, allow_pickle=False) as archive:
            time = np.asarray(archive["slow_time_ms"], float)
            q_mean = np.asarray(archive["slow_q_mean"], float)
            q_core = np.asarray(archive["slow_q_core_mean"], float)
            q_min = np.asarray(archive["slow_q_min"], float)
            m = np.asarray(archive["slow_m_mean"], float)
            rate = smoothed_rate_at(
                archive["time_ms"], archive["rate_E_hz"], time,
                window_ms=20.0)
        onset = float(metadata["scientific_onset_ms"])
        m_star = float(args.tau_m_ms) * rate / 1000.0
        post_mask = time >= onset + 300.0
        if not np.any(post_mask):
            raise RuntimeError(f"seed {seed} has no late plateau window")
        q_mean_cross = first_downcrossing(time, q_mean, q_fold)
        q_core_cross = first_downcrossing(time, q_core, q_fold)
        q_min_cross = first_downcrossing(time, q_min, q_fold)
        if None in (q_mean_cross, q_core_cross, q_min_cross):
            raise RuntimeError(f"seed {seed} never crosses the reduced fold q")
        q_mean_onset = _value_at(time, q_mean, onset)
        q_core_onset = _value_at(time, q_core, onset)
        m_onset = _value_at(time, m, onset)
        rate_onset = _value_at(time, rate, onset)
        late_m_abs_error = float(np.median(np.abs(m[post_mask] - m_star[post_mask])))
        late_m_relative_error = float(late_m_abs_error / max(
            np.median(np.abs(m_star[post_mask])), 1e-12))
        post_rate = float(metadata["state_rate"]["median_post_hz"])
        record = {
            "seed": int(seed),
            "source": {
                "json": str(json_path), "json_sha256": _sha256(json_path),
                "npz": str(npz_path), "npz_sha256": _sha256(npz_path),
            },
            "scientific_onset_ms": onset,
            "fold_crossings_ms": {
                "q_core": q_core_cross,
                "q_min": q_min_cross,
                "q_mean": q_mean_cross,
            },
            "crossing_to_onset_lag_ms": {
                "q_core": onset - q_core_cross,
                "q_min": onset - q_min_cross,
                "q_mean": onset - q_mean_cross,
            },
            "at_onset": {
                "q_core": q_core_onset,
                "q_mean": q_mean_onset,
                "M": m_onset,
                "M_star_from_20ms_rate": float(args.tau_m_ms) * rate_onset / 1000.0,
                "rate_E_20ms_hz": rate_onset,
            },
            "late_plateau": {
                "window_start_ms": onset + 300.0,
                "median_rate_E_hz_from_confirmation": post_rate,
                "high_branch_rate_at_qmin_hz": high_rate_at_qmin,
                "relative_rate_difference_from_high_branch": abs(
                    post_rate - high_rate_at_qmin) / high_rate_at_qmin,
                "median_abs_M_minus_Mstar": late_m_abs_error,
                "relative_median_abs_M_minus_Mstar": late_m_relative_error,
            },
        }
        prefix = f"seed{seed}"
        arrays.update({
            f"{prefix}_time_ms": time,
            f"{prefix}_time_relative_onset_ms": time - onset,
            f"{prefix}_q_mean": q_mean,
            f"{prefix}_q_core": q_core,
            f"{prefix}_q_min": q_min,
            f"{prefix}_M": m,
            f"{prefix}_M_star": m_star,
            f"{prefix}_rate_E_20ms_hz": rate,
        })
        runs.append(record)

    thresholds = {
        "crossing_must_precede_onset_ms": 0.0,
        "late_plateau_relative_rate_difference_max": 0.15,
        "late_plateau_relative_M_tracking_error_max": 0.05,
    }
    gates = {
        "all_q_core_crossings_precede_onset": all(
            row["crossing_to_onset_lag_ms"]["q_core"] > 0.0 for row in runs),
        "all_q_mean_crossings_precede_onset": all(
            row["crossing_to_onset_lag_ms"]["q_mean"] > 0.0 for row in runs),
        "all_onsets_are_beyond_uniform_q_fold": all(
            row["at_onset"]["q_core"] < q_fold
            and row["at_onset"]["q_mean"] < q_fold for row in runs),
        "late_plateau_rate_matches_high_branch_scale": all(
            row["late_plateau"]["relative_rate_difference_from_high_branch"]
            < thresholds["late_plateau_relative_rate_difference_max"]
            for row in runs),
        "late_M_tracks_stationary_manifold": all(
            row["late_plateau"]["relative_median_abs_M_minus_Mstar"]
            < thresholds["late_plateau_relative_M_tracking_error_max"]
            for row in runs),
    }
    _atomic_npz(array_path, **arrays)
    payload = {
        "status": ("SNN_TRAJECTORIES_CONSISTENT_WITH_REDUCED_FOLD_ORGANIZER"
                   if all(gates.values()) else
                   "SNN_TRAJECTORY_PROJECTION_HAS_FAILED_GATES"),
        "scientific_scope": (
            "prospective tonic_b0_v2 seeds 1841-1843 under stationary spatial "
            "OU, projected onto the patient-matched 1-mm frozen-uniform-q Z/M "
            "critical manifold"),
        "manifold": {
            "q_fold": q_fold,
            "D_fold": 1.0 - q_fold,
            "mean_rate_e_hz_at_fold": rate_fold,
            "eta_m": float(args.eta_m),
            "tau_m_ms": float(args.tau_m_ms),
            "stationary_relation": "M*=tau_M*r_E with r_E in spikes/ms",
            "source_bifurcation": {
                "json": str(bifurcation_json),
                "json_sha256": _sha256(bifurcation_json),
                "npz": str(bifurcation_npz),
                "npz_sha256": _sha256(bifurcation_npz),
            },
            "source_saddle_node": {
                "json": str(saddle_json), "json_sha256": _sha256(saddle_json),
            },
        },
        "runs": runs,
        "thresholds": thresholds,
        "gates": gates,
        "interpretation": (
            "All three spatial SNN trajectories enter the permissive side of "
            "the scalar reduced fold before the operational tonic onset, and "
            "their late rate/M scale approaches the high-rate manifold. This "
            "supports the saddle-node as an organizing skeleton, not as an "
            "identified finite-SNN switching threshold."),
        "claim_boundary": (
            "q_core and q_mean bracket different spatial summaries and cross at "
            "different times. Stationary OU, finite-size fluctuations and spatial "
            "heterogeneity are absent from the reduced manifold; temporal ordering "
            "and scale agreement are not a causal or thermodynamic phase-transition "
            "test."),
        "arrays": {"path": str(array_path), "sha256": _sha256(array_path)},
    }
    _atomic_json(payload, output)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "q_fold": q_fold, "gates": gates,
        "runs": [{"seed": row["seed"],
                  "lags": row["crossing_to_onset_lag_ms"],
                  "late": row["late_plateau"]} for row in runs],
    }, indent=2))


if __name__ == "__main__":
    main()
