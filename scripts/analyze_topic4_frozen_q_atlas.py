#!/usr/bin/env python3
"""Diagnose whether the frozen fast substrate has a rhythmic high-state band."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    fixed_state_contact_rhythm_metrics,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _late_rate_metrics(rate_hz, *, dt_ms, start_ms, window_ms=250.0,
                       n_windows=4):
    rate = np.asarray(rate_hz, float)
    smooth = uniform_filter1d(
        rate, size=max(1, int(round(20.0 / float(dt_ms)))), mode="nearest")
    time = np.arange(len(rate), dtype=float) * float(dt_ms)
    medians = []
    for window in range(int(n_windows)):
        lo = float(start_ms) + window * float(window_ms)
        selected = (time >= lo) & (time < lo + float(window_ms))
        if not np.any(selected):
            raise ValueError("fixed-state rate window is incomplete")
        medians.append(float(np.median(smooth[selected])))
    return {
        "median_rate_hz": float(np.median(medians)),
        "minimum_subwindow_median_rate_hz": float(np.min(medians)),
        "per_subwindow_median_rate_hz": medians,
    }


def _late_recruitment_metrics(artifact, *, start_ms, stop_ms):
    time = np.asarray(artifact["full_field_time_ms"], float)
    selected = (time >= float(start_ms)) & (time < float(stop_ms))
    if not np.any(selected):
        raise ValueError("fixed-state recruitment window is incomplete")
    neurons = np.asarray(
        artifact["active_neuron_fraction_20ms"], float)[selected]
    sheet = np.asarray(
        artifact["recruited_spatial_fraction_1mm"], float)[selected]
    return {
        "median_active_neuron_fraction_20ms": float(np.median(neurons)),
        "median_recruited_spatial_fraction_1mm": float(np.median(sheet)),
        "joint_global_recruitment_duty": float(
            np.mean((neurons >= 0.5) & (sheet >= 0.5))),
    }


def _classify_static_mode(q_init, rate, recruitment, rhythm):
    checks = {
        "high_rate_in_all_four_windows": (
            rate["minimum_subwindow_median_rate_hz"] >= 120.0),
        "global_recruitment_duty": (
            recruitment["joint_global_recruitment_duty"] >= 0.75),
        "rhythm_is_global": (
            rhythm["contact_fraction_consistently_rhythmic"] >= 0.80),
        "rhythm_is_frequency_locked": (
            30.0 <= rhythm["median_contact_peak_hz"] <= 80.0
            and rhythm["contact_peak_mad_hz"] <= 8.0),
        "rhythm_is_narrowband": rhythm["median_peak_power_fraction"] >= 0.20,
        "band_power_exceeds_q1_reference": bool(
            np.isclose(float(q_init), 1.0)
            or rhythm["median_band_power_ratio_over_q1_reference"] >= 2.0),
    }
    return {
        "status": ("STATIC_FAST_SUBSYSTEM_SUPPORT"
                   if all(checks.values())
                   else "NO_STATIC_FAST_SUBSYSTEM_SUPPORT"),
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
    }


def analyze_atlas(input_dir: Path, reference_json: Path | None = None):
    records = []
    loaded = []
    for json_path in sorted(input_dir.glob("seed*_q*.json")):
        payload = json.loads(json_path.read_text())
        if payload.get("status") != "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE":
            continue
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        config = payload.get("hybrid_config") or {}
        loaded.append((
            float(config["q_init"]),
            float(config.get("q_init_h_gain", 0.0)),
            float(config.get("q_endpoint_gain", 0.0)),
            float(config.get("q_endpoint_sigma_mm", 2.0)),
            json_path,
            payload,
            npz_path,
        ))
    if not loaded:
        raise RuntimeError("no complete frozen-q artifacts found")
    if reference_json is None:
        references = [item for item in loaded
                      if np.isclose(item[0], 1.0)
                      and np.isclose(item[1], 0.0)]
        if len(references) != 1:
            raise RuntimeError("atlas must contain exactly one q_init=1 reference")
        _, _, _, _, reference_json, reference_payload, reference_path = references[0]
    else:
        reference_payload = json.loads(reference_json.read_text())
        reference_path = reference_json.with_suffix(".npz")
        if (reference_payload.get("status")
                != "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE"
                or not reference_path.exists()):
            raise RuntimeError("external q_init=1 reference is incomplete")
    with np.load(reference_path) as reference_artifact:
        reference_lfp = np.asarray(reference_artifact["lfp_trace"], float)
        reference_contacts = np.asarray(reference_artifact["contact_names"]).astype(str)
        reference_dt = float(reference_artifact["lfp_dt_ms"])
    reference_end_ms = len(reference_lfp) * reference_dt
    reference_start_ms = reference_end_ms - 1000.0
    if reference_start_ms < 0.0:
        raise RuntimeError("q_init=1 reference is shorter than 1000 ms")

    for (q_init, q_init_h_gain, q_endpoint_gain, q_endpoint_sigma_mm,
         json_path, payload, npz_path) in sorted(
            loaded, reverse=True):
        with np.load(npz_path) as artifact:
            dt_ms = float(artifact["lfp_dt_ms"])
            if not np.isclose(dt_ms, reference_dt):
                raise RuntimeError(f"dt mismatch: {npz_path}")
            contacts = np.asarray(artifact["contact_names"]).astype(str)
            if not np.array_equal(contacts, reference_contacts):
                raise RuntimeError(f"contact mismatch: {npz_path}")
            trace = np.asarray(artifact["lfp_trace"], float)
            rate_e = np.asarray(artifact["rate_E_hz"], float)
            if "q_grid_initial" in artifact.files:
                q_grid_initial = np.asarray(artifact["q_grid_initial"], float)
                q_initial_range = [
                    float(np.min(q_grid_initial)),
                    float(np.max(q_grid_initial)),
                ]
                q_initial_mean = float(np.mean(q_grid_initial))
            else:
                q_initial_range = [float(q_init), float(q_init)]
                q_initial_mean = float(q_init)
            stop_ms = len(rate_e) * dt_ms
            start_ms = stop_ms - 1000.0
            if start_ms < 0.0:
                raise RuntimeError(f"trajectory shorter than 1000 ms: {npz_path}")
            rate = _late_rate_metrics(rate_e, dt_ms=dt_ms, start_ms=start_ms)
            recruitment = _late_recruitment_metrics(
                artifact, start_ms=start_ms, stop_ms=stop_ms)
            rhythm = fixed_state_contact_rhythm_metrics(
                trace,
                reference_lfp,
                dt_ms=dt_ms,
                start_ms=start_ms,
                reference_start_ms=reference_start_ms,
            )
        classification = _classify_static_mode(
            q_init, rate, recruitment, rhythm)
        hybrid_config = payload.get("hybrid_config") or {}
        spatial_basis = payload.get("spatial_basis_contract") or {}
        records.append({
            "q_init": q_init,
            "q_init_h_gain": q_init_h_gain,
            "q_endpoint_gain": q_endpoint_gain,
            "q_endpoint_sigma_mm": q_endpoint_sigma_mm,
            "q_source_gain": float(hybrid_config.get("q_source_gain", 0.0)),
            "q_sink_gain": float(hybrid_config.get("q_sink_gain", 0.0)),
            "q_support_gain": float(hybrid_config.get(
                "q_support_gain", 0.0)),
            "q_endpoint_side": spatial_basis.get(
                "active_endpoint_side", "union"),
            "q_min": float((payload.get("hybrid_config") or {}).get(
                "q_min", 0.0)),
            "q_initial_grid_range": q_initial_range,
            "q_initial_grid_mean": q_initial_mean,
            "m_build_gain": float((payload.get("hybrid_config") or {}).get(
                "m_build_gain", 1.0)),
            "eta_m": float((payload.get("hybrid_config") or {}).get(
                "eta_m", 0.0)),
            "tau_m_ms": float((payload.get("hybrid_config") or {}).get(
                "tau_m_ms", 0.0)),
            "m_state_ceiling": float((payload.get("hybrid_config") or {}).get(
                "m_state_ceiling", 0.0)),
            "m_spatial_mix": float((payload.get("hybrid_config") or {}).get(
                "m_spatial_mix", 0.0)),
            "sigma_m_mm": float((payload.get("hybrid_config") or {}).get(
                "sigma_m_mm", 0.0)),
            "eta_m_h_gain": float(hybrid_config.get("eta_m_h_gain", 0.0)),
            "eta_m_source_add": float(hybrid_config.get(
                "eta_m_source_add", 0.0)),
            "eta_m_sink_add": float(hybrid_config.get(
                "eta_m_sink_add", 0.0)),
            "eta_m_gk_add": float(hybrid_config.get(
                "eta_m_gk_add", 0.0)),
            "gk_support_sigma_mm": float(hybrid_config.get(
                "gk_support_sigma_mm", 0.0)),
            "gk_support": (spatial_basis.get("gk_support") or {}).get(
                "rule"),
            "m_current_threshold": float(hybrid_config.get(
                "m_current_threshold", 0.0)),
            "hybrid_config": hybrid_config,
            "seed": int(payload["seed"]),
            "json_path": str(json_path.relative_to(ROOT)),
            "npz_path": str(npz_path.relative_to(ROOT)),
            "trajectory_stop_ms": stop_ms,
            "rate": rate,
            "global_recruitment": recruitment,
            "contact_rhythm": rhythm,
            "classification": classification,
        })
    supporting = [record for record in records
                  if record["classification"]["all_checks_pass"]]
    return {
        "status": "FROZEN_Q_FAST_SUBSYSTEM_ATLAS_COMPLETE",
        "purpose": (
            "diagnose the frozen fast substrate; this is not transition, "
            "dwell, recovery, or patient-waveform evidence"),
        "reference": {
            "q_init": 1.0,
            "q_init_h_gain": 0.0,
            "json_path": str(reference_json.relative_to(ROOT)),
            "verdict": reference_payload.get("verdict"),
        },
        "n_q_levels": len(records),
        "n_supporting_q_levels": len(supporting),
        "supporting_q_levels": [record["q_init"] for record in supporting],
        "supporting_parameter_pairs": [
            {
                "q_init": record["q_init"],
                "q_init_h_gain": record["q_init_h_gain"],
                "q_endpoint_gain": record["q_endpoint_gain"],
                "q_endpoint_sigma_mm": record["q_endpoint_sigma_mm"],
                "q_source_gain": record["q_source_gain"],
                "q_sink_gain": record["q_sink_gain"],
                "q_support_gain": record["q_support_gain"],
                "q_endpoint_side": record["q_endpoint_side"],
            }
            for record in supporting
        ],
        "supporting_parameter_sets": [
            record["hybrid_config"]
            for record in supporting
        ],
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--reference-json",
        help="Optional completed q_init=1 JSON outside the input directory.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    reference_json = None
    if args.reference_json:
        reference_json = Path(args.reference_json)
        if not reference_json.is_absolute():
            reference_json = ROOT / reference_json
    payload = analyze_atlas(input_dir, reference_json=reference_json)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_json_safe(payload), str(out))
    csv_path = out.with_suffix(".csv")
    fieldnames = [
        "q_init", "q_init_h_gain", "q_endpoint_gain",
        "q_endpoint_sigma_mm", "q_endpoint_side", "q_min",
        "q_source_gain", "q_sink_gain", "q_support_gain",
        "q_initial_grid_mean",
        "q_initial_grid_min",
        "q_initial_grid_max", "m_build_gain", "eta_m", "tau_m_ms",
        "m_state_ceiling", "m_spatial_mix", "sigma_m_mm",
        "eta_m_h_gain",
        "eta_m_source_add", "eta_m_sink_add",
        "eta_m_gk_add", "gk_support_sigma_mm", "gk_support",
        "m_current_threshold",
        "all_checks_pass", "median_rate_hz",
        "minimum_subwindow_median_rate_hz", "joint_global_recruitment_duty",
        "contact_fraction_consistently_rhythmic", "median_contact_peak_hz",
        "contact_peak_mad_hz", "median_peak_power_fraction",
        "median_band_power_ratio_over_q1_reference", "json_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in payload["records"]:
            writer.writerow({
                "q_init": record["q_init"],
                "q_init_h_gain": record["q_init_h_gain"],
                "q_endpoint_gain": record["q_endpoint_gain"],
                "q_endpoint_sigma_mm": record["q_endpoint_sigma_mm"],
                "q_endpoint_side": record["q_endpoint_side"],
                "q_source_gain": record["q_source_gain"],
                "q_sink_gain": record["q_sink_gain"],
                "q_support_gain": record["q_support_gain"],
                "q_min": record["q_min"],
                "q_initial_grid_mean": record["q_initial_grid_mean"],
                "q_initial_grid_min": record["q_initial_grid_range"][0],
                "q_initial_grid_max": record["q_initial_grid_range"][1],
                "m_build_gain": record["m_build_gain"],
                "eta_m": record["eta_m"],
                "tau_m_ms": record["tau_m_ms"],
                "m_state_ceiling": record["m_state_ceiling"],
                "m_spatial_mix": record["m_spatial_mix"],
                "sigma_m_mm": record["sigma_m_mm"],
                "eta_m_h_gain": record["eta_m_h_gain"],
                "eta_m_source_add": record["eta_m_source_add"],
                "eta_m_sink_add": record["eta_m_sink_add"],
                "eta_m_gk_add": record["eta_m_gk_add"],
                "gk_support_sigma_mm": record["gk_support_sigma_mm"],
                "gk_support": record["gk_support"],
                "m_current_threshold": record["m_current_threshold"],
                "all_checks_pass": record["classification"]["all_checks_pass"],
                "median_rate_hz": record["rate"]["median_rate_hz"],
                "minimum_subwindow_median_rate_hz": record["rate"][
                    "minimum_subwindow_median_rate_hz"],
                "joint_global_recruitment_duty": record[
                    "global_recruitment"]["joint_global_recruitment_duty"],
                "contact_fraction_consistently_rhythmic": record[
                    "contact_rhythm"]["contact_fraction_consistently_rhythmic"],
                "median_contact_peak_hz": record[
                    "contact_rhythm"]["median_contact_peak_hz"],
                "contact_peak_mad_hz": record[
                    "contact_rhythm"]["contact_peak_mad_hz"],
                "median_peak_power_fraction": record[
                    "contact_rhythm"]["median_peak_power_fraction"],
                "median_band_power_ratio_over_q1_reference": record[
                    "contact_rhythm"][
                        "median_band_power_ratio_over_q1_reference"],
                "json_path": record["json_path"],
            })
    print(json.dumps({
        "n_q_levels": payload["n_q_levels"],
        "supporting_parameter_pairs": payload["supporting_parameter_pairs"],
        "out": str(out),
    }))


if __name__ == "__main__":
    main()
