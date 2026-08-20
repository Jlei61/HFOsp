#!/usr/bin/env python3
"""Rescore archived and Stage-1 Z/M candidates against the frozen target."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fig5_target_informed_bridge import (  # noqa: E402
    SCHEMA_ID,
    exact_contact_reorder,
    jsonable,
    log_band_power,
    lse,
    nonoverlap_log_power_windows,
    robust_z_against_reference,
    score_energy_burden,
    score_energy_field,
    select_state_defined_readout,
    smooth_rate,
)


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_rows(base_root):
    roots = [
        base_root / "zm_joint_morphology_calibration",
        base_root / "zm_joint_morphology_calibration_v2r",
        base_root / "zm_joint_morphology_calibration_v3_etoi",
        base_root / "zm_joint_morphology_calibration_v4_etoi_refine",
        base_root / "zm_joint_morphology_calibration_v5_etoi_boundary",
        base_root / "target_informed_bridge_v1/fit",
    ]
    for directory in roots:
        for path in sorted(directory.glob("*.json")):
            npz = path.with_suffix(".npz")
            if npz.exists() and path.name != "calibration_summary.json":
                yield path, npz, _load_json(path)


def _slice(values, dt_ms, start_ms, stop_ms):
    lo = int(round(float(start_ms) / float(dt_ms)))
    hi = int(round(float(stop_ms) / float(dt_ms)))
    if lo < 0 or hi > len(values) or hi <= lo:
        raise ValueError("requested model window is incomplete")
    return np.asarray(values)[lo:hi]


def _score_one(json_path, npz_path, payload, baseline, target_payload, target_npz,
               readout_cfg):
    parameters = dict(payload.get("parameters") or {})
    ee = float(parameters.get("E_to_E_dose", 1.0))
    etoi = float(parameters.get("E_to_I_dose", 1.0))
    row = {
        "candidate_id": json_path.stem,
        "source_json": str(json_path.relative_to(ROOT)),
        "source_npz": str(npz_path.relative_to(ROOT)),
        "parameters": parameters,
        "primary_zm_only": bool(np.isclose(ee, 1.0) and np.isclose(etoi, 1.0)),
        "edge_dose_comparator": bool(not (np.isclose(ee, 1.0) and np.isclose(etoi, 1.0))),
    }
    onset_op = payload.get("operational_onset_ms")
    if onset_op is None:
        row.update(status="MODEL_ICTAL_NOT_ELIGIBLE", reason="operational detector not reached")
        return row
    t_ictal = float(onset_op) - 100.0
    with np.load(npz_path, allow_pickle=False) as data:
        required = ["lfp_trace", "lfp_dt_ms", "contact_names", "rate_E_hz",
                    "full_field_time_ms", "active_neuron_fraction_20ms",
                    "recruited_spatial_fraction_1mm"]
        missing = [key for key in required if key not in data]
        if missing:
            row.update(status="NOT_EVALUABLE", reason=f"missing arrays: {missing}")
            return row
        trace = np.asarray(data["lfp_trace"], float)
        dt = float(data["lfp_dt_ms"])
        names = np.asarray(data["contact_names"]).astype(str)
        read = select_state_defined_readout(
            trace=trace,
            dt_ms=dt,
            full_field_time_ms=data["full_field_time_ms"],
            active_fraction=data["active_neuron_fraction_20ms"],
            spatial_fraction=data["recruited_spatial_fraction_1mm"],
            t_ictal_ms=t_ictal,
            baseline_trace=baseline["centroid_reference_trace"],
            window_ms=float(readout_cfg["window_ms"]),
            step_ms=float(readout_cfg["step_ms"]),
            activity_threshold=float(readout_cfg["activity_threshold"]),
            duty_threshold=float(readout_cfg["joint_duty_threshold"]),
            frequency_shift_hz=float(readout_cfg["contact_frequency_shift_hz"]),
            frequency_ratio=float(readout_cfg["contact_frequency_ratio"]),
            band_hz=readout_cfg["primary_band_hz"],
        )
        if read is None:
            row.update(status="MODEL_ICTAL_NOT_ELIGIBLE",
                       reason="no state-qualified 500 ms readout window")
            return row
        read_trace = _slice(trace, dt, read.start_ms, read.stop_ms)
        pre_trace = _slice(trace, dt, t_ictal - 500.0, t_ictal)
        model_early_log = log_band_power(read_trace, dt, readout_cfg["primary_band_hz"])
        model_pre_log = log_band_power(pre_trace, dt, readout_cfg["primary_band_hz"])
        model_early_log = exact_contact_reorder(
            model_early_log, names, baseline["contact_names"])
        model_pre_log = exact_contact_reorder(
            model_pre_log, names, baseline["contact_names"])
        model_early_z, _ = robust_z_against_reference(
            baseline["log_power_windows"], model_early_log)
        model_pre_z, _ = robust_z_against_reference(
            baseline["log_power_windows"], model_pre_log)
        smooth_candidate = smooth_rate(data["rate_E_hz"], dt, 20.0)
        read_rate = float(np.median(_slice(
            smooth_candidate, dt, read.start_ms, read.stop_ms)))
        if read_rate < 2.0 * baseline["median_smoothed_rate_hz"]:
            row.update(status="MODEL_ICTAL_NOT_ELIGIBLE",
                       reason="20 ms-smoothed population rate ratio below 2",
                       rate_ratio=read_rate / max(baseline["median_smoothed_rate_hz"], 1e-12))
            return row
    target_names = target_npz["contact_names"].astype(str)
    model_early_z = exact_contact_reorder(
        model_early_z, baseline["contact_names"], target_names)
    model_pre_z = exact_contact_reorder(
        model_pre_z, baseline["contact_names"], target_names)
    target = target_payload["summaries"]["sensitivity_10_150"]
    shaft_ids = target_npz["shaft_ids"].astype(str)
    field = score_energy_field(model_pre_z, model_early_z, target, shaft_ids)
    energy = score_energy_burden(model_early_z, target)
    j_bridge = float(np.mean([energy["D_energy"], field["J_field"]])
                     + lse([energy["D_energy"], field["J_field"]]))
    row.update({
        "status": "BRIDGE_EVALUABLE",
        "readout_window": read.__dict__,
        "rate_ratio": read_rate / baseline["median_smoothed_rate_hz"],
        "model_pre_robust_z": model_pre_z,
        "model_early_robust_z": model_early_z,
        "energy": energy,
        "field": field,
        "J_bridge_without_time": j_bridge,
        "time_component_status": "PENDING_FINE_GRAINED_TARGET",
    })
    return row


def build_paired_baseline(baseline_path, readout_cfg):
    baseline_path = Path(baseline_path)
    with np.load(baseline_path, allow_pickle=False) as z:
        trace = np.asarray(z["lfp_trace"], float)
        dt = float(z["lfp_dt_ms"])
        names = np.asarray(z["contact_names"]).astype(str)
        rate = np.asarray(z["rate_E_hz"], float)
    windows = nonoverlap_log_power_windows(
        trace, dt, window_ms=float(readout_cfg["baseline_window_ms"]),
        band_hz=readout_cfg["primary_band_hz"])
    rate_smoothed = smooth_rate(rate, dt, 20.0)
    return {
        "log_power_windows": windows,
        "contact_names": names,
        "centroid_reference_trace": _slice(trace, dt, 500.0, 1000.0),
        "median_smoothed_rate_hz": float(np.median(
            _slice(rate_smoothed, dt, 500.0, 1000.0))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    args = parser.parse_args()
    config = _load_json(ROOT / args.config)
    out = ROOT / config["output_root"]
    target_payload = _load_json(out / "clinical_target.json")
    baseline_path = out / "paired_baseline/seed1801_zmoff.npz"
    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    target_data = np.load(out / "clinical_target_vectors.npz", allow_pickle=False)
    baseline = build_paired_baseline(baseline_path, config["model_readout"])
    records = [
        _score_one(jpath, npath, payload, baseline, target_payload, target_data,
                   config["model_readout"])
        for jpath, npath, payload in _candidate_rows(
            ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition")
    ]
    records.sort(key=lambda row: (
        row["status"] != "BRIDGE_EVALUABLE",
        row.get("J_bridge_without_time", float("inf")),
        row["candidate_id"],
    ))
    payload = {
        "schema_id": SCHEMA_ID,
        "status": "EXISTING_AND_STAGE1_CANDIDATES_RESCORED",
        "primary_matched_band_hz": [10.0, 150.0],
        "patient_target_role": "development target; display seizure 2 excluded",
        "baseline": {
            "path": str(baseline_path.relative_to(ROOT)),
            "n_nonoverlap_windows": int(len(baseline["log_power_windows"])),
            "median_smoothed_rate_hz": baseline["median_smoothed_rate_hz"],
        },
        "n_candidates": len(records),
        "n_bridge_evaluable": sum(row["status"] == "BRIDGE_EVALUABLE" for row in records),
        "records": records,
    }
    (out / "existing_candidate_rescore.json").write_text(
        json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    with (out / "existing_candidate_rescore.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_id", "status", "primary_zm_only", "edge_dose_comparator",
            "J_bridge_without_time", "D_energy", "D_contact", "D_increment",
            "early_spearman", "rate_ratio", "reason"])
        writer.writeheader()
        for row in records:
            writer.writerow({
                "candidate_id": row["candidate_id"],
                "status": row["status"],
                "primary_zm_only": row["primary_zm_only"],
                "edge_dose_comparator": row["edge_dose_comparator"],
                "J_bridge_without_time": row.get("J_bridge_without_time"),
                "D_energy": (row.get("energy") or {}).get("D_energy"),
                "D_contact": (row.get("field") or {}).get("D_contact"),
                "D_increment": (row.get("field") or {}).get("D_increment"),
                "early_spearman": (row.get("field") or {}).get("early_spearman"),
                "rate_ratio": row.get("rate_ratio"),
                "reason": row.get("reason"),
            })
    print(json.dumps({"status": payload["status"],
                      "n_bridge_evaluable": payload["n_bridge_evaluable"]}))


if __name__ == "__main__":
    main()
