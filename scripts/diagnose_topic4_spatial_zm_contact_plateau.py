#!/usr/bin/env python3
"""Attribute the 9/15 contact plateau to a contact-level mechanism.

The previous round reported an aggregate "60% of contacts are rhythmic" and
stopped there.  An aggregate cannot say *why* the remaining contacts fail, and
the two candidate reasons need different fixes: a contact can miss the gate
because the 30-80 Hz rhythm never reaches its tissue (a spread problem) or
because its own low-state band power is already large (a contrast problem).

This script reads an existing frozen-state artifact plus its q=1 reference and
reports, per contact: absolute 20-100 Hz band power in both states, the peak
frequency in each 250 ms window, distance to the high-h core, and the clause
that actually failed.  It re-uses ``fixed_state_contact_rhythm_metrics`` so the
numbers are the same ones the gate saw.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_global_recruited_oscillation import (  # noqa: E402
    _window_spectrum,
    fixed_state_contact_rhythm_metrics,
)

BAND_HZ = (20.0, 100.0)
TARGET_HZ = (30.0, 80.0)


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


def _periodic_distance(a, b, sheet_l_mm):
    delta = np.abs(np.asarray(a, float) - np.asarray(b, float))
    delta = np.minimum(delta, float(sheet_l_mm) - delta)
    return float(np.hypot(*delta))


def diagnose(candidate_npz: Path, reference_npz: Path, *, window_ms=250.0,
             n_windows=4, sheet_l_mm=20.0):
    with np.load(candidate_npz) as candidate, np.load(reference_npz) as reference:
        dt = float(candidate["lfp_dt_ms"])
        lfp = np.asarray(candidate["lfp_trace"], float)
        reference_lfp = np.asarray(reference["lfp_trace"], float)
        names = np.asarray(candidate["contact_names"]).astype(str)
        shafts = np.asarray(candidate["shaft_ids"]).astype(str)
        contact_xy = np.asarray(candidate["contact_xy_mm"], float)
        positions = np.asarray(candidate["positions_E"], float)
        h_e = np.asarray(candidate["h_E"], float)

    span_ms = float(n_windows) * float(window_ms)
    start = len(lfp) * dt - span_ms
    reference_start = len(reference_lfp) * dt - span_ms
    if start < 0.0 or reference_start < 0.0:
        raise RuntimeError("a trace is shorter than the analysis span")
    rhythm = fixed_state_contact_rhythm_metrics(
        lfp, reference_lfp, dt_ms=dt, start_ms=start,
        reference_start_ms=reference_start, window_ms=window_ms,
        n_windows=n_windows, band_hz=BAND_HZ, target_hz=TARGET_HZ)

    core = h_e >= 0.5
    core_centroid = positions[core].mean(axis=0) if np.any(core) else None
    fs_hz = 1000.0 / dt
    time = np.arange(len(lfp), dtype=float) * dt
    reference_time = np.arange(len(reference_lfp), dtype=float) * dt
    peak_hz = np.asarray(rhythm["per_window_contact_peak_hz"], float)
    fraction = np.asarray(
        rhythm["per_window_contact_peak_power_fraction"], float)
    ratio = np.asarray(
        rhythm["per_window_contact_band_power_ratio_over_q1"], float)
    passing = np.asarray(rhythm["per_contact_consistently_rhythmic"], bool)

    # Absolute band power must be measured on exactly the windows the gate
    # used.  A single 1000 ms periodogram is NOT equivalent: with only linear
    # detrending, the low state's large slow interictal transients leak into
    # 20-100 Hz and inflate the reference by ~70x, which would invert the
    # contrast the gate actually saw.
    def _window_powers(trace, span_start, contact):
        powers = []
        trace_time = np.arange(len(trace), dtype=float) * dt
        for window in range(int(n_windows)):
            lo = float(span_start) + window * float(window_ms)
            selected = (trace_time >= lo) & (trace_time < lo + float(window_ms))
            powers.append(_window_spectrum(
                trace[selected, contact], fs_hz, BAND_HZ)[2])
        return np.asarray(powers, float)

    rows = []
    for index, name in enumerate(names):
        candidate_powers = _window_powers(lfp, start, index)
        reference_powers = _window_powers(reference_lfp, reference_start, index)
        candidate_power = float(np.median(candidate_powers))
        reference_power = float(np.median(reference_powers))
        distance = np.linalg.norm(positions - contact_xy[index], axis=1)
        failed = []
        window_peak = peak_hz[:, index]
        if not np.all((window_peak >= TARGET_HZ[0])
                      & (window_peak <= TARGET_HZ[1])):
            failed.append("peak_outside_30_80_hz_in_some_window")
        if not np.all(fraction[:, index] >= 0.20):
            failed.append("peak_power_fraction_below_0.20")
        if not np.all(ratio[:, index] >= 2.0):
            failed.append("band_power_ratio_below_2")
        rows.append({
            "contact": str(name),
            "shaft": str(shafts[index]),
            "x_mm": float(contact_xy[index][0]),
            "y_mm": float(contact_xy[index][1]),
            "periodic_distance_to_core_mm": (
                None if core_centroid is None
                else _periodic_distance(contact_xy[index], core_centroid,
                                        sheet_l_mm)),
            "mean_h_within_1mm": float(h_e[distance <= 1.0].mean())
            if np.any(distance <= 1.0) else None,
            "median_window_band_power_20_100hz": candidate_power,
            "median_window_reference_band_power_20_100hz": reference_power,
            "median_window_band_power_ratio": float(
                candidate_power / max(reference_power, 1e-20)),
            "per_window_band_power_20_100hz": candidate_powers.tolist(),
            "per_window_reference_band_power_20_100hz":
                reference_powers.tolist(),
            "per_window_peak_hz": window_peak.tolist(),
            "per_window_peak_power_fraction": fraction[:, index].tolist(),
            "per_window_band_power_ratio": ratio[:, index].tolist(),
            "n_windows_in_target_band": int(np.sum(
                (window_peak >= TARGET_HZ[0]) & (window_peak <= TARGET_HZ[1]))),
            "first_window_in_target_band": (
                int(np.argmax((window_peak >= TARGET_HZ[0])
                              & (window_peak <= TARGET_HZ[1])))
                if np.any((window_peak >= TARGET_HZ[0])
                          & (window_peak <= TARGET_HZ[1])) else None),
            "consistently_rhythmic": bool(passing[index]),
            "failing_clauses": failed,
        })

    failing = [row for row in rows if not row["consistently_rhythmic"]]
    passing_rows = [row for row in rows if row["consistently_rhythmic"]]

    def _median(items, key):
        values = [item[key] for item in items if item[key] is not None]
        return float(np.median(values)) if values else None

    summary = {
        "n_contacts": len(rows),
        "n_rhythmic": len(passing_rows),
        "median_distance_to_core_mm_passing": _median(
            passing_rows, "periodic_distance_to_core_mm"),
        "median_distance_to_core_mm_failing": _median(
            failing, "periodic_distance_to_core_mm"),
        "median_band_power_ratio_passing": _median(
            passing_rows, "median_window_band_power_ratio"),
        "median_band_power_ratio_failing": _median(
            failing, "median_window_band_power_ratio"),
        "median_reference_band_power_passing": _median(
            passing_rows, "median_window_reference_band_power_20_100hz"),
        "median_reference_band_power_failing": _median(
            failing, "median_window_reference_band_power_20_100hz"),
        "band_power_statistic_note": (
            "all band powers are medians over the same 250 ms windows the gate "
            "used; a single 1000 ms periodogram is not comparable because slow "
            "low-state transients leak into 20-100 Hz"),
        "failing_clause_counts": {
            clause: int(sum(clause in row["failing_clauses"] for row in rows))
            for clause in ("peak_outside_30_80_hz_in_some_window",
                           "peak_power_fraction_below_0.20",
                           "band_power_ratio_below_2")},
        "failing_contacts_whose_band_arrives_late": [
            row["contact"] for row in failing
            if row["first_window_in_target_band"] not in (None, 0)],
        "failing_contacts_never_in_band": [
            row["contact"] for row in failing
            if row["first_window_in_target_band"] is None],
    }
    return {"rows": rows, "summary": summary,
            "aggregate_rhythm": {k: v for k, v in rhythm.items()
                                 if not k.startswith("per_")},
            "analysis_window_ms": [start, start + span_ms],
            "reference_analysis_window_ms": [reference_start,
                                             reference_start + span_ms]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-npz", required=True)
    parser.add_argument("--reference-npz", required=True)
    parser.add_argument("--sheet-l-mm", type=float, default=20.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    candidate = Path(args.candidate_npz)
    reference = Path(args.reference_npz)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    if not reference.is_absolute():
        reference = ROOT / reference
    report = diagnose(candidate, reference, sheet_l_mm=float(args.sheet_l_mm))
    report["status"] = "SPATIAL_ZM_CONTACT_PLATEAU_DIAGNOSIS_COMPLETE"
    report["candidate_npz"] = str(candidate.relative_to(ROOT))
    report["reference_npz"] = str(reference.relative_to(ROOT))
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_json_safe(report), str(out.with_suffix(".json")))
    with out.with_suffix(".csv").open("w", newline="") as handle:
        fields = [key for key in report["rows"][0]
                  if not key.startswith("per_window")]
        writer = csv.DictWriter(handle, fieldnames=[*fields, "failing_clauses_joined"])
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({**{key: row[key] for key in fields},
                             "failing_clauses_joined": ";".join(
                                 row["failing_clauses"])})
    print(json.dumps(report["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
