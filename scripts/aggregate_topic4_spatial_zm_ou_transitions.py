#!/usr/bin/env python3
"""Stage B/C table: every dynamic trajectory scored on the full Fig5A gate.

The nine LFP clauses and criterion 10 are reported as separate columns and as a
joint verdict, because they fail for different reasons and the distinction is
the scientific content of this round: a state can have a textbook 40-50 Hz
contact spectrum while its population firing rate is a near-constant plateau.

Criterion 10 is recomputed from the stored population rate for any trajectory
produced before it was wired into the runner, so old and new artifacts are
scored identically.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.topic4_tonic_fixed_point import classify_tonic_fixed_point  # noqa: E402

SETTLE_MS = 300.0
POST_GATE_MS = 1000.0


def _criterion10(payload, npz_path: Path):
    stored = payload.get("criterion10_tonic_exclusion")
    if stored is not None:
        return stored, False
    onset = payload.get("scientific_onset_ms")
    if onset is None or not npz_path.exists():
        return None, False
    with np.load(npz_path) as artifact:
        rate = np.asarray(artifact["rate_E_hz"], float)
        dt = float(artifact["lfp_dt_ms"])
        time = np.asarray(artifact["full_field_time_ms"], float)
        active = np.asarray(artifact["active_neuron_fraction_20ms"], float)
    selected = ((time >= float(onset) + SETTLE_MS)
                & (time < float(onset) + SETTLE_MS + POST_GATE_MS))
    try:
        return classify_tonic_fixed_point(
            rate, dt_ms=dt, onset_ms=float(onset),
            active_fraction_20ms=active[selected]), True
    except ValueError:
        return None, False


def _row(json_path: Path):
    payload = json.loads(json_path.read_text())
    config = payload.get("hybrid_config") or {}
    classification = payload.get("classification") or {}
    checks = classification.get("checks") or {}
    rhythm = payload.get("contact_rhythm") or {}
    rates = payload.get("state_rate") or {}
    recruitment = payload.get("global_recruitment") or {}
    stability = payload.get("numerical_stability") or {}
    runtime = payload.get("ou_runtime_evidence") or {}
    stationarity = payload.get("ou_stationarity_across_transition") or {}
    tonic, recomputed = _criterion10(payload, json_path.with_suffix(".npz"))
    high = ((tonic or {}).get("detail") or {}).get("high_state") or {}
    low = ((tonic or {}).get("detail") or {}).get("pre_transition_state") or {}
    nine = bool(classification.get("all_checks_pass", False))
    ten = None if tonic is None else bool(tonic["all_checks_pass"])
    return {
        "run": json_path.stem,
        "path": str(json_path.relative_to(ROOT)),
        "seed": payload.get("seed"),
        "mode": payload.get("mode"),
        "run_role": payload.get("run_role"),
        "parameter_set_id": payload.get("parameter_set_id"),
        "verdict": payload.get("verdict"),
        "k_q_per_ms": config.get("k_q_per_ms"),
        "q_min": config.get("q_min"),
        "q_a50": config.get("q_a50"),
        "q_hill_n": config.get("q_hill_n"),
        "tau_m_ms": config.get("tau_m_ms"),
        "eta_m": config.get("eta_m"),
        "m_spatial_mix": config.get("m_spatial_mix"),
        "scientific_onset_ms": payload.get("scientific_onset_ms"),
        "median_rate_pre_hz": rates.get("median_pre_hz"),
        "q95_rate_pre_hz": rates.get("q95_pre_hz"),
        "median_rate_post_hz": rates.get("median_post_hz"),
        "joint_global_recruitment_duty": recruitment.get(
            "joint_global_recruitment_duty"),
        "n_rhythmic_contacts": payload.get("n_rhythmic_contacts"),
        "contact_fraction_consistently_rhythmic": rhythm.get(
            "contact_fraction_consistently_rhythmic"),
        "median_contact_peak_hz": rhythm.get("median_contact_peak_hz"),
        "contact_peak_mad_hz": rhythm.get("contact_peak_mad_hz"),
        "median_peak_power_fraction": rhythm.get("median_peak_power_fraction"),
        "median_band_power_ratio_post_over_pre": rhythm.get(
            "median_band_power_ratio_post_over_pre"),
        "population_rate_dominant_hz": high.get("dominant_hz"),
        "population_rate_mean_hz": high.get("mean_rate_hz"),
        "population_rate_modulation_depth": high.get("modulation_depth"),
        "pre_transition_modulation_depth": low.get("modulation_depth"),
        "criterion10_recomputed_from_npz": recomputed,
        "nine_clause_lfp_gate_pass": nine,
        "criterion10_tonic_exclusion_pass": ten,
        "fig5a_full_gate_pass": bool(nine and (ten is True)),
        "failed_lfp_clauses": ";".join(
            sorted(key for key, value in checks.items() if not value)),
        "numerically_stable": stability.get("all_checks_pass"),
        "ou_called_every_membrane_step": runtime.get(
            "called_every_membrane_step"),
        "ou_sd_ratio_after_over_before": stationarity.get(
            "sd_ratio_after_over_before"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    rows = []
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "SPATIAL_ZM_OU_TRANSITION_COMPLETE":
            continue
        rows.append(_row(path))
    if not rows:
        raise RuntimeError("no completed transition artifacts found")
    rows.sort(key=lambda row: (
        -int(row["fig5a_full_gate_pass"]),
        -float(row["population_rate_modulation_depth"] or -1.0),
        -float(row["contact_fraction_consistently_rhythmic"] or -1.0),
        row["run"]))
    payload = {
        "status": "SPATIAL_ZM_OU_TRANSITION_AGGREGATE_COMPLETE",
        "n_runs": len(rows),
        "n_nine_clause_lfp_pass": sum(
            row["nine_clause_lfp_gate_pass"] for row in rows),
        "n_full_gate_pass": sum(row["fig5a_full_gate_pass"] for row in rows),
        "gate_note": (
            "the nine LFP clauses score the shape of the detrended contact "
            "spectrum; criterion 10 scores how much the population firing rate "
            "actually moves. Both must pass for Fig5A."),
        "selection_used_image_pixels": False,
        "records": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "records"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
