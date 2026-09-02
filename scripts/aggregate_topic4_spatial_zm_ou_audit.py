#!/usr/bin/env python3
"""Apply the frozen Stage A rule and freeze one OU working point.

The selection rule was declared before any run: take the declared baseline rung
if it qualifies; otherwise take the qualifying rung with the smallest amplitude,
breaking ties toward the declared correlation length.  Only slow-off low-state
clauses enter the decision -- no post-transition rhythm score is read here, and
none is available in these artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE_SIGMA = 0.10
BASELINE_ELL = 0.38


def _row(path: Path):
    payload = json.loads(path.read_text())
    ou = payload["applied_spatial_ou"]
    runtime = payload["ou_runtime_evidence"]
    stationarity = payload["ou_stationarity"]
    clauses = payload["low_state_qualification"]["clauses"]
    return {
        "path": str(path.relative_to(ROOT)),
        "seed": payload["seed"],
        "duration_ms": payload["duration_ms"],
        "sigma_rate_per_ms": ou["sigma_rate_per_ms"],
        "tau_ms": ou["tau_ms"],
        "ell_mm": ou["ell_mm"],
        "called_every_membrane_step": runtime["called_every_membrane_step"],
        "n_step_calls": runtime["n_step_calls"],
        "n_field_updates_recorded": runtime["n_field_updates_recorded"],
        "measured_sd_rate_per_ms": stationarity["whole_run"]["sd_rate_per_ms"],
        "measured_mean_rate_per_ms": stationarity["whole_run"]["mean_rate_per_ms"],
        "sd_ratio_after_over_before": stationarity.get(
            "sd_ratio_after_over_before"),
        "measured_tau_ms": payload["ou_measured_temporal"]["tau_hat_ms"],
        "measured_correlation_length_mm": payload["ou_measured_spatial"][
            "correlation_length_mm_1_over_e"],
        "negative_rate_clip_fraction": payload["ou_negative_rate_clipping"][
            "negative_rate_clip_fraction"],
        "onset_ms": clauses["no_sustained_global_high_state"]["onset_ms"],
        "median_rate_hz": clauses["low_state_rate_bounded"]["median_hz"],
        "q95_rate_hz": clauses["low_state_rate_bounded"]["q95_hz"],
        "joint_global_recruitment_duty": clauses[
            "intermittent_not_continuous_recruitment"][
            "joint_global_recruitment_duty"],
        "n_detected_events": clauses["enough_readable_events"][
            "n_detected_events"],
        "n_returned_events": clauses["enough_readable_events"][
            "n_returned_events"],
        "all_clauses_pass": payload["low_state_qualification"][
            "all_clauses_pass"],
    }


def runtime_certified(row, *, tau_tolerance=0.25, length_tolerance=0.30):
    """The measured field must match what the config declared.

    A Gaussian smoothing kernel of width ``ell`` produces an autocorrelation
    whose 1/e crossing sits near ``2*ell``; that identity, not ``ell`` itself,
    is the correct target for the measured length.
    """
    declared_tau = float(row["tau_ms"])
    declared_length = 2.0 * float(row["ell_mm"])
    checks = {
        "stepped_every_membrane_step": bool(row["called_every_membrane_step"]),
        "sd_matches_declared_amplitude": bool(
            abs(float(row["measured_sd_rate_per_ms"])
                - float(row["sigma_rate_per_ms"]))
            <= 0.15 * float(row["sigma_rate_per_ms"])),
        "field_is_zero_mean": bool(
            abs(float(row["measured_mean_rate_per_ms"])) <= 1e-6),
        "tau_matches_declared": bool(
            abs(float(row["measured_tau_ms"]) - declared_tau)
            <= tau_tolerance * declared_tau),
        "correlation_length_matches_declared": bool(
            abs(float(row["measured_correlation_length_mm"]) - declared_length)
            <= length_tolerance * declared_length),
        "stationary_across_split": bool(
            row["sd_ratio_after_over_before"] is None
            or 0.9 <= float(row["sd_ratio_after_over_before"]) <= 1.1),
    }
    return {"all_pass": bool(all(checks.values())), "checks": checks}


def select_working_point(rows, *, minimum_passing_seeds=2):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(float(row["sigma_rate_per_ms"]), float(row["tau_ms"]),
                 float(row["ell_mm"]))].append(row)
    rungs = []
    for (sigma, tau, ell), records in sorted(grouped.items()):
        certified = [runtime_certified(row) for row in records]
        n_pass = sum(bool(row["all_clauses_pass"]) for row in records)
        rungs.append({
            "sigma_rate_per_ms": sigma, "tau_ms": tau, "ell_mm": ell,
            "n_seeds": len(records),
            "n_seeds_low_state_eligible": n_pass,
            "n_seeds_runtime_certified": sum(
                item["all_pass"] for item in certified),
            "runtime_certification": certified,
            "qualifies": bool(n_pass >= int(minimum_passing_seeds)
                              and all(item["all_pass"] for item in certified)),
            "is_declared_baseline": bool(
                abs(sigma - BASELINE_SIGMA) < 1e-12
                and abs(ell - BASELINE_ELL) < 1e-12),
            "seeds": sorted(int(row["seed"]) for row in records),
        })
    qualifying = [rung for rung in rungs if rung["qualifies"]]
    baseline = next((rung for rung in qualifying
                     if rung["is_declared_baseline"]), None)
    if baseline is not None:
        selected, reason = baseline, "declared baseline qualifies"
    elif qualifying:
        selected = min(qualifying, key=lambda rung: (
            rung["sigma_rate_per_ms"], abs(rung["ell_mm"] - BASELINE_ELL)))
        reason = "baseline failed; lowest qualifying amplitude taken"
    else:
        selected, reason = None, "no rung qualified"
    return {"rungs": rungs, "selected": selected, "selection_reason": reason,
            "minimum_passing_seeds": int(minimum_passing_seeds)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-passing-seeds", type=int, default=2)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    rows = [_row(path) for path in sorted(input_dir.glob("*.json"))
            if json.loads(path.read_text()).get("status")
            == "SPATIAL_ZM_OU_AUDIT_COMPLETE"]
    if not rows:
        raise RuntimeError("no completed Stage A artifacts found")
    decision = select_working_point(
        rows, minimum_passing_seeds=args.minimum_passing_seeds)
    payload = {
        "status": "SPATIAL_ZM_OU_AUDIT_AGGREGATE_COMPLETE",
        "selection_rule": (
            "declared baseline if it qualifies, else the lowest-amplitude "
            "qualifying rung; only slow-off low-state clauses and OU runtime "
            "certification enter this decision"),
        "post_transition_rhythm_used_in_selection": False,
        "n_runs": len(rows),
        **decision,
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
    print(json.dumps({
        "n_runs": len(rows),
        "selected": decision["selected"] and {
            key: decision["selected"][key] for key in
            ("sigma_rate_per_ms", "tau_ms", "ell_mm",
             "n_seeds_low_state_eligible", "n_seeds_runtime_certified")},
        "selection_reason": decision["selection_reason"],
        "out": str(out),
    }, indent=1))


if __name__ == "__main__":
    main()
