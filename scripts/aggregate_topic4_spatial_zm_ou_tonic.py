#!/usr/bin/env python3
"""Aggregate the tonic-global-runaway endpoint for spatial Z/M + OU runs.

This is intentionally separate from the oscillatory Fig. 5A aggregate.  A run
passes here when it makes a persistent, near-saturated global plateau under the
unchanged stationary OU environment.  Frequency and modulation depth are not
selection variables.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(ROOT))

from src.topic4_global_recruited_oscillation import (  # noqa: E402
    TONIC_GLOBAL_RUNAWAY_THRESHOLDS,
    classify_global_tonic_runaway,
)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _full_edge(payload) -> bool:
    edge = payload.get("full_edge_contract") or {}
    return bool(
        edge.get("E_to_E_dose") == 1.0
        and edge.get("E_to_I_dose") == 1.0
        and edge.get("learned_edges_modified") is False
    )


def _contact_recruitment(payload):
    rows = payload.get("per_contact_diagnosis") or []
    passed = [
        bool(
            float(row.get("local_rate_post_hz", 0.0)) >= 120.0
            and float(row.get("local_rate_ratio_post_over_pre", 0.0)) >= 2.0
        )
        for row in rows
    ]
    return {
        "n_contacts": len(rows),
        "n_contacts_recruited": int(sum(passed)),
        "contact_fraction_recruited": (
            float(sum(passed) / len(passed)) if passed else None),
        "all_15_virtual_contacts_recruited": len(rows) == 15 and all(passed),
        "definition": "local post rate >=120 Hz and post/pre rate ratio >=2",
    }


def compact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rates = payload.get("state_rate") or {}
    recruitment = payload.get("global_recruitment") or {}
    onset = payload.get("scientific_onset_ms")
    duration = payload.get("trajectory_duration_ms")
    morphology = payload.get("tonic_global_runaway")
    required_rate_fields = {
        "median_pre_hz", "q95_pre_hz", "median_post_hz", "q05_post_hz",
        "median_ratio_post_over_pre",
    }
    required_recruitment_fields = {
        "median_active_neuron_fraction_20ms",
        "median_recruited_spatial_fraction_1mm",
        "joint_global_recruitment_duty",
    }
    if (morphology is None and onset is not None and duration is not None
            and required_rate_fields.issubset(rates)
            and required_recruitment_fields.issubset(recruitment)):
        morphology = classify_global_tonic_runaway(
            onset_ms=float(onset),
            observed_post_transition_ms=float(duration) - float(onset),
            rates=rates,
            recruitment=recruitment,
        )

    config = payload.get("hybrid_config") or {}
    parameter_contract = {
        "candidate_id": payload.get("candidate_id"),
        "mode": payload.get("mode"),
        "full_edge_contract": payload.get("full_edge_contract") or {},
        "hybrid_config": config,
        "applied_spatial_ou": payload.get("applied_spatial_ou") or {},
        "protocol_contract": payload.get("protocol_contract") or {},
        "tonic_morphology_thresholds": (
            (payload.get("tonic_global_runaway") or {}).get("thresholds")
            or TONIC_GLOBAL_RUNAWAY_THRESHOLDS),
    }
    runtime = payload.get("ou_runtime_evidence") or {}
    stationarity = payload.get("ou_stationarity_across_transition") or {}
    stability = payload.get("numerical_stability") or {}
    ratio = stationarity.get("sd_ratio_after_over_before")
    contact = _contact_recruitment(payload)
    execution_checks = {
        "full_learned_EE_and_EI": _full_edge(payload),
        "hybrid_ZM_mode": payload.get("mode") == "hybrid",
        "OU_called_every_membrane_step": (
            runtime.get("called_every_membrane_step") is True),
        "OU_stationary_across_transition": (
            ratio is not None and 0.9 <= float(ratio) <= 1.1),
        "numerically_stable": stability.get("all_checks_pass") is True,
        "all_15_virtual_contacts_recruited": contact[
            "all_15_virtual_contacts_recruited"],
    }
    morphology_pass = bool(morphology and morphology.get("all_checks_pass"))
    all_pass = morphology_pass and all(execution_checks.values())
    observed = (morphology or {}).get("observed") or {}
    return {
        "path": str(path.relative_to(ROOT)),
        "seed": payload.get("seed"),
        "mode": payload.get("mode"),
        "run_role": payload.get("run_role"),
        "parameter_set_id": payload.get("parameter_set_id"),
        "parameter_contract_sha256": _sha256_json(parameter_contract),
        "scientific_onset_ms": onset,
        "trajectory_duration_ms": duration,
        "morphology_pass": morphology_pass,
        "all_checks_pass": bool(all_pass),
        "morphology_status": (
            "UNSCORABLE_MISSING_STATE_METRICS" if morphology is None
            else morphology.get("status")),
        "failed_morphology_checks": (
            ["UNSCORABLE_MISSING_STATE_METRICS"] if morphology is None else sorted(
                key for key, value in (morphology.get("checks") or {}).items()
                if not value)),
        "failed_execution_checks": sorted(
            key for key, value in execution_checks.items() if not value),
        "median_rate_pre_hz": rates.get("median_pre_hz"),
        "q95_rate_pre_hz": rates.get("q95_pre_hz"),
        "median_rate_post_hz": rates.get("median_post_hz"),
        "q05_rate_post_hz": rates.get("q05_post_hz"),
        "median_rate_ratio_post_over_pre": rates.get(
            "median_ratio_post_over_pre"),
        "median_active_neuron_fraction_20ms": recruitment.get(
            "median_active_neuron_fraction_20ms"),
        "median_recruited_spatial_fraction_1mm": recruitment.get(
            "median_recruited_spatial_fraction_1mm"),
        "joint_global_recruitment_duty": recruitment.get(
            "joint_global_recruitment_duty"),
        "observed_post_transition_ms": observed.get(
            "observed_post_transition_ms"),
        **contact,
        "k_q_per_ms": config.get("k_q_per_ms"),
        "q_min": config.get("q_min"),
        "q_a50": config.get("q_a50"),
        "q_hill_n": config.get("q_hill_n"),
        "tau_m_ms": config.get("tau_m_ms"),
        "eta_m": config.get("eta_m"),
        "m_spatial_mix": config.get("m_spatial_mix"),
        "OU_called_every_membrane_step": execution_checks[
            "OU_called_every_membrane_step"],
        "OU_sd_ratio_after_over_before": ratio,
        "numerically_stable": execution_checks["numerically_stable"],
    }


def _rank(row):
    def number(name, fallback):
        value = row.get(name)
        return fallback if value is None else float(value)

    return (
        -int(row["all_checks_pass"]),
        -number("joint_global_recruitment_duty", -1.0),
        -number("median_active_neuron_fraction_20ms", -1.0),
        number("eta_m", 1e9),
        number("m_spatial_mix", 1e9),
        -number("q05_rate_post_hz", -1.0),
        str(row["path"]),
    )


def confirmation_families(rows, minimum_seeds=3):
    grouped = defaultdict(list)
    for row in rows:
        if (row.get("run_role") == "confirmation"
                and row.get("parameter_set_id")
                and row.get("mode") == "hybrid"):
            grouped[row["parameter_set_id"]].append(row)
    families = []
    for parameter_set_id, records in grouped.items():
        records = sorted(records, key=lambda row: (int(row["seed"]), row["path"]))
        seeds = sorted({int(row["seed"]) for row in records})
        passed = sorted({int(row["seed"]) for row in records
                         if row["all_checks_pass"]})
        hashes = sorted({row["parameter_contract_sha256"] for row in records})
        eligible = bool(
            len(seeds) >= int(minimum_seeds)
            and len(passed) == len(seeds)
            and len(hashes) == 1
        )
        families.append({
            "parameter_set_id": parameter_set_id,
            "n_unique_seeds": len(seeds),
            "seeds": seeds,
            "n_passed_seeds": len(passed),
            "passed_seeds": passed,
            "minimum_confirmation_seeds": int(minimum_seeds),
            "single_frozen_config": len(hashes) == 1,
            "parameter_contract_sha256": hashes[0] if len(hashes) == 1 else None,
            "eligible_multi_seed_family": eligible,
            "records": records,
        })
    families.sort(key=lambda family: (
        -int(family["eligible_multi_seed_family"]),
        -int(family["n_passed_seeds"]),
        str(family["parameter_set_id"]),
    ))
    return families


def representative_from_family(family):
    passing = [row for row in family["records"] if row["all_checks_pass"]]
    if not passing:
        return None
    ordered_onsets = sorted(float(row["scientific_onset_ms"]) for row in passing)
    middle = len(ordered_onsets) // 2
    median = (ordered_onsets[middle] if len(ordered_onsets) % 2
              else 0.5 * (ordered_onsets[middle - 1] + ordered_onsets[middle]))
    return min(passing, key=lambda row: (
        abs(float(row["scientific_onset_ms"]) - median), int(row["seed"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-confirmation-seeds", type=int, default=3)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    rows = []
    for path in sorted(input_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "SPATIAL_ZM_OU_TRANSITION_COMPLETE":
            rows.append(compact(path))
    if not rows:
        raise RuntimeError("no completed spatial Z/M + OU transition artifacts")
    rows.sort(key=_rank)
    families = confirmation_families(rows, args.minimum_confirmation_seeds)
    primary_family = next((item for item in families
                           if item["eligible_multi_seed_family"]), None)
    primary_confirmation = (
        None if primary_family is None else representative_from_family(primary_family))
    primary_discovery = next((row for row in rows
                              if row["all_checks_pass"]
                              and row["run_role"] == "discovery"), None)
    payload = {
        "status": "SPATIAL_ZM_OU_TONIC_RUNAWAY_AGGREGATE_COMPLETE",
        "endpoint": "persistent near-saturated tonic global runaway",
        "endpoint_does_not_require": [
            "30-80 Hz contact rhythm",
            "deep population-rate modulation",
        ],
        "tonic_morphology_thresholds": TONIC_GLOBAL_RUNAWAY_THRESHOLDS,
        "virtual_contact_recruitment_thresholds": {
            "minimum_local_post_rate_hz": 120.0,
            "minimum_local_post_over_pre_rate_ratio": 2.0,
            "required_contacts": "15/15",
        },
        "selection_used_image_pixels": False,
        "discovery_selection_rule": (
            "full tonic/execution gate, then maximum global duty and active "
            "fraction, then the weakest M coupling and spatial mixing; image "
            "pixels and contact rhythm are never inspected"),
        "n_runs": len(rows),
        "n_morphology_pass": sum(row["morphology_pass"] for row in rows),
        "n_full_execution_pass": sum(row["all_checks_pass"] for row in rows),
        "minimum_confirmation_seeds": int(args.minimum_confirmation_seeds),
        "confirmation_families": families,
        "primary_confirmation_family": primary_family,
        "primary_confirmation_candidate": primary_confirmation,
        "primary_discovery_candidate": primary_discovery,
        "records": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    with out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **row,
                "failed_morphology_checks": ";".join(
                    row["failed_morphology_checks"]),
                "failed_execution_checks": ";".join(
                    row["failed_execution_checks"]),
            })
    print(json.dumps({
        "n_runs": payload["n_runs"],
        "n_morphology_pass": payload["n_morphology_pass"],
        "n_full_execution_pass": payload["n_full_execution_pass"],
        "primary_discovery": (
            None if primary_discovery is None else primary_discovery["path"]),
        "primary_confirmation": (
            None if primary_confirmation is None else primary_confirmation["path"]),
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
