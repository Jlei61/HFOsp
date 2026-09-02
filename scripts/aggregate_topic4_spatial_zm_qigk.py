#!/usr/bin/env python3
"""Aggregate spatial Z/qI--M runs without inspecting rendered figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The archived frozen-q / canary artifacts and the persistent-OU transition
# artifacts share this schema, so one selection path serves both.
ACCEPTED_STATUSES = (
    "SPATIAL_ZQIM_HYBRID_CANARY_COMPLETE",
    "SPATIAL_ZM_OU_TRANSITION_COMPLETE",
)


def _compact(path):
    payload = json.loads(path.read_text())
    classification = payload.get("classification") or {}
    checks = classification.get("checks") or {}
    rhythm = payload.get("contact_rhythm") or {}
    recruitment = payload.get("global_recruitment") or {}
    rates = payload.get("state_rate") or {}
    config = payload.get("hybrid_config") or {}
    full_edge = payload.get("full_edge_contract") or {}
    spatial_basis = payload.get("spatial_basis_contract") or {}
    parameter_contract = {
        "candidate_id": payload.get("candidate_id"),
        "mode": payload.get("mode"),
        "full_edge_contract": full_edge,
        "hybrid_config": config,
        "spatial_basis_contract": spatial_basis,
        "protocol_contract": payload.get("protocol_contract") or {},
    }
    parameter_contract_sha256 = hashlib.sha256(
        json.dumps(parameter_contract, sort_keys=True,
                   separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    tonic = payload.get("criterion10_tonic_exclusion") or {}
    tonic_detail = (tonic.get("detail") or {}).get("high_state") or {}
    # Criterion 10 is part of the Fig5A gate.  A persistent-OU transition
    # artifact must carry it and pass it -- a missing field there means the run
    # predates the clause, not that the clause does not apply, so it fails
    # closed.  The archived frozen-q canaries never had the clause and keep
    # their historical nine-clause meaning.
    nine_clause_pass = bool(classification.get("all_checks_pass", False))
    is_transition = payload.get("status") == "SPATIAL_ZM_OU_TRANSITION_COMPLETE"
    criterion10 = tonic.get("all_checks_pass", None)
    full_gate_pass = bool(
        nine_clause_pass
        and (criterion10 is True if is_transition
             else criterion10 is not False))
    ou_applied = payload.get("applied_spatial_ou") or {}
    ou_runtime = payload.get("ou_runtime_evidence") or {}
    ou_stationarity = payload.get("ou_stationarity_across_transition") or {}
    stability = payload.get("numerical_stability") or {}
    return {
        "path": str(path.relative_to(ROOT)),
        "seed": payload.get("seed"),
        "mode": payload.get("mode"),
        "run_role": payload.get("run_role", "discovery"),
        "parameter_set_id": payload.get("parameter_set_id"),
        "parameter_contract_sha256": parameter_contract_sha256,
        "verdict": payload.get("verdict"),
        "all_checks_pass": full_gate_pass,
        "nine_clause_lfp_gate_pass": nine_clause_pass,
        "criterion10_tonic_exclusion_pass": tonic.get("all_checks_pass"),
        "population_rate_modulation_depth": tonic_detail.get(
            "modulation_depth"),
        "population_rate_dominant_hz": tonic_detail.get("dominant_hz"),
        "n_checks_pass": int(sum(bool(value) for value in checks.values())),
        "n_checks": int(len(checks)),
        "failed_checks": sorted(key for key, value in checks.items() if not value),
        "scientific_onset_ms": payload.get("scientific_onset_ms"),
        "k_q_per_ms": config.get("k_q_per_ms"),
        "q_min": config.get("q_min"),
        "q_init": config.get("q_init", 1.0),
        "q_init_h_gain": config.get("q_init_h_gain", 0.0),
        "q_endpoint_gain": config.get("q_endpoint_gain", 0.0),
        "q_source_gain": config.get("q_source_gain", 0.0),
        "q_sink_gain": config.get("q_sink_gain", 0.0),
        "q_support_gain": config.get("q_support_gain", 0.0),
        "q_endpoint_sigma_mm": config.get("q_endpoint_sigma_mm", 2.0),
        "q_endpoint_side": spatial_basis.get("active_endpoint_side", "union"),
        "freeze_q": config.get("freeze_q", False),
        "q_a0": config.get("q_a0"),
        "q_a50": config.get("q_a50"),
        "q_hill_n": config.get("q_hill_n", 1.0),
        "q_floor_h_gain": config.get("q_floor_h_gain"),
        "k_q_h_gain": config.get("k_q_h_gain"),
        "k_q_source_gain": config.get("k_q_source_gain", 0.0),
        "k_q_sink_gain": config.get("k_q_sink_gain", 0.0),
        "k_q_support_gain": config.get("k_q_support_gain", 0.0),
        "tau_m_ms": config.get("tau_m_ms"),
        "m_build_gain": config.get("m_build_gain", 1.0),
        "eta_m": config.get("eta_m"),
        "m_current_threshold": config.get("m_current_threshold", 0.0),
        "m_current_saturation_width": config.get(
            "m_current_saturation_width", 0.0),
        "m_current_hill_n": config.get("m_current_hill_n", 1.0),
        "m_state_ceiling": config.get("m_state_ceiling", 0.0),
        "m_spatial_mix": config.get("m_spatial_mix", 0.0),
        "sigma_m_mm": config.get("sigma_m_mm"),
        "eta_m_h_gain": config.get("eta_m_h_gain"),
        "eta_m_source_add": config.get("eta_m_source_add", 0.0),
        "eta_m_sink_add": config.get("eta_m_sink_add", 0.0),
        "eta_m_gk_add": config.get("eta_m_gk_add", 0.0),
        "gk_support_sigma_mm": config.get("gk_support_sigma_mm", 0.0),
        "gk_support": (spatial_basis.get("gk_support") or {}).get("rule"),
        "full_edge": bool(
            full_edge.get("E_to_E_dose") == 1.0
            and full_edge.get("E_to_I_dose") == 1.0
            and full_edge.get("learned_edges_modified") is False),
        "median_rate_pre_hz": rates.get("median_pre_hz"),
        "median_rate_post_hz": rates.get("median_post_hz"),
        "joint_global_recruitment_duty": recruitment.get(
            "joint_global_recruitment_duty"),
        "contact_fraction_consistently_rhythmic": rhythm.get(
            "contact_fraction_consistently_rhythmic"),
        "median_contact_peak_hz": rhythm.get("median_contact_peak_hz"),
        "contact_peak_mad_hz": rhythm.get("contact_peak_mad_hz"),
        "median_peak_power_fraction": rhythm.get("median_peak_power_fraction"),
        "median_band_power_ratio_post_over_pre": rhythm.get(
            "median_band_power_ratio_post_over_pre"),
        "n_rhythmic_contacts": payload.get("n_rhythmic_contacts"),
        "ou_sigma_rate_per_ms": ou_applied.get("sigma_rate_per_ms"),
        "ou_tau_ms": ou_applied.get("tau_ms"),
        "ou_ell_mm": ou_applied.get("ell_mm"),
        "ou_seed_offset": ou_applied.get("seed_offset"),
        "ou_called_every_membrane_step": ou_runtime.get(
            "called_every_membrane_step"),
        "ou_sd_ratio_after_over_before": ou_stationarity.get(
            "sd_ratio_after_over_before"),
        "numerically_stable": stability.get("all_checks_pass"),
        "ou_realisation_sha256": payload.get("ou_realisation_sha256"),
    }


def _rank(row):
    def value(name, default):
        item = row.get(name)
        return default if item is None else float(item)
    return (
        -int(row["all_checks_pass"] and row["mode"] == "hybrid" and row["full_edge"]),
        -int(row["all_checks_pass"]),
        -int(row["n_checks_pass"]),
        -value("contact_fraction_consistently_rhythmic", -1.0),
        value("contact_peak_mad_hz", 1e9),
        -value("joint_global_recruitment_duty", -1.0),
        -value("median_peak_power_fraction", -1.0),
        str(row["path"]),
    )


def _stationary_noise_evidence(row):
    """A confirmation record must prove the noise ran and did not change.

    ``None`` means the field was never recorded, which is exactly the situation
    the runtime audit exists to rule out, so it fails rather than defaults.
    """
    ratio = row.get("ou_sd_ratio_after_over_before")
    return {
        "ou_called_every_membrane_step": row.get(
            "ou_called_every_membrane_step") is True,
        "ou_sd_stationary_across_transition": (
            ratio is not None and 0.9 <= float(ratio) <= 1.1),
        "numerically_stable": row.get("numerically_stable") is True,
    }


def _confirmation_families(rows, minimum_seeds):
    grouped = defaultdict(list)
    for row in rows:
        if (row["run_role"] == "confirmation"
                and row["parameter_set_id"]
                and row["mode"] == "hybrid"
                and row["full_edge"]):
            grouped[row["parameter_set_id"]].append(row)
    families = []
    for parameter_set_id, records in grouped.items():
        records = sorted(records, key=lambda row: (int(row["seed"]), row["path"]))
        seeds = sorted({int(row["seed"]) for row in records})
        passed_seeds = sorted({int(row["seed"]) for row in records
                               if row["all_checks_pass"]})
        contract_sha256 = sorted({str(row["parameter_contract_sha256"])
                                  for row in records})
        single_frozen_config = len(contract_sha256) == 1
        all_completed_pass = bool(records) and all(
            row["all_checks_pass"] for row in records)
        evidence = [_stationary_noise_evidence(row) for row in records]
        all_noise_evidence_pass = bool(records) and all(
            all(item.values()) for item in evidence)
        eligible = (
            len(seeds) >= int(minimum_seeds)
            and all_completed_pass
            and single_frozen_config
            and all_noise_evidence_pass
        )
        families.append({
            "parameter_set_id": parameter_set_id,
            "n_unique_seeds": len(seeds),
            "seeds": seeds,
            "n_passed_seeds": len(passed_seeds),
            "passed_seeds": passed_seeds,
            "all_completed_seeds_pass": all_completed_pass,
            "all_seeds_stationary_noise_and_stable": all_noise_evidence_pass,
            "per_seed_noise_and_stability_evidence": evidence,
            "single_frozen_config": single_frozen_config,
            "parameter_contract_sha256": (
                contract_sha256[0] if single_frozen_config else None),
            "observed_parameter_contract_sha256": contract_sha256,
            "minimum_confirmation_seeds": int(minimum_seeds),
            "eligible_multi_seed_family": eligible,
            "records": records,
        })
    families.sort(key=lambda family: (
        -int(family["eligible_multi_seed_family"]),
        -int(family["n_passed_seeds"]),
        str(family["parameter_set_id"]),
    ))
    return families


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-confirmation-seeds", type=int, default=3)
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = ROOT / input_dir
    paths = sorted(input_dir.rglob("*.json"))
    rows = [_compact(path) for path in paths
            if json.loads(path.read_text()).get("status") in ACCEPTED_STATUSES]
    rows.sort(key=_rank)
    families = _confirmation_families(rows, args.minimum_confirmation_seeds)
    primary_family = next((family for family in families
                           if family["eligible_multi_seed_family"]), None)
    primary = None
    if primary_family is not None:
        # The representative is the passing seed whose transition time is
        # closest to the family median onset, fixed without viewing pixels.
        family_rows = [row for row in primary_family["records"]
                       if row["all_checks_pass"]
                       and row["scientific_onset_ms"] is not None]
        onsets = sorted(float(row["scientific_onset_ms"]) for row in family_rows)
        median_onset = float(onsets[len(onsets) // 2]) if len(onsets) % 2 else (
            0.5 * (onsets[len(onsets) // 2 - 1] + onsets[len(onsets) // 2]))
        primary = min(family_rows, key=lambda row: (
            abs(float(row["scientific_onset_ms"]) - median_onset),
            int(row["seed"])))
        primary_family["representative_selection"] = {
            "rule": ("passing confirmation seed whose scientific onset is "
                     "closest to the family median onset; ties broken by the "
                     "smaller seed"),
            "family_median_onset_ms": median_onset,
            "representative_onset_ms": float(primary["scientific_onset_ms"]),
            "image_pixels_used": False,
        }
    payload = {
        "status": "SPATIAL_ZQIM_HYBRID_AGGREGATE_COMPLETE",
        "selection_used_image_pixels": False,
        "n_runs": len(rows),
        "n_full_pass": sum(row["all_checks_pass"] for row in rows),
        "n_hybrid_full_edge_pass": sum(
            row["all_checks_pass"] and row["mode"] == "hybrid" and row["full_edge"]
            for row in rows),
        "minimum_confirmation_seeds": int(args.minimum_confirmation_seeds),
        "confirmation_families": families,
        "primary_hybrid_family": primary_family,
        "primary_hybrid_candidate": primary,
        "records": rows,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    csv_path = out.with_suffix(".csv")
    if rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "failed_checks": ";".join(row["failed_checks"])})
    print(json.dumps({
        "n_runs": len(rows),
        "n_hybrid_full_edge_pass": payload["n_hybrid_full_edge_pass"],
        "primary": None if primary is None else primary["path"],
        "out": str(out),
    }))


if __name__ == "__main__":
    main()
