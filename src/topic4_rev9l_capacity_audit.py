"""Derived audits for the rev9-L repeated-event finite-library oracle.

Every fact reported about the L3b negative result is recomputed here from the
frozen oracle payload instead of being transcribed by hand. Three questions are
separated because they carry different scientific weight:

* how far the whole library sits from patient-training variability;
* whether a descriptor was never improved or is simply inert;
* how much of a descriptor's error is structurally unreachable for the tested
  edge family, and therefore cannot be attributed to it.
"""
from __future__ import annotations

import numpy as np

from src.topic4_component_pair_search import DESCRIPTOR_NAMES, score_candidate


MODES = ("A", "B")


def patient_equivalent_objective(floor, *, readable_weight, tau, ood_weight):
    """Objective a fully patient-like model would score against its own floor.

    Reuses ``score_candidate`` so the reference shares the exact aggregation
    path (softplus, weakest-mode log-sum-exp, OOD weight) with the model rows.
    A synthetic model that reproduces the floor centre scores the median
    reference; one sitting on the floor's 95th percentile scores the q95
    reference. Together they bound the band a patient subsample occupies.
    """
    modes = floor["modes"]
    references = {}
    for key in ("median", "q95"):
        descriptors = {
            "modes": {
                mode: {name: float(modes[mode][name][key])
                       for name in DESCRIPTOR_NAMES}
                for mode in MODES
            }
        }
        references[key] = score_candidate(
            descriptors, floor,
            {mode: 1.0 for mode in MODES}, {mode: 0.0 for mode in MODES},
            readable_weight=readable_weight, tau=tau, ood_weight=ood_weight)
    return {
        "floor_median_objective": float(references["median"]["objective"]),
        "floor_q95_objective": float(references["q95"]["objective"]),
        "mode_scores_at_floor_q95": {
            mode: float(value)
            for mode, value in references["q95"]["mode_scores"].items()},
        "interpretation": (
            "objective a patient-training subsample of the same event count "
            "would score at its own floor centre and 95th percentile"),
    }


def _raw(row, mode, name):
    return float(row["mode_descriptors"]["modes"][mode][name])


def library_descriptor_extremes(payload, *, baseline_id="sobol_000",
                                tolerance=1e-12):
    """Separate "never improved below baseline" from "inert descriptor"."""
    rows = payload["candidate_network_rows"]
    baseline = {
        (row["candidate_id"], int(row["network_seed"])): row for row in rows
        if row["candidate_id"] == baseline_id
    }
    if not baseline:
        raise ValueError("descriptor extremes need the baseline candidate")
    summary = {}
    for mode in MODES:
        summary[mode] = {}
        for name in DESCRIPTOR_NAMES:
            values = np.asarray([_raw(row, mode, name) for row in rows], float)
            baseline_values = np.asarray(
                [_raw(row, mode, name) for row in baseline.values()], float)
            minimum = float(values.min())
            baseline_is_min = bool(
                np.all(np.abs(baseline_values - minimum) <= tolerance))
            summary[mode][name] = {
                "minimum": minimum,
                "maximum": float(values.max()),
                "n_rows": int(values.size),
                "n_rows_at_minimum": int(
                    np.sum(np.abs(values - minimum) <= tolerance)),
                "n_rows_worse_than_baseline_minimum": int(
                    np.sum(values > minimum + tolerance)),
                "n_distinct_values": int(
                    np.unique(np.round(values, 12)).size),
                "baseline_equals_library_minimum_in_every_network":
                    baseline_is_min,
                "statement": (
                    "never improved below the scalar baseline; the library "
                    "only ever degraded it" if baseline_is_min
                    else "improved below the scalar baseline somewhere"),
            }
    return summary


def recruitment_reachability(payload):
    """Contacts no candidate/network/mode ever recruits in the forced assay.

    A contact the assay can never reach caps the recruitment error at a value
    no edge parameter can lower, so its share of the error is not evidence
    about the edge family.
    """
    rows = payload["candidate_network_rows"]
    model = {
        mode: np.asarray([
            row["mode_descriptors"]["modes"][mode]["model_recruitment_probability"]
            for row in rows], float)
        for mode in MODES
    }
    patient = {}
    for mode in MODES:
        values = np.asarray([
            row["mode_descriptors"]["modes"][mode]["patient_recruitment_probability"]
            for row in rows], float)
        if not np.allclose(values, values[0], atol=0.0, rtol=0.0):
            raise ValueError("patient recruitment reference changed across rows")
        patient[mode] = values[0]

    never_any = np.ones(patient["A"].shape, bool)
    result = {"modes": {}}
    for mode in MODES:
        never = model[mode].max(axis=0) <= 0.0
        never_any &= never
        best = model[mode][np.argmin([
            _raw(row, mode, "recruitment_mean_absolute_error") for row in rows])]
        error = np.abs(best - patient[mode])
        total = float(error.sum())
        result["modes"][mode] = {
            "n_contacts": int(patient[mode].size),
            "never_recruited_contact_indices": np.flatnonzero(never).tolist(),
            "n_never_recruited": int(never.sum()),
            "patient_recruitment_at_never_recruited": [
                float(value) for value in patient[mode][never]],
            "patient_recruitment_probability": [
                float(value) for value in patient[mode]],
            "best_model_recruitment_probability": [
                float(value) for value in best],
            "best_recruitment_mean_absolute_error": float(
                total / patient[mode].size),
            "share_of_best_error_from_never_recruited": (
                float(error[never].sum() / total) if total > 0.0 else 0.0),
            "best_recruitment_error_if_never_recruited_were_matched": float(
                error[~never].sum() / patient[mode].size),
        }
    result["never_recruited_in_both_modes"] = np.flatnonzero(never_any).tolist()
    result["interpretation"] = (
        "contacts listed here are unreachable for every tested parameter, so "
        "their share of the recruitment error is a forced-assay readout limit, "
        "not evidence about the component-pair edge family")
    return result


def descriptor_support_audit(payload):
    """Model and patient support behind precedence and rank-profile errors.

    The patient floor is built from draws that recruit almost every contact,
    while the model omits the contacts it never reaches. Model shape errors are
    therefore averaged over an easier, smaller support than the floor they are
    standardized against, which biases them downward.
    """
    rows = payload["candidate_network_rows"]
    summary = {}
    for mode in MODES:
        profile = np.asarray([
            np.isfinite(np.asarray(
                row["mode_descriptors"]["modes"][mode]["model_mean_normalized_rank"],
                float)).sum() for row in rows], int)
        patient_profile = np.asarray([
            np.isfinite(np.asarray(
                row["mode_descriptors"]["modes"][mode]["patient_mean_normalized_rank"],
                float)).sum() for row in rows], int)
        pairs = np.asarray([
            row["mode_descriptors"]["modes"][mode]["model_precedence_pairs_with_support"]
            for row in rows], int)
        patient_pairs = np.asarray([
            row["mode_descriptors"]["modes"][mode]["patient_precedence_pairs_with_support"]
            for row in rows], int)
        summary[mode] = {
            "model_profile_contacts": [int(profile.min()), int(profile.max())],
            "patient_profile_contacts": [
                int(patient_profile.min()), int(patient_profile.max())],
            "model_precedence_pairs": [int(pairs.min()), int(pairs.max())],
            "patient_precedence_pairs": [
                int(patient_pairs.min()), int(patient_pairs.max())],
            "model_support_is_smaller": bool(
                profile.max() < patient_profile.min()
                or pairs.max() < patient_pairs.min()),
        }
    summary["bias_direction"] = (
        "model shape errors are averaged over fewer contacts and pairs than "
        "the patient floor, which flatters the model; the reported negative is "
        "therefore conservative")
    return summary


def descriptor_event_count_consistency(payload):
    """Descriptor event count must equal the count-matched floor it used."""
    mismatches = []
    for row in payload["candidate_network_rows"]:
        descriptors = row["mode_descriptors"]
        matched = row["score"].get("matched_floor_event_count_by_mode")
        if descriptors is None or matched is None:
            continue
        for mode in MODES:
            used = int(descriptors["modes"][mode]["n_model_events"])
            floor_count = int(matched[mode])
            if used != floor_count:
                mismatches.append({
                    "candidate_id": row["candidate_id"],
                    "network_seed": int(row["network_seed"]),
                    "mode": mode,
                    "n_model_events": used,
                    "matched_floor_event_count": floor_count,
                })
    return {
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "consistent": not mismatches,
    }


def per_network_mode_descriptor_ratios(payload, floors):
    """Both modes' descriptor errors relative to their count-matched q95."""
    rows = {
        (row["candidate_id"], int(row["network_seed"])): row
        for row in payload["candidate_network_rows"]
    }
    output = []
    for oracle_row in payload["oracle"]["per_network"]:
        seed = int(oracle_row["network_seed"])
        candidate_id = oracle_row["representative_candidate_id"]
        row = rows[(candidate_id, seed)]
        counts = row["score"]["matched_floor_event_count_by_mode"]
        record = {"network_seed": seed, "candidate_id": candidate_id, "modes": {}}
        for mode in MODES:
            count = int(counts[mode])
            if count not in floors:
                raise ValueError(f"missing mode-{mode} floor for n={count}")
            calibration = floors[count]["floor"]["modes"][mode]
            ratios = {
                name: float(_raw(row, mode, name) / calibration[name]["q95"])
                for name in DESCRIPTOR_NAMES
            }
            shape = [name for name in DESCRIPTOR_NAMES
                     if name != "recruitment_mean_absolute_error"]
            record["modes"][mode] = {
                "matched_floor_event_count": count,
                "raw_over_q95": ratios,
                "n_descriptors_above_q95": int(
                    sum(value > 1.0 for value in ratios.values())),
                "n_shape_descriptors_above_q95": int(
                    sum(ratios[name] > 1.0 for name in shape)),
                "recruitment_above_q95": bool(
                    ratios["recruitment_mean_absolute_error"] > 1.0),
            }
        output.append(record)
    return output


def audit_finite_library_capacity(payload, floors, *, objective,
                                  baseline_id="sobol_000"):
    """Assemble the derived facts the L3b negative conclusion depends on."""
    counts = sorted({
        int(row["score"]["matched_floor_event_count_by_mode"][mode])
        for row in payload["candidate_network_rows"]
        if row["score"].get("matched_floor_event_count_by_mode")
        for mode in MODES
    })
    reference = {
        f"n{count}": patient_equivalent_objective(
            floors[count]["floor"],
            readable_weight=objective["readable_fraction_penalty_weight"],
            tau=objective["weakest_mode_lse_tau"],
            ood_weight=objective["ood_weight"])
        for count in counts if count in floors
    }
    ratios = per_network_mode_descriptor_ratios(payload, floors)
    modal = reference[f"n{max(counts)}"]
    values = np.asarray([
        row["score"]["objective"] for row in payload["candidate_network_rows"]
    ], float)
    gap = float(np.median(values) - modal["floor_q95_objective"])
    return {
        "patient_equivalent_objective": reference,
        "library_objective_median": float(np.median(values)),
        "library_objective_minimum": float(values.min()),
        "median_gap_above_patient_q95_objective": gap,
        "descriptor_extremes": library_descriptor_extremes(
            payload, baseline_id=baseline_id),
        "recruitment_reachability": recruitment_reachability(payload),
        "descriptor_support": descriptor_support_audit(payload),
        "descriptor_event_count_consistency":
            descriptor_event_count_consistency(payload),
        "per_network_mode_ratios": ratios,
        "delta_network_is_non_negative_by_construction": True,
        "delta_network_noise_null_tested": False,
        "per_network_oracle_gain_is_library_minimum": (
            "the library contains the scalar baseline, so a per-network "
            "minimum can never exceed it; the count of improved networks is "
            "not an independent finding, only the gain magnitude is"),
    }


__all__ = [
    "audit_finite_library_capacity",
    "descriptor_event_count_consistency",
    "descriptor_support_audit",
    "library_descriptor_extremes",
    "patient_equivalent_objective",
    "per_network_mode_descriptor_ratios",
    "recruitment_reachability",
]
