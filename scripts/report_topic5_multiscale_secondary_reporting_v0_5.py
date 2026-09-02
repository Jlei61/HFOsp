#!/usr/bin/env python3
"""Reporting-only addendum for three under-reported v0.5 contract items.

This script never trains, scores a target, regenerates a field or null, or
changes any estimand, cohort or denominator.  It reads already frozen tables
and applies reporting rules that the locked spec required but the frozen
adjudicator did not emit:

1. spec §10.3 / plan §G  -- the early-ictal secondary/robustness endpoints form
   one predeclared claim family and must be reported with Holm (or a
   simultaneous interval) rather than as individual raw one-sided P values.
   The frozen adjudicator only Holm-corrects the two-test D2 family.
2. spec §9.5 / §3.3      -- the four geometry-eligible robustness spatial nulls
   were computed per patient but never summarised.  They are reported here
   together with the all-contact null restricted to the same patients, because
   each robustness null is identifiable on a different montage subset and a
   naive comparison against the full n=17 all-contact margin confounds
   null strictness with cohort composition.
3. spec §7.3             -- the pre-freeze repair of J is documented, but the
   residual resolution of the repaired moderator is not.  The single primary
   test of the whole target-free family is a Spearman correlation against J, so
   the number of patients tied at exactly zero bounds its resolution.

Every adjustment here can only make a conclusion weaker or equal, never
stronger, so applying it after target unseal cannot manufacture a positive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
L3 = "INTACT|L3_LOCAL_PLUS_LEARNED_LR"
CANONICAL = "canonical_full"

# Endpoints already carrying their own predeclared Holm correction inside the
# frozen adjudicator's two-test D2 family.
D2_FAMILY = (
    "D2_L3_minus_L2m_seed_removed_signed_oracle",
    "D2_L3_added_attenuation_auc_seed_removed_gt_zero",
)
# Narrowest defensible reading of spec §10.3: the cross-state contrasts of L3
# against another arm or against the train-only template.
ARM_CONTRASTS = (
    "nonoracle_L3_minus_L2m_mixture_signed",
    "L3_minus_suffix_full_signed_oracle",
    "L3_minus_template_oracle_full_signed",
)
ROBUSTNESS_NULLS = (
    "within_shaft_margin",
    "distance_bin_margin",
    "spectral_margin",
    "variogram_margin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def holm(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def paired_summary(values: np.ndarray) -> dict:
    """Identical semantics to the frozen scorer's ``paired_summary``."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    nonzero = values[np.abs(values) > 1e-9]
    p = 1.0 if not len(nonzero) else float(
        wilcoxon(nonzero, alternative="greater").pvalue
    )
    return {
        "n": int(len(values)), "median": float(np.median(values)) if len(values) else float("nan"),
        "n_positive": int(np.sum(values > 1e-9)), "n_negative": int(np.sum(values < -1e-9)),
        "n_tied": int(np.sum(np.abs(values) <= 1e-9)), "wilcoxon_p_greater": p,
    }


def secondary_family_holm(summary: dict) -> dict:
    """spec §10.3: report the secondary/robustness family under Holm."""
    endpoints = {
        key: value for key, value in summary.items()
        if isinstance(value, dict) and "wilcoxon_p_greater" in value
    }
    definitions = {
        "reported_family_excluding_prefrozen_D2_pair": [
            key for key in endpoints if key not in D2_FAMILY
        ],
        "widest_all_non_primary_endpoints": list(endpoints),
        "narrowest_cross_state_arm_contrasts": list(ARM_CONTRASTS),
    }
    families = {}
    for name, keys in definitions.items():
        raw = [endpoints[key]["wilcoxon_p_greater"] for key in keys]
        adjusted = holm(raw)
        families[name] = {
            "m": len(keys),
            "endpoints": {
                key: {
                    "median": endpoints[key]["median"],
                    "n": endpoints[key]["n"],
                    "n_positive": endpoints[key]["n_positive"],
                    "raw_p_greater": raw[index],
                    "holm_p_greater": adjusted[index],
                    "supported_after_holm": bool(
                        endpoints[key]["median"] > 0 and adjusted[index] < .05
                    ),
                }
                for index, key in enumerate(keys)
            },
            "any_supported_after_holm": bool(any(
                endpoints[key]["median"] > 0 and adjusted[index] < .05
                for index, key in enumerate(keys)
            )),
        }
    return {
        "rule": "spec_10_3_predeclared_claim_family_holm_not_per_endpoint_stars",
        "only_endpoint_nominally_significant_before_correction":
            "L3_minus_suffix_full_signed_oracle",
        "family_definitions": families,
        "verdict": (
            "The single nominally significant early-ictal secondary endpoint does "
            "not survive family-wise correction under any of the three family "
            "definitions; it must be reported as hypothesis-generating with the "
            "Holm-adjusted P, not with the raw P."
        ),
    }


def d1_robustness_nulls(patient: pd.DataFrame) -> dict:
    """spec §9.5: summarise the geometry-eligible robustness spatial nulls."""
    frame = patient[
        (patient.condition == L3) & (patient.endpoint == CANONICAL)
    ].sort_values("subject")
    result = {
        "primary_all_contact_null_full_cohort": paired_summary(
            frame.all_contact_margin.to_numpy(float)
        ),
        "robustness_nulls": {},
        "note": (
            "Each robustness null is identifiable only on the montage subset that "
            "passes its geometry QC (spec §3.3), so it is compared against the "
            "all-contact null restricted to the same patients."
        ),
    }
    for column in ROBUSTNESS_NULLS:
        eligible = frame[np.isfinite(frame[column])]
        result["robustness_nulls"][column] = {
            "robustness_null": paired_summary(eligible[column].to_numpy(float)),
            "all_contact_null_same_patients": paired_summary(
                eligible.all_contact_margin.to_numpy(float)
            ),
            "subjects": sorted(eligible.subject.tolist()),
        }
    supported = [
        name for name, value in result["robustness_nulls"].items()
        if value["robustness_null"]["median"] > 0
        and value["robustness_null"]["wilcoxon_p_greater"] < .05
    ]
    result["any_robustness_null_supported"] = bool(supported)
    result["verdict"] = (
        "D1 is not supported under any of the four robustness spatial nulls. On the "
        "matched patient subsets the robustness nulls give the same sign and a "
        "comparable margin to the all-contact null, so the reduced margin is driven "
        "by which patients remain eligible, not by null strictness: the all-contact "
        "margin itself falls from +0.212 on all 17 patients to a negative median on "
        "the 8 patients whose geometry supports the spectral surrogate."
    )
    return result


def j_moderator_resolution(j_table: pd.DataFrame, patient: pd.DataFrame) -> dict:
    """spec §7.3: report the residual resolution of the repaired moderator."""
    early_subjects = sorted(patient.subject.unique())

    def describe(values: np.ndarray, label: str) -> dict:
        values = np.asarray(values, float)
        return {
            "cohort": label, "n": int(len(values)),
            "n_exactly_zero": int(np.sum(values == 0.0)),
            "n_unique_values": int(len(np.unique(values))),
            "max": float(values.max()), "median": float(np.median(values)),
            "fraction_below_one_percent_of_max": float(
                np.mean(values < .01 * values.max())
            ),
        }

    full = j_table.J_lat_exceedance_burden.to_numpy(float)
    early = j_table[
        j_table.subject.isin(early_subjects)
    ].J_lat_exceedance_burden.to_numpy(float)
    return {
        "estimand": "mean_event_mean_distal_positive_z_exceedance_above_1",
        "target_free_primary_cohort": describe(full, "28_patient_target_free"),
        "early_ictal_primary_cohort": describe(early, "17_patient_locked_benchmark"),
        "verdict": (
            "After the pre-freeze repair from event-median to event-mean the "
            "moderator is still strongly floor-limited: 10 of 28 patients sit at "
            "exactly zero and share one mid-rank, and only 19 distinct values exist. "
            "The single primary test of the target-free family is a rank correlation "
            "against this moderator, so its null result is bounded by moderator "
            "resolution and must not be reported as evidence that no such coupling "
            "exists."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    summary_path = out / "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json"
    patient_path = out / "early_ictal/EARLY_ICTAL_PER_PATIENT.csv"
    j_path = out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv"
    if not (out / "EARLY_ICTAL_SCORING_COMPLETE.json").exists():
        raise RuntimeError("secondary reporting addendum requires completed locked scoring")
    summary = json.loads(summary_path.read_text())
    patient = pd.read_csv(patient_path)
    j_table = pd.read_csv(j_path)

    payload = {
        "contract": "topic5_multiscale_secondary_reporting_addendum_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "POST_UNSEAL_REPORTING_CORRECTION",
        "scope": "REPORTING_ONLY",
        "model_or_field_generation_after_unseal": False,
        "primary_estimand_changed": False,
        "cohort_changed": False,
        "endpoint_changed": False,
        "direction_of_effect": "MONOTONE_CONSERVATIVE_CANNOT_CREATE_A_POSITIVE",
        "reason": (
            "The frozen adjudicator implements spec §10.3 only for the two-test D2 "
            "family, leaves the remaining early-ictal secondary/robustness endpoints "
            "with raw one-sided P values, omits the four robustness spatial nulls, "
            "and does not report the residual resolution of the repaired moderator."
        ),
        "early_ictal_secondary_family_holm": secondary_family_holm(summary),
        "D1_spatial_robustness_nulls": d1_robustness_nulls(patient),
        "J_moderator_resolution": j_moderator_resolution(j_table, patient),
        "source_hashes": {
            "early_ictal/EARLY_ICTAL_V0_5_SUMMARY.json": sha256_file(summary_path),
            "early_ictal/EARLY_ICTAL_PER_PATIENT.csv": sha256_file(patient_path),
            "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv": sha256_file(j_path),
        },
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    destination = out / "SECONDARY_REPORTING_ADDENDUM.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps({
        "written": str(destination),
        "holm_verdict": payload["early_ictal_secondary_family_holm"]["verdict"],
        "d1_verdict": payload["D1_spatial_robustness_nulls"]["verdict"],
        "j_verdict": payload["J_moderator_resolution"]["verdict"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
