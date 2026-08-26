#!/usr/bin/env python3
"""Pair T1 and T0 by patient and seed; seeds are reproducibility, not patients."""
from __future__ import annotations

import json
import os
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import REGULAR_T1_REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-variant", choices=("spectral", "raw", "both"),
                        default="spectral")
    args = parser.parse_args()
    result_root = (
        contract.RESULT_ROOT / "regular_t1"
        if args.observation_variant == "spectral"
        else contract.RESULT_ROOT / f"regular_t1/{args.observation_variant}_e0"
    )
    rows = []
    for path in sorted((result_root / "runs").glob("*.json")):
        row = json.loads(path.read_text())
        if (row.get("contract") == contract.REVISION
                and row.get("regular_t1_revision") == REGULAR_T1_REVISION
                and row.get("observation_variant", "spectral") == args.observation_variant):
            rows.append(row)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["subject"], int(row["seed"]))][row["arm"]] = row
    paired = []
    for (subject, seed), arms in grouped.items():
        if not all(arm in arms for arm in (
            "t0_no_observation_state", "t1_regular_observation"
        )):
            continue
        row = {"subject": subject, "seed": seed, "contrasts": {}}
        for layer in ("validation_filtered", "validation_correction_off_from_split_start"):
            row["contrasts"][layer] = {}
            for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
                row["contrasts"][layer][endpoint] = float(
                    arms["t1_regular_observation"][layer][endpoint]
                    - arms["t0_no_observation_state"][layer][endpoint]
                )
        row["contrasts"]["post_anchor_correction_off"] = {}
        for horizon in ("5", "10", "20"):
            row["contrasts"]["post_anchor_correction_off"][horizon] = {}
            for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
                row["contrasts"]["post_anchor_correction_off"][horizon][endpoint] = float(
                    arms["t1_regular_observation"]["post_anchor_correction_off"][horizon][endpoint]
                    - arms["t0_no_observation_state"]["post_anchor_correction_off"][horizon][endpoint]
                )
        row["state_swap"] = {}
        for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
            row["state_swap"][endpoint] = {
                "t1_wrong_minus_correct": float(
                    arms["t1_regular_observation"]["matched_wrong_time_state_swap"]
                    ["endpoints"][endpoint]["wrong_minus_correct"]
                ),
                "t0_wrong_minus_correct": float(
                    arms["t0_no_observation_state"]["matched_wrong_time_state_swap"]
                    ["endpoints"][endpoint]["wrong_minus_correct"]
                ),
            }
        paired.append(row)
    per_subject = []
    for subject in sorted({row["subject"] for row in paired}):
        found = [row for row in paired if row["subject"] == subject]
        summary = {"subject": subject, "n_seed_pairs": len(found), "contrasts": {}}
        for layer in ("validation_filtered", "validation_correction_off_from_split_start"):
            summary["contrasts"][layer] = {}
            for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
                values = [row["contrasts"][layer][endpoint] for row in found]
                summary["contrasts"][layer][endpoint] = {
                    "median_t1_minus_t0": float(np.median(values)),
                    "n_t1_better": int(np.sum(np.asarray(values) < 0)),
                    "seed_values": values,
                }
        summary["contrasts"]["post_anchor_correction_off"] = {}
        for horizon in ("5", "10", "20"):
            summary["contrasts"]["post_anchor_correction_off"][horizon] = {}
            for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
                values = [
                    row["contrasts"]["post_anchor_correction_off"][horizon][endpoint]
                    for row in found
                ]
                summary["contrasts"]["post_anchor_correction_off"][horizon][endpoint] = {
                    "median_t1_minus_t0": float(np.median(values)),
                    "n_t1_better": int(np.sum(np.asarray(values) < 0)),
                    "seed_values": values,
                }
        summary["state_swap"] = {}
        for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
            t1_values = [row["state_swap"][endpoint]["t1_wrong_minus_correct"]
                         for row in found]
            t0_values = [row["state_swap"][endpoint]["t0_wrong_minus_correct"]
                         for row in found]
            summary["state_swap"][endpoint] = {
                "median_t1_wrong_minus_correct": float(np.median(t1_values)),
                "n_t1_correct_better": int(np.sum(np.asarray(t1_values) > 0)),
                "t1_seed_values": t1_values,
                "max_abs_t0_control": float(np.max(np.abs(t0_values))),
            }
        per_subject.append(summary)
    cohort = {"contrasts": {}, "state_swap": {}}
    for layer in ("validation_filtered", "validation_correction_off_from_split_start"):
        cohort["contrasts"][layer] = {}
        for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
            values = np.asarray([
                row["contrasts"][layer][endpoint]["median_t1_minus_t0"]
                for row in per_subject
            ], dtype=float)
            nonzero = values[values != 0]
            cohort["contrasts"][layer][endpoint] = {
                "n_patients": int(len(values)),
                "median_patient_t1_minus_t0": (
                    float(np.median(values)) if len(values) else None
                ),
                "n_patients_t1_better": int(np.sum(values < 0)),
                "two_sided_exact_sign_p_unadjusted": (
                    float(binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5).pvalue)
                    if len(nonzero) else None
                ),
                "patient_values": {
                    row["subject"]: row["contrasts"][layer][endpoint]
                    ["median_t1_minus_t0"] for row in per_subject
                },
            }
    cohort["contrasts"]["post_anchor_correction_off"] = {}
    for horizon in ("5", "10", "20"):
        cohort["contrasts"]["post_anchor_correction_off"][horizon] = {}
        for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
            values = np.asarray([
                row["contrasts"]["post_anchor_correction_off"][horizon][endpoint]
                ["median_t1_minus_t0"] for row in per_subject
            ], dtype=float)
            nonzero = values[values != 0]
            cohort["contrasts"]["post_anchor_correction_off"][horizon][endpoint] = {
                "n_patients": int(len(values)),
                "median_patient_t1_minus_t0": (
                    float(np.median(values)) if len(values) else None
                ),
                "n_patients_t1_better": int(np.sum(values < 0)),
                "two_sided_exact_sign_p_unadjusted": (
                    float(binomtest(int(np.sum(nonzero < 0)), len(nonzero), 0.5).pvalue)
                    if len(nonzero) else None
                ),
                "patient_values": {
                    row["subject"]: row["contrasts"]
                    ["post_anchor_correction_off"][horizon][endpoint]
                    ["median_t1_minus_t0"] for row in per_subject
                },
            }
    for endpoint in ("joint_nll", "timing_nll", "mark_nll"):
        values = np.asarray([
            row["state_swap"][endpoint]["median_t1_wrong_minus_correct"]
            for row in per_subject
        ], dtype=float)
        nonzero = values[values != 0]
        cohort["state_swap"][endpoint] = {
            "n_patients": int(len(values)),
            "median_patient_wrong_minus_correct": (
                float(np.median(values)) if len(values) else None
            ),
            "n_patients_correct_better": int(np.sum(values > 0)),
            "two_sided_exact_sign_p_unadjusted": (
                float(binomtest(int(np.sum(nonzero > 0)), len(nonzero), 0.5).pvalue)
                if len(nonzero) else None
            ),
            "max_abs_t0_control": (
                float(max(
                    row["state_swap"][endpoint]["max_abs_t0_control"]
                    for row in per_subject
                )) if per_subject else None
            ),
        }
    output = {
        "contract": contract.REVISION,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "observation_variant": args.observation_variant,
        "n_runs": len(rows), "n_paired": len(paired),
        "n_subjects": len(per_subject), "per_subject": per_subject,
        "cohort_patient_unit": cohort,
        "sealed_opened": False,
        "claim_boundary": (
            "Fixed-epoch development prototype. Filtered improvement shows "
            "observation-conditioned predictive information; correction-off "
            "from split start is too long to replace 5/10/20-event H1 tests. "
            "Exact sign p-values are descriptive and unadjusted; seeds measure "
            "stability and are never treated as independent patients."
        ),
    }
    path = result_root / "REGULAR_T1_SUMMARY.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps({"n_runs": len(rows), "n_paired": len(paired),
                      "n_subjects": len(per_subject)}))


if __name__ == "__main__":
    main()
