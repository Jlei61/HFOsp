#!/usr/bin/env python3
"""Summarize direct first-entry damage from the post-unblinding inbound-edge repair."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_topic5_ecog_patch_necessity_v0_1 import stratified_randomization_test


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/patch_necessity_inbound"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/summary_inbound"
    ))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    patch_rows: list[dict[str, str]] = []
    control_rows: list[dict[str, str]] = []
    unit_audit = []
    for subject in ("958", "1084"):
        for seed_index in range(3):
            root = args.patch_root / subject / f"seed{seed_index}/patch_2x2"
            if not (root / "SUMMARY.json").exists():
                if args.allow_incomplete:
                    continue
                raise FileNotFoundError(root)
            summary = json.loads((root / "SUMMARY.json").read_text())
            if summary.get("lesion_mode") != "inbound_first_entry":
                raise RuntimeError(f"wrong lesion mode: {root}")
            if summary.get("first_entry_contract") != "no_patch_contact_recruited_before_next_rank_v0.1":
                raise RuntimeError(f"stale first-entry definition: {root}")
            patch_rows.extend(read_csv(root / "PATCH_RESULTS.csv"))
            control_rows.extend(read_csv(root / "MATCHED_CONTROL_RESULTS.csv"))
            unit_audit.append({
                "subject": subject, "seed_index": seed_index,
                "n_train_eligible": summary["n_patches_train_eligible"],
                "n_matching_eligible": summary["n_patches_matching_eligible_evaluated"],
                "n_matching_ineligible": summary["n_patches_matching_ineligible"],
                "parameter_hash_unchanged": summary["parameter_hash_unchanged"],
            })
    aggregated = []
    patients = []
    for subject in ("958", "1084"):
        subject_rows = [row for row in patch_rows if row["subject"] == subject]
        if not subject_rows:
            continue
        patch_ids = sorted({row["patch_id"] for row in subject_rows})
        paired_ids = [
            patch_id for patch_id in patch_ids
            if len([row for row in subject_rows if row["patch_id"] == patch_id]) == 3
        ]
        strata = []
        per_patch = []
        for patch_id in paired_ids:
            matched = sorted([
                row for row in subject_rows if row["patch_id"] == patch_id
            ], key=lambda row: int(row["seed_index"]))
            dose_values = {
                dose: np.asarray([float(row[f"entry_damage_contrast_dose_{dose}"]) for row in matched])
                for dose in ("0.75", "0.5", "0")
            }
            row = {
                "subject": subject, "patch_id": patch_id, "patch_nodes": matched[0]["patch_nodes"],
                "n_test_first_entry_decisions": int(matched[0]["n_test_decisions_entering_patch"]),
                **{
                    f"entry_damage_contrast_dose_{dose}_median_seed": float(np.median(values))
                    for dose, values in dose_values.items()
                },
            }
            row["dose_curve_monotonic"] = bool(
                0 <= row["entry_damage_contrast_dose_0.75_median_seed"]
                <= row["entry_damage_contrast_dose_0.5_median_seed"]
                <= row["entry_damage_contrast_dose_0_median_seed"]
            )
            aggregated.append(row); per_patch.append(row)
            seeds = []
            for seed_index, focal in enumerate(matched):
                controls = sorted([
                    control for control in control_rows
                    if control["subject"] == subject and control["patch_id"] == patch_id
                    and int(control["seed_index"]) == seed_index and float(control["dose"]) == 0.0
                ], key=lambda control: int(control["control_index"]))
                if len(controls) != 32:
                    raise RuntimeError(f"{subject} {patch_id} seed{seed_index}: need 32 controls")
                seeds.append([
                    float(focal["delta_nll_entering_patch_dose_0"]),
                    *[float(control["delta_nll_entering_patch"]) for control in controls],
                ])
            strata.append(seeds)
        values = np.asarray([
            row["entry_damage_contrast_dose_0_median_seed"] for row in per_patch
        ])
        observed, p_value, null_low, null_high = stratified_randomization_test(
            np.asarray(strata), seed=2026081800 + int(subject)
        )
        medians = {
            dose: float(np.median([
                row[f"entry_damage_contrast_dose_{dose}_median_seed"] for row in per_patch
            ])) for dose in ("0.75", "0.5", "0")
        }
        patients.append({
            "subject": subject,
            "role": "development" if subject == "1084" else "independent_confirmation",
            "n_paired_patches": len(per_patch),
            "n_patch_ids_missing_seed": len(patch_ids) - len(paired_ids),
            "entry_damage_contrast_dose_0.75_median_patch": medians["0.75"],
            "entry_damage_contrast_dose_0.5_median_patch": medians["0.5"],
            "entry_damage_contrast_dose_0_median_patch": medians["0"],
            "positive_patch_count": int(np.sum(values > 0)),
            "negative_patch_count": int(np.sum(values < 0)),
            "patient_dose_curve_monotonic": bool(0 <= medians["0.75"] <= medians["0.5"] <= medians["0"]),
            "stratified_randomization_observed": observed,
            "stratified_randomization_p_one_sided": p_value,
            "stratified_randomization_null_q025": null_low,
            "stratified_randomization_null_q975": null_high,
        })
    args.output_root.mkdir(parents=True, exist_ok=True)
    if aggregated:
        write_csv(args.output_root / "INBOUND_ENTRY_PATCH_RESULTS.csv", aggregated)
    if patients:
        write_csv(args.output_root / "INBOUND_ENTRY_PATIENT_RESULTS.csv", patients)
    if unit_audit:
        write_csv(args.output_root / "INBOUND_ENTRY_UNIT_AUDIT.csv", unit_audit)
    payload = {
        "schema": "topic5_ecog_inbound_entry_damage_summary_v0.1",
        "complete": len(unit_audit) == 6,
        "n_units": len(unit_audit),
        "patient_results": patients,
        "status": "POST_UNBLINDING_P0_REPAIR",
        "estimand": (
            "NLL damage on first-entry decisions from attenuating outside-to-patch directed edges "
            "minus the median damage from 32 matched dispersed directed-edge lesions on the same decisions."
        ),
    }
    (args.output_root / "INBOUND_ENTRY_SUMMARY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
