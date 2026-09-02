#!/usr/bin/env python3
"""Patient-first summary of ECoG local-patch edge attenuation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    draw = rng.choice(data, size=(20000, len(data)), replace=True)
    return tuple(float(value) for value in np.quantile(np.median(draw, axis=1), [0.025, 0.975]))


def stratified_randomization_test(
    strata: np.ndarray, *, seed: int, n_randomizations: int = 20000,
) -> tuple[float, float, float, float]:
    """Exchange the focal label among one patch and 32 controls within every patch x seed."""
    values = np.asarray(strata, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (3, 33):
        raise ValueError("strata must be patch x 3 seeds x 33 candidate edge sets")
    observed_by_patch = np.median(
        values[:, :, 0] - np.median(values[:, :, 1:], axis=-1), axis=1
    )
    observed = float(np.median(observed_by_patch))
    rng = np.random.default_rng(int(seed))
    null = np.empty(int(n_randomizations), dtype=float)
    for draw in range(int(n_randomizations)):
        focal_index = rng.integers(0, 33, size=values.shape[:2])
        focal = np.take_along_axis(values, focal_index[..., None], axis=-1)[..., 0]
        remaining = np.where(
            np.arange(33)[None, None, :] == focal_index[..., None], np.nan, values
        )
        effect = focal - np.nanmedian(remaining, axis=-1)
        null[draw] = np.median(np.median(effect, axis=1))
    p = float((1 + np.sum(null >= observed)) / (len(null) + 1))
    low, high = (float(value) for value in np.quantile(null, [0.025, 0.975]))
    return observed, p, low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/patch_necessity"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/summary"
    ))
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--patch-sides", nargs="+", type=int, default=(2, 3), choices=(2, 3))
    parser.add_argument(
        "--lesion-mode", default="symmetric_incident",
        choices=("symmetric_incident", "inbound_first_entry"),
    )
    args = parser.parse_args()

    raw_rows: list[dict[str, str]] = []
    raw_control_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        for seed_index in range(3):
            for side in args.patch_sides:
                root = args.patch_root / subject / f"seed{seed_index}" / f"patch_{side}x{side}"
                result_path = root / "PATCH_RESULTS.csv"
                summary_path = root / "SUMMARY.json"
                if not result_path.exists() or not summary_path.exists():
                    if args.allow_incomplete:
                        continue
                    raise FileNotFoundError(f"missing patch unit: {root}")
                summary = json.loads(summary_path.read_text())
                if not summary["parameter_hash_unchanged"]:
                    raise RuntimeError(f"parameter hash changed: {root}")
                if summary.get("lesion_mode") != args.lesion_mode:
                    raise RuntimeError(f"lesion mode mismatch: {root}")
                unit_patch_rows = read_csv(result_path)
                unit_control_rows = read_csv(root / "MATCHED_CONTROL_RESULTS.csv")
                raw_rows.extend(unit_patch_rows)
                raw_control_rows.extend(unit_control_rows)
                weight_errors = np.asarray([
                    float(row["weight_quantile_mean_absolute_error"])
                    for row in unit_control_rows
                ])
                audit_rows.append({
                    "subject": subject, "seed_index": seed_index, "patch_side": side,
                    "n_patches_possible": summary["n_patches_possible"],
                    "n_patches_train_eligible": summary["n_patches_train_eligible"],
                    "n_patches_matching_eligible_evaluated": summary["n_patches_matching_eligible_evaluated"],
                    "n_patches_matching_ineligible": summary["n_patches_matching_ineligible"],
                    "parameter_hash_unchanged": summary["parameter_hash_unchanged"],
                    "n_control_rows": len(unit_control_rows),
                    "expected_control_rows": len(unit_patch_rows) * 3 * 32,
                    "weight_quantile_mae_median": float(np.median(weight_errors)) if len(weight_errors) else float("nan"),
                    "weight_quantile_mae_max": float(np.max(weight_errors)) if len(weight_errors) else float("nan"),
                    "runtime_sec": summary["runtime_sec"],
                })

    patch_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        for side in args.patch_sides:
            selected = [row for row in raw_rows if row["subject"] == subject and int(row["patch_side"]) == side]
            patch_ids_all = sorted({row["patch_id"] for row in selected})
            patch_ids = [
                patch_id for patch_id in patch_ids_all
                if len([row for row in selected if row["patch_id"] == patch_id]) == 3
            ]
            for patch_id in patch_ids:
                matched = [row for row in selected if row["patch_id"] == patch_id]
                matched.sort(key=lambda row: int(row["seed_index"]))
                did = {
                    dose: np.asarray([
                        float(row[f"difference_in_difference_dose_{dose}"]) for row in matched
                    ])
                    for dose in ("0.75", "0.5", "0")
                }
                patch_rows.append({
                    "subject": subject,
                    "patch_side": side,
                    "patch_id": patch_id,
                    "patch_nodes": matched[0]["patch_nodes"],
                    "n_test_decisions_entering_patch": int(matched[0]["n_test_decisions_entering_patch"]),
                    "n_test_decisions_outside_patch": int(matched[0]["n_test_decisions_outside_patch"]),
                    "difference_in_difference_dose_0.75_median_seed": float(np.median(did["0.75"])),
                    "difference_in_difference_dose_0.5_median_seed": float(np.median(did["0.5"])),
                    "difference_in_difference_dose_0_median_seed": float(np.median(did["0"])),
                    "monotonic_damage": bool(
                        np.median(did["0.75"]) <= np.median(did["0.5"]) <= np.median(did["0"])
                    ),
                })
            patient_patch = [
                row for row in patch_rows if row["subject"] == subject and row["patch_side"] == side
            ]
            if not patient_patch:
                continue
            values = np.asarray([
                row["difference_in_difference_dose_0_median_seed"] for row in patient_patch
            ])
            low, high = bootstrap_ci(values, 2026081600 + int(subject) + side)
            positive = int(np.sum(values > 0))
            negative = int(np.sum(values < 0))
            nonzero = positive + negative
            dose_medians = {
                dose: float(np.median([
                    row[f"difference_in_difference_dose_{dose}_median_seed"]
                    for row in patient_patch
                ]))
                for dose in ("0.75", "0.5", "0")
            }
            permutation_p = float("nan")
            permutation_low = float("nan")
            permutation_high = float("nan")
            permutation_observed = float("nan")
            if side == 2:
                strata = []
                for patch_id in [row["patch_id"] for row in patient_patch]:
                    seed_values = []
                    for seed_index in range(3):
                        focal_row = next(
                            row for row in selected
                            if row["patch_id"] == patch_id and int(row["seed_index"]) == seed_index
                        )
                        controls = sorted([
                            row for row in raw_control_rows
                            if row["subject"] == subject
                            and int(row["patch_side"]) == side
                            and row["patch_id"] == patch_id
                            and int(row["seed_index"]) == seed_index
                            and float(row["dose"]) == 0.0
                        ], key=lambda row: int(row["control_index"]))
                        if len(controls) != 32:
                            raise RuntimeError(f"{subject} {patch_id} seed{seed_index}: need 32 controls")
                        seed_values.append([
                            float(focal_row["selectivity_dose_0"]),
                            *[float(row["selectivity"]) for row in controls],
                        ])
                    strata.append(seed_values)
                permutation_observed, permutation_p, permutation_low, permutation_high = stratified_randomization_test(
                    np.asarray(strata), seed=2026081600 + int(subject)
                )
            patient_rows.append({
                "subject": subject,
                "patch_side": side,
                "n_eligible_patches": len(values),
                "full_attenuation_difference_in_difference_median_patch": float(np.median(values)),
                "ci95_low": low,
                "ci95_high": high,
                "positive_patch_count": positive,
                "negative_patch_count": negative,
                "one_sided_sign_p": float(binomtest(positive, nonzero, 0.5, alternative="greater").pvalue) if nonzero else float("nan"),
                "monotonic_patch_count": int(np.sum([row["monotonic_damage"] for row in patient_patch])),
                "n_patch_ids_missing_one_or_more_seeds": len(patch_ids_all) - len(patch_ids),
                "dose_0.75_difference_in_difference_median_patch": dose_medians["0.75"],
                "dose_0.5_difference_in_difference_median_patch": dose_medians["0.5"],
                "patient_median_dose_curve_monotonic": bool(
                    0.0 <= dose_medians["0.75"] <= dose_medians["0.5"] <= dose_medians["0"]
                ),
                "stratified_randomization_observed": permutation_observed,
                "stratified_randomization_p_one_sided": permutation_p,
                "stratified_randomization_null_q025": permutation_low,
                "stratified_randomization_null_q975": permutation_high,
                "n_stratified_randomizations": 20000 if side == 2 else 0,
            })
    write_csv(args.output_root / "PATCH_SEED_AGGREGATED_RESULTS.csv", patch_rows)
    write_csv(args.output_root / "PATCH_PATIENT_RESULTS.csv", patient_rows)
    write_csv(args.output_root / "PATCH_UNIT_AUDIT.csv", audit_rows)
    payload = {
        "schema": "topic5_ecog_patch_necessity_summary_v0.1",
        "complete": len(audit_rows) == 2 * 3 * len(args.patch_sides),
        "n_units": len(audit_rows),
        "lesion_mode": args.lesion_mode,
        "patient_results": patient_rows,
        "positive_estimand": (
            "Removing local edges incident to a contiguous patch selectively worsens held-out "
            "next-contact prediction into that patch beyond equally sized dispersed-edge lesions."
        ),
        "primary_inference": (
            "For 2x2 patches, inference is a within-patch-by-seed focal-label randomization "
            "among the contiguous lesion and 32 matched dispersed lesions; overlapping patches "
            "are not treated as independent observations."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "PATCH_NECESSITY_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
