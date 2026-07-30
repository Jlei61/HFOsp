#!/usr/bin/env python3
"""Run the target-blind 34-patient static-scaffold reliability audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records
from src.topic5_scaffold_reliability import (
    event_count_saturation,
    field_comparison,
    participation_field,
    rank_correlation,
)
from src.topic5_static_scaffold_validation import coherent_index_null


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_median(values: np.ndarray, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = values[
        rng.integers(0, len(values), size=(20000, len(values)))
    ]
    medians = np.nanmedian(draws, axis=1)
    return [
        float(np.nanquantile(medians, 0.025)),
        float(np.nanquantile(medians, 0.975)),
    ]


def _signed_rank(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values) & (values != 0)]
    if not len(values):
        return float("nan")
    return float(wilcoxon(values, alternative="greater").pvalue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "config/topic5_static_scaffold_reliability_history_necessity_v0_1.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text())
    dataset_dir = ROOT / cfg["inputs"]["dataset"]
    output_dir = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / args.output_dir
        if args.output_dir is not None
        else ROOT / cfg["outputs"]["static_reliability"]
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    records = load_records(dataset_dir)
    static = cfg["static_reliability"]
    cohort_rows = []
    null_rows = []
    saturation_rows = []
    field_rows = []

    for subject_index, subject in enumerate(sorted(records)):
        record = records[subject]
        train = record.group_ids[record.train_indices]
        heldout = record.group_ids[record.eval_indices]
        split = len(train) // 2
        train_field = participation_field(train)
        heldout_field = participation_field(heldout)
        first_field = participation_field(train[:split])
        second_field = participation_field(train[split:])
        odd_field = participation_field(train[::2])
        even_field = participation_field(train[1::2])

        train_heldout = field_comparison(train_field, heldout_field)
        chronological = field_comparison(first_field, second_field)
        odd_even = field_comparison(odd_field, even_field)
        permutations, null_metadata = coherent_index_null(
            record.contact_names,
            n_draws=int(static["null_draws"]),
            seed=int(static["seed"]) + subject_index * 10_007,
            mode=str(static["null_mode"]),
        )
        null_rho = np.asarray(
            [
                rank_correlation(train_field[index], heldout_field)
                for index in permutations
            ],
            dtype=np.float64,
        )
        finite_null = null_rho[np.isfinite(null_rho)]
        p_null = (
            float(
                (1 + np.sum(finite_null >= train_heldout["spearman_rho"]))
                / (1 + len(finite_null))
            )
            if len(finite_null)
            else float("nan")
        )
        cohort_rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_contacts": len(record.contact_names),
                "n_train_events": len(train),
                "n_heldout_events": len(heldout),
                **{
                    f"train80_heldout20_{key}": value
                    for key, value in train_heldout.items()
                },
                **{
                    f"chronological_half_{key}": value
                    for key, value in chronological.items()
                },
                **{
                    f"odd_even_{key}": value
                    for key, value in odd_even.items()
                },
                "structured_null_median_rho": float(np.nanmedian(null_rho)),
                "structured_null_excess_rho": float(
                    train_heldout["spearman_rho"] - np.nanmedian(null_rho)
                ),
                "structured_null_p": p_null,
                "structured_null_eligible": bool(null_metadata["eligible"]),
                "structured_null_movable_fraction": float(
                    null_metadata["movable_fraction"]
                ),
                "ictal_target_read": False,
            }
        )
        null_rows.extend(
            {
                "subject": subject,
                "dataset": record.dataset,
                "draw": draw,
                "spearman_rho": float(value),
            }
            for draw, value in enumerate(null_rho)
        )
        for row in event_count_saturation(
            train,
            heldout_field,
            event_counts=static["event_counts"],
            n_subsamples=int(static["subsamples_per_count"]),
            seed=int(static["seed"]) + subject_index * 1_000_003,
        ):
            saturation_rows.append(
                {"subject": subject, "dataset": record.dataset, **row}
            )
        for index, name in enumerate(record.contact_names.astype(str)):
            field_rows.append(
                {
                    "subject": subject,
                    "dataset": record.dataset,
                    "contact_index": index,
                    "contact_name": name,
                    "train80_participation": float(train_field[index]),
                    "heldout20_participation": float(heldout_field[index]),
                    "chronological_first_half": float(first_field[index]),
                    "chronological_second_half": float(second_field[index]),
                    "odd_train_events": float(odd_field[index]),
                    "even_train_events": float(even_field[index]),
                }
            )
        print(
            json.dumps(
                {
                    "subject": subject,
                    "status": "complete",
                    "rho": train_heldout["spearman_rho"],
                    "null_p": p_null,
                    "ictal_target_read": False,
                }
            ),
            flush=True,
        )

    cohort = pd.DataFrame(cohort_rows)
    null_frame = pd.DataFrame(null_rows)
    saturation = pd.DataFrame(saturation_rows)
    fields = pd.DataFrame(field_rows)
    cohort.to_csv(output_dir / "patient_reliability.csv", index=False)
    null_frame.to_csv(
        output_dir / "within_shaft_circular_null_draws.csv", index=False
    )
    saturation.to_csv(output_dir / "event_count_saturation.csv", index=False)
    fields.to_csv(output_dir / "contact_participation_fields.csv", index=False)
    saturation_patient = (
        saturation.groupby(
            ["subject", "dataset", "event_count"], as_index=False
        )
        .agg(
            spearman_rho=("spearman_rho", "median"),
            top_quartile_jaccard=("top_quartile_jaccard", "median"),
            mean_absolute_error=("mean_absolute_error", "median"),
        )
        .merge(
            cohort[
                ["subject", "train80_heldout20_spearman_rho"]
            ],
            on="subject",
            validate="many_to_one",
        )
    )
    saturation_patient["rho_delta_to_full_train80"] = (
        saturation_patient.spearman_rho
        - saturation_patient.train80_heldout20_spearman_rho
    )
    saturation_patient.to_csv(
        output_dir / "event_count_patient_summary.csv", index=False
    )
    saturation_cohort_rows = []
    for event_count, frame in saturation_patient.groupby("event_count"):
        rho = frame.spearman_rho.to_numpy(float)
        delta = frame.rho_delta_to_full_train80.to_numpy(float)
        saturation_cohort_rows.append(
            {
                "event_count": int(event_count),
                "n_patients": int(len(frame)),
                "median_spearman_rho": float(np.median(rho)),
                "spearman_rho_ci95": json.dumps(
                    _bootstrap_median(rho, 20260728 + int(event_count))
                ),
                "median_rho_delta_to_full_train80": float(
                    np.median(delta)
                ),
                "fraction_within_0_05_of_full_train80": float(
                    np.mean(delta >= -0.05)
                ),
            }
        )
    pd.DataFrame(saturation_cohort_rows).to_csv(
        output_dir / "event_count_cohort_summary.csv", index=False
    )

    primary = cohort.train80_heldout20_spearman_rho.to_numpy(float)
    eligible_null = cohort[cohort.structured_null_eligible].copy()
    excess = eligible_null.structured_null_excess_rho.to_numpy(float)
    summary = {
        "status": "complete",
        "n_patients": len(cohort),
        "n_epilepsiae": int(np.sum(cohort.dataset == "epilepsiae")),
        "n_yuquan": int(np.sum(cohort.dataset == "yuquan")),
        "primary": {
            "metric": "train80_vs_heldout20_contact_spearman",
            "median": float(np.nanmedian(primary)),
            "ci95": _bootstrap_median(primary, 20260728),
            "n_positive": int(np.sum(primary > 0)),
            "wilcoxon_greater_p": _signed_rank(primary),
        },
        "structured_null_excess": {
            "n_patients": int(len(eligible_null)),
            "median": float(np.nanmedian(excess)),
            "ci95": _bootstrap_median(excess, 20260729),
            "n_positive": int(np.sum(excess > 0)),
            "wilcoxon_greater_p": _signed_rank(excess),
            "n_patient_null_p_lt_0_05": int(
                np.sum(eligible_null.structured_null_p < 0.05)
            ),
        },
        "median_chronological_half_rho": float(
            np.nanmedian(cohort.chronological_half_spearman_rho)
        ),
        "median_odd_even_rho": float(
            np.nanmedian(cohort.odd_even_spearman_rho)
        ),
        "config_sha256": _sha256(config_path),
        "input_fingerprints": {
            subject: record.input_sha256
            for subject, record in sorted(records.items())
        },
        "ictal_target_read": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True)
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False)
    )
    (output_dir / "DONE.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "n_patients": len(cohort),
                "ictal_target_read": False,
            },
            indent=2,
        )
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
