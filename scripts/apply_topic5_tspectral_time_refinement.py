#!/usr/bin/env python3
"""Apply reviewed time-only refinements to the existing T_spectral timing table.

The script never changes ``simple_phenotype``.  By default it only promotes
Yuquan events that already have one of the three frozen frequency labels, are
absent from the old aligned cache, and pass the pre-existing timing-stability
limits (bootstrap support >=0.70, width <=5 s, consistency within 1 s >=0.50).
Existing accepted events retain their original times.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMING = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/yuquan/refinement_v1p2/per_seizure_subject_refined_onset.csv"
)
DEFAULT_AUDIT = (
    ROOT
    / "results/topic5_ictal_recruitment/peri_onset_energy_timing/early_spectral_phenotype/tspectral_time_refinement_yuquan_bootstrap.csv"
)


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def apply_refinement(
    timing_csv: Path,
    audit_csv: Path,
    out_csv: Path,
    *,
    mark_raw_reviewed: bool,
) -> pd.DataFrame:
    timing = pd.read_csv(timing_csv)
    audit = pd.read_csv(audit_csv)
    eligible = audit[
        audit["refined_candidate_detected"].map(_truth)
        & audit["simple_phenotype"].isin(
            ["broadband_1_150", "gamma_nonbroadband", "low_frequency_only"]
        )
        & audit["bootstrap_support_fraction"].ge(0.70)
        & audit["bootstrap_width_sec"].le(5.0)
        & audit["bootstrap_consistency_1s"].ge(0.50)
    ].copy()
    already_refined = {
        (str(row.subject), int(row.seizure_idx))
        for row in timing[
            timing["timing_status"].eq("accepted_frozen_type_refined")
        ].itertuples(index=False)
    }
    required = eligible[
        (~eligible["in_existing_aligned_cache"].map(_truth))
        | pd.Series(
            [
                (str(row.subject), int(row.seizure_idx)) in already_refined
                for row in eligible.itertuples(index=False)
            ],
            index=eligible.index,
        )
    ].copy()
    if required.empty:
        raise RuntimeError("no stable frozen-type Yuquan extension events")

    if "tspectral_previous_timing_status" not in timing:
        timing["tspectral_previous_timing_status"] = timing["timing_status"]
    if "tspectral_previous_has_accepted_t_best" not in timing:
        timing["tspectral_previous_has_accepted_t_best"] = timing[
            "has_accepted_t_best"
        ]
    for column, default in (
        ("tspectral_refinement_label_version", ""),
        ("tspectral_refinement_simple_phenotype", ""),
        ("tspectral_refinement_source", ""),
        ("tspectral_refinement_raw_reviewed", False),
    ):
        if column not in timing:
            timing[column] = default
    for column in (
        "manual_notes",
        "manual_accept_t_best",
        "tspectral_previous_timing_status",
        "tspectral_refinement_label_version",
        "tspectral_refinement_simple_phenotype",
        "tspectral_refinement_source",
    ):
        timing[column] = timing[column].astype(object)

    found: set[tuple[str, int]] = set()
    for row in required.itertuples(index=False):
        key = (str(row.subject), int(row.seizure_idx))
        mask = timing["subject"].eq(key[0]) & timing["seizure_idx"].eq(key[1])
        if int(mask.sum()) != 1:
            raise RuntimeError(f"timing row cardinality for {key}: {int(mask.sum())}")
        found.add(key)
        value = float(row.refined_candidate_rel_eeg_sec)
        q05 = float(row.bootstrap_q05_rel_eeg_sec)
        q95 = float(row.bootstrap_q95_rel_eeg_sec)
        timing.loc[mask, "analysis_version"] = (
            "topic5_tspectral_subject_v1p2_frozen_type_time_refinement"
        )
        timing.loc[mask, "phenotype_status"] = "frozen_frequency_type_present"
        timing.loc[mask, "timing_status"] = "accepted_frozen_type_refined"
        timing.loc[mask, "has_candidate_t"] = True
        timing.loc[mask, "has_accepted_t_best"] = True
        for column in (
            "t_spectral_candidate_rel_eeg_sec",
            "t_spectral_candidate_rel_cache_zero_sec",
            "t_spectral_best_rel_eeg_sec",
            "t_spectral_best_rel_cache_zero_sec",
        ):
            timing.loc[mask, column] = value
        timing.loc[mask, "t_spectral_candidate_rel_clinical_sec"] = np.nan
        timing.loc[mask, "t_spectral_best_rel_clinical_sec"] = np.nan
        timing.loc[mask, "bootstrap_q05_rel_eeg_sec"] = q05
        timing.loc[mask, "bootstrap_q95_rel_eeg_sec"] = q95
        timing.loc[mask, "bootstrap_width_sec"] = q95 - q05
        timing.loc[mask, "selection_consistency_1s"] = float(
            row.bootstrap_consistency_1s
        )
        timing.loc[mask, "manual_accept_t_best"] = (
            True if mark_raw_reviewed else np.nan
        )
        timing.loc[mask, "manual_t_best_rel_eeg_sec"] = (
            value if mark_raw_reviewed else np.nan
        )
        timing.loc[mask, "manual_notes"] = (
            "raw_qc_low_eeg reviewed; time refined inside frozen frequency label"
            if mark_raw_reviewed
            else "raw_qc_low_eeg review pending; frozen frequency label unchanged"
        )
        timing.loc[mask, "tspectral_refinement_label_version"] = str(
            row.label_version
        )
        timing.loc[mask, "tspectral_refinement_simple_phenotype"] = str(
            row.simple_phenotype
        )
        timing.loc[mask, "tspectral_refinement_source"] = (
            "tspectral_time_refinement_yuquan_bootstrap.csv"
        )
        timing.loc[mask, "tspectral_refinement_raw_reviewed"] = bool(
            mark_raw_reviewed
        )

    expected = {
        (str(row.subject), int(row.seizure_idx))
        for row in required.itertuples(index=False)
    }
    if found != expected:
        raise RuntimeError(f"applied rows differ from stable audit rows: {found ^ expected}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    timing.to_csv(out_csv, index=False)
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-csv", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_TIMING)
    parser.add_argument("--mark-raw-reviewed", action="store_true")
    args = parser.parse_args()
    result = apply_refinement(
        args.timing_csv.resolve(),
        args.audit_csv.resolve(),
        args.out.resolve(),
        mark_raw_reviewed=bool(args.mark_raw_reviewed),
    )
    print(args.out.resolve())
    print(int(result["has_accepted_t_best"].map(_truth).sum()))


if __name__ == "__main__":
    main()
