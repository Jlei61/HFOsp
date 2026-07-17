#!/usr/bin/env python3
"""Derive a bounded narrow-cache sensitivity tier from the frozen T0 audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT / "results/topic5_ictal_recruitment/t0_eligibility_audit_narrow_expanded.csv"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic5_ictal_recruitment/t0_eligibility_audit_narrow_cache_sensitivity.csv"
)


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "t"})


def run(input_csv: Path, output_csv: Path) -> Path:
    rows = pd.read_csv(input_csv)
    rows["narrow_cache_eligible"] = (
        _bool(rows["cacheable"])
        & _bool(rows["baseline_valid"])
        & _bool(rows["has_complete_eeg_interval"])
        & _bool(rows["gap_prev_ok"])
        & (pd.to_numeric(rows["n_montage_resolved"], errors="coerce") >= 6)
    )
    rows["narrow_cache_tier"] = rows["narrow_cache_eligible"].map(
        {True: "sensitivity_min6_overlap", False: "ineligible"}
    )
    rows["narrow_cache_contract"] = (
        "cacheable+valid_baseline+complete_interval+gap>=300s+>=6_resolved; "
        "80pct montage fraction waived for sensitivity cache only"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_csv, index=False)
    eligible = rows[rows["narrow_cache_eligible"]]
    summary = {
        "status": "narrow_cache_sensitivity_only_not_primary_analysis_eligibility",
        "source_audit": str(input_csv),
        "contract": (
            "cacheable, valid baseline, complete interval, >=300 s previous-seizure "
            "gap, and >=6 narrow-axis contacts resolved; the primary >=80% montage "
            "fraction is not waived for scientific inference"
        ),
        "n_rows": int(len(rows)),
        "n_subjects": int(rows["subject_id"].nunique()),
        "n_eligible_seizures": int(len(eligible)),
        "n_subjects_with_eligible_seizure": int(eligible["subject_id"].nunique()),
        "eligible_by_subject": {
            str(key): int(value)
            for key, value in eligible.groupby("subject_id").size().items()
        },
    }
    output_csv.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.input.resolve(), args.output.resolve()))


if __name__ == "__main__":
    main()
