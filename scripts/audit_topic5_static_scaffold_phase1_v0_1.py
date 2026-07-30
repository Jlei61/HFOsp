#!/usr/bin/env python3
"""Completeness and parity audit for static-scaffold Phase 1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
OLD = ROOT / "results/topic5_rnn_internal_state_reduction"


def main() -> None:
    frame = pd.read_csv(BASE / "phase1_existing_fields_patient_metrics.csv")
    expected_rows = 16 * 6 * 5
    if len(frame) != expected_rows:
        raise RuntimeError(f"row count drift: {len(frame)} != {expected_rows}")
    expected_eligible = {
        "all_contact": 16 * 6,
        "within_shaft_circular": 16 * 6,
        "within_shaft_dihedral": 16 * 6,
        "equal_size_shaft_profile": 2 * 6,
        "geometry_smooth_rbf": 13 * 6,
    }
    actual_eligible = (
        frame.groupby("null_mode").eligible.sum().astype(int).to_dict()
    )
    if actual_eligible != expected_eligible:
        raise RuntimeError(
            f"null eligibility drift: {actual_eligible} != {expected_eligible}"
        )
    score_columns = [
        "observed_signed_rho",
        "null_signed_median",
        "signed_margin",
        "signed_empirical_p",
        "observed_absolute_rho",
        "null_absolute_median",
        "absolute_margin",
        "absolute_empirical_p",
    ]
    if frame.loc[~frame.eligible, score_columns].notna().any().any():
        raise RuntimeError("ineligible null rows contain finite scores")

    current = frame.loc[
        (frame.null_mode == "all_contact") & frame.eligible,
        ["subject", "model", "observed_signed_rho"],
    ]
    previous = pd.read_csv(OLD / "early_ictal_fixed_readback_patient_metrics.csv")
    previous = previous.loc[
        (previous.seizure_split == "all")
        & (previous.field == "participation"),
        ["subject", "model", "signed_rho"],
    ]
    joined = current.merge(previous, on=["subject", "model"], how="inner")
    if len(joined) != 16 * 6:
        raise RuntimeError("old/new fixed field parity denominator drifted")
    maximum_signed_difference = float(
        np.max(np.abs(joined.observed_signed_rho - joined.signed_rho))
    )
    if maximum_signed_difference > 1.0e-12:
        raise RuntimeError("signed fixed-field score did not reproduce")

    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "existing_fields_fixed_signed_readout",
        "status": "PASS",
        "n_rows": len(frame),
        "n_patients": int(frame.subject.nunique()),
        "n_models": int(frame.model.nunique()),
        "null_eligibility_rows": actual_eligible,
        "maximum_old_new_signed_rho_difference": maximum_signed_difference,
        "ineligible_rows_are_nan": True,
        "target_context": "reused target; internal validation",
    }
    (BASE / "PHASE1_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
