"""Regression tests for correct-vs-wrong-time alignment in the v0.2 aggregate.

The wrong-time table is built from a strictly smaller seizure population than the
primary table -- a donor must be same-patient, same recorded coverage segment and
exclusion-clear -- so its support tier can legitimately sit one step below the
primary tier for the same patient.  ``epilepsiae_548`` is exactly that case in
the live cohort: ``primary_chronological`` on the primary table,
``sensitivity_loso`` on its matched wrong-time table.

Joining the two tables on ``evaluation_tier`` therefore silently discards the
only correct-vs-wrong-time evidence the cohort has at the primary tier.  The
v0.2 contract §3 names ``correct-time - matched wrong-time`` as a *necessary*
explanatory quantity, and §4 requires a tier downgrade to stay visible rather
than disappear.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.topic5_continuous_marked_state_h2b.aggregate_v02_results import run


PRIMARY_EFFECT = "state_minus_observation_conditional_log_loss"
CARRY_EFFECT = "persistent_minus_memoryless_conditional_log_loss"
WRONG_EFFECT = "correct_minus_wrong_time_conditional_log_loss"


def _primary_rows() -> pd.DataFrame:
    """One primary-tier patient and one LOSO-tier patient, two leads each."""
    rows = []
    for patient, tier in (
        ("epilepsiae_548", "primary_chronological"),
        ("epilepsiae_442", "sensitivity_loso"),
    ):
        for lead in (30, 60):
            rows.append({
                "patient_id": patient,
                "lead_minutes": lead,
                "evaluation_tier": tier,
                PRIMARY_EFFECT: -0.05 if patient == "epilepsiae_548" else 0.02,
                CARRY_EFFECT: -0.01,
                "n_optimizer_seeds": 5,
            })
    return pd.DataFrame(rows)


def _wrong_rows() -> pd.DataFrame:
    """Both patients drop one tier on the wrong-time table."""
    rows = []
    for patient, tier in (
        ("epilepsiae_548", "sensitivity_loso"),
        ("epilepsiae_442", "descriptive_case_series"),
    ):
        for lead in (30, 60):
            rows.append({
                "patient_id": patient,
                "lead_minutes": lead,
                "evaluation_tier": tier,
                WRONG_EFFECT: -0.03 if patient == "epilepsiae_548" else 0.04,
            })
    return pd.DataFrame(rows)


def _write_root(
    tmp_path: Path,
    primary: pd.DataFrame,
    wrong: pd.DataFrame | None,
    *,
    tiers: dict[str, str] | None = None,
) -> Path:
    root = tmp_path / "v0_2"
    (root / "fits/primary").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "manifests").mkdir(parents=True)
    primary.to_csv(root / "fits/primary/patient_median_probe_metrics.csv", index=False)
    if wrong is not None:
        (root / "fits/matched_wrong_time").mkdir(parents=True)
        wrong.to_csv(
            root / "fits/matched_wrong_time/patient_median_probe_metrics.csv",
            index=False,
        )
    (root / "manifests/r1_7_checkpoint_inventory.json").write_text(json.dumps({
        "h1_stable_subjects": ["epilepsiae_442"],
    }))
    resolved = tiers or {
        "epilepsiae_548": "primary_chronological",
        "epilepsiae_442": "sensitivity_loso",
    }
    for subject, tier in resolved.items():
        directory = root / "risk_sets" / subject
        directory.mkdir(parents=True)
        (directory / "input_manifest.json").write_text(json.dumps({
            "subject": subject,
            "support_tier": tier,
            "n_primary_eligible_seizures": 10 if tier == "primary_chronological" else 6,
        }))
    return root


def test_wrong_time_effect_survives_a_tier_downgrade(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    root = _write_root(tmp_path, _primary_rows(), _wrong_rows())

    run(root)

    per_patient = pd.read_csv(root / "reports/per_patient_lead_results.csv")
    kept = per_patient.set_index(["patient_id", "lead_minutes"])
    assert kept.loc[("epilepsiae_548", 30), WRONG_EFFECT] == pytest.approx(-0.03)
    assert kept.loc[("epilepsiae_442", 30), WRONG_EFFECT] == pytest.approx(0.04)


def test_both_support_tiers_are_reported_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    root = _write_root(tmp_path, _primary_rows(), _wrong_rows())

    run(root)

    per_patient = pd.read_csv(root / "reports/per_patient_lead_results.csv")
    row = per_patient[
        (per_patient.patient_id == "epilepsiae_548")
        & (per_patient.lead_minutes == 30)
    ].iloc[0]
    assert row["evaluation_tier"] == "primary_chronological"
    assert row["wrong_time_evaluation_tier"] == "sensitivity_loso"
    assert bool(row["wrong_time_tier_downgraded"]) is True


def test_wrong_time_effect_is_summarised_under_its_own_tier(tmp_path, monkeypatch):
    """A downgrade must not be re-hidden by summarising under the primary tier."""
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    root = _write_root(tmp_path, _primary_rows(), _wrong_rows())

    run(root)

    summary = pd.read_csv(root / "reports/cohort_patient_first_summary.csv")
    wrong = summary[
        (summary.effect == WRONG_EFFECT)
        & (summary.stratum == "all_checkpoint_available")
        & (summary.lead_minutes == 30)
    ]
    assert not wrong.empty
    assert set(wrong.evaluation_tier) == {"sensitivity_loso", "descriptive_case_series"}
    assert "primary_chronological" not in set(wrong.evaluation_tier), (
        "the wrong-time effect was reported at a tier its own table never reached"
    )
    main = summary[
        (summary.effect == PRIMARY_EFFECT) & (summary.lead_minutes == 30)
        & (summary.stratum == "all_checkpoint_available")
    ]
    assert "primary_chronological" in set(main.evaluation_tier)


def test_report_states_whether_primary_chronological_wrong_time_evidence_exists(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    root = _write_root(tmp_path, _primary_rows(), _wrong_rows())

    payload = run(root)

    evidence = payload["correct_vs_wrong_time_evidence"]
    assert evidence["n_patients_with_wrong_time_effect_at_primary_lead"] == 2
    assert evidence["primary_chronological_wrong_time_evidence_exists"] is False
    assert evidence["n_patients_wrong_time_tier_downgraded"] == 2
    assert "sensitivity_loso" in evidence["wrong_time_tiers_present"]


def test_duplicate_wrong_time_key_is_rejected_not_silently_dropped(
        tmp_path, monkeypatch):
    """Paired-cohort discipline: uniqueness is proven before aligning."""
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    wrong = pd.concat([_wrong_rows(), _wrong_rows().head(1)], ignore_index=True)
    root = _write_root(tmp_path, _primary_rows(), wrong)

    with pytest.raises(ValueError, match="wrong-time"):
        run(root)


def test_missing_wrong_time_table_is_reported_as_absent_not_as_zero(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.topic5_continuous_marked_state_h2b.contract.assert_safe_output_path",
        lambda path: Path(path),
    )
    root = _write_root(tmp_path, _primary_rows(), None)

    payload = run(root)

    per_patient = pd.read_csv(root / "reports/per_patient_lead_results.csv")
    assert per_patient[WRONG_EFFECT].isna().all()
    assert per_patient["wrong_time_evaluation_tier"].isna().all()
    evidence = payload["correct_vs_wrong_time_evidence"]
    assert evidence["n_patients_with_wrong_time_effect_at_primary_lead"] == 0
    assert evidence["primary_chronological_wrong_time_evidence_exists"] is False
