from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.topic5_continuous_marked_state_h2b.audit_v02_results import (
    PROBE_PRODUCTS,
    _audit_probe_task,
    _audit_probe_permutation_semantics,
    _expected_probe_tasks,
)
from src.topic5_continuous_marked_state_h2b.contract import H2B_V0_2_REVISION


def _write_probe(tmp_path, *, effect, status, observed, null_values, n_finite):
    pd.DataFrame([{
        "patient_id": "p1", "lead_minutes": 30,
        "evaluation_tier": "descriptive_case_series",
        "state_minus_observation_conditional_log_loss": effect,
    }]).to_csv(tmp_path / "patient_median_probe_metrics.csv", index=False)
    permutation = {
        "status": status,
        "n_permutations": 2,
        "n_finite_permutations": n_finite,
        "observed_state_minus_observation": observed,
        "null_values": null_values,
    }
    (tmp_path / "time_label_permutation.json").write_text(
        json.dumps(permutation), encoding="utf-8",
    )
    (tmp_path / "risk_probe_machine_audit.json").write_text(json.dumps({
        "time_label_permutation": {"status": status},
    }), encoding="utf-8")


def test_v02_audit_accepts_complete_finite_permutation(tmp_path):
    _write_probe(
        tmp_path, effect=-0.2, status="COMPLETE", observed=-0.2,
        null_values=[-0.1, 0.1], n_finite=2,
    )
    assert _audit_probe_permutation_semantics(tmp_path, "p1/primary") == "COMPLETE"


def test_v02_audit_accepts_explicit_nonestimable_primary_lead(tmp_path):
    _write_probe(
        tmp_path, effect=float("nan"),
        status="NOT_ESTIMABLE_AT_PRIMARY_LEAD", observed=None,
        null_values=[], n_finite=0,
    )
    assert _audit_probe_permutation_semantics(
        tmp_path, "p1/primary",
    ) == "NOT_ESTIMABLE_AT_PRIMARY_LEAD"


def test_v02_audit_rejects_complete_status_for_nonfinite_primary_effect(tmp_path):
    _write_probe(
        tmp_path, effect=float("nan"), status="COMPLETE", observed=None,
        null_values=[], n_finite=0,
    )
    with pytest.raises(ValueError, match="presented as inferential"):
        _audit_probe_permutation_semantics(tmp_path, "p1/primary")


def test_expected_probe_tasks_come_from_support_and_wrong_time_risk_tables(tmp_path):
    (tmp_path / "risk_sets/p1").mkdir(parents=True)
    (tmp_path / "risk_sets/p1/matched_wrong_time_risk_sets.csv").write_text(
        "patient_id\n", encoding="utf-8",
    )
    tasks = _expected_probe_tasks(tmp_path, {
        "p0": {"n_primary_eligible_seizures": 1},
        "p1": {"n_primary_eligible_seizures": 2},
        "p2": {"n_primary_eligible_seizures": 5},
    })
    assert tasks == ["p1/primary", "p1/matched_wrong_time", "p2/primary"]


def test_task_audit_accepts_explicit_primary_lead_nonestimability(tmp_path):
    output = tmp_path / "fits/by_subject/p1/primary"
    output.mkdir(parents=True)
    _write_probe(
        output, effect=float("nan"), status="NOT_ESTIMABLE_AT_PRIMARY_LEAD",
        observed=None, null_values=[], n_finite=0,
    )
    permutation_path = output / "time_label_permutation.json"
    permutation = json.loads(permutation_path.read_text(encoding="utf-8"))
    permutation.update({
        "n_permutations_run": 0,
        "null_median": None,
        "null_mean": None,
        "null_q025": None,
        "null_q975": None,
    })
    permutation_path.write_text(json.dumps(permutation), encoding="utf-8")
    audit_path = output / "risk_probe_machine_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update({
        "status": "COMPLETE",
        "execution_status": "COMPLETE",
        "scientific_estimability": "NOT_ESTIMABLE",
        "boundary": {
            "revision": H2B_V0_2_REVISION,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "h3_or_t2_run": False,
            "paper_ready_figures_modified": False,
        },
        "positive_synthetic": {"status": "PASS"},
        "seed_aggregation": "median_within_patient_before_cohort_inference",
    })
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    for name in PROBE_PRODUCTS:
        path = output / name
        if not path.exists():
            path.write_text("header\n", encoding="utf-8")

    result = _audit_probe_task(tmp_path, "p1/primary")

    assert result["scientific_estimability"] == "NOT_ESTIMABLE"
    assert result["permutation_status"] == "NOT_ESTIMABLE_AT_PRIMARY_LEAD"
