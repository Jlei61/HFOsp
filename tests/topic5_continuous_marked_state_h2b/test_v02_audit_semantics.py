from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.topic5_continuous_marked_state_h2b.audit_v02_results import (
    _audit_probe_permutation_semantics,
)


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
