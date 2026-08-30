"""Regression tests for the structurally unestimable probe branch.

Origin: ``epilepsiae_1125/matched_wrong_time`` crashed the v0.2 cohort queue with
``KeyError: 'state_minus_observation_conditional_log_loss'``.  Its wrong-time
table carries a single donor-valid seizure, so no arm reaches ``ok`` and
``run_probe_table`` correctly emits no main-effect column -- but
``time_label_permutation_audit`` indexed that column unconditionally.

The v0.2 contract requires (§4) that one patient being unestimable never blocks
the others, and (§8.5) that unqualified states fail closed without fabricating a
result.  §3 fixes the primary lead at 30 min and forbids substituting a
better-looking lead, so estimability is judged at 30 min only.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.topic5_continuous_marked_state_h2b.contract import (
    CANONICAL_V0_2_RESULT_ROOT,
)
from src.topic5_continuous_marked_state_h2b.risk_probe import (
    build_risk_sets,
    make_positive_synthetic_risk_table,
    primary_lead_estimability,
    run_probe_table,
    time_label_permutation_audit,
)

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def probe_output_dir(request):
    """A writable scratch directory inside the isolated H2b result root.

    ``assert_safe_output_path`` refuses any target outside a versioned H2b root,
    and the worktree's own ``results/`` is a read-only bind mount, so CLI probe
    tests have to land in the canonical root and clean up after themselves.
    """
    target = CANONICAL_V0_2_RESULT_ROOT / "fits/_pytest_scratch" / request.node.name
    if target.exists():
        shutil.rmtree(target)
    try:
        yield target
    finally:
        if target.exists():
            shutil.rmtree(target)


def _single_seizure_not_estimable_table() -> pd.DataFrame:
    """Reproduce the epilepsiae_1125 wrong-time shape: one seizure, all leads."""
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=6)
    frame = frame[frame["seizure_id"] == "sz000"].copy()
    frame["evaluation_tier"] = "not_estimable"
    frame["split"] = "NOT_ESTIMABLE"
    return frame


def _multi_lead_anchor_table(n_seizures: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors, seizures = [], []
    leads = (5, 15, 30, 60, 120)
    for index in range(n_seizures):
        onset = 100_000.0 + 20_000.0 * index
        seizure_id = f"sz{index:02d}"
        seizures.append({
            "patient_id": "p1", "seizure_id": seizure_id,
            "onset_time": np.float64(onset), "segment_id": "segment0",
        })
        times = [(f"case_{index}_{lead}", onset - 60.0 * lead) for lead in leads]
        times += [(f"control_{index}_{c}", onset - 15_000.0 - 100.0 * c)
                  for c in range(8)]
        for anchor_id, anchor_time in times:
            anchors.append({
                "patient_id": "p1", "seed": 0, "anchor_id": anchor_id,
                "anchor_time": np.float64(anchor_time), "segment_id": "segment0",
                "segment_start": np.float64(0.0),
                "segment_end": np.float64(400_000.0),
                "observation_available": True,
                "observation_signature": "complete_10m",
                "in_ictal_or_postictal": False,
                "wrong_time_donor_valid": True,
                "wrong_time_same_segment": True,
                "wrong_time_exclusion_clear": True,
                "history__recent_count": float(index),
                "observation__spectral": float(anchor_time % 17),
                "state__persistent_0": float(anchor_time % 23),
                "memoryless__code_0": float(anchor_time % 19),
                "wrong_time__state_0": float(anchor_time % 29),
            })
    return pd.DataFrame(anchors), pd.DataFrame(seizures)


def _primary_lead_starved_table() -> pd.DataFrame:
    """Estimable at the other leads, one seizure at the fixed 30 min lead."""
    anchors, seizures = _multi_lead_anchor_table(3)
    frame, _ = build_risk_sets(anchors, seizures, controls_per_case=3, random_seed=7)
    starved = ~(
        (frame["lead_minutes"] == 30) & (frame["seizure_id"].astype(str) != "sz00")
    )
    return frame[starved].copy()


def test_permutation_audit_reports_structured_not_estimable_instead_of_keyerror():
    frame = _single_seizure_not_estimable_table()

    audit = time_label_permutation_audit(frame, n_permutations=8, random_seed=5)

    assert audit["status"] == "NOT_ESTIMABLE_AT_PRIMARY_LEAD"
    assert audit["n_permutations_run"] == 0
    assert audit["n_finite_permutations"] == 0
    assert audit["primary_lead_minutes"] == 30
    assert "insufficient eligible seizures" in audit["reason"]


def test_not_estimable_permutation_never_fabricates_an_effect_or_a_null():
    frame = _single_seizure_not_estimable_table()

    audit = time_label_permutation_audit(frame, n_permutations=8, random_seed=5)

    for explicit_null in (
        "observed_state_minus_observation",
        "null_median", "null_mean", "null_q025", "null_q975", "null_values",
    ):
        value = audit[explicit_null]
        assert value is None or value == [], f"{explicit_null} fabricated a result"


def test_estimability_is_judged_at_the_fixed_primary_lead_only():
    """A table that fits at 5/15 min but not at 30 min is NOT estimable.

    v0.2 contract §3: the primary lead is fixed at 30 min and may not be
    replaced by whichever lead happens to fit.
    """
    frame = _primary_lead_starved_table()
    fitted = run_probe_table(frame)

    # The column exists, because the shorter leads did fit.
    assert "state_minus_observation_conditional_log_loss" in fitted.patient_medians
    thirty = fitted.patient_medians["lead_minutes"] == 30
    assert thirty.any(), "the 30 min rows must still be present"
    assert not np.isfinite(
        fitted.patient_medians.loc[
            thirty, "state_minus_observation_conditional_log_loss"
        ].to_numpy(dtype=float)
    ).any()

    audit = time_label_permutation_audit(frame, n_permutations=8, random_seed=5)
    assert audit["status"] == "NOT_ESTIMABLE_AT_PRIMARY_LEAD"
    assert audit["n_permutations_run"] == 0


def test_estimable_table_still_runs_the_permutation_null():
    frame = make_positive_synthetic_risk_table(
        n_seizures=12, state_strength=4.0, random_seed=4,
    )

    audit = time_label_permutation_audit(frame, n_permutations=4, random_seed=5)

    assert audit["status"] == "COMPLETE"
    assert audit["n_permutations_run"] == 4
    assert len(audit["null_values"]) == 4
    assert np.isfinite(audit["observed_state_minus_observation"])


def test_primary_lead_estimability_reports_every_contract_effect():
    """v0.2 §3 names two necessary explanatory effects alongside the main one."""
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=6)
    report = primary_lead_estimability(run_probe_table(frame).patient_medians)

    assert report["estimable"] is True
    assert set(report["effects"]) == {
        "state_minus_observation_conditional_log_loss",
        "persistent_minus_memoryless_conditional_log_loss",
        "correct_minus_wrong_time_conditional_log_loss",
    }
    assert report["effects"][
        "state_minus_observation_conditional_log_loss"
    ]["n_finite_patients"] >= 1


def test_probe_cli_completes_execution_but_records_not_estimable(
        tmp_path, probe_output_dir):
    """Engineering completion and scientific estimability are separate fields."""
    table = tmp_path / "matched_wrong_time_risk_sets.csv"
    _single_seizure_not_estimable_table().to_csv(table, index=False)
    output = probe_output_dir

    completed = subprocess.run(
        [sys.executable,
         "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
         "--risk-table", str(table), "--output-dir", str(output),
         "--arms", "B_history", "B_observation", "B_state", "memoryless",
         "wrong_time",
         "--h2b-revision", "continuous_marked_state_h2b_cross_task_v0_2",
         "--n-permutations", "8", "--overwrite"],
        cwd=REPO, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    audit = json.loads((output / "risk_probe_machine_audit.json").read_text())
    assert audit["status"] == "COMPLETE"
    assert audit["execution_status"] == "COMPLETE"
    assert audit["scientific_estimability"] == "NOT_ESTIMABLE"
    assert audit["time_label_permutation"]["status"] == (
        "NOT_ESTIMABLE_AT_PRIMARY_LEAD"
    )
    assert audit["time_label_permutation"]["n_permutations_run"] == 0
    for name in (
        "per_seed_probe_metrics.csv", "patient_median_probe_metrics.csv",
        "lead_curve.csv", "time_label_permutation.json",
        "positive_synthetic.json",
    ):
        assert (output / name).stat().st_size > 0, f"{name} is empty"
    lead_curve = pd.read_csv(output / "lead_curve.csv")
    assert list(lead_curve.columns), "lead curve lost its header"
    assert lead_curve.empty


def test_estimable_probe_cli_records_estimable(tmp_path, probe_output_dir):
    table = tmp_path / "primary_risk_sets.csv"
    make_positive_synthetic_risk_table(
        n_seizures=12, state_strength=4.0, random_seed=4,
    ).to_csv(table, index=False)
    output = probe_output_dir

    completed = subprocess.run(
        [sys.executable,
         "scripts/topic5_continuous_marked_state_h2b/run_risk_probe.py",
         "--risk-table", str(table), "--output-dir", str(output),
         "--arms", "B_history", "B_observation", "B_state", "memoryless",
         "wrong_time",
         "--h2b-revision", "continuous_marked_state_h2b_cross_task_v0_2",
         "--n-permutations", "4", "--overwrite"],
        cwd=REPO, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr[-3000:]
    audit = json.loads((output / "risk_probe_machine_audit.json").read_text())
    assert audit["execution_status"] == "COMPLETE"
    assert audit["scientific_estimability"] == "ESTIMABLE"
    assert audit["time_label_permutation"]["status"] == "COMPLETE"
