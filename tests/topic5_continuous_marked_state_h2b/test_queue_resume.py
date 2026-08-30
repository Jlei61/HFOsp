from __future__ import annotations

import json

from scripts.topic5_continuous_marked_state_h2b.run_v02_cohort_queue import (
    _completed_probe_task,
)
from src.topic5_continuous_marked_state_h2b.contract import (
    H2B_V0_2_REVISION,
    sha256_file,
)


def test_completed_probe_task_requires_all_outputs_and_exact_input_hash(tmp_path):
    risk_table = tmp_path / "risk.csv"
    risk_table.write_text("risk_set_id,is_case\nr0,1\n", encoding="utf-8")
    output = tmp_path / "fits/by_subject/patient_a/primary"
    output.mkdir(parents=True)
    for name in (
        "per_seed_probe_metrics.csv",
        "patient_median_probe_metrics.csv",
        "lead_curve.csv",
        "time_label_permutation.json",
        "positive_synthetic.json",
    ):
        (output / name).write_text("{}\n", encoding="utf-8")
    audit_path = output / "risk_probe_machine_audit.json"
    audit = {
        "status": "COMPLETE",
        "boundary": {"revision": H2B_V0_2_REVISION},
        "input": {"risk_table_sha256": sha256_file(risk_table)},
        "positive_synthetic": {"status": "PASS"},
    }
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    row = _completed_probe_task(
        tmp_path, "patient_a/primary", risk_table,
    )
    assert row is not None
    assert row["status"] == "SKIPPED_COMPLETE_INPUT_BOUND"

    risk_table.write_text("risk_set_id,is_case\nr0,0\n", encoding="utf-8")
    assert _completed_probe_task(
        tmp_path, "patient_a/primary", risk_table,
    ) is None


def test_completed_probe_task_rejects_partial_output_set(tmp_path):
    risk_table = tmp_path / "risk.csv"
    risk_table.write_text("risk_set_id,is_case\nr0,1\n", encoding="utf-8")
    output = tmp_path / "fits/by_subject/patient_a/primary"
    output.mkdir(parents=True)
    (output / "risk_probe_machine_audit.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8",
    )
    assert _completed_probe_task(
        tmp_path, "patient_a/primary", risk_table,
    ) is None
