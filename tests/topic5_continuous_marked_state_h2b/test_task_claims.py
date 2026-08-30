"""Per-patient/arm task ownership for the durable v0.2 cohort queue.

The v0.2 run lost its orchestrator to a SIGSTOP while three probe children kept
computing, and a second operator started a fourth probe by hand outside the
queue.  A queue-level ``flock`` cannot prevent that: it serialises *coordinators*,
not *tasks*.  Ownership therefore has to live next to each task's own output, be
taken by an atomic exclusive create, and be reclaimable only when the holder is
provably gone.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.topic5_continuous_marked_state_h2b.run_v02_cohort_queue import (
    HEARTBEAT_STALE_SECONDS,
    _claim_path,
    _claim_probe_task,
    _completed_probe_task,
    _record_task_completion,
    _release_claim,
    _write_heartbeat,
)
from src.topic5_continuous_marked_state_h2b.contract import (
    H2B_V0_2_REVISION,
    sha256_file,
)


PRODUCTS = (
    "per_seed_probe_metrics.csv",
    "patient_median_probe_metrics.csv",
    "lead_curve.csv",
    "time_label_permutation.json",
    "positive_synthetic.json",
)


def _risk_table(tmp_path: Path, body: str = "risk_set_id,is_case\nr0,1\n") -> Path:
    table = tmp_path / "risk.csv"
    table.write_text(body, encoding="utf-8")
    return table


def _finished_output(
    root: Path,
    label: str,
    risk_table: Path,
    *,
    estimability: str = "ESTIMABLE",
) -> Path:
    subject, analysis = label.split("/", maxsplit=1)
    output = root / "fits/by_subject" / subject / analysis
    output.mkdir(parents=True, exist_ok=True)
    for name in PRODUCTS:
        (output / name).write_text("{}\n", encoding="utf-8")
    (output / "risk_probe_machine_audit.json").write_text(json.dumps({
        "status": "COMPLETE",
        "execution_status": "COMPLETE",
        "scientific_estimability": estimability,
        "boundary": {"revision": H2B_V0_2_REVISION},
        "input": {"risk_table_sha256": sha256_file(risk_table)},
        "positive_synthetic": {"status": "PASS"},
    }), encoding="utf-8")
    return output


def test_two_workers_race_and_only_one_claim_wins(tmp_path):
    table = _risk_table(tmp_path)

    first = _claim_probe_task(tmp_path, "p/primary", table, command=["a"])
    second = _claim_probe_task(tmp_path, "p/primary", table, command=["a"])

    assert first is not None
    assert second is None, "a live claim was handed to a second worker"


def test_claim_records_every_ownership_field(tmp_path):
    table = _risk_table(tmp_path)

    claim = _claim_probe_task(tmp_path, "p/primary", table, command=["cmd", "x"])

    for field in (
        "patient_id", "analysis", "pid", "pgid", "hostname", "source_commit",
        "risk_table_path", "risk_table_sha256", "command", "claimed_at_utc",
        "heartbeat_utc", "output_dir",
    ):
        assert field in claim, f"claim is missing {field}"
    assert claim["pid"] == os.getpid()
    assert claim["pgid"] == os.getpgid(0)
    assert claim["risk_table_sha256"] == sha256_file(table)
    assert claim["patient_id"] == "p"
    assert claim["analysis"] == "primary"


def test_a_claim_held_by_a_live_pid_is_never_preempted(tmp_path):
    table = _risk_table(tmp_path)
    _claim_probe_task(tmp_path, "p/primary", table, command=["a"])
    # Age the heartbeat far past the staleness window; the holder is still alive.
    path = _claim_path(tmp_path, "p/primary")
    claim = json.loads(path.read_text())
    claim["heartbeat_epoch"] -= 10 * HEARTBEAT_STALE_SECONDS
    path.write_text(json.dumps(claim), encoding="utf-8")

    assert _claim_probe_task(tmp_path, "p/primary", table, command=["a"]) is None


def test_stale_claim_from_a_dead_pid_is_reclaimed(tmp_path):
    table = _risk_table(tmp_path)
    path = _claim_path(tmp_path, "p/primary")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "patient_id": "p", "analysis": "primary",
        "pid": 2 ** 22, "pgid": 2 ** 22, "hostname": "gone",
        "risk_table_sha256": sha256_file(table),
        "heartbeat_epoch": 0.0,
    }), encoding="utf-8")

    claim = _claim_probe_task(tmp_path, "p/primary", table, command=["a"])

    assert claim is not None
    assert claim["pid"] == os.getpid()
    assert claim["reclaimed_from"]["pid"] == 2 ** 22


def test_dead_holder_with_a_fresh_heartbeat_is_not_reclaimed(tmp_path):
    """Both conditions are required: the PID gone AND the heartbeat expired."""
    import time

    table = _risk_table(tmp_path)
    path = _claim_path(tmp_path, "p/primary")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "patient_id": "p", "analysis": "primary",
        "pid": 2 ** 22, "pgid": 2 ** 22, "hostname": "gone",
        "risk_table_sha256": sha256_file(table),
        "heartbeat_epoch": time.time(),
    }), encoding="utf-8")

    assert _claim_probe_task(tmp_path, "p/primary", table, command=["a"]) is None


def test_stale_claim_with_a_changed_input_is_not_reclaimed_blindly(tmp_path):
    table = _risk_table(tmp_path)
    path = _claim_path(tmp_path, "p/primary")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "patient_id": "p", "analysis": "primary",
        "pid": 2 ** 22, "pgid": 2 ** 22, "hostname": "gone",
        "risk_table_sha256": "0" * 64,
        "heartbeat_epoch": 0.0,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="input"):
        _claim_probe_task(tmp_path, "p/primary", table, command=["a"])


def test_released_claim_can_be_taken_again(tmp_path):
    table = _risk_table(tmp_path)
    _claim_probe_task(tmp_path, "p/primary", table, command=["a"])
    _release_claim(tmp_path, "p/primary")

    assert _claim_probe_task(tmp_path, "p/primary", table, command=["a"]) is not None


def test_heartbeat_advances_without_losing_ownership(tmp_path):
    table = _risk_table(tmp_path)
    claim = _claim_probe_task(tmp_path, "p/primary", table, command=["a"])
    before = json.loads(_claim_path(tmp_path, "p/primary").read_text())

    _write_heartbeat(tmp_path, "p/primary")

    after = json.loads(_claim_path(tmp_path, "p/primary").read_text())
    assert after["heartbeat_epoch"] >= before["heartbeat_epoch"]
    assert after["pid"] == claim["pid"]


def test_completed_task_is_skipped_and_binds_output_hashes(tmp_path):
    table = _risk_table(tmp_path)
    output = _finished_output(tmp_path, "p/primary", table)
    _record_task_completion(tmp_path, "p/primary", table)

    row = _completed_probe_task(tmp_path, "p/primary", table)

    assert row is not None
    assert row["status"] == "SKIPPED_COMPLETE_INPUT_BOUND"
    receipt = json.loads((output / ".task_complete.json").read_text())
    assert set(receipt["output_sha256"]) == {*PRODUCTS, "risk_probe_machine_audit.json"}


def test_changed_input_forces_a_rerun(tmp_path):
    table = _risk_table(tmp_path)
    _finished_output(tmp_path, "p/primary", table)
    _record_task_completion(tmp_path, "p/primary", table)

    table.write_text("risk_set_id,is_case\nr0,0\n", encoding="utf-8")

    assert _completed_probe_task(tmp_path, "p/primary", table) is None


def test_tampered_output_is_not_accepted_as_complete(tmp_path):
    table = _risk_table(tmp_path)
    output = _finished_output(tmp_path, "p/primary", table)
    _record_task_completion(tmp_path, "p/primary", table)

    (output / "lead_curve.csv").write_text("tampered\n", encoding="utf-8")

    assert _completed_probe_task(tmp_path, "p/primary", table) is None


def test_partial_output_set_is_never_accepted(tmp_path):
    table = _risk_table(tmp_path)
    output = tmp_path / "fits/by_subject/p/primary"
    output.mkdir(parents=True)
    (output / "risk_probe_machine_audit.json").write_text(
        json.dumps({"status": "COMPLETE"}), encoding="utf-8",
    )

    assert _completed_probe_task(tmp_path, "p/primary", table) is None


def test_not_estimable_probe_counts_as_a_legitimate_completion(tmp_path):
    """A structurally unestimable patient finishes; it is not retried forever."""
    table = _risk_table(tmp_path)
    _finished_output(tmp_path, "p/matched_wrong_time", table,
                     estimability="NOT_ESTIMABLE")
    _record_task_completion(tmp_path, "p/matched_wrong_time", table)

    row = _completed_probe_task(tmp_path, "p/matched_wrong_time", table)

    assert row is not None
    assert row["scientific_estimability"] == "NOT_ESTIMABLE"
    assert row["returncode"] == 0


def test_legacy_output_without_a_receipt_adopts_its_digests_once(tmp_path):
    """The 11 probes already on disk predate any stored output digest."""
    table = _risk_table(tmp_path)
    output = _finished_output(tmp_path, "p/primary", table)

    row = _completed_probe_task(tmp_path, "p/primary", table)

    assert row is not None
    receipt = json.loads((output / ".task_complete.json").read_text())
    assert receipt["hash_baseline_adopted_from_existing_output"] is True
    # Once adopted the digests are enforced like any other.
    (output / "lead_curve.csv").write_text("tampered\n", encoding="utf-8")
    assert _completed_probe_task(tmp_path, "p/primary", table) is None


def test_complete_legacy_null_gets_metadata_only_denominator_migration(tmp_path):
    table = _risk_table(tmp_path)
    output = _finished_output(tmp_path, "p/primary", table)
    permutation_path = output / "time_label_permutation.json"
    permutation_path.write_text(json.dumps({
        "status": "COMPLETE", "n_permutations": 3,
        "observed_state_minus_observation": -0.2,
        "null_values": [-0.1, 0.0, 0.1],
    }), encoding="utf-8")
    audit_path = output / "risk_probe_machine_audit.json"
    audit = json.loads(audit_path.read_text())
    audit["time_label_permutation"] = {"status": "COMPLETE"}
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    _record_task_completion(tmp_path, "p/primary", table)
    before = json.loads((output / ".task_complete.json").read_text())

    row = _completed_probe_task(tmp_path, "p/primary", table)

    assert row is not None
    assert row["permutation_denominator_metadata_migrated"] is True
    permutation = json.loads(permutation_path.read_text())
    assert permutation["n_permutations_run"] == 3
    assert permutation["n_finite_permutations"] == 3
    audit = json.loads(audit_path.read_text())
    assert audit["time_label_permutation"]["n_permutations_run"] == 3
    assert audit["execution_status"] == "COMPLETE"
    assert audit["scientific_estimability"] == "ESTIMABLE"
    assert audit["permutation_denominator_schema_migration"][
        "scientific_values_changed"
    ] is False
    after = json.loads((output / ".task_complete.json").read_text())
    assert after["metadata_migration"]["n_null_values"] == 3
    assert after["output_sha256"] != before["output_sha256"]


def test_foreign_writer_scan_finds_a_live_probe_for_the_same_output(tmp_path):
    """A live writer with no claim file must still be seen (the orphan case)."""
    import subprocess
    import sys
    import time

    from scripts.topic5_continuous_marked_state_h2b.run_v02_cohort_queue import (
        _foreign_writer_pids, _task_output_dir,
    )

    output = _task_output_dir(tmp_path, "p/primary")
    output.mkdir(parents=True)
    impostor = tmp_path / "run_risk_probe.py"
    impostor.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, str(impostor), "--output-dir", str(output)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.pid in _foreign_writer_pids(tmp_path, "p/primary"):
                break
            time.sleep(0.1)
        assert process.pid in _foreign_writer_pids(tmp_path, "p/primary")
        assert _foreign_writer_pids(tmp_path, "p/matched_wrong_time") == []

        table = _risk_table(tmp_path)
        assert _claim_probe_task(
            tmp_path, "p/primary", table, command=["a"],
        ) is None, "claimed a task another live process is already writing"
    finally:
        process.kill()
        process.wait()


def test_foreign_writer_scan_is_empty_when_nobody_is_writing(tmp_path):
    from scripts.topic5_continuous_marked_state_h2b.run_v02_cohort_queue import (
        _foreign_writer_pids,
    )

    assert _foreign_writer_pids(tmp_path, "p/primary") == []
