"""The consumer side must not launder an unaudited upstream into a clean release.

v0.1 contract §5 admits an R1.7 release into H2b only on a COMPLETE machine audit
covering exactly 50 fits.  v0.2 reads R1.7B instead -- 17 subjects x 5 seeds, an
exploratory extension with no machine audit of its own.  These tests pin the two
things that matter: the checks actually re-verify the upstream rather than trust
its summary, and the unmet gate is reported rather than quietly satisfied.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.topic5_continuous_marked_state_h2b.audit_r1_7b_release import (
    EXPECTED_CELLS,
    EXPECTED_SUBJECTS,
    RELEASE_LABEL,
    V0_1_STAGE2_REQUIRED_FITS,
    _audit_cells,
    _audit_h1_is_not_a_filter,
)
from src.topic5_continuous_marked_state_h2b.contract import (
    CANONICAL_V0_2_RESULT_ROOT,
)


LIVE_AUDIT = CANONICAL_V0_2_RESULT_ROOT / "reports/r1_7b_consumer_acceptance_audit.json"


def _cell(subject: str, seed: int, *, available: bool = True) -> dict:
    return {
        "subject": subject, "seed": seed,
        "analysis_status": "SCORED" if available else "NONFINITE_GRADIENT",
        "checkpoint_available": available,
        "checkpoint_path": False, "checkpoint_sha256": None,
        "result_path": False, "result_sha256": None,
        "source_revision": "r1_7b_extended_development_cohort_v1",
        "formal_test_partition_opened": False, "sealed_opened": False,
        "state_source_uses_seizure_labels": False,
        "seizure_gradient_path": False,
    }


def _inventory(**overrides) -> dict:
    entries = [
        _cell(f"s{index:02d}", seed, available=False)
        for index in range(EXPECTED_SUBJECTS) for seed in range(5)
    ]
    payload = {"entries": entries}
    payload.update(overrides)
    return payload


def test_cell_audit_accepts_the_frozen_shape():
    summary = _audit_cells(_inventory())

    assert summary["n_cells"] == EXPECTED_CELLS
    assert summary["n_subjects"] == EXPECTED_SUBJECTS
    assert summary["cells_reference_a_temporary_worktree"] is False


def test_cell_audit_rejects_a_missing_cell():
    inventory = _inventory()
    inventory["entries"] = inventory["entries"][:-1]

    with pytest.raises(ValueError, match="cells"):
        _audit_cells(inventory)


def test_cell_audit_rejects_a_checkpoint_inside_a_temporary_worktree():
    inventory = _inventory()
    inventory["entries"][0]["checkpoint_path"] = "/tmp/hfosp_r17_20260827/model.pt"

    with pytest.raises(ValueError, match="temporary worktree"):
        _audit_cells(inventory)


def test_cell_audit_rejects_a_seizure_label_path_in_the_state_source():
    inventory = _inventory()
    inventory["entries"][3]["state_source_uses_seizure_labels"] = True

    with pytest.raises(ValueError, match="seizure labels"):
        _audit_cells(inventory)


def test_cell_audit_rejects_a_missing_checkpoint_dressed_as_scored():
    inventory = _inventory()
    inventory["entries"][2]["analysis_status"] = "SCORED"

    with pytest.raises(ValueError, match="no checkpoint"):
        _audit_cells(inventory)


def test_cell_audit_rejects_a_mixed_source_revision():
    inventory = _inventory()
    inventory["entries"][7]["source_revision"] = "some_other_release"

    with pytest.raises(ValueError, match="source revisions"):
        _audit_cells(inventory)


def _h1_root(tmp_path: Path, reasons: dict[str, str], probed: list[str]) -> Path:
    root = tmp_path / "v0_2"
    (root / "manifests").mkdir(parents=True)
    for subject in probed:
        directory = root / "risk_sets" / subject
        directory.mkdir(parents=True)
        (directory / "input_manifest.json").write_text("{}", encoding="utf-8")
    rows = [
        {"subject": subject, "exclusion_or_deferred_reason": "ready"}
        for subject in probed
    ] + [
        {"subject": subject, "exclusion_or_deferred_reason": reason}
        for subject, reason in reasons.items()
    ]
    pd.DataFrame(rows).to_csv(
        root / "manifests/patient_support_census.csv", index=False,
    )
    return root


def test_h1_check_accepts_a_mixed_stratum_run(tmp_path):
    root = _h1_root(
        tmp_path, {"c": "no_frozen_seizures"}, ["a", "b"],
    )

    report = _audit_h1_is_not_a_filter(root, {"h1_stable_subjects": ["a"]})

    assert report["h1_used_as_entry_gate"] is False
    assert report["probed_h1_stable"] == ["a"]
    assert report["probed_h1_unstable"] == ["b"]


def test_h1_check_rejects_a_run_that_kept_only_the_stable_stratum(tmp_path):
    root = _h1_root(tmp_path, {"c": "no_frozen_seizures"}, ["a", "b"])

    with pytest.raises(ValueError, match="one H1 stratum"):
        _audit_h1_is_not_a_filter(root, {"h1_stable_subjects": ["a", "b"]})


def test_h1_check_rejects_an_outcome_flavoured_exclusion_reason(tmp_path):
    root = _h1_root(tmp_path, {"c": "h1_unstable_subject"}, ["a", "b"])

    with pytest.raises(ValueError, match="H1-flavoured"):
        _audit_h1_is_not_a_filter(root, {"h1_stable_subjects": ["a"]})


@pytest.mark.skipif(not LIVE_AUDIT.is_file(), reason="live audit not produced yet")
def test_live_audit_declares_the_unmet_v0_1_gate_rather_than_claiming_it():
    payload = json.loads(LIVE_AUDIT.read_text())
    gate = payload["v0_1_stage2_release_gate"]

    assert payload["status"] == "PASS_EXPLORATORY_DEVELOPMENT_SOURCE"
    assert payload["release_label"] == RELEASE_LABEL
    assert gate["gate_met_by_consumed_release"] is False
    assert gate["r1_7b_machine_audit_present"] is False
    assert gate["r1_7b_n_fits"] != V0_1_STAGE2_REQUIRED_FITS
    assert gate["weakening_is_declared_not_silent"] is True
    assert payload["provenance"]["r1_7_worktree_still_present"] is False
    assert all(row["pushed"] for row in payload["provenance"]["r1_7b_commits"])
    assert payload["h1_stratification"]["probed_h1_stable"]
    assert payload["h1_stratification"]["probed_h1_unstable"]
