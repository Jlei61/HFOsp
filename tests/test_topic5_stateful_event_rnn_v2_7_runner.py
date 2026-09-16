import json
from pathlib import Path

import yaml

from scripts import run_topic5_stateful_event_rnn_v2_7 as runner
from scripts import refine_topic5_stateful_event_rnn_v2_7_epoch_boundary as boundary


def test_v27_config_matches_v26_except_namespace():
    child = runner.assert_repair_only_config(runner.DEFAULT_CONFIG)
    parent = yaml.safe_load(runner.PARENT_CONFIG.open())
    assert child["contract"] == "topic5_stateful_event_sequence_rnn_v2_7"
    assert child["output_root"].endswith("/v2_7")
    for key in ("contract", "output_root"):
        child.pop(key)
        parent.pop(key)
    assert child == parent


def test_validation_runner_is_bound_to_repaired_fit():
    assert runner.fit_stateful_event_rnn.__module__ == (
        "src.topic5_stateful_event_rnn_v2_7"
    )


def test_provenance_manifest_verifies_parent_and_new_namespace():
    manifest = runner.provenance_manifest(runner.DEFAULT_CONFIG)
    parent_state = json.load(runner.PARENT_FROZEN_STATE.open())
    assert manifest["repair_only_grid_match"] is True
    assert manifest["parent_v2_6"]["config_sha256"] == parent_state[
        "config_sha256"
    ]
    assert manifest["parent_v2_6"]["module_sha256"] == parent_state[
        "module_sha256"
    ]
    assert manifest["parent_v2_6"]["runner_sha256"] == parent_state[
        "runner_sha256"
    ]
    assert manifest["v2_7"]["module_sha256"] == runner.sha256(
        runner.V27_MODULE
    )
    assert manifest["v2_7"]["runner_sha256"] == runner.sha256(
        runner.V27_RUNNER
    )
    assert manifest["v2_7"]["cohort_worker_sha256"] == runner.sha256(
        runner.V27_WORKER
    )


def test_incomplete_freeze_is_fail_closed_and_records_hashes(tmp_path: Path):
    config = runner.assert_repair_only_config(runner.DEFAULT_CONFIG)
    state = runner.freeze_screen(
        config,
        runner.DEFAULT_CONFIG,
        tmp_path,
        subjects=["intentionally_missing_subject"],
    )
    assert state["status"] == "INCOMPLETE"
    assert state["n_completed"] == 0
    assert state["n_failed"] == 1
    assert state["test_results_read_during_selection"] is False
    assert state["repair_only_grid_match"] is True
    assert state["parent_v2_6"]["frozen_status"] == (
        "ALL_PATIENT_VALIDATION_PROFILES_FROZEN"
    )
    written = json.load(
        (tmp_path / "validation_screen/FROZEN_VALIDATION_STATE.json").open()
    )
    assert written == state


def test_epoch_boundary_rebinds_training_to_v27(monkeypatch):
    observed = {}

    def fake_refine(subject, config, output, **kwargs):
        observed["fit"] = boundary.parent_boundary.fit_profile
        return {"subject": subject, "status": "NOT_TRIGGERED", "profile_changed": False}

    monkeypatch.setattr(boundary.parent_boundary, "refine_subject", fake_refine)
    result = boundary.refine_subject(
        "subject", {}, Path("unused"), top_n=3, trigger_epoch=35,
        maximum_epochs=100, patience=16
    )
    assert result["status"] == "NOT_TRIGGERED"
    assert observed["fit"] is boundary.fit_profile


def test_epoch_boundary_adapter_accepts_parent_trigger_signature(monkeypatch):
    observed = {}

    def fake_fit(profile, datasets, encoder, config, scales, seed):
        observed.update(
            profile=profile,
            datasets=datasets,
            encoder=encoder,
            config=config,
            scales=scales,
            seed=seed,
        )
        return "fitted", 1.25

    monkeypatch.setattr(boundary, "fit_profile_v27", fake_fit)
    result = boundary.fit_profile(
        "subject", "profile", "datasets", "encoder", "config", "scales", 7
    )
    assert result == ("fitted", 1.25)
    assert observed == {
        "profile": "profile",
        "datasets": "datasets",
        "encoder": "encoder",
        "config": "config",
        "scales": "scales",
        "seed": 7,
    }
