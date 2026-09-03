from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic5_group_event_state.v033_training_lab.views import ViewHeld, view_for_request


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    subject = "epilepsiae_tuning"
    commit = "a" * 40
    evaluator = tmp_path / "canonical.json"
    evaluator.write_text(json.dumps({"status": "ACTIVE_DEVELOPMENT_ONLY"}))
    evaluator_hash = _sha(evaluator)
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "status": "RELEASED_DEVELOPMENT_ONLY", "canonical_sha256": evaluator_hash,
        "sealed_partition_opened": False,
    }))

    h_hash = "b" * 64
    h_path = tmp_path / "h_mark.npz"
    np.savez_compressed(h_path, artifact_hash=np.asarray(h_hash))

    anchor_time = np.asarray([1000, 2000, 3000, 4000, 6000, 7000, 8000, 9000, 10000, 11000], dtype=float)
    phase = np.asarray([
        "CALIBRATION", "CALIBRATION", "STATE_TRAIN", "STATE_TRAIN", "STATE_TRAIN",
        "STATE_SELECTION", "STATE_SELECTION", "DEVELOPMENT_EVALUATION",
        "DEVELOPMENT_EVALUATION", "DEVELOPMENT_EVALUATION",
    ])
    event_time = np.asarray([100, 900, 1500, 2500, 3500, 4500, 6100, 6500, 7500, 8500,
                             9500, 10500, 11500, 12500], dtype=float)
    target_bounds = np.asarray([[0, 6000], [6000, 14000]], dtype=float)
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    event_target = np.where(event_time < 6000, 0, 1)
    bins = ((0, 300), (300, 900), (900, 1800))
    counts = np.zeros((anchor_time.size, 3), dtype=np.int64)
    for row, (time, segment) in enumerate(zip(anchor_time, target)):
        for column, (left, right) in enumerate(bins):
            counts[row, column] = np.sum(
                (event_target == segment) & (event_time >= time + left) & (event_time < time + right)
            )
    last = np.searchsorted(event_time, anchor_time, side="left") - 1
    split_hash = "c" * 64
    registry = "d" * 64
    content_hash = "e" * 64
    metadata = {
        "format": "group_event_state_v0_3_3_materialized_human_r0_input", "schema_version": 1,
        "subject": subject, "sealed": False, "bin_convention": "left_closed_right_open_[t+a,t+b)",
        "science_code_commit": commit, "builder_code_commit": commit,
        "data_registry_key": registry, "split_hash": split_hash,
        "h_mark_artifact_hash": h_hash, "event_feature_names_r0": ["size", "energy"],
    }
    artifact = tmp_path / "human.npz"
    np.savez_compressed(
        artifact, metadata_json=np.asarray(json.dumps(metadata)), artifact_hash=np.asarray(content_hash),
        anchor_id=np.asarray([f"a{i}" for i in range(anchor_time.size)]), anchor_time=anchor_time,
        phase=phase, anchor_carry=np.zeros(anchor_time.size, dtype=np.int64), target_segment=target,
        carry_bounds=np.asarray([[0, 14000]], dtype=float), target_segment_bounds=target_bounds,
        last_event_pos=last, eligible_by_horizon=np.ones((anchor_time.size, 3), dtype=bool),
        target_counts=counts, log_mu_h_mark=np.zeros_like(counts, dtype=float),
        nb_log_r_train_frozen=np.zeros(3), event_id=np.asarray([f"e{i}" for i in range(event_time.size)]),
        event_time=event_time, event_carry=np.zeros(event_time.size, dtype=np.int64),
        event_features_r0=np.ones((event_time.size, 2), dtype=np.float32),
        event_feature_valid=np.ones((event_time.size, 2), dtype=bool),
        train_event_mask=event_time < 7000,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "format": "group_event_state_v0_3_3_human_r0_input_manifest", "subject": subject, "role": "tuning",
        "sealed": False, "development_evaluation_used_for_fitting": False,
        "input_path": str(artifact), "input_npz_sha256": _sha(artifact), "input_artifact_hash": content_hash,
        "h_mark_path": str(h_path), "h_mark_npz_sha256": _sha(h_path), "h_mark_artifact_hash": h_hash,
        "evaluator_contract": str(evaluator), "evaluator_contract_sha256": evaluator_hash,
        "science_code_commit": commit, "builder_code_commit": commit,
        "data_registry_key": registry, "split_hash": split_hash,
    }))
    request = {
        "input_view": {
            "kind": "R0", "subject": subject, "data_registry_key": registry,
            "materialized_arrays_only": True, "artifact_path": str(artifact),
            "artifact_sha256": _sha(artifact), "artifact_content_sha256": content_hash,
            "artifact_manifest": str(manifest), "artifact_manifest_sha256": _sha(manifest),
            "evaluator_contract": str(evaluator), "evaluator_contract_sha256": evaluator_hash,
            "canonical_evaluator_release": str(release), "canonical_evaluator_release_sha256": _sha(release),
        },
        "scientific_target": {"bins_seconds": [list(value) for value in bins],
                              "bin_convention": "left_closed_right_open_[t+a,t+b)"},
        "baseline_H": {"name": "H_mark", "hash": h_hash},
        "input_hash": _sha(artifact), "split_hash": split_hash, "science_code_commit": commit,
    }
    return request, phase


def test_materialized_human_r0_uses_continuous_carry_but_seizure_cut_targets_and_hides_development(tmp_path):
    request, phase = _fixture(tmp_path)
    view, meta = view_for_request(request, release_present=True, scaling="robust")
    assert view.n("train") == 3 and view.n("inner_val") == 2
    assert np.unique(view.anchor_segment).size == 1
    assert np.unique(view.target_segment).size == 2
    assert np.all(view.counts[phase == "DEVELOPMENT_EVALUATION"] == -1)
    assert np.isnan(view.log_mu_h[phase == "DEVELOPMENT_EVALUATION"]).all()
    assert view.h_source == "materialized_agent_c_H_mark"
    assert meta["development_evaluation_exposed"] is False


def test_materialized_human_r0_refuses_an_unreleased_or_changed_evaluator(tmp_path):
    request, _phase = _fixture(tmp_path)
    Path(request["input_view"]["canonical_evaluator_release"]).write_text(json.dumps({
        "status": "HOLD", "canonical_sha256": request["input_view"]["evaluator_contract_sha256"],
        "sealed_partition_opened": False,
    }))
    request["input_view"]["canonical_evaluator_release_sha256"] = _sha(
        Path(request["input_view"]["canonical_evaluator_release"])
    )
    with pytest.raises(ViewHeld, match="not released"):
        view_for_request(request, release_present=True, scaling="robust")


def test_non_tuning_human_r0_requires_explicit_trainability_expansion_authorization(tmp_path):
    request, _phase = _fixture(tmp_path)
    manifest_path = Path(request["input_view"]["artifact_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["role"] = "explicit_non_tuning_override"
    manifest_path.write_text(json.dumps(manifest))
    request["input_view"]["artifact_manifest_sha256"] = _sha(manifest_path)
    with pytest.raises(ViewHeld, match="trainability-expansion"):
        view_for_request(request, release_present=True, scaling="robust")


def test_non_tuning_human_r0_is_allowed_only_as_optimization_diagnostic(tmp_path):
    request, _phase = _fixture(tmp_path)
    manifest_path = Path(request["input_view"]["artifact_manifest"])
    manifest = json.loads(manifest_path.read_text())
    manifest["role"] = "explicit_non_tuning_override"
    manifest_path.write_text(json.dumps(manifest))
    request["input_view"]["artifact_manifest_sha256"] = _sha(manifest_path)
    request["human_trainability_experiment"] = {
        "cohort_expansion_authorized": True,
        "scientific_interpretation": "optimization_diagnostic_only",
    }
    view, meta = view_for_request(request, release_present=True, scaling="robust")
    assert view.n("train") == 3 and view.n("inner_val") == 2
    assert meta["development_evaluation_exposed"] is False
