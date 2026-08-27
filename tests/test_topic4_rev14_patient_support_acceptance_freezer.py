from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic4_rev14_patient_support import PatientSupportSchemaError
from src.topic4_shaft_aware import build_event_features, contract_groups
from src.topic4_shaft_aware_direction import (
    assign_direction_modes,
    fit_direction_classifier,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_topic4_rev14_patient_support_acceptance.py"
SPEC = importlib.util.spec_from_file_location("rev14_support_freezer", SCRIPT)
assert SPEC and SPEC.loader
freezer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freezer)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, poison_target: str | None = None,
             poison_classifier: str | None = None) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    shafts = np.asarray(["ICL", "ICL", "SCL", "SCL"])
    groups = {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])}
    onsets, labels, blocks = [], [], []
    for block in range(24):
        jitter = 0.01 * (block % 5)
        onsets.extend((
            [0.0, 1.0 + jitter, 2.0, 3.0 - jitter],
            [3.0 - jitter, 2.0, 1.0 + jitter, 0.0],
        ))
        labels.extend((0, 1))
        blocks.extend((block, block))
    onsets = np.asarray(onsets, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    blocks = np.asarray(blocks, dtype=np.int64)
    n_features = build_event_features(onsets, groups)["features"].shape[1]
    embedding = {
        "center": np.zeros(n_features),
        "scale": np.ones(n_features),
        "components": np.eye(n_features),
    }
    classifier = fit_direction_classifier(
        onsets, labels, blocks, groups=groups, embedding=embedding,
        n_splits=6, ood_quantile=0.95,
    )

    target_path = tmp_path / "patient_training_target.npz"
    arrays = {
        "contact_names": names,
        "shaft_ids": shafts,
        "patient_train_onsets": onsets,
        "patient_train_old_labels": labels,
        "patient_train_block_ids": blocks,
        "feature_center": embedding["center"],
        "feature_scale": embedding["scale"],
        "pca_components": embedding["components"],
    }
    if poison_target:
        arrays[poison_target] = np.asarray([1])
    np.savez(target_path, **arrays)

    classifier_path = tmp_path / "classifier.json"
    classifier_json = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in classifier.items()
    }
    payload = {"direction_classifier": classifier_json}
    if poison_classifier:
        payload[poison_classifier] = {"value": 1}
    classifier_path.write_text(json.dumps(payload))

    contacts = [
        {"contact_index": index, "contact_name": str(name), "shaft_id": str(shaft)}
        for index, (name, shaft) in enumerate(zip(names, shafts))
    ]
    contract_path = tmp_path / "contact_contract.json"
    contract_path.write_text(json.dumps({"contacts": contacts}))

    config_path = tmp_path / "config.json"
    config = {
        "schema_id": "topic4_rev14_patient_support_acceptance_config_v2",
        "scientific_role": "training_only_patient_support_acceptance",
        "subject": "synthetic",
        "inputs": {
            "patient_training_target": {
                "path": str(target_path), "sha256": _sha256(target_path),
            },
            "old_ab_train_only_classifier": {
                "path": str(classifier_path), "sha256": _sha256(classifier_path),
            },
            "contact_contract": {
                "path": str(contract_path), "sha256": _sha256(contract_path),
            },
        },
        "classifier": {
            "semantics": "FULL_TIMING",
            "accessed_keys": list(freezer.CLASSIFIER_KEYS),
            "prediction_role": "assign_formal_patient_train_ood_class",
            "ood_reference_label": "classifier_assigned_labels",
            "primary_label_key": "patient_train_old_labels",
            "replace_primary_labels": False,
            "refit_n_splits": 6,
            "refit_atol": 1e-12,
            "refit_rtol": 0.0,
        },
        "floor": {
            "sample_size_per_side": 6,
            "draws": 4096,
            "joint_draws": 8,
            "joint_inner_draws": 8,
            "seed": 17,
            "tie_tolerance": 1e-12,
            "quantile": 0.95,
            "quantile_method": "linear",
            "sampling": "one event per distinct recording block; left and right blocks disjoint",
        },
        "output": {
            "directory": str(tmp_path / "output"),
            "manifest": "manifest.json",
            "npz": "floors.npz",
        },
        "forbidden": {},
    }
    config_path.write_text(json.dumps(config))
    return config_path, tmp_path / "output"


def _patient_mapping_from_fixture(config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    target_path = Path(config["inputs"]["patient_training_target"]["path"])
    classifier_path = Path(config["inputs"]["old_ab_train_only_classifier"]["path"])
    contract_path = Path(config["inputs"]["contact_contract"]["path"])
    with np.load(target_path, allow_pickle=False) as loaded:
        target = {key: np.asarray(loaded[key]) for key in loaded.files}
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    contract = json.loads(contract_path.read_text())
    embedding = {
        "center": np.asarray(target["feature_center"], dtype=np.float64),
        "scale": np.asarray(target["feature_scale"], dtype=np.float64),
        "components": np.asarray(target["pca_components"], dtype=np.float64),
    }
    assigned = assign_direction_modes(
        np.asarray(target["patient_train_onsets"], dtype=np.float64),
        groups=contract_groups(contract), embedding=embedding,
        classifier=classifier,
    )
    return {
        "contact_names": target["contact_names"],
        "shaft_ids": target["shaft_ids"],
        "patient_train_onsets": target["patient_train_onsets"],
        "patient_train_old_labels": target["patient_train_old_labels"],
        "patient_train_classifier_labels": np.asarray(
            assigned["labels"], dtype=np.int8,
        ),
        "patient_train_block_ids": target["patient_train_block_ids"],
        "patient_train_ood": np.asarray(assigned["ood"], dtype=bool),
        "primary_label_key": "patient_train_old_labels",
        "source_sha256": _sha256(target_path),
    }


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    root = tmp_path_factory.mktemp("support_freezer")
    config_path, output = _fixture(root)
    manifest = freezer.freeze_acceptance(
        artifact_root=root, config_path=config_path,
    )
    return config_path, output, manifest


def test_freezes_4096_draw_training_only_floor_and_generated_ood(frozen):
    _, output, manifest = frozen
    assert manifest["status"] == "PATIENT_TRAINING_SUPPORT_ACCEPTANCE_FROZEN"
    assert manifest["floor_contract"]["draws"] == 4096
    assert manifest["floor_contract"]["primary_labels"] == (
        "patient_train_old_labels"
    )
    assert manifest["patient_training_contract"][
        "primary_labels_replaced_by_classifier"
    ] is False
    assert manifest["classifier_contract"]["training_cv_summary_keys_accessed"] is False
    assert manifest["forbidden_input_audit"] == {
        "only_three_scientific_inputs": True,
        "patient_evaluation_arrays_loaded": False,
        "seizure_arrays_loaded": False,
        "model_artifacts_loaded": False,
        "candidate_scores_loaded": False,
        "candidate_selection_performed": False,
        "simulation_performed": False,
    }
    npz_path = output / "floors.npz"
    assert _sha256(npz_path) == manifest["artifact"]["npz_sha256"]
    with np.load(npz_path, allow_pickle=False) as loaded:
        assert loaded["patient_train_ood"].dtype == np.bool_
        assert len(loaded["patient_train_ood"]) == 48
        assert "patient_train_classifier_labels" in loaded.files
        assert "patient_train_old_label_conditioned_ood_diagnostic" in loaded.files
        floor_keys = manifest["floor_contract"]["floor_distribution_array_keys"]
        assert set(floor_keys) == set(manifest["floor_contract"]["floor_q95"])
        for endpoint, key in floor_keys.items():
            assert loaded[key].shape == (4096,), endpoint

    assert manifest["classifier_contract"]["refit_performed"] is True
    assert manifest["classifier_contract"]["refit_all_parameters_match"] is True
    assert all(
        row["matches"]
        for row in manifest["classifier_contract"]["refit_audit"][
            "parameters"
        ].values()
    )
    assert "classifier-assigned class" in manifest["classifier_contract"][
        "patient_train_ood_semantics"
    ]
    assert manifest["patient_training_contract"]["joint_null_label_contract"] == {
        "pseudo_model_side": "patient_train_classifier_labels",
        "patient_reference_side": "patient_train_old_labels",
    }


def test_formal_patient_ood_exactly_matches_classifier_assigned_class(frozen):
    config_path, output, _ = frozen
    config = json.loads(config_path.read_text())
    target_path = Path(config["inputs"]["patient_training_target"]["path"])
    classifier_path = Path(config["inputs"]["old_ab_train_only_classifier"]["path"])
    contract_path = Path(config["inputs"]["contact_contract"]["path"])
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], dtype=np.float64)
        embedding = {
            "center": np.asarray(loaded["feature_center"], dtype=np.float64),
            "scale": np.asarray(loaded["feature_scale"], dtype=np.float64),
            "components": np.asarray(loaded["pca_components"], dtype=np.float64),
        }
    classifier = json.loads(classifier_path.read_text())["direction_classifier"]
    contract = json.loads(contract_path.read_text())
    assigned = assign_direction_modes(
        onsets, groups=contract_groups(contract), embedding=embedding,
        classifier=classifier,
    )
    with np.load(output / "floors.npz", allow_pickle=False) as loaded:
        np.testing.assert_array_equal(
            loaded["patient_train_classifier_labels"], assigned["labels"],
        )
        np.testing.assert_array_equal(
            loaded["patient_train_ood"], assigned["ood"],
        )
        np.testing.assert_allclose(
            loaded["patient_train_ood_distance"], assigned["ood_distance"],
            rtol=0.0, atol=0.0,
        )


def test_rebuild_is_byte_deterministic(frozen):
    config_path, output, first = frozen
    second = freezer.freeze_acceptance(
        artifact_root=config_path.parent, config_path=config_path,
    )
    assert second == first
    assert _sha256(output / "floors.npz") == first["artifact"]["npz_sha256"]
    assert {path.name for path in output.iterdir()} == {"manifest.json", "floors.npz"}


def test_old_label_conditioned_ood_is_diagnostic_only():
    embedding = np.asarray([[0.0], [10.0]])
    old_labels = np.asarray([0, 1], dtype=np.int8)
    classifier = {
        "class_centers": np.asarray([[0.0], [10.0]]),
        "class_precisions": np.asarray([[[1.0]], [[1.0]]]),
        "ood_distance_thresholds": np.asarray([1.0, 1.0]),
    }
    distance, ood = freezer._old_label_conditioned_ood(
        embedding, old_labels, classifier,
    )
    np.testing.assert_array_equal(distance, np.asarray([0.0, 0.0]))
    np.testing.assert_array_equal(ood, np.asarray([False, False]))

    swapped_old_labels = 1 - old_labels
    distance, ood = freezer._old_label_conditioned_ood(
        embedding, swapped_old_labels, classifier,
    )
    np.testing.assert_array_equal(distance, np.asarray([100.0, 100.0]))
    np.testing.assert_array_equal(ood, np.asarray([True, True]))


@pytest.mark.parametrize("parameter", freezer.CLASSIFIER_PARAMETER_KEYS)
def test_refit_rejects_every_tampered_classifier_parameter(tmp_path, parameter):
    config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    record = config["inputs"]["old_ab_train_only_classifier"]
    classifier_path = Path(record["path"])
    payload = json.loads(classifier_path.read_text())
    value = payload["direction_classifier"][parameter]
    if isinstance(value, list):
        array = np.asarray(value, dtype=np.float64)
        array.flat[0] += 0.125
        payload["direction_classifier"][parameter] = array.tolist()
    elif parameter == "n_train":
        payload["direction_classifier"][parameter] = int(value) + 1
    elif parameter == "ood_quantile":
        payload["direction_classifier"][parameter] = float(value) - 0.125
    else:
        payload["direction_classifier"][parameter] = float(value) + 0.125
    classifier_path.write_text(json.dumps(payload))
    record["sha256"] = _sha256(classifier_path)
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match="classifier refit parameter differs"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)
    assert not output.exists()


def test_freezer_supplies_classifier_labels_to_joint_patient_mapping(
        tmp_path, monkeypatch):
    config_path, output = _fixture(tmp_path)
    captured = {}

    def capture_mapping(value, **_kwargs):
        captured.update(value)
        raise RuntimeError("mapping captured")

    monkeypatch.setattr(freezer, "patient_training_from_mapping", capture_mapping)
    with pytest.raises(RuntimeError, match="mapping captured"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)
    assert "patient_train_classifier_labels" in captured
    assert captured["patient_train_classifier_labels"].shape == (
        len(captured["patient_train_old_labels"]),
    )
    assert captured["patient_train_ood"].shape == (
        len(captured["patient_train_old_labels"]),
    )
    assert not output.exists()


def test_patient_mapping_requires_classifier_assigned_labels(tmp_path):
    config_path, _ = _fixture(tmp_path)
    mapping = _patient_mapping_from_fixture(config_path)
    mapping.pop("patient_train_classifier_labels")
    with pytest.raises(
            PatientSupportSchemaError,
            match="patient_train_classifier_labels"):
        freezer.patient_training_from_mapping(mapping)


def test_classifier_labels_change_joint_null_but_not_old_label_floors(tmp_path):
    config_path, _ = _fixture(tmp_path)
    base_mapping = _patient_mapping_from_fixture(config_path)
    old_labels = np.asarray(
        base_mapping["patient_train_old_labels"], dtype=np.int8,
    )
    block_ids = np.asarray(base_mapping["patient_train_block_ids"])
    onsets = np.asarray(base_mapping["patient_train_onsets"], dtype=np.float64).copy()
    for row, (mode, block) in enumerate(zip(old_labels, block_ids)):
        onsets[row] += 0.03 * ((int(block) % 7) - 3) * np.asarray(
            [0.0, 1.0, -1.0, 0.5]
        )
        if int(block) % 4 == 0:
            onsets[row, 1] = np.nan
        if int(block) % 5 == 0:
            onsets[row, 3] = np.nan
        if int(block) % 6 == 0 and int(mode) == 0:
            onsets[row, 2:] = np.nan
        if int(block) % 7 == 0 and int(mode) == 1:
            onsets[row, :2] = np.nan
    base_mapping["patient_train_onsets"] = onsets
    base_mapping["patient_train_classifier_labels"] = old_labels.copy()
    base_mapping["patient_train_ood"] = (block_ids % 5 == 0)
    changed_mapping = {
        key: np.asarray(value).copy() if isinstance(value, np.ndarray) else value
        for key, value in base_mapping.items()
    }
    changed_mapping["patient_train_classifier_labels"] = (
        1 - old_labels
    )
    base = freezer.patient_training_from_mapping(base_mapping)
    changed = freezer.patient_training_from_mapping(changed_mapping)
    kwargs = {
        "sample_size": 6,
        "floor_draws": 64,
        "joint_draws": 24,
        "joint_inner_draws": 24,
        "seed": 91,
        "tie_tolerance": 1e-12,
    }
    base_calibration = freezer.build_patient_support_calibration(base, **kwargs)
    changed_calibration = freezer.build_patient_support_calibration(changed, **kwargs)

    assert base_calibration.floor_q95 == changed_calibration.floor_q95
    for endpoint in base_calibration.floor_distributions:
        np.testing.assert_array_equal(
            base_calibration.floor_distributions[endpoint],
            changed_calibration.floor_distributions[endpoint],
        )
    assert not np.array_equal(
        base_calibration.joint_distribution,
        changed_calibration.joint_distribution,
    )
    assert base_calibration.joint_distribution_sha256 != (
        changed_calibration.joint_distribution_sha256
    )


def test_refit_rejects_same_length_training_target_drift(tmp_path):
    config_path, output = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    record = config["inputs"]["patient_training_target"]
    target_path = Path(record["path"])
    with np.load(target_path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["patient_train_onsets"] = arrays["patient_train_onsets"].copy()
    arrays["patient_train_onsets"][0, 1] += 0.25
    np.savez(target_path, **arrays)
    record["sha256"] = _sha256(target_path)
    config_path.write_text(json.dumps(config))

    with pytest.raises(RuntimeError, match="classifier refit parameter differs"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)
    assert not output.exists()


def test_manifest_and_npz_are_published_by_one_directory_rename(
        tmp_path, monkeypatch):
    config_path, output = _fixture(tmp_path)
    original_replace = freezer.os.replace
    replacements = []

    def observed_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        return original_replace(source, destination)

    monkeypatch.setattr(freezer.os, "replace", observed_replace)
    freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)
    assert len(replacements) == 1
    source, destination = replacements[0]
    assert ".staging-" in source.name
    assert destination == output
    assert {path.name for path in output.iterdir()} == {"manifest.json", "floors.npz"}


@pytest.mark.parametrize("failure_point", ["manifest", "source_drift"])
def test_partial_failure_leaves_no_output_or_staging(
        tmp_path, monkeypatch, failure_point):
    config_path, output = _fixture(tmp_path)
    if failure_point == "manifest":
        def fail_manifest(*_args, **_kwargs):
            raise RuntimeError("injected manifest failure")

        monkeypatch.setattr(freezer, "_write_json", fail_manifest)
        expected = "injected manifest failure"
    else:
        def fail_source_drift(*_args, **_kwargs):
            raise RuntimeError("freeze source drifted during build: injected")

        monkeypatch.setattr(
            freezer, "_assert_source_snapshot_unchanged", fail_source_drift,
        )
        expected = "source drifted"
    with pytest.raises(RuntimeError, match=expected):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.staging-*"))


def test_public_calibration_serializer_carries_future_joint_fields():
    class FutureCalibration:
        floor_distributions = {"mode_0.x": np.asarray([0.1, 0.2])}
        joint_null_distributions = {"joint_max": np.asarray([0.8, 1.2])}

        def to_dict(self):
            return {
                "schema_id": "future",
                "joint_threshold_q95": 1.37,
                "floor_distributions": self.floor_distributions,
                "joint_null_distributions": self.joint_null_distributions,
            }

    metadata, arrays, indices = freezer._calibration_artifact_parts(
        FutureCalibration()
    )
    assert metadata["joint_threshold_q95"] == 1.37
    assert set(indices) == {"floor_distributions", "joint_null_distributions"}
    np.testing.assert_array_equal(
        arrays[indices["joint_null_distributions"]["joint_max"]],
        np.asarray([0.8, 1.2]),
    )


@pytest.mark.parametrize("field", ["patient_heldout_onsets", "ictal_power"])
def test_rejects_forbidden_patient_target_fields(tmp_path, field):
    config_path, _ = _fixture(tmp_path, poison_target=field)
    with pytest.raises(RuntimeError, match="forbidden patient field"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)


@pytest.mark.parametrize("field", ["patient_heldout_target", "ictal_target"])
def test_rejects_forbidden_classifier_payloads(tmp_path, field):
    config_path, _ = _fixture(tmp_path, poison_classifier=field)
    with pytest.raises(RuntimeError, match="forbidden patient field"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)


def test_rejects_extra_scientific_input_and_contact_drift(tmp_path):
    config_path, _ = _fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["inputs"]["model_candidate"] = config["inputs"]["contact_contract"]
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="exactly three"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)

    config_path, _ = _fixture(tmp_path / "drift")
    config = json.loads(config_path.read_text())
    contract_record = config["inputs"]["contact_contract"]
    contract_path = Path(contract_record["path"])
    contract = json.loads(contract_path.read_text())
    contract["contacts"][0]["shaft_id"] = "SCL"
    contract_path.write_text(json.dumps(contract))
    contract_record["sha256"] = _sha256(contract_path)
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="shaft identities differ"):
        freezer.freeze_acceptance(artifact_root=tmp_path, config_path=config_path)


def test_canonical_contract_locks_formal_draws_and_only_three_inputs():
    config = json.loads(
        (ROOT / "config/topic4_rev14_patient_support_acceptance.json").read_text()
    )
    assert config["floor"]["draws"] == 4096
    assert config["floor"]["joint_draws"] == 512
    assert config["floor"]["joint_inner_draws"] == 512
    assert set(config["inputs"]) == freezer.ALLOWED_INPUTS
    assert config["classifier"]["ood_reference_label"] == (
        "classifier_assigned_labels"
    )
    assert config["classifier"]["refit_n_splits"] == 6
    assert config["classifier"]["refit_atol"] <= 1e-12
    assert config["classifier"]["refit_rtol"] == 0.0
    source = SCRIPT.read_text()
    assert "snn_engine" not in source
    assert "run_topic4" not in source
