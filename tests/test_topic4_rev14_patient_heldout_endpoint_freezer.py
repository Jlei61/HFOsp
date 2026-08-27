from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.interictal_propagation import load_subject_propagation_events
from src.topic4_core_field_profile import profile_grid, split_by_block
from src.topic4_shaft_aware import (
    build_event_features,
    contract_groups,
    fit_patient_embedding,
)
from src.topic4_shaft_aware_direction import fit_direction_classifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/freeze_topic4_rev14_patient_heldout_endpoint.py"
SPEC = importlib.util.spec_from_file_location("rev14_heldout_freezer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
freezer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freezer)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable_classifier(classifier: dict) -> dict:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in classifier.items()
    }


def _contract(names: list[str]) -> dict:
    contacts = []
    for index, name in enumerate(names):
        shaft = "ICL" if name.startswith("ICL") else "SCL"
        contacts.append({
            "contact_name": name,
            "contact_index": index,
            "shaft_id": shaft,
            "shared_axis_coordinate_mm": float(index),
            "sheet_xy_mm": [float(index), 0.0 if shaft == "ICL" else 5.0],
        })
    return {
        "schema": "synthetic_contact_contract",
        "contacts": contacts,
        "readout_parameters": {"readability_part_min": 5},
        "hashes": {"synthetic": "test-only"},
    }


def _build_fixture(tmp_path: Path, *, sparse_heldout: bool = False) -> dict:
    artifact_root = tmp_path / "artifact_root"
    raw_root = tmp_path / "raw_interictal"
    raw_root.mkdir(parents=True)
    names = [
        "SCL9", "ICL11", "SCL8", "ICL10", "SCL7", "ICL9", "ICL8",
        "SCL6", "ICL7", "ICL6", "ICL5", "ICL4", "ICL3", "ICL2", "ICL1",
    ]
    source_order = list(reversed(names))
    source_lookup = np.asarray([names.index(name) for name in source_order])
    n_events_per_block = 16
    rng = np.random.default_rng(20260828)
    _, heldout_fixture_indices = split_by_block(
        np.repeat(np.arange(6), n_events_per_block), 1.0 / 3.0, 17,
    )
    heldout_fixture_blocks = set(
        np.repeat(np.arange(6), n_events_per_block)[heldout_fixture_indices].tolist()
    )
    for block in range(6):
        labels = np.arange(n_events_per_block) % 2
        event_rows = []
        event_masks = []
        for event_in_block, mode in enumerate(labels):
            direction = (
                np.arange(len(names), dtype=float)
                if mode == 0 else np.arange(len(names) - 1, -1, -1, dtype=float)
            )
            row = direction + rng.normal(0.0, 0.35, size=len(names))
            mask = rng.random(len(names)) > 0.18
            if np.sum(mask) < 8:
                mask[np.argsort(rng.random(len(names)))[:8]] = True
            if sparse_heldout and block in heldout_fixture_blocks and event_in_block == 0:
                mask[:] = False
                mask[:5] = True
            mask[0] = True
            mask[1] = True
            event_rows.append(row)
            event_masks.append(mask)
        ranks = np.asarray(event_rows).T
        onsets = ranks * 0.25 + 0.01 * block
        bools = np.asarray(event_masks, dtype=np.int8).T
        stem = f"synthetic_{block:02d}"
        np.savez_compressed(
            raw_root / f"{stem}_lagPat_withFreqCent.npz",
            lagPatRank=ranks[source_lookup],
            lagPatRaw=onsets[source_lookup],
            eventsBool=bools[source_lookup],
            chnNames=np.asarray(source_order),
            start_t=np.asarray(float(block * 1000.0)),
        )
        packed = np.column_stack([
            np.arange(n_events_per_block, dtype=float) * 10.0,
            np.arange(n_events_per_block, dtype=float) * 10.0 + 5.0,
        ])
        np.save(raw_root / f"{stem}_packedTimes_withFreqCent.npy", packed)

    contract = _contract(names)
    contract_path = tmp_path / "contact_contract.json"
    contract_path.write_text(json.dumps(contract, sort_keys=True))
    axial = {
        row["contact_name"]: row["shared_axis_coordinate_mm"]
        for row in contract["contacts"]
    }
    old_reference_path = tmp_path / "old_rank_curve_reference.npz"
    np.savez_compressed(old_reference_path, grid=profile_grid(axial))

    raw = load_subject_propagation_events(raw_root)
    train, heldout = split_by_block(raw["block_ids"], 1.0 / 3.0, 17)
    assert set(raw["block_ids"][train]).isdisjoint(raw["block_ids"][heldout])
    raw_names = np.asarray(raw["channel_names"]).astype(str)
    reorder = np.asarray([int(np.flatnonzero(raw_names == name)[0]) for name in names])
    ranks = np.asarray(raw["ranks"], float)[reorder].T
    onsets = np.asarray(raw["lag_raw"], float)[reorder].T
    masks = np.asarray(raw["bools"], bool)[reorder].T
    ranks[~masks] = np.nan
    onsets[~masks] = np.nan
    labels = np.tile(np.arange(n_events_per_block) % 2, 6)
    train_labels = labels[train]
    # The canonical target stores float32 onsets and the frozen classifier was
    # fitted after loading that target, so the fixture must reproduce that path.
    train_onsets = onsets[train].astype(np.float32).astype(np.float64)
    groups = contract_groups(contract)
    feature_parts = build_event_features(train_onsets, groups)
    embedding = fit_patient_embedding(
        feature_parts["features"], variance_fraction=0.99,
        max_components=10, reference_n=32, n_directions=8, seed=31,
    )
    classifier = fit_direction_classifier(
        train_onsets, train_labels, raw["block_ids"][train],
        groups=groups, embedding=embedding, n_splits=3,
        regularization_c=1.0, ood_quantile=0.99,
    )

    training_target_path = tmp_path / "training_target.npz"
    np.savez_compressed(
        training_target_path,
        contact_names=np.asarray(names),
        shaft_ids=np.asarray([
            "ICL" if name.startswith("ICL") else "SCL" for name in names
        ]),
        sheet_xy_mm=np.asarray([row["sheet_xy_mm"] for row in contract["contacts"]]),
        shared_axis_coordinate_mm=np.asarray([
            row["shared_axis_coordinate_mm"] for row in contract["contacts"]
        ]),
        patient_train_event_indices=train,
        patient_train_block_ids=raw["block_ids"][train],
        patient_train_old_labels=train_labels,
        patient_train_onsets=train_onsets.astype(np.float32),
        feature_center=embedding["center"],
        feature_scale=embedding["scale"],
        pca_components=embedding["components"],
    )
    classifier_path = tmp_path / "classifier_manifest.json"
    classifier_path.write_text(json.dumps({
        "direction_classifier": _jsonable_classifier(classifier),
    }, sort_keys=True))
    split_config_path = tmp_path / "split_config.json"
    split_config_path.write_text(json.dumps({
        "subject": "epilepsiae_1146",
        "inputs": {
            "patient_root": str(raw_root),
            "old_rank_curve_reference": str(old_reference_path),
        },
        "patient_split": {
            "unit": "recording_block",
            "heldout_fraction": 1.0 / 3.0,
            "seed": 17,
        },
    }, sort_keys=True))
    expected = {
        "split_config": _sha256(split_config_path),
        "training_target": _sha256(training_target_path),
        "contact_contract": _sha256(contract_path),
        "classifier_manifest": _sha256(classifier_path),
        "old_rank_curve_reference": _sha256(old_reference_path),
    }
    return {
        "artifact_root": artifact_root,
        "raw_root": raw_root,
        "names": names,
        "train": train,
        "heldout": heldout,
        "raw": raw,
        "split_config": split_config_path,
        "training_target": training_target_path,
        "contact_contract": contract_path,
        "classifier_manifest": classifier_path,
        "output": tmp_path / "frozen_endpoint",
        "expected": expected,
    }


def _freeze(fixture: dict) -> dict:
    return freezer.freeze_endpoint(
        artifact_root=fixture["artifact_root"],
        split_config_path=fixture["split_config"],
        training_target_path=fixture["training_target"],
        contact_contract_path=fixture["contact_contract"],
        classifier_manifest_path=fixture["classifier_manifest"],
        output_dir=fixture["output"],
        expected_hashes=fixture["expected"],
        evaluation_overrides={
            "sample_size": 4,
            "draws": 4,
            "projection_count": 8,
            "calibration_draws": 8,
            "calibration_seed": 41,
        },
    )


def test_freezer_is_deterministic_and_manifest_npz_hashes_are_self_consistent(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    first = _freeze(fixture)
    manifest_path = fixture["output"] / freezer.MANIFEST_NAME
    npz_path = fixture["output"] / freezer.NPZ_NAME
    first_manifest_bytes = manifest_path.read_bytes()
    first_npz_bytes = npz_path.read_bytes()

    second = _freeze(fixture)
    assert second == first
    assert manifest_path.read_bytes() == first_manifest_bytes
    assert npz_path.read_bytes() == first_npz_bytes
    assert _sha256(npz_path) == first["artifact"]["npz_sha256"]
    assert first["evaluation_schema_sha256"] == freezer._canonical_sha256(
        first["evaluation_contract"]
    )
    with np.load(npz_path, allow_pickle=False) as loaded:
        assert sorted(loaded.files) == first["artifact"]["array_keys"]
        for key in loaded.files:
            assert freezer._array_sha256(loaded[key]) == first["artifact"][
                "array_sha256"
            ][key]


def test_split_is_block_disjoint_contact_ordered_and_contains_no_training_events(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = _freeze(fixture)
    npz_path = fixture["output"] / freezer.NPZ_NAME
    with np.load(npz_path, allow_pickle=False) as loaded:
        keys = loaded.files
        assert not any(key.startswith("patient_train") or key.startswith("training_")
                       for key in keys)
        np.testing.assert_array_equal(loaded["contact_names"], fixture["names"])
        heldout_events = loaded["heldout_event_indices"]
        assert not np.intersect1d(heldout_events, fixture["train"]).size
        np.testing.assert_array_equal(heldout_events, fixture["heldout"])
        heldout_blocks = np.unique(loaded["heldout_block_ids"])
    split = manifest["split_contract"]
    assert split["blocks_disjoint"] is True
    assert set(split["train_block_ids"]).isdisjoint(split["heldout_block_ids"])
    assert set(heldout_blocks.tolist()) == set(split["heldout_block_ids"])
    assert split["heldout_block_ids_sha256"] == freezer._array_sha256(
        np.asarray(sorted(split["heldout_block_ids"]))
    )
    assert manifest["contact_contract"]["contact_order_sha256"] == (
        freezer._canonical_sha256(fixture["names"])
    )
    assert manifest["disjointness_audit"] == {
        "training_event_overlap": 0,
        "training_block_overlap": 0,
        "output_contains_training_event_arrays": False,
    }


def test_source_drift_is_a_hard_failure_and_does_not_overwrite_endpoint(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    _freeze(fixture)
    manifest_path = fixture["output"] / freezer.MANIFEST_NAME
    npz_path = fixture["output"] / freezer.NPZ_NAME
    frozen_manifest = manifest_path.read_bytes()
    frozen_npz = npz_path.read_bytes()
    packed = sorted(fixture["raw_root"].glob("*_packedTimes_withFreqCent.npy"))[0]
    with packed.open("ab") as handle:
        handle.write(b"source-drift")
    fixture["expected"] = {
        **fixture["expected"],
        # Raw file hashes are frozen from the existing manifest, not this map.
    }
    with pytest.raises(RuntimeError, match="source drift"):
        _freeze(fixture)
    assert manifest_path.read_bytes() == frozen_manifest
    assert npz_path.read_bytes() == frozen_npz


def test_frozen_classifier_and_evaluation_contract_are_training_only(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    manifest = _freeze(fixture)
    labels = manifest["old_A_B_label_definition"]
    assert labels["classifier_fit_data"] == "patient training recording blocks only"
    assert labels["heldout_classifier_refit"] is False
    assert manifest["training_runner_dependency"]["required"] is False
    endpoint = manifest["evaluation_contract"]["patient_endpoint"]
    assert endpoint.startswith("interictal_")
    assert "seizure" not in endpoint

    training_sources = [
        ROOT / "src/topic4_rev14_static_node_objective.py",
        ROOT / "scripts/rescore_topic4_rev13_exact_off_static_node.py",
    ]
    for path in training_sources:
        source = path.read_text()
        assert freezer.NPZ_NAME not in source
        assert freezer.MANIFEST_NAME not in source


def test_sparse_heldout_events_are_retained_and_only_flagged_unreadable(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path, sparse_heldout=True)
    manifest = _freeze(fixture)
    with np.load(fixture["output"] / freezer.NPZ_NAME, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(
            loaded["heldout_event_indices"], fixture["heldout"],
        )
        readable = loaded["heldout_old_curve_readable"]
        assert readable.shape == fixture["heldout"].shape
        assert np.any(~readable)
        assert np.all(loaded["heldout_finite_contact_count"][~readable] == 5)
        assert np.all(loaded["heldout_readout_contract_readable"][~readable])
    assert manifest["endpoint_counts"]["n_events"] == len(fixture["heldout"])
    assert manifest["endpoint_counts"]["n_old_curve_unreadable"] > 0
    assert manifest["old_A_B_label_definition"]["readability_part_min"] == 6
    assert manifest["old_A_B_label_definition"]["readout_contract_part_min"] == 5


def test_classifier_with_same_training_count_but_changed_fit_is_rejected(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    payload = json.loads(fixture["classifier_manifest"].read_text())
    payload["direction_classifier"]["coef"][0] += 0.25
    fixture["classifier_manifest"].write_text(json.dumps(payload, sort_keys=True))
    fixture["expected"]["classifier_manifest"] = _sha256(
        fixture["classifier_manifest"]
    )
    with pytest.raises(RuntimeError, match="not reproducible from frozen training blocks"):
        _freeze(fixture)


def test_training_event_membership_must_exactly_reconstruct_frozen_split(
        tmp_path: Path) -> None:
    fixture = _build_fixture(tmp_path)
    with np.load(fixture["training_target"], allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    arrays["patient_train_event_indices"][0] = arrays[
        "patient_train_event_indices"
    ][1]
    np.savez_compressed(fixture["training_target"], **arrays)
    fixture["expected"]["training_target"] = _sha256(fixture["training_target"])
    with pytest.raises(RuntimeError, match="training events differ"):
        _freeze(fixture)


def test_failed_manifest_write_leaves_no_partial_freeze(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_fixture(tmp_path)

    def fail_json(*_args, **_kwargs):
        raise OSError("synthetic manifest failure")

    monkeypatch.setattr(freezer, "_atomic_json", fail_json)
    with pytest.raises(OSError, match="synthetic manifest failure"):
        _freeze(fixture)
    assert not fixture["output"].exists()
    assert not list(fixture["output"].parent.glob(
        f".{fixture['output'].name}.staging.*"
    ))


def test_source_drift_during_build_aborts_atomic_publication(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _build_fixture(tmp_path)
    packed = sorted(fixture["raw_root"].glob("*_packedTimes_withFreqCent.npy"))[0]
    original_atomic_json = freezer._atomic_json

    def write_then_drift(path, payload):
        original_atomic_json(path, payload)
        with packed.open("ab") as handle:
            handle.write(b"mid-build-source-drift")

    monkeypatch.setattr(freezer, "_atomic_json", write_then_drift)
    with pytest.raises(RuntimeError, match="before atomic endpoint publication"):
        _freeze(fixture)
    assert not fixture["output"].exists()
    assert not list(fixture["output"].parent.glob(
        f".{fixture['output'].name}.staging.*"
    ))
