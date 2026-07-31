"""Lossless counterfactual migration from Phase-C to Phase-D state."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src import topic4_zm_fast_carrier_contract as C
from src import topic4_zm_fast_carrier_state as S
from src.topic4_zm_checkpoint import load_state_npz, state_hash


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def input_manifest():
    return C.build_input_manifest(ROOT)


@pytest.fixture(scope="module")
def source(input_manifest):
    row = input_manifest["source_panel"][0]
    state, _ = load_state_npz(
        ROOT / row["path"],
        expected_config_sha=input_manifest["source"]["canonical_config_sha"],
        expected_engine_sha=row["source_state_manifest"]["engine_sha"],
        expected_dt=input_manifest["source"]["dt_ms"],
    )
    return row, state


def test_real_source_migrates_losslessly_with_only_zero_phi(input_manifest):
    row = input_manifest["source_panel"][0]
    migrated, record = S.load_and_migrate(
        ROOT, input_manifest, row_id=("pre_entry", "natural")
    )
    original, _ = load_state_npz(
        ROOT / row["path"],
        expected_config_sha=input_manifest["source"]["canonical_config_sha"],
        expected_engine_sha=row["source_state_manifest"]["engine_sha"],
        expected_dt=input_manifest["source"]["dt_ms"],
    )
    assert set(migrated) == set(original) | {"slow.phi_increment"}
    for key in original:
        if key == "rng_state":
            assert migrated[key] == original[key]
        else:
            np.testing.assert_array_equal(migrated[key], original[key])
            assert migrated[key].dtype == original[key].dtype
            assert migrated[key].shape == original[key].shape
    phi = migrated["slow.phi_increment"]
    assert phi.shape == (40000,)
    assert phi.dtype == np.float64
    assert np.count_nonzero(phi) == 0
    assert np.count_nonzero(phi[32000:]) == 0
    assert record["source_state_hash"] == row["state_hash"]
    assert record["migrated_state_hash"] == state_hash(migrated)
    assert record["inserted_fields"] == ["slow.phi_increment"]
    assert record["source_config_sha"] != record["phaseD_arm_config_sha256"]


def test_migration_is_deterministic(input_manifest):
    a, ra = S.load_and_migrate(
        ROOT,
        input_manifest,
        row_id=("bounded_mid", "rising"),
        contract_already_validated=True,
    )
    b, rb = S.load_and_migrate(
        ROOT,
        input_manifest,
        row_id=("bounded_mid", "rising"),
        contract_already_validated=True,
    )
    assert ra == rb
    assert state_hash(a) == state_hash(b)


@pytest.mark.parametrize("missing", ["ring_sE", "ring_sI", "slow.z", "slow.m"])
def test_missing_carried_field_fails_closed(input_manifest, source, missing):
    row, state = source
    changed = dict(state)
    changed.pop(missing)
    with pytest.raises(S.StateMigrationError, match="field inventory"):
        S.migrate_state(changed, row, input_manifest)


def test_unknown_source_field_fails_closed(input_manifest, source):
    row, state = source
    changed = dict(state)
    changed["mystery"] = np.zeros(1)
    with pytest.raises(S.StateMigrationError, match="field inventory"):
        S.migrate_state(changed, row, input_manifest)


def test_existing_or_nonzero_phi_fails_closed(input_manifest, source):
    row, state = source
    changed = dict(state)
    changed["slow.phi_increment"] = np.ones(40000)
    with pytest.raises(S.StateMigrationError, match="phi"):
        S.migrate_state(changed, row, input_manifest)


def test_row_identity_and_file_hash_are_load_bearing(input_manifest):
    changed = copy.deepcopy(input_manifest)
    changed["source_panel"][0]["file_sha256"] = "0" * 64
    changed["manifest_sha256"] = C.canonical_sha(
        {key: value for key, value in changed.items() if key != "manifest_sha256"}
    )
    with pytest.raises(S.StateMigrationError, match="file hash"):
        S.load_and_migrate(
            ROOT,
            changed,
            row_id=("pre_entry", "natural"),
            contract_already_validated=True,
        )
    with pytest.raises(S.StateMigrationError, match="not locked"):
        S.load_and_migrate(
            ROOT, input_manifest, row_id=("pre_entry", "peak")
        )


def test_transformation_record_contains_field_level_fingerprints(
    input_manifest,
):
    _, record = S.load_and_migrate(
        ROOT,
        input_manifest,
        row_id=("bounded_late", "peak"),
        contract_already_validated=True,
    )
    assert set(record["carried_field_fingerprints"]) == set(
        input_manifest["state_migration"]["carried_fields"]
    )
    for row in record["carried_field_fingerprints"].values():
        assert set(row) == {"dtype", "shape", "sha256"}
        assert len(row["sha256"]) == 64
    json.dumps(record, allow_nan=False)


def test_observable_fingerprint_is_exact_and_excludes_only_wall_time():
    base = {
        "rate": np.asarray([1.0, 2.0], dtype=np.float32),
        "spikes": np.asarray([[True, False]], dtype=bool),
        "runaway": None,
        "count": 2,
        "wall_s": 1.0,
    }
    same = copy.deepcopy(base)
    same["wall_s"] = 99.0
    assert S.fingerprint_observables(base) == S.fingerprint_observables(same)
    changed = copy.deepcopy(base)
    changed["rate"][1] = np.nextafter(changed["rate"][1], np.float32(3.0))
    assert S.fingerprint_observables(base) != S.fingerprint_observables(changed)


def test_exact_continuation_comparator_fails_closed():
    S.require_exact_continuation({"x": 1}, {"x": 1}, label="arm A")
    with pytest.raises(S.StateMigrationError, match="byte-identical"):
        S.require_exact_continuation({"x": 1}, {"x": 2}, label="arm A")
