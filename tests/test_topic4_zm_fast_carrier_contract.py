"""Fail-closed Phase-D source-state and migration contract tests."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from src import topic4_zm_fast_carrier_contract as C


ROOT = Path(__file__).resolve().parents[1]


def _rehash(payload):
    out = copy.deepcopy(payload)
    out.pop("manifest_sha256", None)
    out["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            out,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return out


@pytest.fixture(scope="module")
def manifest():
    return C.build_input_manifest(ROOT)


def test_real_phasec_evidence_builds_a_deterministic_input_lock(manifest):
    assert C.build_input_manifest(ROOT) == manifest
    C.validate_input_manifest(manifest, ROOT)
    assert manifest["schema"] == C.INPUT_SCHEMA
    assert manifest["production_authorized"] is False
    assert manifest["source"]["phasec_status"] == (
        "post_result_futility_stopped_incomplete"
    )
    assert manifest["source"]["completed_phasec_runs"] == 59
    assert manifest["source"]["complete_phasec1_negative"] is False
    assert manifest["claim_boundary"]["fast_carrier_supported"] is False
    assert manifest["claim_boundary"]["ictal_lifecycle_established"] is False


def test_source_panel_uses_only_real_fast_states(manifest):
    rows = manifest["source_panel"]
    assert [
        (row["bin_name"], row["fast_phase"]) for row in rows
    ] == [
        ("pre_entry", "natural"),
        ("bounded_mid", "rising"),
        ("bounded_mid", "peak"),
        ("bounded_late", "rising"),
        ("bounded_late", "peak"),
    ]
    pre = rows[0]
    assert [x["replicate"] for x in pre["first_pass_noise_banks"]] == [
        "noise_replay",
        "noise_resample_1",
    ]
    assert all(x["is_paired"] for x in pre["first_pass_noise_banks"])
    for row in rows[1:]:
        assert [x["replicate"] for x in row["first_pass_noise_banks"]] == [
            "noise_replay"
        ]
    assert not any(
        row["bin_name"] == "pre_entry"
        and row["fast_phase"] in {"rising", "peak"}
        for row in rows
    )


def test_source_states_and_futility_evidence_are_reverified_from_disk(manifest):
    assert len(manifest["source"]["futility_evidence_set_sha256"]) == 3
    for value in manifest["source"]["futility_evidence_set_sha256"].values():
        assert len(value) == 64
    for row in manifest["source_panel"]:
        path = ROOT / row["path"]
        assert path.is_file()
        assert row["file_sha256"] == C.sha256_file(path)
        assert row["source_state_manifest"]["state_hash"] == row["state_hash"]
        assert row["source_state_manifest"]["seed"] == 1
        assert row["source_state_manifest"]["config_sha"] == (
            manifest["source"]["canonical_config_sha"]
        )


def test_migration_is_explicit_and_inserts_only_zero_phi(manifest):
    migration = manifest["state_migration"]
    assert migration["source_schema"] == "zm_sim_state_v1"
    assert migration["population_sizes"] == {
        "N": 40000,
        "NE": 32000,
        "NI": 8000,
    }
    assert migration["inserted_fields"] == {
        "slow.phi_increment": {
            "dtype": "float64",
            "fill": 0.0,
            "shape": [40000],
            "target": "E_active_I_exact_zero",
        }
    }
    assert "ring_sE" in migration["carried_fields"]
    assert "ring_sI" in migration["carried_fields"]
    assert "rng_state" in migration["carried_fields"]
    assert "slow.z" in migration["carried_fields"]
    assert "slow.m" in migration["carried_fields"]
    assert set(migration["carried_fields"]) == set(
        manifest["source_panel"][0]["source_state_manifest"]["keys"]
    ) | {"rng_state"}


@pytest.mark.parametrize(
    "mutator,match",
    [
        (
            lambda d: d["source_panel"][0]["source_state_manifest"].__setitem__(
                "engine_sha", "0" * 64
            ),
            "source engine",
        ),
        (
            lambda d: d["state_migration"]["carried_fields"].remove("ring_sE"),
            "migration carried-field",
        ),
        (
            lambda d: d["source_panel"][0]["source_state_manifest"].__setitem__(
                "seed", 3
            ),
            "seed",
        ),
        (
            lambda d: d["source_panel"][0]["first_pass_noise_banks"][0].__setitem__(
                "bank_sha", "0" * 64
            ),
            "noise bank",
        ),
        (
            lambda d: d["state_migration"]["inserted_fields"][
                "slow.phi_increment"
            ].__setitem__("fill", 0.1),
            "phi",
        ),
        (
            lambda d: d["source_panel"][0].__setitem__("fast_phase", "rising"),
            "source panel",
        ),
        (
            lambda d: d["source_semantic_hashes"].__setitem__(
                "connectivity", "0" * 64
            ),
            "connectivity",
        ),
        (
            lambda d: d["source_semantic_hashes"].__setitem__(
                "threshold_field", "0" * 64
            ),
            "threshold",
        ),
    ],
)
def test_semantic_or_provenance_drift_fails_closed(manifest, mutator, match):
    changed = copy.deepcopy(manifest)
    mutator(changed)
    changed = _rehash(changed)
    with pytest.raises(C.ContractInputError, match=match):
        C.validate_input_manifest(changed, ROOT, expected=manifest)


def test_manifest_self_hash_fails_closed(manifest):
    changed = copy.deepcopy(manifest)
    changed["resource_policy"]["max_full_snn_workers"] = 13
    with pytest.raises(C.ContractInputError, match="self-hash"):
        C.validate_input_manifest(changed, ROOT, expected=manifest)


def test_arm_configs_keep_ee_immutable_and_separate_source_from_intervention(
    manifest,
):
    arms = manifest["arms"]
    assert list(arms) == ["A", "B", "C", "D"]
    assert arms["A"]["mode"] == "current_exact_control"
    assert arms["B"]["gamma_global_gaba"] == 0.0
    assert arms["C"]["gamma_global_gaba"] == pytest.approx(1 / 6)
    assert len(arms["D"]["phi_grid"]) == 6
    assert all(row["ee_mutation_allowed"] is False for row in arms.values())
    assert manifest["source"]["canonical_config_sha"] != (
        manifest["phaseD_arm_config_sha256"]
    )


def test_write_once_publication_is_idempotent_and_rejects_drift(
    manifest, tmp_path
):
    path = tmp_path / "input.json"
    C.publish_once(path, manifest)
    first = path.read_bytes()
    C.publish_once(path, manifest)
    assert path.read_bytes() == first
    changed = copy.deepcopy(manifest)
    changed["schema"] = "drift"
    with pytest.raises(C.ContractInputError, match="overwrite"):
        C.publish_once(path, changed)
