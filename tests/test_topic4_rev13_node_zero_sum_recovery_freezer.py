import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev13_node_zero_sum_recovery import (
    _load_hashed_inputs,
    build_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG_PATH = ROOT / "config/topic4_rev13_node_zero_sum_recovery.json"


def _config():
    return json.loads(CONFIG_PATH.read_text())


def _inputs(config=None):
    config = _config() if config is None else config
    loaded, _ = _load_hashed_inputs(config, ARTIFACT_ROOT)
    return loaded


def _build(config=None, loaded=None):
    config = _config() if config is None else config
    loaded = _inputs(config) if loaded is None else loaded
    return build_candidates(
        loaded["stage_ak_manifest"], loaded["stage_al_manifest"],
        loaded["stage_ak_config"], config,
    )


def test_rev13_frozen_config_has_exact_arms_seeds_and_closed_pathways():
    config = _config()
    assert [row["arm_id"] for row in config["arms"]] == [
        "exact_off", "zero_sum_c010", "zero_sum_c020", "zero_sum_c040",
        "raise_only_c020", "spatial_shift_c020",
    ]
    assert config["search"]["canary_network_seeds"] == [2311]
    assert config["search"]["fit_network_seeds"] == [2312, 2313]
    assert config["search"]["engineering_parity_network_seeds"] == [2291]
    assert config["search"]["simulation"]["duration_ms"] == 10000.0
    assert set(config["pathways"].values()) == {"off"}
    contract = config["node_accessibility_contract"]
    assert contract["tau_ms"] == 250.0
    assert contract["reference_rate_hz"] == 50.0
    assert contract["a_max_multiplier"] == 2.0


def test_freezer_hashes_only_model_internal_inputs():
    config = _config()
    assert set(config["inputs"]) == {
        "stage_ak_config", "stage_ak_manifest", "stage_al_manifest",
    }
    forbidden = ("patient", "prototype", "classifier", "target", "heldout")
    for name, record in config["inputs"].items():
        text = f"{name} {record['path']}".lower()
        assert not any(term in text for term in forbidden)
        assert len(record["sha256"]) == 64
    loaded, audit = _load_hashed_inputs(config, ARTIFACT_ROOT)
    assert set(loaded) == set(config["inputs"])
    assert {name: row["sha256"] for name, row in audit.items()} == {
        name: row["sha256"] for name, row in config["inputs"].items()
    }


def test_freezer_builds_six_arms_on_one_canonical_ak_substrate():
    config = _config()
    candidates, audit = _build(config)
    assert len(candidates) == 6
    assert {row["primary_substrate_id"] for row in candidates} == {
        "stage_ak_mean_g10_p_disp_g08_m"
    }
    assert {row["node_mapping"]["mapping_sha256"] for row in candidates} == {
        config["primary_substrate"]["mapping_sha256"]
    }
    assert {row["node_field"]["field_sha256"] for row in candidates} == {
        config["primary_substrate"]["mean_field_sha256"]
    }
    assert {
        row["node_dispersion_field"]["field_sha256"] for row in candidates
    } == {config["primary_substrate"]["dispersion_field_sha256"]}
    assert audit["n_unique_substrates"] == 1
    assert audit["stage_al_alias"]["counted_as_additional_substrate"] is False
    assert audit["stage_al_alias"]["maximum_coefficient_absolute_error"] <= 1e-12
    assert not any(row["candidate_id"].startswith("stage_al_") for row in candidates)


def test_exact_off_is_unique_and_literal_controller_absence():
    candidates, _ = _build()
    off = [row for row in candidates if row["node_accessibility"] is None]
    assert len(off) == 1
    assert off[0]["candidate_id"] == "exact_off"
    assert all(row["selection_eligible"] is False for row in candidates)


def test_event_unit_and_source_topology_are_copied_exactly_from_ak():
    config = _config()
    loaded = _inputs(config)
    _build(config, loaded)
    assert config["event_unit"] == loaded["stage_ak_config"]["event_unit"]
    assert config["event_unit"] == loaded["stage_ak_manifest"]["event_unit"]
    assert config["source_topology"] == loaded["stage_ak_config"]["source_topology"]


def test_support_contract_and_controller_scale_are_frozen():
    config = _config()
    candidates, audit = _build(config)
    support = config["primary_substrate"]["support"]
    assert support["field_names"] == ["node_field", "node_dispersion_field"]
    assert audit["support"] == support
    by_id = {row["candidate_id"]: row for row in candidates}
    assert by_id["zero_sum_c010"]["node_accessibility"]["c"] == 0.1
    assert by_id["zero_sum_c020"]["node_accessibility"]["c"] == 0.2
    assert by_id["zero_sum_c040"]["node_accessibility"]["c"] == 0.4
    assert by_id["raise_only_c020"]["node_accessibility"]["mode"] == "raise_only"
    shifted = by_id["spatial_shift_c020"]["node_accessibility"]
    assert shifted["spatial_shift"] == (
        config["node_accessibility_contract"]["spatial_shift"]
    )
    k2 = config["model_internal_k2_contract"]
    assert k2["formal_feature"].startswith("signed_causal_family_displacement")
    assert k2["minimum_temporal_blocks_per_direction"] == 2
    assert set(k2["matched_control_pairs"]["zero_sum_c020"]) == {
        "exact_off", "raise_only_c020", "spatial_shift_c020",
    }


def test_freezer_rejects_spatial_control_or_k2_contract_drift():
    config = _config()
    config["node_accessibility_contract"]["spatial_shift"]["shift_bins"] = 3
    with pytest.raises(RuntimeError, match="spatial-shift"):
        _build(config)

    config = _config()
    config["model_internal_k2_contract"]["formal_feature"] = "onset_map"
    with pytest.raises(RuntimeError, match="formal K2"):
        _build(config)


def test_freezer_rejects_duplicate_or_implicit_off():
    config = _config()
    config["arms"][1]["controller"] = None
    with pytest.raises(RuntimeError, match="exact-off|arm set"):
        _build(config)


def test_freezer_rejects_primary_mapping_or_support_drift():
    config = _config()
    config["primary_substrate"]["mapping_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="mapping hash"):
        _build(config)

    config = _config()
    config["primary_substrate"]["support"]["field_names"] = ["node_field"]
    with pytest.raises(RuntimeError, match="support fields"):
        _build(config)


def test_freezer_rejects_stage_al_as_a_distinct_substrate():
    config = _config()
    loaded = _inputs(config)
    altered = copy.deepcopy(loaded["stage_al_manifest"])
    alias = next(
        row for row in altered["candidates"]
        if row["candidate_id"] == "stage_al_m100_d100"
    )
    alias["node_dispersion_field"]["coefficients"][0][0] += 1e-5
    with pytest.raises(RuntimeError, match="not numerically equivalent"):
        _build(config, {**loaded, "stage_al_manifest": altered})


def test_freezer_rejects_patient_target_input_or_open_pathway():
    config = _config()
    record = config["inputs"].pop("stage_al_manifest")
    config["inputs"]["patient_target"] = record
    with pytest.raises(RuntimeError, match="input set|patient"):
        _build(config)

    config = _config()
    config["pathways"]["Z_M"] = "active"
    with pytest.raises(RuntimeError, match="must remain off"):
        _build(config)
