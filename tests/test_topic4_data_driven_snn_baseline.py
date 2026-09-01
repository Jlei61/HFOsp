"""Shared baseline contracts for current and future data-driven SNN searches."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev10_d6_continuous_field_kmeans_screen import (
    candidate_library,
    fixed_runtime_contract,
    shared_runtime_baseline,
)
from scripts.freeze_topic4_rev10_d6_2_joint_continuous_field_surface import (
    candidate_library as joint_candidate_library,
)
from src.topic4_data_driven_snn_baseline import (
    DEFAULT_BASELINE,
    EXPECTED_ID,
    apply_data_driven_snn_baseline,
    baseline_record,
    load_data_driven_snn_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
D6_CONFIG = ROOT / "config/topic4_rev10_d6_continuous_field_kmeans_screen.json"
D62_CONFIG = ROOT / "config/topic4_rev10_d6_2_joint_continuous_field_surface.json"
D7_CONFIG = ROOT / "config/topic4_rev10_d7_active_zm_continuous_field_canary.json"


def test_shared_baseline_is_hash_locked_and_has_no_implicit_runtime():
    baseline = load_data_driven_snn_baseline()
    assert baseline["baseline_id"] == EXPECTED_ID
    assert baseline["consumer_contract"]["default_runtime_mode"] is None
    assert baseline["active_slow_state"]["reference_status"] == (
        "UNSAFE_ON_CURRENT_WARM_H_FIELD"
    )
    assert baseline["negative_evidence"]["runaway_networks_per_candidate"] == 3
    for record in baseline["inputs"].values():
        digest = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert digest == record["sha256"]


def test_baseline_application_is_explicit_and_preserves_candidate_field():
    baseline = load_data_driven_snn_baseline()
    candidate = {
        "candidate_id": "field_a",
        "node_field": {"field_sha256": "field-hash"},
    }
    active = apply_data_driven_snn_baseline(
        candidate, baseline, runtime_mode="active_z_plus_m",
    )
    control = apply_data_driven_snn_baseline(
        candidate, baseline, runtime_mode="paired_slow_off",
    )
    assert active["node_field"] == control["node_field"] == candidate["node_field"]
    assert active["mz"]["use_z"] is active["mz"]["use_m"] is True
    assert control["mz"]["mode"] == "off"
    assert active["spatial_ou"] == control["spatial_ou"]
    with pytest.raises(ValueError, match="runtime_mode must be explicit"):
        apply_data_driven_snn_baseline(candidate, baseline, runtime_mode=None)


def test_free_field_library_can_inherit_shared_zm_baseline():
    config = json.loads(D6_CONFIG.read_text())
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    anchor = next(
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )
    record = baseline_record(DEFAULT_BASELINE)
    config["data_driven_snn_baseline"] = {
        **record, "runtime_mode": "active_z_plus_m",
    }
    with pytest.raises(RuntimeError, match="cannot audit delayed Z/M runaway"):
        candidate_library(config, anchor)
    config["search"]["simulation"]["duration_ms"] = 20000.0
    rows, _ = candidate_library(config, anchor)
    assert rows
    assert all(row["mz"]["mode"] == "z_plus_m" for row in rows)
    assert all(
        row["data_driven_snn_baseline"]["baseline_id"] == EXPECTED_ID
        for row in rows
    )
    assert len({
        json.dumps(row["mz"], sort_keys=True) for row in rows
    }) == 1


def test_shared_baseline_rejects_mixed_dynamic_protocols():
    baseline = load_data_driven_snn_baseline()
    candidate = {"adaptation": {"mode": "local"}}
    with pytest.raises(RuntimeError, match="active adaptation"):
        apply_data_driven_snn_baseline(
            candidate, baseline, runtime_mode="active_z_plus_m",
        )


def test_joint_free_field_interpolation_preserves_inherited_zm_runtime():
    config = json.loads(D62_CONFIG.read_text())
    source = json.loads(
        (ROOT / config["inputs"]["d6_1_manifest"]["path"]).read_text()
    )
    baseline = load_data_driven_snn_baseline()
    source = copy.deepcopy(source)
    source["candidate_set"]["candidates"] = [
        apply_data_driven_snn_baseline(
            row, baseline, runtime_mode="active_z_plus_m",
        )
        for row in source["candidate_set"]["candidates"]
    ]
    rows, _ = joint_candidate_library(config, source)
    assert all(row["mz"]["mode"] == "z_plus_m" for row in rows)
    assert all(
        row["data_driven_snn_baseline"]["baseline_id"] == EXPECTED_ID
        for row in rows
    )


def test_d7_canary_uses_active_zm_for_twenty_seconds_on_fresh_networks():
    config = json.loads(D7_CONFIG.read_text())
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    anchor = next(
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )
    assert config["data_driven_snn_baseline"]["runtime_mode"] == (
        "active_z_plus_m"
    )
    assert config["search"]["simulation"]["duration_ms"] >= 20000.0
    assert config["search"]["fit_network_seeds"] == [1421, 1422]
    rows, _ = candidate_library(config, anchor)
    assert len(rows) == 49
    assert all(row["mz"]["mode"] == "z_plus_m" for row in rows)
    assert all(row["spatial_ou"]["mode"] == "local" for row in rows)


def test_d7_manifest_exposes_duration_and_late_runaway_contract():
    config = json.loads(D7_CONFIG.read_text())
    contract = fixed_runtime_contract(
        config, shared_runtime_baseline(config),
    )
    assert contract["duration_ms"] == 20000.0
    assert contract["late_runaway_is_invalid"] is True
