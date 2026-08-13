"""Scientific and engineering contracts for rev10-ZM1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.freeze_topic4_rev10_zm1_data_driven_h_zm as freezer
from scripts.audit_topic4_rev10_zm1_data_driven_h_zm import _delta


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev10_zm1_data_driven_h_zm.json"


def test_zm1_config_inputs_are_hash_locked_and_exploratory():
    config = json.loads(CONFIG.read_text())
    assert config["scientific_role"] == (
        "development_only_data_driven_h_zm_consistency"
    )
    assert config["search"]["acceptance"]["role"].startswith(
        "reported benchmarks only"
    )
    assert config["search"]["confirmation_network_seeds"] == [
        1351, 1352, 1353, 1354, 1355, 1356,
    ]
    for record in config["inputs"].values():
        digest = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        assert digest == record["sha256"]


def test_freezer_builds_common_random_number_pair(monkeypatch):
    monkeypatch.setattr(freezer, "_runtime_provenance", lambda commit: {
        "runtime_modules_dirty": False,
        "runtime_modules_match_expected_commit": True,
    })

    def fake_check_output(command, **kwargs):
        del kwargs
        if command[:3] == ["git", "rev-parse", "test-commit"]:
            return "a" * 40 + "\n"
        if command[:3] == ["git", "status", "--porcelain"]:
            return ""
        raise AssertionError(command)

    monkeypatch.setattr(freezer.subprocess, "check_output", fake_check_output)
    manifest = freezer.build_manifest(CONFIG, "test-commit")
    control, active = manifest["candidate_set"]["candidates"]
    assert control["candidate_id"] == "h_spou_slow_off"
    assert active["candidate_id"] == "h_spou_zm_transfer"
    assert control["coefficients"] == active["coefficients"]
    assert control["spatial_ou"] == active["spatial_ou"]
    assert control["mz"]["mode"] == "off"
    assert active["mz"]["mode"] == "z_plus_m"
    assert active["mz"]["use_z"] is active["mz"]["use_m"] is True
    assert manifest["fixed_contract"][
        "same_network_and_dynamics_seeds_across_arms"
    ] is True
    assert manifest["selection_freeze"]["selected_before_fresh_networks"] is True


def test_paired_delta_uses_equal_network_candidate_rows():
    active = {"activity": {"mean_network_returned_events_scored": 7.5}}
    control = {"activity": {"mean_network_returned_events_scored": 5.0}}
    assert _delta(
        active, control, "mean_network_returned_events_scored"
    ) == 2.5
