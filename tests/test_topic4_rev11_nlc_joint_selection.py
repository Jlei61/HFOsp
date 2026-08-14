import json
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev11_nlc_joint_node_connectivity_selection import (
    _score_network_sample,
)
from scripts.freeze_topic4_rev11_nlc_joint_node_connectivity_selection import (
    selection_library,
)
from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds
from scripts.run_topic4_rev9l_forced_source_worker import _sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_selection.json"


def _inputs():
    config = json.loads(CONFIG.read_text())
    fit_manifest = json.loads(
        (ROOT / config["inputs"]["nlc2_fit_manifest"]["path"]).read_text()
    )
    fit_verdict = json.loads(
        (ROOT / config["inputs"]["nlc2_fit_verdict"]["path"]).read_text()
    )
    return config, fit_manifest, fit_verdict


def test_nlc3_inputs_and_fresh_network_contract_are_frozen():
    config, *_ = _inputs()
    assert active_network_seeds(config) == [1541, 1542, 1543, 1544, 1545, 1546]
    assert config["search"]["phase"] == "selection"
    assert config["search"]["simulation"]["duration_ms"] == 16000.0
    assert config["search"]["beta"] == "closed"
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_nlc3_library_copies_fit_candidates_without_refit():
    config, fit_manifest, fit_verdict = _inputs()
    candidates = selection_library(config, fit_manifest, fit_verdict)
    assert [row["candidate_id"] for row in candidates] == [
        "nlc2_joint_04_02", "nlc2_joint_03_15", "joint_04_control",
        "nlc2_joint_03_14", "nlc2_joint_03_11", "joint_03_control",
        "node_baseline",
    ]
    source = {
        row["candidate_id"]: row
        for row in fit_manifest["candidate_set"]["candidates"]
    }
    for candidate in candidates:
        assert candidate == source[candidate["candidate_id"]]


def test_network_sample_score_rewards_better_three_way_geometry():
    config, fit_manifest, _ = _inputs()
    candidate = next(
        row for row in fit_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == "joint_04_control"
    )
    good = [{
        "natural": 0.9, "crossfit": 0.6,
        "recruitment_A": 0.1, "recruitment_B": 0.1,
        "shape_A": 0.7, "shape_B": 0.8,
        "ood": 0.1, "occupancy": 0.2,
    } for _ in range(6)]
    bad = [{**row, "natural": 0.4, "crossfit": -0.2, "shape_A": 4.0}
           for row in good]
    assert _score_network_sample(candidate, good, config) < (
        _score_network_sample(candidate, bad, config)
    )


def test_nlc3_selection_keeps_dynamic_ictal_variables_out():
    config, fit_manifest, fit_verdict = _inputs()
    for candidate in selection_library(config, fit_manifest, fit_verdict):
        assert candidate["mz"]["mode"] == "off"
        assert candidate["adaptation"]["mode"] == "off"
        assert candidate["inhibitory_resource"]["mode"] == "off"
        np.testing.assert_equal(np.asarray(candidate["coefficients"]).shape, (2, 6))
