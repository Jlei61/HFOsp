import json
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev11_nlc_joint_node_connectivity_fit import (
    joint_fit_score,
)
from scripts.freeze_topic4_rev11_nlc_joint_node_connectivity_fit import (
    candidate_library,
)
from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds
from scripts.run_topic4_rev9l_forced_source_worker import _sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_fit.json"


def _inputs():
    config = json.loads(CONFIG.read_text())
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    anchor = next(
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )
    nlc1 = json.loads(
        (ROOT / config["inputs"]["nlc1_manifest"]["path"]).read_text()
    )
    rescore = json.loads(
        (ROOT / config["inputs"]["nlc1_crossfit_rescore"]["path"]).read_text()
    )
    return config, anchor, nlc1, rescore


def test_nlc2_inputs_and_fit_networks_are_frozen():
    config, *_ = _inputs()
    assert active_network_seeds(config) == [1521, 1522, 1523]
    assert config["search"]["beta"] == "closed"
    assert config["search"]["simulation"]["duration_ms"] == 8000.0
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_nlc2_library_is_deterministic_continuous_and_joint():
    config, anchor, nlc1, rescore = _inputs()
    left, residuals = candidate_library(config, anchor, nlc1, rescore)
    right, _ = candidate_library(config, anchor, nlc1, rescore)
    assert left == right
    assert len(left) == 35
    assert len(residuals["coefficients"]) == 12
    assert [row["candidate_id"] for row in left[:3]] == [
        "node_baseline", "joint_03_control", "joint_04_control",
    ]
    assert left[0]["node_field"]["field_type"] == "spline_continuous"
    assert np.all(np.asarray(left[0]["coefficients"]) == 0.0)
    for row in left[3:]:
        assert row["arm"] == "Node+EE+EtoI"
        assert row["node_field"]["field_type"] == "spline_continuous"
        assert row["node_field"]["component_count"] is None
        assert row["node_field"]["peak_count_constraint"] is None
        assert np.any(np.asarray(row["coefficients"])[0] != 0.0)
        assert np.any(np.asarray(row["coefficients"])[1] != 0.0)
        assert row["mz"]["mode"] == "off"


def test_joint_score_penalizes_a_worse_patient_margin():
    config, *_ = _inputs()
    metrics = {
        "natural_balanced_alignment_equal_network": {"equal_network_mean": 0.8},
        "crossfit_margin_equal_network": {"equal_network_mean": 0.4},
        "recruitment_worst_mode_error": 0.1,
    }
    aggregate = {
        "mean_network_shape_A": 1.0,
        "mean_network_shape_B": 1.0,
        "mean_network_ood_fraction": 0.1,
        "mean_network_fraction_time_above_detector": 0.2,
    }
    candidate = {"search_coordinates": {
        "node_amplitude": 0.0,
        "edge_delta": np.zeros((2, 6)).tolist(),
    }}
    good, _ = joint_fit_score(
        metrics, aggregate, candidate, config["search"]["joint_objective"],
        config["field_search"], config["local_connectivity_basis"],
    )
    metrics["crossfit_margin_equal_network"] = {"equal_network_mean": -0.4}
    bad, _ = joint_fit_score(
        metrics, aggregate, candidate, config["search"]["joint_objective"],
        config["field_search"], config["local_connectivity_basis"],
    )
    assert bad > good


def test_candidate_builder_does_not_consume_observation_geometry():
    source = (
        ROOT / "scripts/freeze_topic4_rev11_nlc_joint_node_connectivity_fit.py"
    ).read_text()
    assert "patient_train" not in source
    assert "contact_xy" not in source
    assert "shaft_id" not in source
