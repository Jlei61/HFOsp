import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev11_nlc_local_connectivity_canary import (
    build_candidates,
)
from scripts.audit_topic4_rev11_nlc_local_connectivity_canary import adjudicate
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (
    _returned_summary_filename,
)
from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds
from scripts.run_topic4_rev9l_forced_source_worker import _sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev11_nlc_local_connectivity_canary.json"


def _config():
    return json.loads(CONFIG.read_text())


def test_rev11_nlc_inputs_and_network_pool_are_frozen():
    config = _config()
    assert active_network_seeds(config) == [1501, 1502, 1503]
    assert config["search"]["beta"] == "closed"
    assert config["search"]["simulation"]["duration_ms"] == 8000.0
    for record in config["inputs"].values():
        assert len(record["sha256"]) == 64
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_candidate_library_is_deterministic_and_factorial():
    config = _config()
    left = build_candidates(config)
    right = build_candidates(config)
    assert left == right
    assert len(left) == 13
    assert [row["arm"] for row in left].count("Node") == 1
    assert [row["arm"] for row in left].count("Node+EE") == 3
    assert [row["arm"] for row in left].count("Node+EtoI") == 3
    assert [row["arm"] for row in left].count("Node+EE+EtoI") == 6
    baseline = left[0]
    np.testing.assert_array_equal(baseline["coefficients"], np.zeros((2, 6)))
    for candidate in left[1:4]:
        assert np.any(np.asarray(candidate["coefficients"])[0] != 0.0)
        assert np.all(np.asarray(candidate["coefficients"])[1] == 0.0)
    for candidate in left[4:7]:
        assert np.all(np.asarray(candidate["coefficients"])[0] == 0.0)
        assert np.any(np.asarray(candidate["coefficients"])[1] != 0.0)


def test_primary_canary_keeps_zm_and_gaba_feedback_frozen():
    config = _config()
    assert config["local_connectivity_basis"]["pathways"] == ["E_to_E", "E_to_I"]
    for candidate in build_candidates(config):
        assert candidate["mz"]["mode"] == "off"
        assert candidate["adaptation"]["mode"] == "off"
        assert candidate["inhibitory_resource"]["mode"] == "off"


def test_nlc_loader_uses_canary_summary_contract():
    assert _returned_summary_filename(_config()) == (
        "canary_summary_returned_only.json"
    )


def test_adjudication_is_exploratory_and_reports_best_arm():
    rows = [
        {"candidate_id": "node_baseline", "arm": "Node", "selection_score": 0.8},
        {"candidate_id": "ee", "arm": "Node+EE", "selection_score": 0.6},
        {"candidate_id": "ei", "arm": "Node+EtoI", "selection_score": 0.7},
        {"candidate_id": "joint", "arm": "Node+EE+EtoI", "selection_score": 0.5},
    ]
    result = adjudicate(rows)
    assert result["status"] == "REV11NLC_LOCAL_CONNECTIVITY_CAPACITY_CANDIDATE_FOUND"
    assert result["selected_candidate_id"] == "joint"
    assert np.isclose(result["selected_minus_baseline_score"], -0.3)
