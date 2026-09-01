import json
from pathlib import Path

import numpy as np

from scripts.audit_topic4_rev11_nlc_frozen_substrate_confirmation import (
    ARM_IDS,
    factorial_interaction,
)
from scripts.freeze_topic4_rev11_nlc_frozen_substrate_confirmation import (
    confirmation_library,
)
from scripts.run_topic4_rev10_r_edge_flow_worker import active_network_seeds
from scripts.run_topic4_rev9l_forced_source_worker import _sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"


def _inputs():
    config = json.loads(CONFIG.read_text())
    selection_manifest = json.loads(
        (ROOT / config["inputs"]["nlc3_selection_manifest"]["path"]).read_text()
    )
    selection_verdict = json.loads(
        (ROOT / config["inputs"]["nlc3_selection_verdict"]["path"]).read_text()
    )
    return config, selection_manifest, selection_verdict


def test_confirmation_inputs_final_pool_and_duration_are_frozen():
    config, *_ = _inputs()
    assert active_network_seeds(config) == list(range(1561, 1573))
    assert config["search"]["phase"] == "confirmation"
    assert config["search"]["simulation"]["duration_ms"] == 20000.0
    assert config["search"]["beta"] == "closed"
    for record in config["inputs"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_confirmation_library_is_selected_joint_plus_pathway_ablations():
    config, manifest, verdict = _inputs()
    rows = confirmation_library(config, manifest, verdict)
    assert tuple(row["candidate_id"] for row in rows) == ARM_IDS
    by_id = {row["candidate_id"]: row for row in rows}
    joint = np.asarray(by_id["joint_04_control"]["coefficients"])
    node = np.asarray(by_id["node_baseline"]["coefficients"])
    ee = np.asarray(by_id["joint_04_ee_only"]["coefficients"])
    etoi = np.asarray(by_id["joint_04_etoi_only"]["coefficients"])
    np.testing.assert_array_equal(node, np.zeros((2, 6)))
    np.testing.assert_array_equal(ee[0], joint[0])
    np.testing.assert_array_equal(ee[1], np.zeros(6))
    np.testing.assert_array_equal(etoi[0], np.zeros(6))
    np.testing.assert_array_equal(etoi[1], joint[1])
    assert len({row["node_field"]["field_sha256"] for row in rows}) == 1
    assert all(row["mz"]["mode"] == "off" for row in rows)


def test_factorial_interaction_is_zero_for_additive_endpoints():
    config, *_ = _inputs()
    rows = {}
    offsets = {
        "node_baseline": 0.0,
        "joint_04_ee_only": 0.1,
        "joint_04_etoi_only": 0.2,
        "joint_04_control": 0.3,
    }
    for candidate_id, offset in offsets.items():
        rows[candidate_id] = {
            str(seed): {
                "natural": 0.5 + offset,
                "crossfit": 0.2 + offset,
                "shape_A": 1.0 + offset,
                "shape_B": 1.0 + offset,
                "ood": 0.4 + offset,
                "occupancy": 0.2 + offset,
            }
            for seed in range(12)
        }
    result = factorial_interaction(rows, config)
    assert result["status"] == "OK"
    assert result["n_paired_networks"] == 12
    for endpoint in result["endpoints"].values():
        assert abs(endpoint["observed_interaction"]) < 1e-12
