from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import aggregate_topic4_rev16_joint_m3_m4_response as analysis


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_response_analysis.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_contract_is_node_only_and_training_only():
    config, loaded = analysis._load_inputs(CONFIG, ARTIFACT_ROOT)
    assert config["response_tensor"]["joint_real_coordinates"] == 48
    assert config["progression_rule"]["fresh_selection_network_seeds"] == [2351, 2352, 2353]
    assert config["progression_rule"]["natural_kmeans_used_for_construction"] is False
    assert config["progression_rule"]["patient_heldout_used_for_construction"] is False
    assert config["progression_rule"]["EE_EtoI_ZM"] == "off"
    assert loaded["m3_response_aggregate"][1]["status"] == "COMPLETE"


def test_shell_tensor_uses_exact_sign_pairs():
    rows = []
    for seed in (2331, 2332, 2333):
        for coordinate in range(20):
            for sign in (-1, 1):
                rows.append({
                    "seed": seed, "shell_coordinate_index": coordinate,
                    "full_mode_index": coordinate + 14,
                    "mode_nx": coordinate, "mode_ny": 4,
                    "phase": "cos", "sign": sign,
                    "mode_0_mean": sign * (coordinate + 1),
                    "mode_1_mean": sign * 2 * (coordinate + 1),
                    "mode_0_effective_events": 10 + sign,
                    "mode_1_effective_events": 9 - sign,
                    "j14": sign * 3 * (coordinate + 1),
                })
    tensor = analysis.response_tensor(rows)
    assert np.asarray(tensor["gradients"]["A"]).shape == (3, 20)
    assert np.allclose(np.asarray(tensor["gradients"]["A"])[:, 0], 1.25)
    assert len(tensor["coordinate_pairs"]) == 60


def test_joint_tensor_concatenates_m3_then_shell():
    metrics = ("A", "B", "J14", "support_A", "support_B")
    m3 = {
        "response_tensor": {
            "network_seeds": [2331, 2332, 2333],
            "gradients": {key: np.ones((3, 28)).tolist() for key in metrics},
        },
    }
    shell = {
        "network_seeds": [2331, 2332, 2333],
        "gradients": {key: (2 * np.ones((3, 20))).tolist() for key in metrics},
    }
    result = analysis.joint_tensor(m3, shell)
    matrix = np.asarray(result["gradients"]["A"])
    assert matrix.shape == (3, 48)
    assert np.all(matrix[:, :28] == 1.0)
    assert np.all(matrix[:, 28:] == 2.0)
    assert result["m3_coordinate_slice"] == [0, 28]
    assert result["m4_shell_coordinate_slice"] == [28, 48]


def test_joint_maximin_protects_b_in_all_networks():
    g_a = np.zeros((3, 48), dtype=float)
    g_b = np.zeros((3, 48), dtype=float)
    g_a[:, 0] = [1.0, 1.2, 0.8]
    g_b[:, 28] = [1.0, 0.7, 1.3]
    result = analysis.maximin_direction(g_a, g_b)
    assert result["feasible_positive_margin"] is True
    direction = np.asarray(result["direction"])
    assert direction.shape == (48,)
    assert np.all(g_a @ direction < -0.79)
    assert np.all(g_b @ direction <= 1e-7)


def test_joint_sparse_can_select_m3_and_shell_coordinates():
    g_a = np.zeros((3, 48), dtype=float)
    g_b = np.zeros((3, 48), dtype=float)
    support = np.zeros((3, 48), dtype=float)
    g_a[:, 2] = [1.0, 1.1, 0.9]
    g_a[:, 31] = [-0.8, -0.9, -0.7]
    result = analysis.consensus_sparse_direction(
        g_a, g_b, support, max_coordinates=10,
    )
    assert result["feasible"] is True
    assert result["selected_coordinates"] == [2, 31]


def test_incomplete_inventory_does_not_construct_directions(monkeypatch, tmp_path):
    config = json.loads(CONFIG.read_text())
    manifest = {"provenance": {"git_commit": "worker"}}
    loaded = {
        "m4_shell_config": (tmp_path / "source.json", {"output_root": "workers"}),
        "m4_shell_manifest": (tmp_path / "manifest.json", manifest),
        "m3_response_aggregate": (tmp_path / "m3.json", {"status": "COMPLETE"}),
        "j14_config": (tmp_path / "j14.json", {}),
        "support_config": (tmp_path / "support.json", {}),
    }
    monkeypatch.setattr(analysis, "_load_inputs", lambda *args: (config, loaded))
    monkeypatch.setattr(
        analysis, "_provenance",
        lambda *args: {"formal_ready": True, "snn_simulation_run": False},
    )
    monkeypatch.setattr(
        analysis, "_inventory",
        lambda *args: ([], {
            "expected_runs": 120, "present_validated": 0,
            "missing": ["all"], "invalid_artifact": [],
            "complete_cartesian_product": False,
        }),
    )
    result = analysis.aggregate(tmp_path / "config.json", tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert result["joint_response_tensor"] is None
    assert result["robust_directions"] is None
