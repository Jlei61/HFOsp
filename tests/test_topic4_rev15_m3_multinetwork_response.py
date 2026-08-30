from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import aggregate_topic4_rev15_m3_multinetwork_response as analysis
from scripts import wait_topic4_rev15_m3_multinetwork_then_aggregate as waiter


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev15_m3_multinetwork_response_analysis.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_analysis_contract_is_training_only_and_three_network():
    config, loaded = analysis._load_inputs(CONFIG, ARTIFACT_ROOT)
    assert config["response_tensor"]["network_seeds"] == [2331, 2332, 2333]
    assert config["progression_rule"]["fresh_selection_network_seeds"] == [2341, 2342, 2343]
    assert config["progression_rule"]["natural_kmeans_used_for_construction"] is False
    assert config["progression_rule"]["patient_heldout_used_for_construction"] is False
    assert config["progression_rule"]["EE_EtoI_ZM"] == "off"
    assert config["response_tensor"]["frozen_direction_classifier_input"] == "full_contact_onset_timing"
    assert config["response_tensor"]["natural_kmeans_input"] == "not_used_for_construction"
    assert "seed2331_atlas_aggregate" not in config["inputs"]
    assert "failed_combination_aggregate" not in config["inputs"]
    assert loaded["multinetwork_manifest"][1]["source_atlas"][
        "candidate_payload_exact_copy"
    ] is True


def test_maximin_direction_improves_every_network_and_protects_b():
    g_a = np.zeros((3, 28), dtype=float)
    g_b = np.zeros((3, 28), dtype=float)
    g_a[:, 0] = [1.0, 1.2, 0.8]
    g_b[:, 1] = [1.0, 0.7, 1.3]
    result = analysis.maximin_direction(g_a, g_b)
    assert result["feasible_positive_margin"] is True
    direction = np.asarray(result["direction"])
    assert np.all(g_a @ direction < -0.79)
    assert np.all(g_b @ direction <= 1e-7)
    assert np.isclose(np.linalg.norm(direction), 1.0)


def test_support_protected_maximin_respects_all_halfspaces():
    g_a = np.zeros((3, 28), dtype=float)
    g_b = np.zeros((3, 28), dtype=float)
    support_a = np.zeros((3, 28), dtype=float)
    support_b = np.zeros((3, 28), dtype=float)
    g_a[:, 0] = 1.0
    g_b[:, 1] = 1.0
    support_a[:, 0] = -1.0
    support_b[:, 0] = -0.5
    result = analysis.maximin_direction(
        g_a, g_b, support_a=support_a, support_b=support_b,
    )
    assert result["feasible_positive_margin"] is True
    direction = np.asarray(result["direction"])
    assert np.all(g_a @ direction < 0.0)
    assert np.all(g_b @ direction <= 1e-7)
    assert np.all(support_a @ direction >= -1e-7)
    assert np.all(support_b @ direction >= -1e-7)


def test_consensus_sparse_requires_networkwise_sign_agreement():
    g_a = np.zeros((3, 28), dtype=float)
    g_b = np.zeros((3, 28), dtype=float)
    support_a = np.zeros((3, 28), dtype=float)
    g_a[:, 0] = [1.0, 1.1, 0.9]
    g_b[:, 0] = [0.2, 0.1, -0.1]
    support_a[:, 0] = [-1.0, -0.8, 0.2]
    g_a[:, 1] = [1.0, -1.0, 1.0]
    result = analysis.consensus_sparse_direction(g_a, g_b, support_a)
    assert result["feasible"] is True
    assert result["selected_coordinates"] == [0]
    direction = np.asarray(result["direction"])
    assert direction[0] < 0.0
    assert direction[1] == 0.0


def test_response_tensor_uses_exact_sign_pairs():
    rows = []
    for seed in (2331, 2332, 2333):
        for coordinate in range(28):
            for sign in (-1, 1):
                rows.append({
                    "seed": seed, "coordinate_index": coordinate,
                    "sign": sign, "mode_0_mean": sign * (coordinate + 1),
                    "mode_1_mean": sign * 2 * (coordinate + 1),
                    "mode_0_effective_events": 10 + sign,
                    "mode_1_effective_events": 9 - sign,
                    "j14": sign * 3 * (coordinate + 1),
                })
    tensor = analysis.response_tensor(rows)
    g_a = np.asarray(tensor["gradients"]["A"])
    assert g_a.shape == (3, 28)
    assert np.allclose(g_a[:, 0], 1.25)
    assert np.allclose(g_a[:, 27], 35.0)
    assert len(tensor["coordinate_pairs"]) == 84


def test_aggregate_rescores_seed2331_and_fresh_networks_from_raw_workers(
    monkeypatch, tmp_path,
):
    config = {
        "output_root": "analysis",
        "response_tensor": {"central_difference_denominator": 1.6},
        "robust_direction_construction": {},
        "progression_rule": {},
        "claim_boundary": "test",
    }
    source = {"output_root": "fresh"}
    fresh_manifest = {"provenance": {"git_commit": "worker"}}
    loaded = {
        "multinetwork_config": (tmp_path / "fresh.json", source),
        "multinetwork_manifest": (tmp_path / "fresh_manifest.json", fresh_manifest),
        "j14_config": (tmp_path / "j14.json", {}),
        "support_config": (tmp_path / "support.json", {}),
    }
    complete_58 = {
        "expected_runs": 58, "present_validated": 58,
        "missing": [], "invalid_artifact": [],
        "complete_cartesian_product": True,
    }
    complete_116 = {
        "expected_runs": 116, "present_validated": 116,
        "missing": [], "invalid_artifact": [],
        "complete_cartesian_product": True,
    }
    seed_manifest = {"candidates": [{"candidate_id": "seed_candidate"}]}
    monkeypatch.setattr(analysis, "_load_inputs", lambda *args: (config, loaded))
    monkeypatch.setattr(
        analysis, "_provenance",
        lambda *args: {"formal_ready": True, "snn_simulation_run": False},
    )
    monkeypatch.setattr(
        analysis, "_inventory",
        lambda *args: ([{"source": "fresh_raw_worker"}], complete_116),
    )
    monkeypatch.setattr(
        analysis, "_seed2331_inventory",
        lambda *args: (
            [{"source": "seed2331_raw_worker"}], complete_58, seed_manifest,
        ),
    )
    score_calls = []

    def fake_score(records, manifest, *args):
        score_calls.append((records[0]["source"], manifest))
        return [{"rescored_source": records[0]["source"]}]

    def fake_tensor(rows, denominator):
        assert denominator == 1.6
        assert [row["rescored_source"] for row in rows] == [
            "seed2331_raw_worker", "fresh_raw_worker",
        ]
        return {"coordinate_pairs": []}

    monkeypatch.setattr(analysis, "_score_fresh", fake_score)
    monkeypatch.setattr(analysis, "response_tensor", fake_tensor)
    monkeypatch.setattr(analysis, "construct_directions", lambda *args: {"mean_a": {}})

    result = analysis.aggregate(tmp_path / "config.json", tmp_path)

    assert result["status"] == "COMPLETE"
    assert result["inventory"]["present_validated"] == 174
    assert [row[0] for row in score_calls] == [
        "seed2331_raw_worker", "fresh_raw_worker",
    ]
    assert result["ranking_contract"]["historical_aggregate_numeric_rows_used"] is False


def test_waiter_requires_complete_clean_cartesian_product():
    running = {
        "status": "REV15_M3_MULTINETWORK_QUEUE_RUNNING",
        "n_jobs": 116, "n_complete": 36, "n_failed": 0,
        "n_invalid_artifact": 0,
    }
    assert waiter.classify(running) == "wait"
    complete = {
        **running, "status": waiter.COMPLETE, "n_complete": 116,
    }
    assert waiter.classify(complete) == "complete"
    assert waiter.classify({**complete, "n_invalid_artifact": 1}) == "failed"
    assert waiter.classify({
        **running, "status": "REV15_M3_MULTINETWORK_QUEUE_FAILED",
    }) == "failed"
