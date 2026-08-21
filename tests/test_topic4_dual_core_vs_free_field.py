import json
from pathlib import Path

import numpy as np

from scripts.run_topic4_rev10_sa_spectral_field_worker import _candidate_node
from scripts.aggregate_topic4_dual_core_vs_free_field import _event_masks
from src.topic4_continuous_field import continuous_field_h
from src.topic4_core_field_rev9 import reconstruct_node_from_h
from src.topic4_manual_dual_core import (
    budget_matched_dual_core_h,
    dual_core_query_h,
)


ROOT = Path(__file__).resolve().parents[1]


def _stage():
    return {
        "N_core_manual": 5,
        "quantile_seed": 7,
        "engine": {
            "L": 20.0, "core_mean": -50.0, "core_std": 2.0,
            "v_base": -40.0,
        },
    }


def test_budget_matched_dual_core_is_exact_and_deterministic():
    positions = np.asarray([
        [0.0, 0.0], [0.1, 0.0], [0.2, 0.0],
        [9.8, 0.0], [9.9, 0.0], [10.0, 0.0], [5.0, 5.0],
    ])
    centers = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    first, audit = budget_matched_dual_core_h(
        positions, centers, target_count=5,
    )
    second, second_audit = budget_matched_dual_core_h(
        positions, centers, target_count=5,
    )
    assert first.sum() == 5
    assert np.array_equal(first, second)
    assert audit == second_audit
    assert sum(audit["selected_per_core"]) == 5


def test_dual_core_query_uses_frozen_e_cutoff():
    centers = np.asarray([[0.0, 0.0], [10.0, 0.0]])
    query = np.asarray([[0.5, 0.0], [1.1, 0.0], [9.2, 0.0]])
    assert np.array_equal(
        dual_core_query_h(query, centers, distance_cutoff_mm=1.0),
        np.asarray([1.0, 0.0, 1.0]),
    )


def test_worker_manual_node_reuses_frozen_signed_depth_contract():
    positions = np.asarray([
        [0.0, 0.0], [0.2, 0.0], [0.4, 0.0],
        [9.6, 0.0], [9.8, 0.0], [10.0, 0.0],
    ])
    candidate = {
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": [[0.0, 0.0], [10.0, 0.0]],
        "target_count": 5,
    }
    observed = _candidate_node(
        candidate, positions, n_total=8, stage=_stage(), config={},
    )
    expected = reconstruct_node_from_h(
        observed["h"], n_total=8, quantile_seed=7,
        core_mean=-50.0, core_std=2.0, v_base=-40.0,
    )
    assert np.array_equal(observed["d"], expected["d"])
    assert np.array_equal(observed["vtheta"], expected["vtheta"])
    assert observed["field_audit"]["selected_count"] == 5


def test_existing_spline_candidate_path_is_unchanged():
    rng = np.random.default_rng(9)
    positions = rng.uniform(0.0, 20.0, size=(40, 2))
    coefficients = rng.normal(size=(6, 6))
    candidate = {
        "field_type": "spline_continuous",
        "coefficients": coefficients.tolist(),
        "n_basis": 6,
        "degree": 3,
    }
    stage = _stage()
    stage["N_core_manual"] = 5
    observed = _candidate_node(
        candidate, positions, n_total=50, stage=stage, config={},
    )
    h, _ = continuous_field_h(
        coefficients, positions, n_basis=6, degree=3,
        target_count=5, L=20.0,
    )
    expected = reconstruct_node_from_h(
        h, n_total=50, quantile_seed=7,
        core_mean=-50.0, core_std=2.0, v_base=-40.0,
    )
    assert np.array_equal(observed["h"], expected["h"])
    assert np.array_equal(observed["vtheta"], expected["vtheta"])


def test_comparison_config_freezes_equal_budget_and_zero_edge_rows():
    config = json.loads(
        (ROOT / "config/topic4_dual_core_vs_free_field.json").read_text()
    )
    assert config["manual_dual_core"]["target_count"] == 1129
    assert config["search"]["simulation"]["duration_ms"] == 20000.0
    assert config["search"]["confirmation_network_seeds"] == list(range(1561, 1573))
    assert config["search"]["beta"] == "closed"


def test_single_shaft_event_remains_in_distribution_but_not_kmeans():
    onsets = np.asarray([
        [0.0, 1.0, 2.0, np.nan, np.nan],
        [0.0, 1.0, 2.0, 3.0, np.nan],
    ])
    masks = _event_masks(
        onsets, np.asarray([True, True]), np.asarray([False, False]),
        {"ICL": np.asarray([0, 1, 2]), "SCL": np.asarray([3, 4])},
    )
    assert np.array_equal(masks["distribution"], [True, True])
    assert np.array_equal(masks["formal_kmeans"], [False, True])
