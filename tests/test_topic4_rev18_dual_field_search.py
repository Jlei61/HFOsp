from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.topic4_rev18_dual_field_search import (
    evaluate_candidate,
    global_blueprints,
    network_loss,
    nominate,
)
from scripts.run_topic4_rev12_node_worker import _validate_scientific_role


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return json.loads((
        ROOT / "config/topic4_rev18_dual_field_global_screen.json"
    ).read_text())


def _row(seed: int, *, j14=2.0, a=1.5, b=1.2, support_a=6.0,
         support_b=8.0, alignment=0.9, clusters=(10, 10), ood=4) -> dict:
    return {
        "seed": seed,
        "run_status": "VALID",
        "j14": j14,
        "mode_0_mean": a,
        "mode_1_mean": b,
        "mode_0_effective_events": support_a,
        "mode_1_effective_events": support_b,
        "ood_count": ood,
        "n_contact_primary": 20,
        "natural_kmeans": {
            "status": "OK",
            "direction_balanced_alignment": alignment,
            "cluster_counts": list(clusters),
        },
    }


def test_global_blueprints_are_uniform_sheet_multicoordinate_and_deterministic():
    design = _config()["dual_field_global_screen"]
    first = global_blueprints(design)
    second = global_blueprints(design)
    assert first == second
    assert len(first) == 51
    assert len({row["candidate_id"] for row in first}) == len(first)
    assert all(row["observation_coordinates_used"] is False for row in first)
    assert all(np.isclose(np.linalg.norm(row["direction"]), 1.0) for row in first)
    sobol = [row for row in first if row["family"].startswith("sobol")]
    assert len(sobol) == 48
    assert all(np.count_nonzero(np.abs(row["direction"]) > 1e-12) > 20 for row in sobol)


def test_network_loss_rewards_complete_and_weak_mode_improvement():
    contract = _config()["robust_objective"]
    anchor = _row(1)
    improved = _row(1, j14=1.6, a=1.1)
    degraded = _row(1, j14=2.4, a=1.9)
    assert network_loss(improved, anchor, contract)["loss"] \
        < network_loss(anchor, anchor, contract)["loss"]
    assert network_loss(degraded, anchor, contract)["loss"] \
        > network_loss(anchor, anchor, contract)["loss"]


def test_network_loss_penalizes_kmeans_collapse_and_support_loss():
    contract = _config()["robust_objective"]
    anchor = _row(1)
    collapsed = _row(
        1, j14=1.6, a=1.1, support_a=1.0, alignment=0.4, clusters=(19, 1),
    )
    preserved = _row(
        1, j14=1.6, a=1.1, support_a=7.0, alignment=0.9, clusters=(9, 11),
    )
    assert network_loss(collapsed, anchor, contract)["loss"] \
        > network_loss(preserved, anchor, contract)["loss"]


def test_candidate_aggregation_is_equal_network_and_robust():
    contract = _config()["robust_objective"]
    anchors = [_row(seed) for seed in (1, 2, 3)]
    stable = [_row(seed, j14=1.8, a=1.3) for seed in (1, 2, 3)]
    unstable = [
        _row(1, j14=1.3, a=0.9),
        _row(2, j14=1.3, a=0.9),
        _row(3, j14=3.0, a=2.5),
    ]
    stable_result = evaluate_candidate(stable, anchors, contract)
    unstable_result = evaluate_candidate(unstable, anchors, contract)
    assert stable_result["robust_loss"] < unstable_result["robust_loss"]
    ranked = nominate([
        {"candidate_id": "unstable", **unstable_result},
        {"candidate_id": "stable", **stable_result},
    ], maximum_candidates=1)
    assert [row["candidate_id"] for row in ranked] == ["stable"]


def test_invalid_candidate_cannot_be_nominated():
    contract = _config()["robust_objective"]
    anchors = [_row(seed) for seed in (1, 2, 3)]
    candidate = [_row(seed) for seed in (1, 2, 3)]
    candidate[1]["run_status"] = "INVALID_RUNAWAY"
    result = evaluate_candidate(candidate, anchors, contract)
    assert result["valid_all_networks"] is False
    assert nominate([{"candidate_id": "bad", **result}], maximum_candidates=1) == []


def test_shared_worker_accepts_only_the_new_node_only_screen_role():
    _validate_scientific_role("development_only_dual_continuous_node_global_screen")
