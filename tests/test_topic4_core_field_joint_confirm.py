"""Selection contracts for the unseen-network joint confirmation producer."""
import numpy as np
import pytest

from scripts.run_topic4_core_field_stage3_joint_confirm import (
    _distance_to_target,
    confirmation_seeds,
    evaluation_errors,
    select_candidates,
)
from src.topic4_core_field_profile import fit_rank_curve_reference
from src.topic4_core_field_stage3 import n_free


def _checkpoint():
    theta0 = np.arange(n_free(3), dtype=float).tolist()
    theta1 = (np.arange(n_free(3), dtype=float) + 1.0).tolist()
    return {
        "K": 3,
        "history": [
            {"generation": 0, "distance": 0.8, "n_usable": 20,
             "theta": theta0, "seeds": [601, 602]},
            {"generation": 1, "distance": 0.6, "n_usable": 20,
             "theta": theta1, "seeds": [603, 604]},
        ],
        "optimizer": {"mean": np.zeros(n_free(3)).tolist()},
    }


def test_confirmation_candidates_are_frozen_without_using_confirmation_scores():
    selected = select_candidates(_checkpoint(), 20.0)
    assert [row["roles"] for row in selected] == [
        ["training_global_best", "final_generation_best"],
        ["final_optimizer_mean"],
    ]
    assert selected[0]["training"]["distance"] == 0.6


def test_confirmation_seed_pool_is_disjoint_from_every_fit_generation():
    seeds, fit = confirmation_seeds(_checkpoint(), 6)
    assert seeds == [501, 502, 503, 504, 505, 506]
    assert set(seeds).isdisjoint(fit)
    with pytest.raises(ValueError):
        confirmation_seeds(_checkpoint(), 100)


def test_worker_errors_keep_candidate_and_network_identity():
    candidates = [dict(candidate_id="a", roles=["global"]),
                  dict(candidate_id="b", roles=["final"])]
    raw = [dict(seed=501, events=[]), dict(seed=502, error="boom"),
           dict(seed=501, error="bad"), dict(seed=502, events=[])]
    assert evaluation_errors(raw, candidates, [501, 502]) == [
        dict(candidate_id="a", roles=["global"], seed=502, error="boom"),
        dict(candidate_id="b", roles=["final"], seed=501, error="bad"),
    ]


def test_target_distance_uses_exactly_the_frozen_source_count():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(80, 6))
    reference = fit_rank_curve_reference(
        train, n_components=4, n_reference=60, n_projections=8, seed=2)
    target = reference["reference_z"]
    first = _distance_to_target(train[:20], reference, target, n_events=20)
    extended = np.vstack((train[:20], np.full((10, 6), 999.0)))
    second = _distance_to_target(extended, reference, target, n_events=20)
    assert np.isfinite(first) and np.isfinite(second)
    assert _distance_to_target(train[:19], reference, target, n_events=20) is None
