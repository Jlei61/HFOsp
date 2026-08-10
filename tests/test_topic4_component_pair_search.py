import json
from pathlib import Path

import numpy as np

from src.topic4_component_pair_search import (
    patient_descriptor_floor,
    score_candidate,
    selection_candidates_with_baseline,
    sobol_candidates,
)
from src.topic4_component_pair_edge import gamma_matrix


def test_frozen_phase1_candidates_are_all_six_dimensional():
    config = json.loads(Path(
        "config/topic4_rev9l_component_pair_edge.json").read_text())
    candidates = config["component_pair_family"]["phase1_candidates"]
    assert len(candidates) == 13
    assert len({row["candidate_id"] for row in candidates}) == 13
    for row in candidates:
        assert gamma_matrix(row["gamma"]).shape == (3, 2)


def test_sobol_candidates_are_deterministic_bounded_and_reserve_zero():
    search = {
        "n_candidates": 64, "dimension": 6, "scramble": True,
        "seed": 17, "candidate_zero_reserved_for_gamma0": True,
    }
    first = sobol_candidates(search, [-1.0, 1.0])
    second = sobol_candidates(search, [-1.0, 1.0])
    assert first == second
    np.testing.assert_array_equal(first[0]["gamma"], np.zeros(6))
    values = np.asarray([row["gamma"] for row in first])
    assert values.shape == (64, 6)
    assert np.all((values >= -1.0) & (values <= 1.0))


def test_selection_candidates_append_zero_residual_baseline_once():
    top = [{"candidate_id": "sobol_015", "gamma": [0.5] * 6}]
    selected = selection_candidates_with_baseline(top)
    assert [row["candidate_id"] for row in selected] == [
        "sobol_015", "sobol_000"]
    np.testing.assert_array_equal(selected[-1]["gamma"], np.zeros(6))
    assert selection_candidates_with_baseline(selected) == selected


def test_weakest_mode_objective_protects_the_worse_mode():
    floor = {"modes": {mode: {name: {"median": 0.0, "scale_iqr": 1.0}
                                      for name in (
                                          "recruitment_mean_absolute_error",
                                          "precedence_mean_absolute_error",
                                          "mean_rank_profile_absolute_error",
                                          "event_distribution_sliced_wasserstein")}
                        for mode in ("A", "B")}}
    descriptors = {"modes": {
        "A": {name: 2.0 for name in floor["modes"]["A"]},
        "B": {name: 0.0 for name in floor["modes"]["B"]},
    }}
    score = score_candidate(
        descriptors, floor, {"A": 1.0, "B": 1.0}, {"A": 0.0, "B": 0.0},
        readable_weight=2.0, tau=0.25, ood_weight=0.1)
    assert score["weak_mode"] == "A"
    assert score["mode_scores"]["A"] > score["mode_scores"]["B"]
    assert score["mode_scores"]["B"] < score["weakest_mode_shape"] \
        < score["mode_scores"]["A"]


def test_patient_floor_uses_distinct_block_draws_and_returns_finite_scales():
    rng = np.random.default_rng(4)
    curves = rng.normal(size=(24, 3))
    ranks = np.asarray([rng.permutation(4) for _ in range(24)], float)
    labels = np.repeat([0, 1], 12)
    blocks = np.asarray([f"A{i}" for i in range(12)]
                        + [f"B{i}" for i in range(12)])
    reference = {
        "center": np.zeros(3),
        "components": np.eye(2, 3),
        "score_center": np.zeros(2),
        "score_scale": np.ones(2),
        "directions": np.eye(2),
    }
    floor, samples, sampled = patient_descriptor_floor(
        curves, ranks, labels, blocks, reference,
        n_per_mode=6, repeats=10, seed=7)
    assert sampled.shape == (10, 2, 6)
    assert all(len(set(sampled[draw, mode])) == 6
               for draw in range(10) for mode in range(2))
    for mode in ("A", "B"):
        for metric, summary in floor["modes"][mode].items():
            assert np.isfinite(samples[mode][metric]).all()
            assert summary["scale_iqr"] >= 1e-6
