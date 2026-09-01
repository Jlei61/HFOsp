import numpy as np

from src.topic4_zm_fig5 import (
    select_positive_identity_candidate,
    stratified_random_sites,
    sustained_fraction_around,
)


def test_stratified_random_sites_are_deterministic_and_cover_every_stratum():
    left = stratified_random_sites(
        n_side=4, extent_mm=(0.0, 20.0), margin_mm=1.2, seed=17)
    right = stratified_random_sites(
        n_side=4, extent_mm=(0.0, 20.0), margin_mm=1.2, seed=17)
    assert np.array_equal(left, right)
    assert left.shape == (16, 2)
    edges = np.linspace(1.2, 18.8, 5)
    for index, point in enumerate(left):
        row, column = divmod(index, 4)
        assert edges[column] <= point[0] <= edges[column + 1]
        assert edges[row] <= point[1] <= edges[row + 1]


def test_sustained_fraction_requires_joint_global_recruitment():
    time = np.arange(0.0, 100.0, 10.0)
    active = np.ones(10)
    spatial = np.ones(10)
    spatial[4] = 0.49
    assert sustained_fraction_around(
        time, active, spatial, center_ms=50.0, window_ms=100.0,
        threshold=0.5) == 0.9


def test_positive_identity_selection_does_not_use_mirror_or_absolute_score():
    selected = select_positive_identity_candidate([
        {"time_ms": 10.0, "ta_identity_r": 0.4, "tb_identity_r": -0.9},
        {"time_ms": 20.0, "ta_identity_r": 0.3, "tb_identity_r": 0.8},
    ])
    assert selected["time_ms"] == 20.0
    assert selected["winning_template"] == "TB"
    assert selected["selection_score"] == 0.8
