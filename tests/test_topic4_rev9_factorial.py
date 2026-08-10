import numpy as np
import pytest

from src.topic4_rev9_factorial import (
    ARM_ORDER,
    arm_contract,
    event_equal_density,
    factorial_effects,
    normalized_event_ranks,
    pairwise_precedence,
)


def test_arm_contract_is_exact_two_factor_design():
    assert [(arm_contract(arm)["node"], arm_contract(arm)["edge"])
            for arm in ARM_ORDER] == [
        (False, False), (True, False), (False, True), (True, True)]
    with pytest.raises(ValueError):
        arm_contract("Core")


def test_event_equal_density_weights_events_not_spike_count():
    histograms = np.asarray([
        [[9.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
    ])
    density, count = event_equal_density(histograms)
    assert count == 2
    np.testing.assert_allclose(density, [[0.5, 0.0], [0.0, 0.5]])


def test_rank_and_precedence_ignore_missing_contacts():
    ranks = np.asarray([[0.0, 2.0, np.nan], [2.0, 0.0, 1.0]])
    normalized = normalized_event_ranks(ranks)
    np.testing.assert_allclose(normalized[0, :2], [0.0, 1.0])
    probability, support = pairwise_precedence(ranks)
    assert probability[0, 1] == pytest.approx(0.5)
    assert support[0, 2] == 1


def test_factorial_effects_preserve_pairing_and_interaction():
    values = {
        "Null": np.asarray([1.0, 2.0, 3.0]),
        "Node": np.asarray([2.0, 3.0, 4.0]),
        "Edge": np.asarray([3.0, 4.0, 5.0]),
        "Node+Edge": np.asarray([5.0, 6.0, 7.0]),
    }
    result = factorial_effects(values, seed=3, repeats=100)
    assert result["delta_node"]["estimate"] == pytest.approx(1.0)
    assert result["delta_edge"]["estimate"] == pytest.approx(2.0)
    assert result["interaction"]["estimate"] == pytest.approx(1.0)
    assert result["interaction"]["n_paired"] == 3
