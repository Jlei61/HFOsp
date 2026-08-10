import numpy as np
import pytest

from scripts.aggregate_topic4_rev9l_l3b_repeated_oracle import _score
from src.topic4_repeated_network_oracle import summarize_network_oracles


def test_repeated_oracle_separates_per_network_and_shared_capacity():
    result = summarize_network_oracles({
        "a": {1: 1.0, 2: 4.0},
        "b": {1: 4.0, 2: 1.0},
        "c": {1: 2.0, 2: 2.0},
    })
    assert np.isclose(result["C_per_net"], 1.0)
    assert np.isclose(result["shared"]["C_shared"], 2.0)
    assert np.isclose(result["Delta_network"], 1.0)
    assert result["shared"]["selected_candidate_id"] == "c"


def test_repeated_oracle_rejects_misaligned_or_nonfinite_matrix():
    with pytest.raises(ValueError):
        summarize_network_oracles({"a": {1: 1.0}, "b": {2: 1.0}})
    with pytest.raises(ValueError):
        summarize_network_oracles({"a": {1: np.inf}})


def test_missing_repeated_mode_is_retained_as_finite_failure():
    result = _score(
        {"mode_descriptors": None}, {}, {}, failure_objective=100.0)
    assert result["objective"] == 100.0
    assert result["readout_failure"] is True
