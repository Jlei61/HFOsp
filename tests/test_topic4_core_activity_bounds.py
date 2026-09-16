import itertools
import numpy as np
import pytest
from scripts.audit_topic4_xy_core_activity_bounds import activity_bounds


def test_bounds_are_sharp_over_all_hidden_neuron_assignments():
    # Two cells: 3 and 4 neurons, with 1 and 2 designated core members.
    by_observation = {}
    for bits in itertools.product([0, 1], repeat=7):
        observed = (sum(bits[:3]), sum(bits[3:]))
        actual_core = bits[0] + bits[3] + bits[4]
        by_observation.setdefault(observed, []).append(actual_core)
    for observed, possibilities in by_observation.items():
        lo, hi = activity_bounds(np.array([observed]), np.array([3, 4]), np.array([1, 2]))
        assert lo[0] == min(possibilities)
        assert hi[0] == max(possibilities)


def test_impossible_activity_is_rejected():
    with pytest.raises(ValueError):
        activity_bounds(np.array([[4]]), np.array([3]), np.array([1]))
