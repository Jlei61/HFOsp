import numpy as np

from scripts.analyze_topic4_dual_core_ood_results import _fixed_indices


def test_secondary_fixed_count_is_deterministic_and_not_a_gate():
    indices = np.arange(20)
    selected = _fixed_indices(indices, 10)
    assert len(selected) == 10
    assert selected[0] == 0 and selected[-1] == 19
    assert _fixed_indices(np.arange(9), 10) is None
