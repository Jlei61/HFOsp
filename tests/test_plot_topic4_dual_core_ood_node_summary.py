import numpy as np

from scripts.plot_topic4_dual_core_ood_node_summary import _even_sample


def test_even_sample_retains_endpoints_without_random_selection():
    source = np.arange(100)
    sampled = _even_sample(source, 11)
    assert len(sampled) == 11
    assert sampled[0] == 0 and sampled[-1] == 99
    assert np.array_equal(_even_sample(np.arange(3), 11), np.arange(3))
