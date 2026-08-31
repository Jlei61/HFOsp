import numpy as np

from scripts.plot_topic4_dual_core_ood_pathways import _paired_significant


def test_paired_star_requires_bootstrap_interval_to_exclude_zero():
    node = np.arange(12, dtype=float)
    assert _paired_significant(node - 2.0, node, seed=1)
    alternating = node + np.asarray([-1, 1] * 6, float)
    assert not _paired_significant(alternating, node, seed=1)
