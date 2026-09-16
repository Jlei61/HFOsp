from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analyze_topic5_spf_multiround import _spearman
from scripts.build_topic5_spf_multiround_verdict import _comparison
from scripts.run_topic5_spf_nested_learning_curve import _nested_order


def test_nested_learning_curve_is_deterministic_and_prefix_nested():
    indices = np.arange(1000, 1200)
    full = _nested_order("example_subject", indices, 120)
    repeated = _nested_order("example_subject", indices, 120)
    assert np.array_equal(full, repeated)
    assert len(full) == 120
    assert len(np.unique(full)) == len(full)
    assert set(full).issubset(set(indices))
    assert np.array_equal(full[:24], full[:48][:24])


def test_nested_learning_curve_subject_order_is_target_blind_but_distinct():
    indices = np.arange(100)
    left = _nested_order("subject_left", indices, 100)
    right = _nested_order("subject_right", indices, 100)
    assert not np.array_equal(left, right)
    assert set(left) == set(right) == set(indices)


def test_spearman_uses_average_ranks_for_ties():
    x = np.asarray([0.0, 0.0, 1.0, 2.0])
    y = np.asarray([0.0, 0.0, 2.0, 1.0])
    # Average ranks are [0.5, 0.5, 2, 3] and [0.5, 0.5, 3, 2].
    expected = np.corrcoef(
        np.asarray([0.5, 0.5, 2.0, 3.0]),
        np.asarray([0.5, 0.5, 3.0, 2.0]),
    )[0, 1]
    assert np.isclose(_spearman(x, y), expected)


def test_comparison_respects_metric_direction():
    values = pd.DataFrame(
        {
            "subject": ["a", "a", "b", "b"],
            "model": ["left", "right", "left", "right"],
            "score": [0.9, 0.8, 0.3, 0.2],
        }
    )
    lower = _comparison(values, "left", "right", "score")
    higher = _comparison(
        values, "left", "right", "score", higher_is_better=True
    )
    assert lower["left_better"] == 0
    assert higher["left_better"] == 2
    assert lower["per_patient"] == higher["per_patient"]
