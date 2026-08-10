import numpy as np

from scripts.audit_topic4_rev9l_forced_source_formal import (
    _matrix_spread,
    _summary,
)


def test_forced_audit_summary_ignores_missing_values():
    result = _summary([1.0, np.nan, 3.0])
    assert result["median"] == 2.0
    assert result["n"] == 2


def test_forced_audit_matrix_spread_is_cellwise_across_arms():
    first = np.asarray([[0.2, -0.5], [-0.8, 0.9]])
    second = np.asarray([[0.3, -0.4], [-0.7, 0.95]])
    np.testing.assert_allclose(
        _matrix_spread([first, second]), [[0.1, 0.1], [0.1, 0.05]])
