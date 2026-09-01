import numpy as np
import pytest
from scipy import sparse

from src.topic4_rev9_edge_structure import (
    field_background_membership,
    summarize_edge_redistribution,
)


def test_field_background_membership_preserves_support_and_tail_background():
    result = field_background_membership(
        h=[1.0, 0.25, 0.0],
        component_contributions=[[3.0, 1.0], [1.0, 1.0], [1e-12, 2e-12]])
    membership = result["membership"]
    np.testing.assert_allclose(membership.sum(axis=1), 1.0)
    np.testing.assert_allclose(membership[0], [0.75, 0.25, 0.0])
    np.testing.assert_allclose(membership[1], [0.125, 0.125, 0.75])
    np.testing.assert_allclose(membership[2], [0.0, 0.0, 1.0])


def test_edge_summary_uses_target_rows_source_columns_and_fixed_delays():
    membership = np.eye(2)
    old = [
        sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]),
        sparse.csr_matrix([[0.0, 0.0], [2.0, 0.0]]),
    ]
    new = [
        sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]),
        sparse.csr_matrix([[0.0, 0.0], [2.0, 0.0]]),
    ]
    result = summarize_edge_redistribution(
        old, new, membership, delay_dt_ms=0.5, h=[1.0, 0.0])
    assert result["old_flow"][0, 1] == pytest.approx(1.0)
    assert result["old_flow"][1, 0] == pytest.approx(2.0)
    assert result["old_pair_delay_ms"][0, 1] == pytest.approx(0.0)
    assert result["old_pair_delay_ms"][1, 0] == pytest.approx(0.5)
    np.testing.assert_allclose(result["flow_ratio"][[0, 1], [1, 0]], 1.0)
    assert result["incoming_max_abs_error"] == pytest.approx(0.0)


def test_edge_summary_detects_source_outgoing_redistribution_with_incoming_conservation():
    membership = np.eye(2)
    old = [sparse.csr_matrix([[1.0, 1.0], [1.0, 1.0]])]
    new = [sparse.csr_matrix([[1.5, 0.5], [1.5, 0.5]])]
    result = summarize_edge_redistribution(
        old, new, membership, delay_dt_ms=1.0, h=[1.0, 0.0])
    np.testing.assert_allclose(result["group_outgoing_ratio"], [1.5, 0.5])
    assert result["incoming_max_abs_error"] == pytest.approx(0.0)
    assert result["total_weight_relative_error"] == pytest.approx(0.0)
