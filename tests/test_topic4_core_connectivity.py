"""Contracts for the field-derived, gain-normalized E-to-E core."""
import numpy as np
from scipy import sparse

from src.topic4_core_connectivity import (ee_field_partition,
                                          field_normalized_ee_core,
                                          incoming_ee_weight)


def _net():
    # Rows are targets; columns are E sources. Last row is an I target.
    b1 = sparse.csc_matrix(np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0],
    ]))
    b2 = sparse.csc_matrix(np.array([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]))
    return dict(NE=3, NI=1, ampa_by_delay=[b1, b2],
                gaba_by_delay=[sparse.csc_matrix((4, 1))],
                ampa_flat=("stale",), marker="kept")


def test_field_core_preserves_each_e_targets_total_incoming_e_weight():
    net = _net()
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    out, diag = field_normalized_ee_core(net, np.array([1.0, 0.8, 0.0]), 2.0)
    after = incoming_ee_weight(out["ampa_by_delay"], 3)
    np.testing.assert_allclose(after, before, rtol=1e-14, atol=1e-14)
    assert diag["max_abs_incoming_E_error"] < 1e-14


def test_field_core_changes_relative_core_coupling_but_not_topology_or_e_to_i():
    net = _net()
    out, _ = field_normalized_ee_core(net, np.array([1.0, 1.0, 0.0]), 2.0)
    old = sum(net["ampa_by_delay"])
    new = sum(out["ampa_by_delay"])
    np.testing.assert_array_equal((new != 0).toarray(), (old != 0).toarray())
    np.testing.assert_array_equal(new[3, :].toarray(), old[3, :].toarray())
    # For target 0, core source 1 gains relative to non-core source 2.
    assert new[0, 1] / old[0, 1] > new[0, 2] / old[0, 2]
    old_part = ee_field_partition(net["ampa_by_delay"],
                                  np.array([1.0, 1.0, 0.0]), 0.5)
    new_part = ee_field_partition(out["ampa_by_delay"],
                                  np.array([1.0, 1.0, 0.0]), 0.5)
    assert new_part["weight"]["within_core"] > old_part["weight"]["within_core"]


def test_field_core_is_noop_at_zero_gain_and_drops_stale_flat_cache():
    net = _net()
    out, _ = field_normalized_ee_core(net, np.array([1.0, 0.5, 0.0]), 0.0)
    for new, old in zip(out["ampa_by_delay"], net["ampa_by_delay"]):
        np.testing.assert_array_equal(new.toarray(), old.toarray())
    assert "ampa_flat" not in out
    assert out["marker"] == "kept"
    assert "ampa_flat" in net


def test_field_core_rejects_invalid_fields_and_gain():
    net = _net()
    for field in (np.ones(2), np.array([1.0, np.nan, 0.0]),
                  np.array([1.0, 1.1, 0.0])):
        try:
            field_normalized_ee_core(net, field, 1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid field was accepted")
    try:
        field_normalized_ee_core(net, np.ones(3), -1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("negative gain was accepted")


def test_default_partition_uses_a_compact_top_five_percent_diagnostic():
    h = np.linspace(0.0, 1.0, 100)
    # This helper only needs valid E-source columns; duplicate the toy bins to
    # 100 targets/sources so the default subset size can be pinned directly.
    matrix = sparse.eye(100, format="csc")
    out = ee_field_partition([matrix], h)
    assert out["core_quantile"] == 0.95
    assert out["n_core"] == 5
