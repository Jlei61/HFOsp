"""Contracts for the field-derived, gain-normalized E-to-E core."""
import numpy as np
from scipy import sparse

from src.topic4_core_connectivity import (ee_field_partition,
                                          field_normalized_ee_core,
                                          field_normalized_ee_pair,
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


def test_pair_mapper_preserves_structure_and_invalidates_all_ampa_caches():
    net = _net()
    net["ampa_source_cache"] = ("also stale",)
    net["gaba_flat"] = ("still valid",)
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    out, diag = field_normalized_ee_pair(
        net, np.array([1.0, 0.6, 0.0]), 2.0,
        active_vth_shift=np.array([-0.5, -0.2, 0.0]))
    np.testing.assert_allclose(
        incoming_ee_weight(out["ampa_by_delay"], 3), before,
        rtol=0.0, atol=1e-14)
    assert diag["topology_unchanged"]
    assert diag["e_to_i_unchanged"]
    assert diag["gaba_unchanged"]
    assert diag["n_zero_incoming_e_targets"] == 0
    assert set(diag["invalidated_ampa_cache_keys"]) == {
        "ampa_flat", "ampa_source_cache"}
    assert "gaba_flat" in out


def test_pair_mapper_locks_target_row_and_source_column_direction():
    net = _net()
    h = np.array([1.0, 0.8, 0.0])
    out, _ = field_normalized_ee_pair(net, h, 2.0)
    old = sum(net["ampa_by_delay"])
    new = sum(out["ampa_by_delay"])
    # Target row 0 redistributes its fixed incoming mass toward source column 1.
    assert new[0, 1] / old[0, 1] > new[0, 2] / old[0, 2]
    # I-target rows are E->I and remain exact.
    np.testing.assert_array_equal(new[3, :].toarray(), old[3, :].toarray())


def test_pair_mapper_target_only_factor_cancels_but_pair_factor_does_not():
    net = _net()
    # Equal h_source values make alpha*h_target*h_source target-only.
    target_only, _ = field_normalized_ee_pair(net, np.ones(3), 3.0)
    for new, old in zip(target_only["ampa_by_delay"], net["ampa_by_delay"]):
        np.testing.assert_allclose(new.toarray(), old.toarray(), rtol=1e-14, atol=1e-14)

    pair, _ = field_normalized_ee_pair(net, np.array([1.0, 0.7, 0.0]), 3.0)
    assert any(not np.allclose(new.toarray(), old.toarray())
               for new, old in zip(pair["ampa_by_delay"], net["ampa_by_delay"]))


def test_pair_mapper_zero_is_exact_noop_and_zero_incoming_target_is_reported():
    net = _net()
    # Remove all E input to target row 2 without changing matrix dimensions.
    net["ampa_by_delay"] = [m.tolil() for m in net["ampa_by_delay"]]
    for matrix in net["ampa_by_delay"]:
        matrix[2, :] = 0.0
    net["ampa_by_delay"] = [m.tocsc() for m in net["ampa_by_delay"]]

    out, diag = field_normalized_ee_pair(net, np.array([1.0, 0.5, 0.0]), 0.0)
    for new, old in zip(out["ampa_by_delay"], net["ampa_by_delay"]):
        np.testing.assert_array_equal(new.data, old.data)
        np.testing.assert_array_equal(new.indices, old.indices)
        np.testing.assert_array_equal(new.indptr, old.indptr)
    assert diag["exact_noop"]
    assert diag["n_zero_incoming_e_targets"] == 1


def test_pair_mapper_beta_path_is_finite_and_budget_preserving():
    net = _net()
    pos = np.array([[0.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    out, diag = field_normalized_ee_pair(
        net, np.array([1.0, 0.8, 0.2]), 1.0,
        beta=0.5, pos_e=pos, l_ee=0.5)
    np.testing.assert_allclose(
        incoming_ee_weight(out["ampa_by_delay"], 3), before,
        rtol=0.0, atol=1e-14)
    assert np.isfinite(diag["edge_ratio"]["max"])
    assert diag["target_groups"]["h_top10_percent"]["n_targets"] >= 1
