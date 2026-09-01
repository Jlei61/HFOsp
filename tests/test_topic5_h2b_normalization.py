"""B0.4 contract tests: TRAIN-only field normalization and route templates.

H2b spec §4: "route/pattern 只能由 TRAIN seizures 或已有临床标签定义;
held-out seizure 不参与 clustering、模板、阈值或归一化."

The point of these tests is that the split is *structurally* enforced -- a
held-out field cannot reach the fit even by accident -- rather than merely
documented.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.topic5_h2b_transfer.normalization import (
    FieldNormalization,
    assign_route,
    fit_field_normalization,
    fit_route_templates,
)


def _fields(n, n_ch=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, n_ch))


# --- TRAIN-only fit -------------------------------------------------------------


def test_normalization_uses_only_the_rows_named_as_train():
    fields = np.array([[0.0, 0.0], [2.0, 2.0], [100.0, 100.0]])
    norm = fit_field_normalization(fields, train_index=[0, 1])
    assert norm.n_train == 2
    assert np.allclose(norm.mean, [1.0, 1.0])


def test_fitting_with_an_out_of_range_index_raises_rather_than_clipping():
    with pytest.raises(ValueError, match="train_index"):
        fit_field_normalization(_fields(3), train_index=[0, 5])


def test_fitting_with_no_train_rows_raises():
    with pytest.raises(ValueError, match="at least one"):
        fit_field_normalization(_fields(3), train_index=[])


def test_a_constant_contact_gets_unit_scale_not_a_division_by_zero():
    fields = np.array([[3.0, 1.0], [3.0, 5.0]])
    norm = fit_field_normalization(fields, train_index=[0, 1])
    assert norm.scale[0] == 1.0
    out = norm.apply(np.array([3.0, 5.0]))
    assert np.isfinite(out).all()
    assert out[0] == 0.0


def test_apply_is_pure_and_does_not_refit_on_the_held_out_field():
    fields = np.array([[0.0, 0.0], [2.0, 2.0]])
    norm = fit_field_normalization(fields, train_index=[0, 1])
    before = (norm.mean.copy(), norm.scale.copy(), norm.n_train)
    norm.apply(np.array([1000.0, -1000.0]))
    assert np.array_equal(norm.mean, before[0])
    assert np.array_equal(norm.scale, before[1])
    assert norm.n_train == before[2]


# --- route templates ------------------------------------------------------------


def test_routes_are_built_from_train_rows_only_and_report_their_support():
    fields = np.vstack([
        np.array([5.0, 0.0, 0.0]), np.array([4.5, 0.2, 0.1]),   # route A
        np.array([0.0, 0.0, 5.0]), np.array([0.1, 0.1, 4.7]),   # route B
        np.array([9.9, 9.9, 9.9]),                              # held out
    ])
    routes = fit_route_templates(fields, train_index=[0, 1, 2, 3], max_routes=2)
    assert routes.n_train == 4
    assert routes.templates.shape[0] == 2
    assert sorted(routes.support) == [2, 2]


def test_a_route_below_minimum_support_is_kept_separate_not_merged():
    """'支持不足的 route 不强行合并' -- an under-supported route stays visible."""
    fields = np.vstack([
        np.array([5.0, 0.0]), np.array([4.8, 0.1]), np.array([5.1, 0.2]),
        np.array([0.0, 5.0]),  # lone route
    ])
    routes = fit_route_templates(fields, train_index=[0, 1, 2, 3], max_routes=2)
    assert sorted(routes.support) == [1, 3]
    assert routes.under_supported == (True,) or True in routes.under_supported


def test_held_out_field_is_assigned_to_a_frozen_template_not_used_to_move_it():
    fields = np.vstack([np.array([5.0, 0.0]), np.array([4.8, 0.1]),
                        np.array([0.0, 5.0]), np.array([0.1, 4.9])])
    routes = fit_route_templates(fields, train_index=[0, 1, 2, 3], max_routes=2)
    frozen = routes.templates.copy()
    label, sim = assign_route(np.array([4.9, 0.05]), routes)
    assert np.array_equal(routes.templates, frozen)
    assert sim > 0.9
    assert label in (0, 1)


def test_route_fitting_refuses_more_routes_than_train_seizures():
    with pytest.raises(ValueError, match="max_routes"):
        fit_route_templates(_fields(3), train_index=[0, 1], max_routes=3)
