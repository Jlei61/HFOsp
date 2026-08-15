from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.topic4_canonical_shaft_layout import (
    balanced_precedence_error,
    balanced_recruitment_error,
    canonical_shaft_layout,
    contact_shaft_contract,
)


def test_canonical_layout_preserves_shaft_and_ordinal_without_patient_values():
    names = ["B3", "A1", "A4", "B1", "A'2"]
    layout = canonical_shaft_layout(names)
    assert layout["shaft_ids"] == ["B", "A", "A", "B", "A'"]
    np.testing.assert_array_equal(layout["within_shaft_ordinals"], [3, 1, 4, 1, 2])
    assert layout["uses_event_ranks"] is False
    assert layout["uses_mode_labels"] is False
    assert layout["anatomical_interpretation"] is False
    assert np.all(layout["coords_sheet"] >= 0.0)
    assert np.all(layout["coords_sheet"] <= 20.0)
    a_indices = [1, 2]
    assert layout["coords_sheet"][a_indices[1], 0] > layout["coords_sheet"][a_indices[0], 0]


def test_canonical_layout_rejects_unparseable_or_duplicate_contacts():
    with pytest.raises(ValueError, match="unparseable"):
        contact_shaft_contract(["A1", "not-a-contact"])
    with pytest.raises(ValueError, match="unique"):
        contact_shaft_contract(["A1", "A1"])


def test_recruitment_loss_gives_each_shaft_equal_weight():
    shafts = ["A", "A", "A", "A", "B"]
    patient = np.zeros(5)
    model = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0])
    assert balanced_recruitment_error(model, patient, shafts) == 0.5


def test_precedence_loss_gives_each_shaft_pair_class_equal_weight():
    names = ["A1", "A2", "A3", "B1"]
    pairs = np.asarray(list(combinations(range(len(names)), 2)), int)
    patient = np.zeros((len(pairs), 3))
    model = np.zeros_like(patient)
    model[:3] = 1.0
    assert balanced_precedence_error(model, patient, names, pairs) == 0.5


def test_canonical_layout_is_invariant_to_the_order_contacts_arrive_in():
    names = ["B3", "A1", "A4", "B1", "A'2"]
    layout = canonical_shaft_layout(names)
    shuffled = [names[index] for index in (3, 0, 4, 2, 1)]
    other = canonical_shaft_layout(shuffled)
    first = dict(zip(layout["contact_names"], layout["coords_sheet"].tolist()))
    second = dict(zip(other["contact_names"], other["coords_sheet"].tolist()))
    assert first == second


def test_canonical_layout_fills_the_ordinal_axis_like_the_real_geometry_fit():
    for names in (["A1", "A2", "B1", "B2", "C1"], ["A3", "A4", "B7", "B9"]):
        coords = canonical_shaft_layout(names)["coords_sheet"]
        span = float(coords[:, 0].max() - coords[:, 0].min())
        assert span == pytest.approx(16.0)


def test_canonical_layout_keeps_dense_montages_inside_the_sheet_and_distinct():
    names = [
        f"{shaft}{index}"
        for shaft in ("A", "B", "C", "D", "E", "F", "G", "H")
        for index in range(1, 8)
    ]
    layout = canonical_shaft_layout(names)
    coords = layout["coords_sheet"]
    assert coords.shape == (len(names), 2)
    assert np.all(coords >= 0.0) and np.all(coords <= 20.0)
    assert len({tuple(row) for row in coords.tolist()}) == len(names)


def test_single_shaft_layout_remains_a_valid_one_dimensional_readout():
    layout = canonical_shaft_layout(["A1", "A3", "A5"])
    assert layout["n_shafts"] == 1
    np.testing.assert_allclose(layout["coords_sheet"][:, 1], 10.0)
    assert np.all(np.diff(layout["coords_sheet"][:, 0]) > 0.0)
