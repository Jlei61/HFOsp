from __future__ import annotations

import numpy as np
import pytest

from src.topic4_cohort_formal_layout import build_subject_layout, subject_null_seed


def _build(subject_id="epilepsiae_9", names=("A1", "A2", "A3", "B1", "B2"),
           real=None, n_permutations=64):
    return build_subject_layout(
        subject_id, list(names), real_coords_sheet=real,
        n_permutations=n_permutations, base_seed=20260820,
        sheet_size_mm=20.0, margin_mm=2.0,
    )


def test_subject_null_seed_is_frozen_and_subject_specific():
    first = subject_null_seed("yuquan_songzishuo", base_seed=20260820)
    assert first == subject_null_seed("yuquan_songzishuo", base_seed=20260820)
    assert first != subject_null_seed("yuquan_zhangbichen", base_seed=20260820)
    assert first != subject_null_seed("yuquan_songzishuo", base_seed=20260821)


def test_layout_carries_canonical_geometry_shafts_ordinals_and_null():
    built = _build()
    arrays = built["arrays"]
    assert arrays["canonical_coords_sheet"].shape == (5, 2)
    assert arrays["canonical_shaft_ids"].tolist() == ["A", "A", "A", "B", "B"]
    np.testing.assert_array_equal(
        arrays["canonical_within_shaft_ordinals"], [1, 2, 3, 1, 2],
    )
    assert "real_coords_sheet" not in arrays
    assert built["record"]["real_geometry_layout"] is None
    null = built["record"]["within_shaft_null"]
    assert null["effective_null_size"] == 11
    assert null["exhaustive"] is True
    permutations = arrays["within_shaft_null_permutations"]
    assert permutations.shape == (11, 5)


def test_layout_keeps_real_geometry_as_a_second_montage_when_available():
    real = np.column_stack([np.linspace(2.0, 18.0, 5), np.full(5, 9.0)])
    built = _build(real=real)
    np.testing.assert_allclose(built["arrays"]["real_coords_sheet"], real)
    assert built["record"]["real_geometry_layout"]["x_span_mm"] == pytest.approx(16.0)
    assert built["record"]["real_geometry_layout"]["y_span_mm"] == pytest.approx(0.0)


def test_layout_rejects_real_geometry_that_does_not_span_the_contact_order():
    with pytest.raises(ValueError, match="does not span the contact order"):
        _build(real=np.zeros((4, 2)))


def test_layout_records_that_it_never_read_patient_values():
    record = _build()["record"]["canonical_layout"]
    assert record["uses_event_ranks"] is False
    assert record["uses_mode_labels"] is False
    assert record["anatomical_interpretation"] is False
