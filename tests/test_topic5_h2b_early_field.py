"""B0.3 contract tests: early ictal field descriptors.

The band-power / robust-z path is already validated upstream
(``src/topic5_ictal_recruitment.py``) and the window slicer already exists
(``src/topic5_t0_features.window_activation``); only the descriptors the H2b
spec §1 asks for on top of the field are new, so only those are pinned here.
"""

from __future__ import annotations

import numpy as np

from src.topic5_h2b_transfer.early_field import (
    first_crossing_time,
    laterality_index,
    spatial_entropy,
    normalize_field,
)


# --- first recruited group / early propagation path ---------------------------


def test_first_crossing_returns_the_time_the_contact_first_exceeds_threshold():
    relt = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    z = np.array([[0.0, 0.0, 9.0, 9.0, 9.0],   # crosses at t=1
                  [0.0, 0.0, 0.0, 0.0, 9.0]])  # crosses at t=3
    out = first_crossing_time(z, relt, threshold=5.0, t0=0.0, t1=5.0)
    assert np.allclose(out, [1.0, 3.0])


def test_a_contact_that_never_crosses_is_nan_not_the_window_end():
    relt = np.array([0.0, 1.0, 2.0])
    z = np.array([[0.0, 1.0, 2.0]])
    out = first_crossing_time(z, relt, threshold=5.0, t0=0.0, t1=5.0)
    assert np.isnan(out[0])


def test_crossings_before_the_window_are_ignored():
    """Pre-onset activity must not be read as early ictal recruitment."""
    relt = np.array([-2.0, -1.0, 0.0, 1.0])
    z = np.array([[9.0, 9.0, 0.0, 0.0]])
    out = first_crossing_time(z, relt, threshold=5.0, t0=0.0, t1=5.0)
    assert np.isnan(out[0])


# --- normalization -------------------------------------------------------------


def test_normalize_field_puts_the_field_on_a_unit_simplex_over_positive_part():
    field = np.array([-3.0, 1.0, 3.0])
    out = normalize_field(field)
    assert np.isclose(out.sum(), 1.0)
    assert out[0] == 0.0
    assert np.isclose(out[2] / out[1], 3.0)


def test_normalize_field_of_an_all_negative_field_is_nan_not_uniform():
    """An all-suppressed field has no recruitment mass; uniform would be a lie."""
    out = normalize_field(np.array([-1.0, -2.0]))
    assert np.all(np.isnan(out))


# --- spatial extent -------------------------------------------------------------


def test_entropy_is_one_for_a_flat_field_and_zero_for_a_single_contact():
    assert np.isclose(spatial_entropy(np.array([1.0, 1.0, 1.0, 1.0])), 1.0)
    assert np.isclose(spatial_entropy(np.array([5.0, 0.0, 0.0, 0.0])), 0.0)


def test_entropy_ignores_contacts_without_coverage():
    a = spatial_entropy(np.array([1.0, 1.0, np.nan]))
    b = spatial_entropy(np.array([1.0, 1.0]))
    assert np.isclose(a, b)


# --- laterality -----------------------------------------------------------------


def test_laterality_is_plus_one_when_all_mass_is_left():
    field = np.array([4.0, 4.0, 0.0])
    hemi = np.array([-1, -1, +1])  # sign of x: negative = left
    assert np.isclose(laterality_index(field, hemi), 1.0)


def test_laterality_is_zero_for_a_symmetric_field():
    field = np.array([2.0, 2.0])
    hemi = np.array([-1, +1])
    assert np.isclose(laterality_index(field, hemi), 0.0)


def test_laterality_without_any_mapped_contact_is_nan():
    field = np.array([2.0, 2.0])
    hemi = np.array([0, 0])  # unmapped
    assert np.isnan(laterality_index(field, hemi))


# --- atomic write ---------------------------------------------------------------


def test_atomic_npz_write_lands_on_the_exact_path_and_leaves_no_temp(tmp_path):
    """np.savez appends '.npz' to its filename argument, which silently breaks
    a write-temp-then-rename scheme. Pin the behaviour so it cannot regress.
    """
    from src.topic5_h2b_transfer.early_field import save_npz_atomic

    target = tmp_path / "subject.npz"
    save_npz_atomic(target, {"a": np.arange(3)})

    assert target.exists()
    assert list(tmp_path.iterdir()) == [target]
    assert np.array_equal(np.load(target)["a"], np.arange(3))


def test_atomic_npz_write_replaces_an_existing_file(tmp_path):
    from src.topic5_h2b_transfer.early_field import save_npz_atomic

    target = tmp_path / "subject.npz"
    save_npz_atomic(target, {"a": np.zeros(2)})
    save_npz_atomic(target, {"a": np.ones(2)})
    assert np.array_equal(np.load(target)["a"], np.ones(2))
    assert list(tmp_path.iterdir()) == [target]
