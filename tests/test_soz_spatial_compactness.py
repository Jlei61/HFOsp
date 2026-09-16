import numpy as np
import pytest

from src.soz_spatial_compactness import (
    all_contact_null,
    analyze_subject_compactness,
    median_pairwise_distance,
    rms_radius,
    shaft_stratified_null,
    spatial_metrics,
)


def test_rms_radius_and_pairwise_distance_known_line():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert rms_radius(coords) == pytest.approx(np.sqrt(8.0 / 3.0))
    assert median_pairwise_distance(coords) == pytest.approx(2.0)


def test_all_contact_null_detects_tight_subset_reproducibly():
    coords = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0],
                       [20.0, 0.0, 0.0], [30.0, 0.0, 0.0], [40.0, 0.0, 0.0]])
    observed = spatial_metrics(coords[:2])
    a = all_contact_null(
        coords, n_selected=2, observed_metrics=observed, n_null=999,
        rng=np.random.default_rng(7),
    )
    b = all_contact_null(
        coords, n_selected=2, observed_metrics=observed, n_null=999,
        rng=np.random.default_rng(7),
    )
    assert a == b
    assert a["rms_radius"]["observed_to_null_median_ratio"] < 0.02
    assert a["rms_radius"]["p_left"] < 0.1


def test_shaft_null_preserves_profile_and_can_be_informative():
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0], [1.0, 20.0, 0.0], [10.0, 20.0, 0.0],
    ])
    selected = np.array([True, True, False, True, True, False])
    observed = spatial_metrics(coords[selected])
    result = shaft_stratified_null(
        coords, ["A", "A", "A", "B", "B", "B"], selected,
        observed_metrics=observed, n_null=499, rng=np.random.default_rng(3),
    )
    assert result["available"] is True
    assert result["n_unique_profiles"] == 9
    assert result["rms_radius"]["observed_to_null_median_ratio"] < 1.0


def test_shaft_null_fails_closed_when_profile_fixes_exact_set():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    selected = np.array([True, True, False])
    observed = spatial_metrics(coords[selected])
    result = shaft_stratified_null(
        coords, ["A", "A", "B"], selected,
        observed_metrics=observed, n_null=100, rng=np.random.default_rng(2),
    )
    assert result["available"] is False
    assert result["reason"] == "shaft_profile_has_no_permutation_freedom"


def test_subject_analysis_reports_primary_and_sensitivity():
    names = ["A1", "A2", "A3", "B1", "B2", "B3"]
    coords = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0], [1.0, 20.0, 0.0], [10.0, 20.0, 0.0],
    ])
    result = analyze_subject_compactness(
        names, coords, ["A1", "A2", "B1", "B2"], ["A", "A", "A", "B", "B", "B"],
        n_null=199, rng=np.random.default_rng(11),
    )
    assert result["n_soz_contacts"] == 4
    assert result["n_nonsoz_contacts"] == 2
    assert result["all_contact_null"]["available"] is True
    assert result["shaft_stratified_null"]["available"] is True


def test_subject_analysis_rejects_single_contact_soz():
    with pytest.raises(ValueError, match=">=2"):
        analyze_subject_compactness(
            ["A1", "A2", "A3"], np.eye(3), ["A1"], ["A", "A", "A"],
            n_null=10, rng=np.random.default_rng(1),
        )
