"""Stage 3 joint observable: retain event-profile geometry without labels."""
import hashlib
import json
import os

import numpy as np
import pytest

from src.topic4_core_field_profile import (
    fixed_count_indices,
    fit_rank_curve_reference,
    normalized_rank_curve,
    rank_curve_reference_summary,
    rank_curve_table,
    sliced_embedding_distance,
    sliced_rank_curve_distance,
    transform_rank_curves,
)


AX = {f"C{i}": float(x) for i, x in enumerate(np.linspace(-8.0, 8.0, 11))}


def _profile(kind, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(list(AX.values()))
    if kind == "forward":
        y = x
    elif kind == "reverse":
        y = -x
    elif kind == "middle":
        y = np.abs(x)
    else:
        raise ValueError(kind)
    y = y + noise * rng.normal(size=len(y))
    return {name: float(value) for name, value in zip(AX, y)}


def test_opposite_profiles_remain_opposite_after_normalization():
    forward = normalized_rank_curve(_profile("forward"), AX)
    reverse = normalized_rank_curve(_profile("reverse"), AX)
    assert np.corrcoef(forward, reverse)[0, 1] == pytest.approx(-1.0, abs=1e-12)


def test_middle_source_is_not_collapsed_into_a_direction_sign():
    forward = normalized_rank_curve(_profile("forward"), AX)
    middle = normalized_rank_curve(_profile("middle"), AX)
    assert abs(float(np.corrcoef(forward, middle)[0, 1])) < 0.2


def test_embedding_distance_is_symmetric_and_rejects_shape_drift():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(20, 3))
    b = rng.normal(size=(30, 3))
    directions = np.eye(3)
    assert sliced_embedding_distance(a, b, directions) == pytest.approx(
        sliced_embedding_distance(b, a, directions))
    with pytest.raises(ValueError, match="matching 2-D shapes"):
        sliced_embedding_distance(a, b[:, :2], directions)


def test_phantom_ranks_are_removed_by_the_same_participation_mask():
    real = {k: v for k, v in list(_profile("forward").items())[:8]}
    polluted = dict(real)
    polluted.update({k: 99.0 for k in list(AX)[8:]})
    a = normalized_rank_curve(real, AX)
    b = normalized_rank_curve(polluted, AX, participating=set(real))
    assert np.allclose(a, b)


def test_reference_fit_and_distance_are_deterministic_and_label_free():
    events = []
    for seed in range(80):
        events.append(_profile("forward", noise=0.5, seed=seed))
        events.append(_profile("reverse", noise=0.5, seed=1000 + seed))
    curves = rank_curve_table(events, AX)
    a = fit_rank_curve_reference(curves, n_components=4, n_reference=100,
                                 n_projections=12, seed=7)
    b = fit_rank_curve_reference(curves, n_components=4, n_reference=100,
                                 n_projections=12, seed=7)
    assert np.allclose(a["directions"], b["directions"])
    assert np.allclose(a["reference_z"], b["reference_z"])
    assert rank_curve_reference_summary(a)["uses_direction_labels"] is False


def test_joint_distance_rejects_a_single_middle_generator():
    train = []
    held_out = []
    for seed in range(100):
        target = train if seed < 80 else held_out
        target.extend((_profile("forward", 0.7, seed),
                       _profile("reverse", 0.7, 1000 + seed)))
    middle = [_profile("middle", 0.7, 2000 + seed) for seed in range(40)]
    train_curves = rank_curve_table(train, AX)
    held_curves = rank_curve_table(held_out, AX)
    middle_curves = rank_curve_table(middle, AX)
    ref = fit_rank_curve_reference(train_curves, n_components=4,
                                   n_reference=120, n_projections=16, seed=3)
    assert sliced_rank_curve_distance(held_curves, ref) < 0.35
    assert sliced_rank_curve_distance(middle_curves, ref) > 2.0 * \
        sliced_rank_curve_distance(held_curves, ref)


def test_transform_refuses_a_different_profile_grid():
    curves = rank_curve_table([
        _profile("forward", 0.2, 1), _profile("forward", 0.2, 2),
        _profile("reverse", 0.2, 3), _profile("reverse", 0.2, 4),
    ], AX)
    ref = fit_rank_curve_reference(curves, n_components=2, n_reference=2,
                                   n_projections=4, seed=0)
    with pytest.raises(ValueError, match="profile grid"):
        transform_rank_curves(np.ones((3, curves.shape[1] - 1)), ref)


def test_fixed_count_selector_is_deterministic_unique_and_covers_the_stream():
    index = fixed_count_indices(103, 20)
    np.testing.assert_array_equal(index, fixed_count_indices(103, 20))
    assert len(np.unique(index)) == 20
    assert index[0] < 5 and index[-1] > 97
    assert fixed_count_indices(19, 20) is None


_SUMMARY = ("results/topic4_sef_hfo/data_driven_core_field_stage3/"
            "joint_observable/calibration_summary.json")


@pytest.mark.integration
@pytest.mark.skipif(not os.path.exists(_SUMMARY), reason="calibration artifact not on disk")
def test_real_artifact_passes_without_hiding_the_stage3_failure():
    summary = json.load(open(_SUMMARY))
    assert all(summary["calibration_gates"].values())
    assert summary["reference"]["uses_direction_labels"] is False
    arms = summary["arms"]
    assert arms["stage3_flexible"]["fixed_count_distance"]["median"] > \
        max(arms["stage2_filament"]["fixed_count_distance"]["median"],
            arms["hand_placed_two_cores"]["fixed_count_distance"]["median"])
    assert arms["stage3_flexible"]["prototype_correlation"] > 0
    assert all(arms[key]["prototype_correlation"] < 0 for key in (
        "patient_heldout", "stage2_filament", "hand_placed_two_cores"))
    with open(summary["reference_npz"], "rb") as fh:
        assert hashlib.sha256(fh.read()).hexdigest() == summary["reference_sha256"]
