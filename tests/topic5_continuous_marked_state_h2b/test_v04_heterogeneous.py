from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.v03_hazard import HazardDesign
from src.topic5_continuous_marked_state_h2b.v04_assay import (
    apply_synthetic_postictal_exclusion,
    inject_slow_state,
    sample_synthetic_onsets,
)
from src.topic5_continuous_marked_state_h2b.v04_heterogeneous import (
    circular_shift_state_within_segment,
    fit_route_feature_map,
    prequential_heterogeneous_hazard,
)


def _design() -> HazardDesign:
    n = 180
    time = np.arange(n, dtype=np.float64) * 300.0
    onset_rows = np.asarray([30, 55, 80, 105, 130, 155, 175])
    phase = np.linspace(0.0, 8.0 * np.pi, n)
    state = np.column_stack([np.sin(phase), np.cos(phase)])
    observation = np.column_stack([np.sin(phase * 0.2), np.cos(phase * 0.2)])
    history = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    return HazardDesign(
        source_index=np.arange(n), time_epoch=time,
        segment=np.zeros(n, dtype=np.int64), history=history,
        current_observation=observation, persistent_state=state,
        memoryless_state=observation,
        onset_time=time[onset_rows], onset_segment=np.zeros(len(onset_rows), dtype=np.int64),
    )


def test_route_map_uses_two_routes_only_with_two_events_per_route() -> None:
    values = np.column_stack([np.r_[np.full(4, -3.0), np.full(4, 3.0)], np.zeros(8)])
    fitted = fit_route_feature_map(
        values, train_rows=np.arange(8),
        prior_event_anchor_rows=np.asarray([0, 1, 6, 7]),
    )
    assert fitted.n_routes == 2
    assert sorted(fitted.route_sizes.tolist()) == [2, 2]
    assert fitted.transform(values).shape == (8, 3)


def test_route_map_falls_back_to_one_route_with_three_prior_seizures() -> None:
    values = np.column_stack([np.linspace(-2, 2, 8), np.zeros(8)])
    fitted = fit_route_feature_map(
        values, train_rows=np.arange(8), prior_event_anchor_rows=np.asarray([0, 4, 7]),
    )
    assert fitted.n_routes == 1
    transformed = fitted.transform(values)
    assert np.all(transformed[:, 1:] == 0.0)


def test_route_map_rejects_tight_artificial_two_means_split() -> None:
    values = np.column_stack([np.linspace(-4.0, 4.0, 101), np.zeros(101)])
    fitted = fit_route_feature_map(
        values, train_rows=np.arange(101),
        prior_event_anchor_rows=np.asarray([75, 76, 77, 78, 79, 80]),
    )
    assert fitted.n_routes == 1


def test_one_route_comparator_remains_one_with_many_seizures() -> None:
    values = np.column_stack([np.r_[np.full(4, -3.0), np.full(4, 3.0)], np.zeros(8)])
    fitted = fit_route_feature_map(
        values, train_rows=np.arange(8),
        prior_event_anchor_rows=np.asarray([0, 1, 6, 7]), maximum_routes=1,
    )
    assert fitted.n_routes == 1
    axis = fitted.transform_single_axis(values)[:, 0]
    assert axis[0] * axis[-1] < 0.0


def test_prior_seizure_anchor_need_not_be_probe_eligible_row() -> None:
    values = np.column_stack([np.linspace(-2, 2, 8), np.zeros(8)])
    fitted = fit_route_feature_map(
        values, train_rows=np.asarray([0, 2, 4, 6]),
        prior_event_anchor_rows=np.asarray([1, 3, 5]),
    )
    assert fitted.n_routes == 1


def test_prequential_routes_are_fit_without_heldout_seizure() -> None:
    result = prequential_heterogeneous_hazard(_design(), initial_k=2, horizon_minutes=30)
    assert result["status"] == "COMPLETE_DEVELOPMENT"
    assert result["n_oof_seizures"] >= 3
    assert result["primary_metric"] == "conditional_risk_set_log_loss"
    assert result["controls_per_risk_set"] == 5
    assert all(row["heldout_seizure_did_not_define_route"] for row in result["folds"])
    assert all(row["n_prior_seizures"] == row["heldout_seizure_rank"] - 1
               for row in result["folds"])
    assert all(row["n_test_controls"] == 5 for row in result["folds"])
    assert all(row["n_test_rows"] == 6 for row in result["folds"])
    assert all(row["n_train_risk_sets"] >= 2 for row in result["folds"])
    assert all(row["train_test_rows_disjoint"] for row in result["folds"])
    assert all(row["identical_risk_set_rows_across_arms"] for row in result["folds"])
    assert "two_route_minus_single_axis_state" in result["equal_seizure_weight_effects"]
    assert "observation_minus_history" in result["equal_seizure_weight_effects"]
    assert "route_state_minus_history" in result["equal_seizure_weight_effects"]


def test_mutating_state_after_heldout_onset_cannot_change_first_fold() -> None:
    design = _design()
    original = prequential_heterogeneous_hazard(design, initial_k=2, horizon_minutes=30)
    first = original["folds"][0]
    state = np.array(design.persistent_state, copy=True)
    state[design.time_epoch > first["heldout_onset_epoch"]] += 1000.0
    changed = prequential_heterogeneous_hazard(
        HazardDesign(**{**design.__dict__, "persistent_state": state}),
        initial_k=2, horizon_minutes=30,
    )
    for key, value in first.items():
        if key.startswith("logloss_") or key in {
            "persistent_route_sizes", "persistent_route_separation_bandwidth",
        }:
            assert np.allclose(value, changed["folds"][0][key])


def test_control_rows_do_not_depend_on_state_values() -> None:
    design = _design()
    original = prequential_heterogeneous_hazard(design, initial_k=2, horizon_minutes=30)
    altered = HazardDesign(**{
        **design.__dict__,
        "persistent_state": np.flip(design.persistent_state, axis=0),
        "memoryless_state": -3.0 * design.memoryless_state,
    })
    changed = prequential_heterogeneous_hazard(altered, initial_k=2, horizon_minutes=30)
    assert [row["test_control_source_indices"] for row in original["folds"]] == [
        row["test_control_source_indices"] for row in changed["folds"]
    ]


def test_invalid_wrong_time_donor_does_not_become_memoryless_comparator() -> None:
    design = _design()
    result = prequential_heterogeneous_hazard(
        design, initial_k=2, horizon_minutes=30,
        wrong_time_state=design.memoryless_state,
        wrong_time_valid=np.zeros(len(design.time_epoch), dtype=bool),
    )
    assert result["n_wrong_time_estimable_folds"] == 0
    assert result["equal_seizure_weight_effects"]["correct_minus_wrong_time"] is None
    assert all("logloss_route_state_wrong_time" not in row for row in result["folds"])


def test_support_rich_default_uses_first_sixty_percent_for_training() -> None:
    design = _design()
    onset_rows = np.asarray([15, 30, 45, 60, 75, 90, 105, 120, 135, 150])
    design = HazardDesign(**{
        **design.__dict__,
        "onset_time": design.time_epoch[onset_rows],
        "onset_segment": np.zeros(len(onset_rows), dtype=np.int64),
    })
    result = prequential_heterogeneous_hazard(design, horizon_minutes=30)
    assert result["initial_k"] == 6
    assert result["initial_training_rule"] == "primary_chronological_60_percent_train"


def test_circular_shift_never_crosses_segment() -> None:
    design = _design()
    segment = np.repeat([0, 1], 90)
    design = HazardDesign(**{**design.__dict__, "segment": segment})
    values = np.column_stack([segment, np.arange(len(segment))])
    shifted = circular_shift_state_within_segment(design, values, 0.3)
    assert np.array_equal(shifted[:, 0], segment)


def test_synthetic_onsets_are_balanced_and_postictal_gaps_are_split() -> None:
    design = _design()
    injected, slow = inject_slow_state(design)
    design = HazardDesign(**{**design.__dict__, "persistent_state": injected})
    onset, group, anchor = sample_synthetic_onsets(
        design, np.abs(slow[:, 0]), rng=np.random.default_rng(4),
        n_seizures=4, minimum_separation_minutes=60,
        balance=slow[:, 0] >= 0,
    )
    assert len(onset) == 4
    assert np.sum(slow[anchor, 0] >= 0) == 2
    excluded, take = apply_synthetic_postictal_exclusion(
        design, onset, group, postictal_minutes=30,
    )
    assert len(take) < len(design.time_epoch)
    for event, segment in zip(excluded.onset_time, excluded.onset_segment):
        before = np.flatnonzero(
            (excluded.segment == segment) & (excluded.time_epoch < event)
        )
        assert len(before)
        assert event - excluded.time_epoch[before[-1]] <= 300.0 + 1e-9
