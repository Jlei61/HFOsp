from __future__ import annotations

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_observer import (
    LADDER_HISTORY,
    balanced_row_weights,
    select_validation_candidate_per_ladder,
    sequence_metadata,
    unresolved_history_result,
)
from src.topic5_event_innovation_data import ContinuitySequence


def _sequence(name: str, start: int, stop: int) -> ContinuitySequence:
    indices = np.arange(start, stop)
    return ContinuitySequence(
        continuity_unit_id=name,
        event_indices=indices,
        event_times=indices.astype(float),
        source_ids=np.repeat(name, len(indices)),
    )


def test_sequence_metadata_resets_progress_and_recent_rate():
    group, position, nuisance = sequence_metadata(
        [_sequence("a", 0, 30), _sequence("b", 30, 55)], 55
    )
    assert position[0] == position[30] == 0
    assert nuisance[0, 0] == nuisance[30, 0] == 0
    assert nuisance[19, 2] == nuisance[49, 2] == 0
    assert nuisance[20, 2] > 0 and nuisance[50, 2] > 0
    assert group[0] != group[30]


def test_balanced_row_weights_give_each_sequence_equal_mass():
    groups = np.array([0, 0, 0, 1, 1])
    indices = np.arange(5)
    weight = balanced_row_weights(indices, groups)
    np.testing.assert_allclose(weight[:3].sum(), 1.0)
    np.testing.assert_allclose(weight[3:].sum(), 1.0)


def test_observer_ladder_selects_within_each_rung_without_skipping_order():
    candidates = [
        {"ladder": "pre20", "dimension": 2, "alpha": 1.0, "validation_rank_mse": 0.2},
        {"ladder": "pre20", "dimension": 1, "alpha": 1.0, "validation_rank_mse": 0.1},
        {"ladder": "four_lag_bins", "dimension": 1, "alpha": 10.0, "validation_rank_mse": 0.05},
    ]
    selected = select_validation_candidate_per_ladder(
        candidates, ["pre20", "four_lag_bins"]
    )
    assert [row["ladder"] for row in selected] == ["pre20", "four_lag_bins"]
    assert selected[0]["dimension"] == 1


def test_full_innovation_validity_ladder_contains_time_and_long_history():
    assert LADDER_HISTORY["four_lag_bins_plus_time"] == 80
    assert list(LADDER_HISTORY)[-1] == "four_lag_bins_plus_time"


def test_insufficient_history_is_recorded_not_promoted_or_failed(tmp_path):
    phase0 = tmp_path / "phase0"
    (phase0 / "per_subject").mkdir(parents=True)
    (phase0 / "per_subject" / "p.json").write_text('{"status":"PHASE0_PATIENT_PASS"}')
    result, row = unresolved_history_result(
        "p", {"contract": "test"}, phase0, tmp_path / "out", "too few rows"
    )
    assert result["status"] == "UNRESOLVED_INSUFFICIENT_HISTORY"
    assert result["crossfit_artifact"] is None
    assert row["n_validation_rows"] == 0
