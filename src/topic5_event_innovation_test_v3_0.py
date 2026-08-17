"""Frozen final-fit helpers for the Topic 5 v3.0 human test."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from scripts.run_topic5_event_innovation_v3_0_observer import (
    LADDER_HISTORY,
    balanced_row_weights,
    history_fields,
    projected_ladder,
    sequence_metadata,
)
from src.topic5_event_innovation_data import ContinuitySequence
from src.topic5_event_innovation_observer_v3_0 import (
    fit_standardized_masked_observer,
)
from src.topic5_event_innovation_v3_0 import RankStateBasis


def fit_final_test_innovations(
    raw: Mapping[str, Any],
    sequences: Mapping[str, Sequence[ContinuitySequence]],
    basis: RankStateBasis,
    selected: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Refit the selected observer on train+validation, then predict test.

    Every feature remains past-only within its own continuity unit.  The test
    rank is read only after prediction and solely to form the held-out
    innovation residual.
    """

    ladder_name = str(selected["ladder"])
    history = int(LADDER_HISTORY[ladder_name])
    fitting_sequences = [*sequences["train"], *sequences["validation"]]
    test_sequences = list(sequences["test"])
    fitting_group, fitting_position, fitting_nuisance = sequence_metadata(
        fitting_sequences, len(raw["rank"])
    )
    _, test_position, test_nuisance = sequence_metadata(
        test_sequences, len(raw["rank"])
    )
    fitting_ladder = projected_ladder(
        basis,
        history_fields(raw["rank"], raw["participation"], fitting_sequences),
        fitting_nuisance,
    )
    test_ladder = projected_ladder(
        basis,
        history_fields(raw["rank"], raw["participation"], test_sequences),
        test_nuisance,
    )
    fitting_rows = np.concatenate([
        np.asarray(sequence.event_indices, dtype=np.int64)
        for sequence in fitting_sequences
    ])
    fitting_rows = fitting_rows[fitting_position[fitting_rows] >= history]
    test_rows = np.concatenate([
        np.asarray(sequence.event_indices, dtype=np.int64)
        for sequence in test_sequences
    ])
    test_rows = test_rows[test_position[test_rows] >= history]
    if not len(fitting_rows) or not len(test_rows):
        raise ValueError("final observer lacks fitting or test history support")
    observer = fit_standardized_masked_observer(
        fitting_ladder[ladder_name][fitting_rows],
        raw["rank"][fitting_rows],
        raw["participation"][fitting_rows],
        alpha=float(selected["alpha"]),
        feature_name=ladder_name,
        minimum_observations=int(config["observer_minimum_observations"]),
        sample_weight=balanced_row_weights(fitting_rows, fitting_group),
    )
    prediction = observer.predict(test_ladder[ladder_name][test_rows])
    valid = (
        raw["participation"][test_rows]
        & np.isfinite(raw["rank"][test_rows])
    )
    residual = np.where(valid, raw["rank"][test_rows] - prediction, 0.0)
    return {
        int(event): (residual[row], valid[row])
        for row, event in enumerate(test_rows)
    }


def _offset_groups(parts) -> list[np.ndarray]:
    output = []
    offset = 0
    for part in parts:
        group = np.asarray(part.group, dtype=np.int64)
        if len(group):
            mapped = group + offset
            offset = int(mapped.max()) + 1
        else:
            mapped = group
        output.append(mapped.astype(np.int32))
    return output


def combine_response_rows(parts):
    """Concatenate local-response rows without merging source group codes."""

    from scripts.run_topic5_event_innovation_v3_0_local_response import ResponseRows

    values = list(parts)
    if not values:
        raise ValueError("at least one response-row object is required")
    groups = _offset_groups(values)
    return ResponseRows(
        event_index=np.concatenate([row.event_index for row in values]),
        group=np.concatenate(groups),
        pre_state=np.vstack([row.pre_state for row in values]),
        future_state=np.vstack([row.future_state for row in values]),
        past_state=np.vstack([row.past_state for row in values]),
        innovation_state=np.vstack([row.innovation_state for row in values]),
        nuisance=np.vstack([row.nuisance for row in values]),
        observed_future_field=np.vstack([
            row.observed_future_field for row in values
        ]),
        future_support=np.vstack([row.future_support for row in values]),
        future_windows=[window for row in values for window in row.future_windows],
    )

def combine_cumulative_rows(parts):
    """Concatenate cumulative-response rows without merging source groups."""

    from scripts.run_topic5_event_innovation_v3_0_cumulative_response import CumulativeRows

    values = list(parts)
    if not values:
        raise ValueError("at least one cumulative-row object is required")
    groups = _offset_groups(values)
    return CumulativeRows(
        anchor_event=np.concatenate([row.anchor_event for row in values]),
        group=np.concatenate(groups),
        pre_state=np.vstack([row.pre_state for row in values]),
        future_state=np.vstack([row.future_state for row in values]),
        cumulative_innovation=np.vstack([
            row.cumulative_innovation for row in values
        ]),
        dose=np.concatenate([row.dose for row in values]),
        alignment=np.concatenate([row.alignment for row in values]),
        nuisance=np.vstack([row.nuisance for row in values]),
        observed_future_field=np.vstack([
            row.observed_future_field for row in values
        ]),
        future_support=np.vstack([row.future_support for row in values]),
        future_windows=[window for row in values for window in row.future_windows],
    )
