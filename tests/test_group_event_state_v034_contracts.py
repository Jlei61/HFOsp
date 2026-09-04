from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from src.topic5_group_event_state.v034_contracts.anchors import (
    AnchorRecord,
    build_fixed_time_anchors,
    independent_window_count,
    validate_anchor_records,
)
from src.topic5_group_event_state.v034_contracts.baseline import build_multiscale_history
from src.topic5_group_event_state.v034_contracts.levels import (
    fit_train_mean_adapter,
    rolling_prefix_level,
    selection_period_mean_input_oracle,
)
from src.topic5_group_event_state.v034_contracts.eligibility import endpoint_rows


@dataclass(frozen=True)
class Segment:
    segment_id: int
    session_id: int
    start_epoch: float
    stop_epoch: float


class Partition:
    boundary_epochs = np.array([1000.0, 2000.0, 3000.0])
    recorded_seconds = {"base_fit": 1000, "inner_val": 1000, "dev_val": 1000, "dev_test": 1000}

    def phase_of(self, epoch: float) -> str:
        return ("base_fit", "inner_val", "dev_val", "dev_test")[int(np.searchsorted(self.boundary_epochs, epoch, side="right"))]

    def bounds(self, phase: str) -> tuple[float, float]:
        i = ("base_fit", "inner_val", "dev_val", "dev_test").index(phase)
        edges = (-np.inf, 1000.0, 2000.0, 3000.0, np.inf)
        return edges[i], edges[i + 1]


def test_fixed_time_anchor_target_and_embargo_never_cross_gap_or_split() -> None:
    segments = [Segment(0, 0, 0.0, 1500.0), Segment(1, 1, 1700.0, 2600.0)]
    partition = Partition()
    rows = build_fixed_time_anchors(
        segments,
        partition,
        horizons_seconds=(100.0,),
        grid_seconds=100.0,
        warmup_seconds=100.0,
        embargo_seconds=300.0,
    )
    assert rows
    assert all(row.embargo_stop <= segments[row.segment_id].stop_epoch for row in rows)
    assert all(partition.phase_of(np.nextafter(row.embargo_stop, -np.inf)) == row.phase for row in rows)
    assert not any(row.epoch in {800.0, 900.0, 1400.0, 1800.0, 1900.0} for row in rows)


def test_validator_rejects_target_that_crosses_a_boundary() -> None:
    segments = [Segment(0, 0, 0.0, 1500.0)]
    bad = [AnchorRecord(950.0, 1050.0, 1050.0, 100.0, 0, 0, "base_fit")]
    with pytest.raises(ValueError, match="phase boundary"):
        validate_anchor_records(bad, segments, Partition())


def test_independent_windows_are_counted_per_real_piece() -> None:
    segments = [Segment(0, 0, 0.0, 900.0), Segment(1, 1, 1100.0, 1900.0)]
    # 900 s in base_fit and 800 s in inner_val; the 200 s gap never contributes.
    assert independent_window_count(segments, Partition(), phase="base_fit", horizon_seconds=300) == 3
    assert independent_window_count(segments, Partition(), phase="inner_val", horizon_seconds=300) == 2


def test_train_mean_adapter_never_uses_nontrain_rows() -> None:
    control = fit_train_mean_adapter(np.array([1.0, 3.0, 1000.0]), np.array([True, True, False]))
    np.testing.assert_allclose(control.values[:, 0], 2.0)
    assert control.provenance["fit_partition"] == "TRAIN_only"


def test_rolling_prefix_waits_until_future_block_is_observed() -> None:
    control = rolling_prefix_level(
        np.array([10.0, 30.0]),
        observation_available_at=np.array([10.0, 30.0]),
        query_times=np.array([5.0, 10.0, 20.0, 30.0]),
        observation_segment=np.array([0, 0]),
        query_segment=np.array([0, 0, 0, 0]),
        initial_level=np.array([2.0]),
    )
    np.testing.assert_allclose(control.values[:, 0], [2.0, 10.0, 10.0, 20.0])
    assert control.provenance["causal_at_evaluation"] is True


def test_rolling_prefix_resets_at_segment() -> None:
    control = rolling_prefix_level(
        np.array([10.0]),
        observation_available_at=np.array([10.0]),
        query_times=np.array([20.0, 20.0]),
        observation_segment=np.array([0]),
        query_segment=np.array([0, 1]),
        initial_level=np.array([2.0]),
    )
    np.testing.assert_allclose(control.values[:, 0], [10.0, 2.0])


def test_selection_period_mean_is_explicitly_noncausal_and_input_only() -> None:
    state = np.array([[1.0], [3.0], [99.0]])
    control = selection_period_mean_input_oracle(
        state, np.array([True, True, False]), source_semantics="input_state"
    )
    np.testing.assert_allclose(control.values[:, 0], 2.0)
    assert control.provenance["causal_at_evaluation"] is False
    assert control.provenance["uses_future_labels"] is False
    with pytest.raises(ValueError, match="input_state only"):
        selection_period_mean_input_oracle(
            state, np.array([True, True, False]), source_semantics="target"
        )


def test_multiscale_history_is_prefix_only_and_resets_across_segments() -> None:
    kwargs = dict(
        anchor_times=np.array([5.0, 15.0, 105.0]),
        anchor_segment=np.array([0, 0, 1]),
        event_times=np.array([0.0, 10.0, 100.0]),
        event_segment=np.array([0, 0, 1]),
        event_features={"extent": np.array([1.0, 3.0, 9.0])},
        segment_bounds={0: (0.0, 20.0), 1: (100.0, 120.0)},
        tau_seconds=(10.0,),
    )
    first = build_multiscale_history(**kwargs)
    changed = build_multiscale_history(
        **{**kwargs, "event_features": {"extent": np.array([1.0, 3000.0, 9.0])}}
    )
    # Changing the event at t=10 cannot alter the anchor at t=5.
    np.testing.assert_allclose(first.values[0], changed.values[0])
    extent_col = first.names.index("extent[0]_tau10")
    assert first.values[2, extent_col] == pytest.approx(9.0)
    assert first.provenance["contains_future_seizure_information"] is False


def test_multiscale_history_rejects_future_seizure_covariate() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        build_multiscale_history(
            anchor_times=np.array([5.0]),
            anchor_segment=np.array([0]),
            event_times=np.array([0.0]),
            event_segment=np.array([0]),
            event_features={"time_to_next_seizure": np.array([1.0])},
            tau_seconds=(10.0,),
        )


def test_endpoint_contract_keeps_long_horizons_exploratory_and_missing_field_explicit() -> None:
    blocks = {
        300: {"dev_test": 100},
        1800: {"dev_test": 25},
        7200: {"dev_test": 4},
        21600: {"dev_test": 1},
    }
    capabilities = {
        "conditional_spatial_grammar_participation": {"available": True},
        "multiband_waveform": {"available": True},
        "early_ictal_field": {"available": False, "reason": "registry missing"},
    }
    rows = endpoint_rows(
        subject="s",
        blocks_by_horizon=blocks,
        prior_support={"seizures": {"development_evaluation": 0}},
        capabilities=capabilities,
        count_requirement_30m=47,
        grammar_requirement_30m=20,
    )
    count_30 = next(r for r in rows if r["endpoint"] == "future_event_count" and r["horizon_seconds"] == 1800)
    grammar_30 = next(r for r in rows if r["endpoint"] == "conditional_spatial_grammar_participation" and r["horizon_seconds"] == 1800)
    count_120 = next(r for r in rows if r["endpoint"] == "future_event_count" and r["horizon_seconds"] == 7200)
    field = next(r for r in rows if r["endpoint"] == "h2b_early_ictal_field")
    assert count_30["status"] == "not_estimable"
    assert grammar_30["status"] == "estimable"
    assert count_120["exploratory"] is True and count_120["core"] is False
    assert field["status"] == "not_yet_measurable"
