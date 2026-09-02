"""Estimability by endpoint × horizon from real coverage segments (v0.3.3 plan Task 7, G1-G4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.topic5_group_event_state.v02.timeline import RecordedSession, build_anchor_grid, build_carry_segments
from src.topic5_group_event_state.v032_eval.partition import eval_partition
from src.topic5_group_event_state.v033_evaluator import eligibility as G


def _toy():
    sessions = [RecordedSession(0, 0.0, 60_000.0), RecordedSession(1, 80_000.0, 130_000.0)]
    seizures = [{"onset_epoch": 30_000.0, "offset_epoch": 30_090.0}, {"onset_epoch": 125_000.0, "offset_epoch": 125_060.0}]
    segments = build_carry_segments(sessions, seizures, postictal_exclusion_seconds=3600.0, min_segment_seconds=300.0)
    partition = eval_partition(segments)
    rng = np.random.default_rng(0)
    events = np.sort(np.concatenate([rng.uniform(s.start_epoch, s.stop_epoch, int(s.duration_seconds * 0.02)) for s in segments]))
    grid = build_anchor_grid(segments, partition, events, horizons_seconds=(300.0, 1800.0, 7200.0),
                             grid_seconds=300.0, min_warmup_seconds=300.0)
    group_count = rng.integers(1, 4, events.size)
    return sessions, segments, partition, grid, events, group_count, seizures


def test_support_blocks_come_from_coverage_segments_not_sessions():
    sessions, segments, partition, grid, events, group_count, seizures = _toy()
    sup = G.subject_support_from_arrays(segments=segments, partition=partition, grid=grid, event_times=events,
                                        group_count=group_count, seizures=seizures, horizons=(300.0, 1800.0, 7200.0))
    lo, hi = partition.bounds("dev_test")
    manual = sum(int(np.floor((min(s.stop_epoch, hi) - max(s.start_epoch, lo)) / 1800.0))
                 for s in segments if min(s.stop_epoch, hi) > max(s.start_epoch, lo))
    assert sup["blocks"]["1800"]["dev_test"] == manual
    assert sup["blocks"]["1800"]["development"] == sup["blocks"]["1800"]["dev_test"]
    assert sup["blocks"]["1800"]["development_evaluation"] == sup["blocks"]["1800"]["dev_test"]
    assert sup["blocks"]["1800"]["development_total"] == sup["blocks"]["1800"]["dev_val"] + sup["blocks"]["1800"]["dev_test"]
    assert sup["n_sessions"] == 2 and sup["blocks"]["1800"]["development"] != sup["n_sessions"]
    assert sup["grammar_positive_anchors"]["1800"]["development"] <= sup["anchors"]["1800"]["development"]
    assert sup["h2a_positive_k_events"]["dev_test"] == int((group_count[partition.labels_of(events) == 3] >= 2).sum())
    assert sup["seizures"]["by_phase"]["dev_test"] + sup["seizures"]["by_phase"]["base_fit"] + \
        sup["seizures"]["by_phase"]["inner_val"] + sup["seizures"]["by_phase"]["dev_val"] == 2


def test_eligibility_rows_read_required_blocks_from_the_power_curve_or_flag_pending():
    sessions, segments, partition, grid, events, group_count, seizures = _toy()
    sup = G.subject_support_from_arrays(segments=segments, partition=partition, grid=grid, event_times=events,
                                        group_count=group_count, seizures=seizures, horizons=(300.0, 1800.0, 7200.0))
    requirements = {("count_profile", 1800): {"required_blocks": 8, "source": "toy_power_curve", "tier": "medium"}}
    rows = G.eligibility_rows("toy", sup, requirements)
    by = {(r["endpoint"], r["horizon_seconds"]): r for r in rows}
    row = by[("count_profile", 1800)]
    assert row["required_blocks"] == 8
    assert row["available_development_evaluation_blocks"] == sup["blocks"]["1800"]["dev_test"]
    assert row["available_development_blocks"] == row["available_development_evaluation_blocks"]
    assert row["estimable"] == (row["available_development_evaluation_blocks"] >= 8)
    pending = by[("conditional_grammar", 1800)]
    assert pending["required_blocks"] is None and pending["estimable"] is None and pending["status"] == "power_curve_pending"
    assert by[("h2a_event_anchor", None)]["support_unit"] == "positive_k_events"
    assert by[("h2b_seizure_risk", None)]["support_unit"] == "seizures_in_development_evaluation"
    assert by[("h2a_event_anchor", None)]["available_development_positive_k_events"] == sup["h2a_positive_k_events"]["dev_test"]
    assert by[("h2b_seizure_risk", None)]["available_development_seizures"] == sup["seizures"]["by_phase"]["dev_test"]
    for r in rows:
        assert not any(k in r for k in ("gain", "nll", "p_value")), "eligibility must never carry a result"


def test_power_requirements_take_conservative_max_across_scaffolds_without_overwrite():
    payload = {
        "format": "curve", "source_commit": "abcdef123456",
        "curves": [
            {"view": "count_profile", "horizon_seconds": 1800, "subject": "low",
             "effect_tiers": {"medium": {"required_blocks_level0": 8,
                                           "required_blocks_by_level": {"0": 8},
                                           "oracle_gain_median": 0.05}}},
            {"view": "count_profile", "horizon_seconds": 1800, "subject": "high",
             "effect_tiers": {"medium": {"required_blocks_level0": 15,
                                           "required_blocks_by_level": {"0": 15},
                                           "oracle_gain_median": 0.052}}},
        ],
    }
    got = G.requirements_from_power_curves(payload)
    row = got[("count_profile", 1800)]
    assert row["required_blocks"] == 15
    assert [x["subject"] for x in row["calibration_scaffolds"]] == ["low", "high"]
    payload["curves"][1]["effect_tiers"]["medium"]["required_blocks_level0"] = None
    assert G.requirements_from_power_curves(payload)[("count_profile", 1800)]["required_blocks"] is None
