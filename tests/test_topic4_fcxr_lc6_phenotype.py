import importlib.util
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc6_phenotype import (
    baseline_tradeoff,
    classify_high_state,
    normalized_theil_sen,
    spatial_slow_flow_readout,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc6a_aggregate", ROOT / "scripts/aggregate_topic4_fcxr_lc6a_phenotypes.py"
)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)


def test_boundedness_uses_complete_one_second_rate_and_independent_drift_fields():
    result = classify_high_state(
        global_onset_ms=1000., local_onset_ms=None, offset_ms=None, total_ms=9000.,
        global_rate_100ms=np.r_[np.zeros(10), np.full(80, 80.)],
        d_trace=np.full(800, .2), h_trace=np.full(800, 1.0), trace_dt_ms=10.,
        max_near_refractory_fraction=.01,
    )
    assert result["headline"] == "BOUNDED_CARRIER_CANDIDATE"
    assert result["bounded_candidate"] is True
    saturated = classify_high_state(
        global_onset_ms=1000., local_onset_ms=None, offset_ms=None, total_ms=9000.,
        global_rate_100ms=np.r_[np.zeros(10), np.full(80, 300.)],
        d_trace=np.full(800, .2), h_trace=np.full(800, 1.0), trace_dt_ms=10.,
        max_near_refractory_fraction=.01,
    )
    assert saturated["headline"] == "SATURATED_HIGH_STATE"


def test_theil_sen_and_baseline_tradeoff_are_continuous_readouts():
    drift = normalized_theil_sen(np.linspace(1, 2, 21), dt_s=.1, tail_s=2.)
    assert drift["slope_per_s"] > 0
    reference = {
        "event_rate_hz": 2., "iei_median_ms": 500.,
        "duration_median_ms": 10., "participation_median": .1,
    }
    close = baseline_tradeoff(dict(reference), reference)
    assert close["tradeoff"] is False
    changed = dict(reference, event_rate_hz=3.)
    assert baseline_tradeoff(changed, reference)["tradeoff"] is True


def test_spatial_readout_reports_d_halo_lead_and_recruitment_speed():
    positions = np.array([[.25, .25], [.75, .25], [1.25, .25], [1.75, .25]])
    bins = np.array([0, 1, 2, 3])
    occupancy = np.ones(4)
    rates = np.array([
        [20., 0., 0., 0.],
        [20., 20., 0., 0.],
        [20., 20., 20., 0.],
    ])
    d_maps = np.array([
        [0., 0., 0., 0.],
        [0., .1, .2, 0.],
        [0., .1, .2, .3],
    ])
    result = spatial_slow_flow_readout(
        rates, d_maps, positions, bins, occupancy, axis_unit=[1., 0.],
        source_xy=[.25, .25], sheet_size_mm=2., local_rate_threshold_hz=10.,
        onset_ms=0.,
    )
    assert result["max_D_halo_lead_mm"] > 0
    assert result["recruitment_front_speed_mm_per_s"] > 0


def test_fork_selection_prefers_margin_then_distinct_phenotype():
    rows = [
        {"condition": "Q1", "effective_onset_ms": 1000., "headline": "A",
         "spatial_phenotype": "STATIONARY", "boundedness": {"boundedness_margin": .1},
         "pinned_checkpoints": {"onset_plus_2s": {}}},
        {"condition": "Q2", "effective_onset_ms": 1000., "headline": "A",
         "spatial_phenotype": "STATIONARY", "boundedness": {"boundedness_margin": .2},
         "pinned_checkpoints": {"onset_plus_2s": {}}},
        {"condition": "Q3", "effective_onset_ms": 1000., "headline": "B",
         "spatial_phenotype": "DYNAMIC", "boundedness": {"boundedness_margin": -.1},
         "pinned_checkpoints": {"onset_plus_2s": {}}},
    ]
    selected = AGG.select_fork_candidates(rows)
    assert [row["condition"] for row in selected] == ["Q2", "Q3"]
