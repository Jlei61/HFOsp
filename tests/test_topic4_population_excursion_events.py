import numpy as np
import pytest

from scripts.audit_topic4_rev12_population_excursion import _seed_delay_map
from src.topic4_node_dualmode import (
    population_excursion_episodes,
    sheet_activity_movie,
)


def _fragment(start, stop, peak=0.05):
    return {
        "t_on": float(start), "t_off": float(stop),
        "dur_ms": float(stop - start + 1), "peak_ext": float(peak),
        "returned": True,
    }


def _episodes(trace, fragments, reset_ms=10.0):
    return population_excursion_episodes(
        fragments, trace, sample_dt_ms=1.0, event_on_threshold=0.02,
        low_threshold_fraction=0.2, reset_ms=reset_ms, pre_roll_ms=2.0,
    )


def test_fragments_without_fast_state_reset_form_one_episode():
    trace = np.zeros(80)
    trace[10:21] = 0.04
    trace[21:27] = 0.006
    trace[27:36] = 0.04
    episodes = _episodes(trace, [_fragment(10, 20), _fragment(27, 35)])
    assert len(episodes) == 1
    assert episodes[0]["detector_fragment_indices"] == [0, 1]
    assert episodes[0]["t_on"] == 8.0
    assert episodes[0]["returned"] is True


def test_long_clock_gap_does_not_split_without_population_reset():
    trace = np.full(250, 0.006)
    trace[10:21] = 0.04
    trace[220:231] = 0.04
    episodes = _episodes(
        trace, [_fragment(10, 20), _fragment(220, 230)], reset_ms=20.0,
    )
    assert len(episodes) == 1
    assert episodes[0]["detector_fragment_indices"] == [0, 1]


def test_full_low_state_dwell_separates_independent_episodes():
    trace = np.zeros(100)
    trace[10:21] = 0.04
    trace[45:56] = 0.04
    episodes = _episodes(trace, [_fragment(10, 20), _fragment(45, 55)])
    assert len(episodes) == 2
    assert all(row["returned"] for row in episodes)
    assert episodes[0]["reset_start_ms"] == 21.0


def test_contact_geometry_cannot_enter_population_boundaries():
    trace = np.zeros(80)
    trace[10:21] = 0.04
    trace[21:27] = 0.006
    trace[27:36] = 0.04
    first = _episodes(trace, [_fragment(10, 20), _fragment(27, 35)])
    second = _episodes(trace.copy(), [_fragment(10, 20), _fragment(27, 35)])
    assert first == second
    assert "contact" not in " ".join(first[0].keys()).lower()


def test_last_excursion_is_nonreturned_without_complete_reset():
    trace = np.zeros(50)
    trace[30:] = 0.03
    episodes = _episodes(trace, [_fragment(30, 49)])
    assert len(episodes) == 1
    assert episodes[0]["returned"] is False


def test_split_invariance_for_one_population_excursion():
    trace = np.zeros(100)
    trace[10:41] = 0.04
    unsplit = _episodes(trace, [_fragment(10, 40)])
    split = _episodes(trace, [_fragment(10, 20), _fragment(22, 40)])
    assert len(unsplit) == len(split) == 1
    for key in ("t_on", "t_off", "returned", "reset_start_ms"):
        assert unsplit[0][key] == split[0][key]


def test_sheet_activity_movie_preserves_full_run_and_spatial_bins():
    spikes = np.zeros((8, 3), bool)
    spikes[0, 0] = True
    spikes[3, 1] = True
    spikes[7, 2] = True
    positions = np.asarray([[0.2, 0.2], [1.2, 0.2], [1.2, 1.2]])
    movie = sheet_activity_movie(
        spikes, positions, dt_ms=0.5, frame_ms=1.0,
        bin_mm=1.0, sheet_mm=2.0,
    )
    assert movie["activity_counts"].shape == (4, 2, 2)
    assert movie["activity_counts"].sum(axis=(1, 2)).tolist() == [1, 1, 0, 1]


def test_seed_delay_map_requires_explicit_unique_physical_delays():
    assert _seed_delay_map(["2161=34.5", "2162=33.6"]) == {
        2161: 34.5, 2162: 33.6,
    }
    with pytest.raises(ValueError):
        _seed_delay_map(["2161=34.5", "2161=34.6"])
    with pytest.raises(ValueError):
        _seed_delay_map(["2161=-1"])
