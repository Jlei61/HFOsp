# tests/test_topic5_slow_state_config.py
from pathlib import Path

import yaml

CONFIG = Path("config/topic5_slow_state_v4_0.yaml")


def _load():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_session_join_is_three_hundred_seconds_on_metadata_gaps():
    assert _load()["session_join_seconds"] == 300.0


def test_primary_event_grid_excludes_five_thousand():
    config = _load()
    assert config["event_window_grid"] == [50, 100, 200, 500, 1000, 2000]
    assert config["diagnostic_only_windows"] == [5000]


def test_clock_grid_covers_five_minutes_to_six_hours():
    assert _load()["clock_window_grid"] == [300, 900, 1800, 3600, 7200, 14400, 21600]


def test_a_scale_needs_a_minimum_number_of_independent_windows():
    # 200 random splits are one window's uncertainty, not 200 replicates
    assert _load()["min_windows_per_scale"] >= 5


def test_support_floors_exist_for_contacts_and_for_pairs():
    config = _load()
    assert config["min_participation_count"] >= 5
    assert config["min_pair_coparticipation_count"] >= 5


def test_at_least_two_primary_families_must_resolve():
    assert _load()["min_resolved_families"] == 2


def test_admission_gate_is_forty_blocks():
    assert _load()["min_blocks_for_admission"] == 40


def test_default_workers_is_conservative_until_peak_rss_is_measured():
    assert 4 <= _load()["default_workers"] <= 8


def test_no_absolute_reliability_threshold_is_configured():
    assert "reliability_threshold" not in CONFIG.read_text(encoding="utf-8")


def test_forbidden_inputs_are_all_declared_true():
    forbidden = _load()["forbidden_inputs"]
    for key in ("old_heldout20", "ictal_or_snn", "soz_or_geometry", "ab_or_axis_labels"):
        assert forbidden[key] is True


def test_synthetic_calibration_grid_spans_states_dwell_noise_and_block_count():
    grid = _load()["synthetic_grid"]
    assert grid["n_states"] == [2, 3, 4, 5]
    assert grid["dwell_blocks"] == [2, 5, 10, 20, 50]
    assert len(grid["noise_levels"]) == 3
    assert grid["n_blocks"] == [40, 80, 160, 320]
