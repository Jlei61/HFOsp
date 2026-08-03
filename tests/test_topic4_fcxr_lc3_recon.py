"""Pure contracts for LC3 no-kick reconnaissance."""
from src.topic4_fcxr_lc3_recon import (
    nearest_snapshot_labels,
    reconnaissance_verdict,
    select_landmark_times,
)


def test_no_onset_uses_only_final_landmark():
    assert select_landmark_times({"label": "INTERICTAL_BASELINE", "bout": None},
                                 win_ms=1000.0, total_ms=32000.0) == {
        "no_onset_final": 32000.0}


def test_recovered_bout_gets_pre_high_offset_and_recovery_landmarks():
    got = select_landmark_times({"label": "RECOVERED_INTERICTAL", "bout": [10, 12]},
                                win_ms=1000.0, total_ms=45000.0)
    assert got == {
        "pre_onset": 9000.0, "onset": 10000.0, "early_high": 10500.0,
        "late_high_pre_offset": 12750.0, "post_offset": 13500.0,
        "recovered": 21000.0,
    }


def test_nearest_snapshot_tie_breaks_to_earlier_time():
    got = nearest_snapshot_labels({"t0": 0.0, "t250": 250.0}, {"x": 125.0})
    assert got["x"]["snapshot_label"] == "t0"


def test_recon_verdict_never_calls_a_recovered_pattern_a_candidate():
    lc = {"label": "RECOVERED_INTERICTAL", "bout": [8, 10]}
    assert reconnaissance_verdict(
        lifecycle=lc, numerical_unsafe=False, refractory_ceiling_fraction=0.0,
        x_activates_after_onset=True) == "RECON_RECOVERED_PATTERN"
    assert reconnaissance_verdict(
        lifecycle={"label": "ICTAL_LIKE_BOUNDED", "bout": [8, 20]},
        numerical_unsafe=False, refractory_ceiling_fraction=0.0,
        x_activates_after_onset=True) == "RECON_HIGH_WITHOUT_OFFSET"
    assert reconnaissance_verdict(
        lifecycle=lc, numerical_unsafe=False, refractory_ceiling_fraction=0.05,
        x_activates_after_onset=True) == "RECON_SATURATED_TONIC_BAD_DATA"
