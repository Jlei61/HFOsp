import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_topic4_zm_lifecycle_sprint.py"
SPEC = importlib.util.spec_from_file_location("topic4_zm_lifecycle_sprint_analysis", SCRIPT)
A = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(A)


def test_event_free_baseline_rejects_initial_high_rate_event():
    core = np.r_[np.zeros(20), np.full(8, 80.0), np.zeros(40)]
    mask = A.event_free_baseline_bins(core)
    assert mask[:20].all()
    assert not mask[20:28].any()


def test_baseline_referenced_intensity_distinguishes_sustained_gain():
    rms = np.ones((40, 2))
    rms[20:] = 10.0
    baseline = np.zeros(40, bool)
    baseline[:20] = True
    got = A.baseline_referenced_intensity(rms, baseline, slice(20, 40))
    assert got["median_gain_db_across_contacts"] == pytest.approx(20.0)
    assert got["occupancy_above_6db"] == 1.0
    assert got["normalized_integrated_energy_per_s"] == 100.0


def test_contact_rms_uses_event_free_baseline_mean():
    raw = np.zeros((400, 1))
    raw[200:] = 4.0
    baseline = np.array([True, True, False, False])
    rms, status = A.contact_rms_from_baseline(raw, 1000.0, baseline, bin_ms=100.0)
    assert status == "insufficient_event_free_baseline"
    baseline[:] = True
    rms, status = A.contact_rms_from_baseline(raw, 1000.0, baseline, bin_ms=100.0)
    assert status == "ok" and rms.shape == (4, 1)


def _event_fixture():
    core = np.zeros(80)
    surround = np.ones(80)
    active = np.zeros(80)
    kymo = np.zeros((4, 80))
    rms = np.ones((80, 3))
    for lo in (8, 24, 40, 56):
        core[lo:lo + 3] = (40.0, 70.0, 45.0)
        active[lo:lo + 3] = 0.2
        kymo[:, lo:lo + 3] = np.asarray(
            [[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0], [0.0, 0.0, 0.5]]
        )
        rms[lo:lo + 3] = np.asarray(
            [[4.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 4.0]]
        )
    return core, surround, active, kymo, rms


def test_returning_event_features_preserve_spatial_and_contact_order():
    core, surround, active, kymo, rms = _event_fixture()
    windows = A.returning_event_windows(core, threshold_hz=30.0)
    feats = A.returning_event_features(
        core, surround, active, kymo, rms, windows, bin_ms=25.0
    )
    assert len(feats) == 4
    assert feats[0]["duration_ms"] == 75.0
    assert feats[0]["peak_core_hz"] == 70.0
    assert feats[0]["axial_direction"] == 1
    assert feats[0]["contact_order"] == [0, 1, 2]


def test_returning_event_match_separates_single_candidate_from_distribution_recovery():
    core, surround, active, kymo, rms = _event_fixture()
    windows = A.returning_event_windows(core, threshold_hz=30.0)
    ref = A.returning_event_features(core, surround, active, kymo, rms, windows)
    one = A.match_returning_events(ref, ref[:1])
    assert one["single_event_candidate"] is True
    assert one["distribution_recovered"] is False
    recovered = A.match_returning_events(ref, ref[:3])
    assert recovered["single_event_candidate"] is True
    assert recovered["distribution_recovered"] is True


def test_returning_event_match_rejects_wrong_long_high_rate_fragment():
    core, surround, active, kymo, rms = _event_fixture()
    windows = A.returning_event_windows(core, threshold_hz=30.0)
    ref = A.returning_event_features(core, surround, active, kymo, rms, windows)
    wrong = [dict(ref[0], duration_ms=500.0, peak_core_hz=250.0)]
    got = A.match_returning_events(ref, wrong)
    assert got["single_event_candidate"] is False
    assert got["distribution_recovered"] is False
