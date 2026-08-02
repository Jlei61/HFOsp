import importlib.util
import json
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


def test_event_free_baseline_does_not_absorb_very_early_persistent_entry():
    core = np.r_[np.zeros(8), np.full(80, 120.0)]
    mask = A.event_free_baseline_bins(core)
    assert mask[:8].all()
    assert not mask[8:].any()
    got = A.detect_episode(core, mask)
    assert got["onset_bin"] is not None


def test_resolve_baseline_drops_intra_episode_gaps_of_a_burst_train():
    """Deep gaps inside a burst train are sub-threshold but are not event-free."""
    burst = np.tile(np.r_[np.full(5, 200.0), np.zeros(3)], 40)
    core = np.r_[np.zeros(20), burst]
    single = A.event_free_baseline_bins(core)
    baseline, episode, audit = A.resolve_episode_baseline(core)

    assert single[20:48].any()                       # first pass keeps burst-train gaps
    assert audit["pass"] == "pre_onset_restricted"
    assert audit["n_bins_dropped_after_onset"] > 0
    assert not baseline[episode["onset_bin"]:].any()
    assert baseline[:20].all()


def test_resolve_baseline_keeps_the_single_pass_when_nothing_follows_onset():
    core = np.r_[np.zeros(20), np.full(80, 120.0)]
    baseline, episode, audit = A.resolve_episode_baseline(core)
    assert audit["n_bins_dropped_after_onset"] == 0
    assert audit["pass"] == "single_already_pre_onset"
    assert baseline.sum() == A.event_free_baseline_bins(core).sum()


def test_resolve_baseline_refuses_to_starve_itself_below_the_rms_minimum():
    core = np.r_[
        np.zeros(3), np.full(17, 300.0),
        np.tile(np.r_[np.zeros(1), np.full(1, 300.0)], 40),
    ]
    baseline, _, audit = A.resolve_episode_baseline(core)
    assert audit["pass"] == "single_insufficient_pre_onset_bins"
    assert audit["n_pre_onset_bins"] < 4
    assert baseline.sum() == A.event_free_baseline_bins(core).sum()


def test_baseline_referenced_intensity_distinguishes_sustained_gain():
    rms = np.ones((40, 2))
    rms[20:] = 10.0
    baseline = np.zeros(40, bool)
    baseline[:20] = True
    got = A.baseline_referenced_intensity(rms, baseline, slice(20, 40))
    assert got["median_gain_db_across_contacts"] == pytest.approx(20.0)
    assert got["occupancy_above_6db"] == 1.0
    assert got["normalized_integrated_energy_per_s"] == 100.0


def test_baseline_referenced_intensity_does_not_epsilon_promote_zero_contact():
    rms = np.ones((40, 2))
    rms[:20, 1] = 0.0
    rms[20:, 0] = 2.0
    rms[20:, 1] = 100.0
    baseline = np.zeros(40, bool)
    baseline[:20] = True
    got = A.baseline_referenced_intensity(rms, baseline, slice(20, 40))
    assert got["n_valid_baseline_contacts"] == 1
    assert got["normalized_integrated_energy_per_s"] == pytest.approx(4.0)
    assert got["max_gain_db_across_contacts"] == pytest.approx(10 * np.log10(4.0))


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


def test_episode_detector_rejects_transient_lull_and_keeps_later_durable_offset():
    core = np.r_[
        np.zeros(80), np.full(120, 90.0), np.zeros(60),
        np.full(100, 90.0), np.zeros(120),
    ]
    baseline = np.zeros(core.size, bool)
    baseline[:60] = True
    got = A.detect_episode(core, baseline)
    assert got["status"] == "onset_durable_offset"
    assert got["offset_bin"] is not None
    assert len(got["transient_offset_bins"]) == 1
    assert len(got["rapid_reentry_bins"]) == 1


def test_episode_detector_does_not_call_trace_truncation_an_offset():
    core = np.r_[np.zeros(80), np.full(120, 90.0), np.zeros(35)]
    baseline = np.zeros(core.size, bool)
    baseline[:60] = True
    got = A.detect_episode(core, baseline)
    assert got["status"] == "offset_unconfirmed_at_trace_end"
    assert got["offset_bin"] is None


def test_phase_map_keeps_only_manifest_registered_fast_trajectories():
    """M-surface forks and control panels share seed1/ and must not leak in."""
    manifest = {"rows": [{
        "config_id": "fast0", "family": "depression_only_lhs", "arm": "i2e",
        "tau_D_ms": 300.7, "d_star": 0.7281, "strength_scale": 1.0,
        "g_M": 1.0, "tau_M_ms": 500.0, "g_Z": 1.0, "T_ms": 12000.0,
    }]}

    def mechanism(tau_m, g_m):
        return {
            "arm": "i2e", "strength_scale": 1.0,
            "i2e_depression": {"tau_D_ms": 300.7, "d_star_nominal": 0.7281},
            "dynamic_slow_flow": {"g_M": g_m, "tau_M_ms": tau_m, "g_Z": 1.0},
        }

    analyses = [
        {"stem": "fast", "mechanism": mechanism(500.0, 1.0)},
        {"stem": "m_fork", "mechanism": mechanism(2000.0, 10.0)},
        {"stem": "controlled", "mechanism": mechanism(500.0, 1.0)},
    ]
    summaries = {
        "fast": {"T_ms": 12000.0},
        "m_fork": {"T_ms": 20000.0},
        "controlled": {"T_ms": 12000.0, "finite_control": {"uplift_mV": 4.0}},
    }

    rows, missing = A.select_batch_rows(manifest, analyses, summaries)
    assert [row["stem"] for row in rows] == ["fast"]
    assert rows[0]["config_id"] == "fast0"
    assert missing == []


def test_phase_map_reports_manifest_cells_without_a_trajectory():
    manifest = {"rows": [{
        "config_id": "not_run", "arm": "i2e", "tau_D_ms": 300.0, "d_star": 0.55,
        "strength_scale": 1.0, "g_M": 1.0, "tau_M_ms": 500.0, "g_Z": 1.0,
        "T_ms": 12000.0,
    }]}
    rows, missing = A.select_batch_rows(manifest, [], {})
    assert rows == []
    assert missing == ["not_run"]


def test_single_trajectory_analysis_writes_separate_artifact(tmp_path, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    (root / "summary.json").write_text("{}\n")
    (root / "traces.npz").write_bytes(b"fixture")
    expected = {"stem": "run", "phenotype": "fixture"}
    monkeypatch.setattr(A, "analyze_one", lambda path: expected)
    output = tmp_path / "single.json"

    assert A.write_single_analysis(root, output) == expected
    assert json.loads(output.read_text()) == expected
