"""The guard that licenses reading a re-simulated entry ledger.

The 20 s entry run exists only to emit measurements the 45 s reconnaissance
runner could not.  That is legitimate exactly as long as it is the *same*
trajectory, so the run refuses to publish unless its events reproduce the
recorded ones bit for bit.  A guard that passes on a drifted trajectory would
silently attribute one run's dose to another run's onset.
"""
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_topic4_fcxr_lc3_entry.py"
    spec = importlib.util.spec_from_file_location("run_topic4_fcxr_lc3_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded(n, t0=100.0, step=1000.0):
    return [dict(t_on_ms=t0 + i * step, t_off_ms=t0 + i * step + 8.0,
                 dur_ms=8.0, peak_ext=0.05 + 0.001 * i, returned=True)
            for i in range(n)]


def _fresh(recorded):
    return [dict(t_on=e["t_on_ms"], t_off=e["t_off_ms"], dur_ms=e["dur_ms"],
                 peak_ext=e["peak_ext"], returned=e["returned"]) for e in recorded]


def test_identical_events_reproduce():
    m = _module()
    rec = _recorded(12)
    ok, detail = m._event_prefix_matches(_fresh(rec), rec)
    assert ok and "12 events" in detail


def test_a_drifted_peak_extent_is_refused_and_named():
    m = _module()
    rec = _recorded(12)
    fresh = _fresh(rec)
    fresh[7]["peak_ext"] += 1e-4
    ok, detail = m._event_prefix_matches(fresh, rec)
    assert not ok and "event 7 peak_ext" in detail


def test_a_drifted_onset_time_is_refused():
    m = _module()
    rec = _recorded(12)
    fresh = _fresh(rec)
    fresh[3]["t_on"] += 0.05
    ok, detail = m._event_prefix_matches(fresh, rec)
    assert not ok and "event 3 t_on" in detail


def test_a_missing_event_is_refused():
    m = _module()
    rec = _recorded(12)
    ok, detail = m._event_prefix_matches(_fresh(rec)[:-1], rec)
    assert not ok and "event count 11 vs recorded 12" in detail


def test_events_past_the_cut_are_excluded_from_both_sides():
    """The recorded run saw 45 s and this one 20 s, so only the shared span counts."""
    m = _module()
    rec = _recorded(12) + _recorded(5, t0=m.T_MS + 500.0)
    ok, detail = m._event_prefix_matches(_fresh(_recorded(12)), rec)
    assert ok and "12 events" in detail


def test_an_event_straddling_the_cut_is_not_a_mismatch():
    """Truncated here, whole there -- excluding it is correct, counting it is not."""
    m = _module()
    rec = _recorded(12)
    straddler = dict(t_on=m.T_MS - 3.0, t_off=m.T_MS + 5.0, dur_ms=8.0,
                     peak_ext=0.05, returned=True)
    ok, _ = m._event_prefix_matches(_fresh(rec) + [straddler], rec)
    assert ok


def test_the_20s_window_is_the_recon_runners_own_checkpoint():
    """Not a free parameter: comparability against the stored label depends on it."""
    m = _module()
    assert m.T_MS == m.RECON.ONSET_CHECK_MS == 20000.0
