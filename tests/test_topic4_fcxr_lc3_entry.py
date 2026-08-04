"""The 20 s entry runner's registered scope and its reproduction guard.

The guard's own contracts live in `tests/test_topic4_fcxr_lc3_reproduction.py`;
what is checked here is that the runner is wired to it and that the window it
compares against is not a free parameter.
"""
import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_topic4_fcxr_lc3_entry.py"
    spec = importlib.util.spec_from_file_location("run_topic4_fcxr_lc3_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recorded(n, t0=100.0, step=1000.0, dur=10.0):
    return [dict(t_on_ms=t0 + i * step, t_off_ms=t0 + i * step + dur,
                 dur_ms=dur, peak_ext=0.05 + 0.001 * i) for i in range(n)]


def _fresh(recorded):
    return [dict(t_on=e["t_on_ms"], t_off=e["t_off_ms"], dur_ms=e["dur_ms"],
                 peak_ext=e["peak_ext"]) for e in recorded]


def test_the_20s_window_is_the_recon_runners_own_checkpoint():
    """Not a free parameter: comparability against the stored label depends on it."""
    m = _module()
    assert m.T_MS == m.RECON.ONSET_CHECK_MS == 20000.0


def test_identical_events_reproduce():
    m = _module()
    rec = _recorded(12)
    ok, detail = m._event_prefix_matches(_fresh(rec), rec)
    assert ok and "12 events" in detail


def test_a_drifted_event_is_refused_and_named():
    m = _module()
    rec = _recorded(12)
    fresh = _fresh(rec)
    fresh[7]["peak_ext"] += 1e-4
    ok, detail = m._event_prefix_matches(fresh, rec)
    assert not ok and "event 7 peak_ext" in detail


def test_the_runner_no_longer_fails_a_truncated_tail_event():
    """The regression: this shape failed two of three bit-identical seeds."""
    m = _module()
    body = _recorded(8, t0=100.0, step=2000.0, dur=30.0)
    whole = dict(t_on_ms=19985.0, t_off_ms=20014.0, dur_ms=30.0, peak_ext=0.33381)
    truncated = dict(t_on=19985.0, t_off=19999.0, dur_ms=15.0, peak_ext=0.33381)
    ok, detail = m._event_prefix_matches(_fresh(body) + [truncated], body + [whole])
    assert ok, detail
