import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_aggregate", ROOT / "scripts/aggregate_topic4_fcxr_lc5v2p1_phase_map.py"
)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)


def _row(outcome, post, end, tau=8, gamma=.01):
    return {
        "outcome": outcome, "post_onset_observed_ms": post, "end_rate_hz": end,
        "tau_ms": tau * 1000, "tau_s": tau, "gamma": gamma,
    }


def test_candidate_prefers_finite_then_observation_then_lower_end_rate():
    rows = [
        _row("CONTAINED_HIGH_NO_OFFSET", 5000, 30, gamma=.007),
        _row("FINITE_EXCURSION_CANDIDATE", 3000, 20, gamma=.008),
        _row("FINITE_EXCURSION_CANDIDATE", 5000, 40, gamma=.009),
        _row("FINITE_EXCURSION_CANDIDATE", 5000, 10, gamma=.010),
    ]
    assert AGG.choose_extension_candidate(rows)["gamma"] == .010


def test_candidate_is_absent_when_map_has_no_contained_or_finite():
    rows = [
        _row("ESCALATING_SATURATION", 7000, 300),
        _row("ENTRY_BLOCKED_WITH_IED", None, .03),
    ]
    assert AGG.choose_extension_candidate(rows) is None
