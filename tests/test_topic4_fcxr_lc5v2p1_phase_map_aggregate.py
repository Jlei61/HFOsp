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
        "source_summary": "/tmp/source/summary.json", "T_ms": 18000.0,
        "onset_ms": None if post is None else 18000.0 - post, "offset_ms": None,
        "n_returning": 12, "mean_rate_hz": 20.0,
        "per_second_mean_rate_hz": [20.0],
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


def test_extension_reclassifies_source_without_erasing_screen_outcome():
    rows = [_row("CONTAINED_HIGH_NO_OFFSET", 7000, 240, tau=3, gamma=.06)]
    extension = {
        "status": "COMPLETE", "source_summary": "/tmp/source/summary.json",
        "source_outcome": "CONTAINED_HIGH_NO_OFFSET", "outcome": "ESCALATING_SATURATION",
        "tau_ms": 3000.0, "gamma_nominal_dose": .06, "T_ms": 19000.0,
        "onset_ms": 11000.0, "offset_ms": None, "n_returning": 84,
        "mean_rate_hz": 41.5, "end_rate_hz": 308.7,
        "per_second_mean_rate_hz": [240.0, 308.7],
        "early_stop_reason": "REGISTERED_SATURATION_REACHED",
    }
    merged, adjudication = AGG.merge_extension(rows, extension)
    assert merged[0]["screen_outcome"] == "CONTAINED_HIGH_NO_OFFSET"
    assert merged[0]["adjudicated_outcome"] == "ESCALATING_SATURATION"
    assert merged[0]["screen_end_rate_hz"] == 240
    assert merged[0]["end_rate_hz"] == 308.7
    assert adjudication["early_stop_reason"] == "REGISTERED_SATURATION_REACHED"


def test_two_extensions_close_both_short_window_labels():
    first = _row("CONTAINED_HIGH_NO_OFFSET", 7000, 240, tau=3, gamma=.06)
    second = _row("CONTAINED_HIGH_NO_OFFSET", 2000, 146, tau=15, gamma=.003)
    second["source_summary"] = "/tmp/second/summary.json"
    extensions = [
        (Path("/tmp/first_extension/summary.json"), {
            "status": "COMPLETE", "source_summary": first["source_summary"],
            "source_outcome": "CONTAINED_HIGH_NO_OFFSET",
            "outcome": "ESCALATING_SATURATION", "tau_ms": 3000.0,
            "gamma_nominal_dose": .06, "T_ms": 19000.0, "onset_ms": 11000.0,
            "offset_ms": None, "n_returning": 84, "mean_rate_hz": 41.5,
            "end_rate_hz": 308.7, "per_second_mean_rate_hz": [240.0, 308.7],
            "early_stop_reason": "REGISTERED_SATURATION_REACHED",
        }),
        (Path("/tmp/second_extension/summary.json"), {
            "status": "COMPLETE", "source_summary": second["source_summary"],
            "source_outcome": "CONTAINED_HIGH_NO_OFFSET",
            "outcome": "ESCALATING_SATURATION", "tau_ms": 15000.0,
            "gamma_nominal_dose": .003, "T_ms": 27000.0, "onset_ms": 23000.0,
            "offset_ms": None, "n_returning": 58, "mean_rate_hz": 39.4,
            "end_rate_hz": 405.9, "per_second_mean_rate_hz": [146.0, 308.5, 405.9],
            "early_stop_reason": "REGISTERED_SATURATION_REACHED",
        }),
    ]
    merged, adjudications = AGG.merge_extensions([first, second], extensions)
    assert [row["adjudicated_outcome"] for row in merged] == [
        "ESCALATING_SATURATION", "ESCALATING_SATURATION",
    ]
    assert [Path(row["extension_summary"]) for row in merged] == [
        extensions[0][0], extensions[1][0],
    ]
    assert len(adjudications) == 2
    assert adjudications[1]["end_rate_hz"] == 405.9


def test_short_post_onset_containment_is_right_censored():
    row = _row("CONTAINED_HIGH_NO_OFFSET", 2000, 146, tau=15, gamma=.003)
    row["adjudicated_outcome"] = row["outcome"]
    assert AGG.evidence_class(row) == "RIGHT_CENSORED_CONTAINMENT_CANDIDATE"
    row["post_onset_observed_ms"] = 7000
    assert AGG.evidence_class(row) == "CONTAINED_HIGH_NO_OFFSET"
