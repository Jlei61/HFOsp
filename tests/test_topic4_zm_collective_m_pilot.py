from scripts.analyze_topic4_zm_collective_m_pilot import _verdict


def _row(status="onset_no_offset", offset=None, returning=False):
    return {
        "episode_status": status,
        "offset_ms": offset,
        "runaway": False,
        "returning_event": returning,
        "returning_distribution": False,
    }


def test_collective_m_verdict_separates_prevention_offset_and_return():
    rows = {
        "H_mdiv2": _row(),
        "H_mdiv4": _row("onset_durable_offset", 5000.0, True),
        "noH_mdiv4": _row("no_onset"),
    }
    assert _verdict(rows)["verdict"] == "LIFECYCLE_CANDIDATE_SEED1"
    rows["H_mdiv4"] = _row("no_onset")
    rows["H_mdiv2"] = _row("no_onset")
    assert _verdict(rows)["verdict"] == "COLLECTIVE_M_PREVENTS_ENTRY"
