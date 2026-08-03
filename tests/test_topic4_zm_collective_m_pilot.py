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
        "H05_mdiv2": _row(),
        "H05_mdiv3": _row(),
        "H05_mdiv4": _row("onset_durable_offset", 5000.0, True),
        "H1_mdiv2": _row(),
        "H1_mdiv3": _row(),
        "H1_mdiv4": _row(),
        "noH_mdiv4": _row("no_onset"),
    }
    assert _verdict(rows)["verdict"] == "LIFECYCLE_CANDIDATE_SEED1"
    for key in list(rows):
        if key != "noH_mdiv4":
            rows[key] = _row("no_onset")
    assert _verdict(rows)["verdict"] == "COLLECTIVE_M_PREVENTS_ENTRY"
