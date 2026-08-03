from scripts.analyze_topic4_zm_collective_m_pilot import _verdict


def _row(status="onset_no_offset", offset=None, returning=False, credible=True):
    return {
        "episode_status": status,
        "offset_ms": offset,
        "runaway": False,
        "returning_event": returning,
        "returning_distribution": False,
        "median_vseeg_gain_db": 25.0 if credible else 17.0,
        "energy_occupancy_6db": 0.75 if credible else 0.20,
        "post_onset_deep_gap_fraction": 0.10 if credible else 0.65,
        "spatial_pc1": 0.90 if credible else 0.98,
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


def test_rate_onset_cannot_override_failed_energy_and_spatial_semantics():
    rows = {
        key: _row("no_onset") for key in (
            "H05_mdiv2", "H05_mdiv3", "H05_mdiv4",
            "H1_mdiv2", "H1_mdiv3", "H1_mdiv4", "noH_mdiv4",
        )
    }
    rows["H1_mdiv2"] = _row("onset_persistent", credible=False)
    assert _verdict(rows)["verdict"] == "NO_CREDIBLE_ICTAL_CARRIER"
