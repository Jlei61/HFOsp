from scripts.analyze_topic4_zm_phase_lag_lifecycle import adjudicate


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


def _panel(**updates):
    rows = {
        "delay2_H_Mmem": _row(credible=False),
        "delay3_H_Mmem": _row(credible=False),
        "delay3_H_noM": _row(credible=False),
        "delay3_noH_noM": _row(credible=False),
    }
    rows.update(updates)
    return rows


def test_no_macro_semantic_carrier_is_a_clean_no_go():
    rows = _panel(delay3_H_Mmem=_row("onset_persistent", credible=False))
    assert adjudicate(rows)["verdict"] == "NO_CREDIBLE_ICTAL_CARRIER"


def test_lifecycle_candidate_requires_m_dependent_offset_and_return():
    rows = _panel(
        delay3_H_Mmem=_row("onset_durable_offset", 6000.0, returning=True),
        delay3_H_noM=_row(),
    )
    assert adjudicate(rows)["verdict"] == "PHASE_LAG_LIFECYCLE_CANDIDATE_SEED1"


def test_m_that_destroys_the_carrier_is_not_counted_as_termination():
    rows = _panel(
        delay3_H_Mmem=_row("no_sustained_onset", credible=False),
        delay3_H_noM=_row(),
    )
    assert adjudicate(rows)["verdict"] == "M_MEMORY_PREVENTS_OR_FRAGMENTS_CARRIER"

