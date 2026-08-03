from scripts.analyze_topic4_zm_conductance_homotopy import adjudicate, credible_carrier


def _row(*, occupancy=.7, gap=.1, pc1=.9, rate=100., rho80=.05, offset=None, returning=False):
    return {
        "episode_status": "onset_persistent" if offset is None else "onset_durable_offset",
        "runaway": False,
        "median_vseeg_gain_db": 30.,
        "energy_occupancy_6db": occupancy,
        "post_onset_deep_gap_fraction": gap,
        "spatial_pc1": pc1,
        "core_mean_hz": rate,
        "core_rho80_active_fraction": rho80,
        "offset_ms": offset,
        "returning_event": returning,
        "returning_distribution": False,
    }


def test_carrier_gate_rejects_fragment_and_common_mode_plateau():
    assert credible_carrier(_row())
    assert not credible_carrier(_row(occupancy=.49))
    assert not credible_carrier(_row(gap=.21))
    assert not credible_carrier(_row(pc1=.96))
    assert not credible_carrier(_row(rate=300.))


def test_long_run_controls_lifecycle_verdict():
    rows = {"short": _row()}
    assert adjudicate(rows, None)["verdict"] == "CARRIER_CANDIDATE_AWAITS_LONG_RUN"
    assert adjudicate(rows, _row())["verdict"] == "DURABLE_CARRIER_WITHOUT_NATIVE_OFFSET"
    assert adjudicate(rows, _row(offset=6000., returning=True))["verdict"] == "LIFECYCLE_CANDIDATE_SEED1"

