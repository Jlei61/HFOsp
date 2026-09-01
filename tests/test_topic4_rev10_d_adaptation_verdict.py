from scripts.audit_topic4_rev10_d_adaptation_canary import adjudicate


def _row(candidate, mode, tau=0.0, increment=0.0, *, a=0, b=3, both=0,
         score=10.0, runaway=0):
    return {
        "candidate_id": candidate,
        "adaptation_mode": mode,
        "adaptation_tau_ms": tau,
        "adaptation_increment_mV": increment,
        "networks_with_clean_A": a,
        "networks_with_clean_B": b,
        "networks_with_both_clean_modes": both,
        "selection_score_equal_network": score,
        "n_runaway_networks": runaway,
    }


def _summary(pass_one=True):
    rows = [_row("edge_noop", "off", a=1, both=1, score=10.0)]
    for tau in (250.0, 750.0, 2000.0):
        for increment in (0.1, 0.25, 0.5):
            passes = pass_one and tau == 750.0 and increment == 0.25
            rows.append(_row(
                f"local_{tau}_{increment}", "local", tau, increment,
                a=2 if passes else 1, both=2 if passes else 1,
                score=8.0 if passes else 11.0,
            ))
            rows.append(_row(
                f"global_{tau}_{increment}", "global", tau, increment,
                a=0, both=0, score=12.0,
            ))
    return {"candidate_rows": rows}


def test_local_candidate_must_exceed_global_and_off_same_network_support():
    verdict = adjudicate(_summary())
    assert verdict["status"] == "REV10D_LOCAL_ADAPTATION_ROUTE_ACCESS_OBSERVED"
    assert verdict["selected_local_candidate_id"] == "local_750.0_0.25"


def test_no_local_specific_candidate_closes_the_canary():
    verdict = adjudicate(_summary(pass_one=False))
    assert verdict["status"] == "REV10D_LOCAL_ADAPTATION_ROUTE_ACCESS_NOT_OBSERVED"
    assert verdict["selected_local_candidate_id"] is None


def test_runaway_candidate_cannot_pass():
    summary = _summary()
    candidate = next(
        row for row in summary["candidate_rows"]
        if row["candidate_id"] == "local_750.0_0.25"
    )
    candidate["n_runaway_networks"] = 1
    assert adjudicate(summary)["selected_local_candidate_id"] is None
