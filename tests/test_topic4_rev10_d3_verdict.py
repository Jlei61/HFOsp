from scripts.audit_topic4_rev10_d3_ee_std_canary import adjudicate


def _row(candidate, mode, u=0.0, tau=0.0, a=0, b=3, both=0, score=10.0):
    return {
        "candidate_id": candidate,
        "ee_std_mode": mode,
        "ee_std_u": u,
        "ee_std_tau_ms": tau,
        "networks_with_clean_A": a,
        "networks_with_clean_B": b,
        "networks_with_both_clean_modes": both,
        "selection_score_equal_network": score,
        "n_runaway_networks": 0,
        "mean_network_minimum_std_availability": 0.8,
    }


def _summary(passing=True):
    rows = [_row("edge_noop", "off")]
    for u in (0.08, 0.2):
        for tau in (500.0, 1500.0):
            is_pass = passing and u == 0.2 and tau == 1500.0
            rows.append(_row(
                f"local_{u}_{tau}", "local", u, tau,
                a=2 if is_pass else 0,
                both=2 if is_pass else 0,
                score=8.0 if is_pass else 11.0,
            ))
            rows.append(_row(
                f"global_{u}_{tau}", "global", u, tau,
                a=0, both=0, score=12.0,
            ))
    return {"candidate_rows": rows}


def test_source_specific_std_must_exceed_global_and_off():
    verdict = adjudicate(_summary())
    assert verdict["selected_local_candidate_id"] == "local_0.2_1500.0"


def test_absent_shared_local_support_closes_d3():
    verdict = adjudicate(_summary(False))
    assert verdict["status"].endswith("NOT_OBSERVED")
