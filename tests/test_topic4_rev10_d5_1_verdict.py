from scripts.audit_topic4_rev10_d5_1_low_amplitude_bracket import adjudicate


def _row(candidate_id, mode, sigma, *, both, score, occupancy=0.1):
    return {
        "candidate_id": candidate_id,
        "spatial_ou_mode": mode,
        "spatial_ou_sigma_rate_per_ms": sigma,
        "n_runaway_networks": 0,
        "networks_with_clean_A": both,
        "networks_with_clean_B": both,
        "networks_with_both_clean_modes": both,
        "selection_score_equal_network": score,
        "mean_network_ood_fraction": 0.2,
        "max_network_spatial_ou_clip_fraction": 0.0,
        "mean_network_detected_events_descriptive": 12.0 + sigma,
        "mean_network_returned_events_scored": 10.0 + sigma,
        "mean_network_returned_fraction": 0.8,
        "mean_network_fraction_time_above_detector": occupancy,
        "max_network_fraction_time_above_detector": occupancy + 0.01,
        "mean_network_peak_active_fraction": 0.1,
    }


def _summary(access_by_sigma, permuted_access_by_sigma=None):
    permuted_access_by_sigma = permuted_access_by_sigma or {}
    rows = [_row("edge_noop", "off", 0.0, both=0, score=12, occupancy=0.05)]
    for sigma in (0.1, 0.2, 0.35):
        rows.append(_row(
            f"local_{sigma}", "local", sigma,
            both=access_by_sigma.get(sigma, 0),
            score={0.1: 9.0, 0.2: 6.0, 0.35: 1.0}[sigma],
        ))
        rows.append(_row(
            f"permuted_{sigma}", "permuted", sigma,
            both=permuted_access_by_sigma.get(sigma, 0), score=8.0,
        ))
    return {"candidate_rows": rows}


def test_d5_1_selects_lowest_accessible_amplitude_not_best_score():
    verdict = adjudicate(_summary({0.2: 2, 0.35: 3}))
    assert verdict["status"] == "REV10D5_1_LOWEST_ACCESSIBLE_AMPLITUDE_FROZEN"
    assert verdict["selected_sigma_rate_per_ms"] == 0.2
    assert verdict["selected_local_candidate_id"] == "local_0.2"


def test_d5_1_keeps_permutation_capacity_as_mechanistic_readout():
    verdict = adjudicate(_summary({0.1: 2}, {0.1: 2}))
    assert verdict["selected_sigma_rate_per_ms"] == 0.1
    assert verdict["selected_marginal_access_also_sufficient"] is True
    assert verdict["comparisons"][0]["local_activity"][
        "fraction_time_above_detector_ratio_to_off"
    ] == 2.0


def test_d5_1_stops_when_no_low_amplitude_opens_both_routes():
    verdict = adjudicate(_summary({0.1: 1, 0.2: 1, 0.35: 1}))
    assert verdict["status"] == "REV10D5_1_LOW_AMPLITUDE_ACCESS_NOT_OBSERVED"
    assert verdict["selected_local_candidate_id"] is None
