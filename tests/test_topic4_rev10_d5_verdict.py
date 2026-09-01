from scripts.audit_topic4_rev10_d5_spatial_ou_canary import adjudicate


def _row(candidate_id, mode, sigma=0.0, ell=0.0, both=0, runaway=0, score=10.0):
    return {
        "candidate_id": candidate_id,
        "spatial_ou_mode": mode,
        "spatial_ou_sigma_rate_per_ms": sigma,
        "spatial_ou_ell_mm": ell,
        "n_runaway_networks": runaway,
        "networks_with_clean_A": both,
        "networks_with_clean_B": both,
        "networks_with_both_clean_modes": both,
        "selection_score_equal_network": score,
        "mean_network_ood_fraction": 0.1,
        "max_network_spatial_ou_clip_fraction": 0.0,
    }


def _summary(local_both=0, permuted_both=0):
    rows = [_row("edge_noop", "off", both=0, score=12.0)]
    for sigma in (0.5, 1.0):
        for ell in (0.38, 0.76):
            special = sigma == 0.5 and ell == 0.38
            rows.append(_row(
                f"local_{sigma}_{ell}", "local", sigma, ell,
                both=local_both if special else 0, score=5.0 if special else 10.0,
            ))
            rows.append(_row(
                f"permuted_{sigma}_{ell}", "permuted", sigma, ell,
                both=permuted_both if special else 0, score=8.0,
            ))
    return {"candidate_rows": rows}


def test_locality_specific_access_requires_advantage_over_permuted_and_off():
    verdict = adjudicate(_summary(local_both=2, permuted_both=1))
    assert verdict["status"] == "REV10D5_SPATIAL_LOCALITY_ACCESS_OBSERVED"
    assert verdict["selected_local_candidate_id"] == "local_0.5_0.38"


def test_equal_local_and_permuted_access_is_not_locality_evidence():
    verdict = adjudicate(_summary(local_both=2, permuted_both=2))
    assert verdict["status"] == "REV10D5_NONLOCAL_MARGINAL_ACCESS_OBSERVED"


def test_no_dual_mode_access_closes_tested_family():
    verdict = adjudicate(_summary(local_both=1, permuted_both=0))
    assert verdict["status"].endswith("ACCESS_NOT_OBSERVED")
