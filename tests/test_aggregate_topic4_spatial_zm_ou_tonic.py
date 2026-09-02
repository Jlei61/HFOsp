from scripts.aggregate_topic4_spatial_zm_ou_tonic import (
    confirmation_families,
    representative_from_family,
)


def _row(seed, *, passed=True, parameter_set_id="tonic_v1", onset=500.0):
    return {
        "path": f"seed{seed}.json",
        "seed": seed,
        "mode": "hybrid",
        "run_role": "confirmation",
        "parameter_set_id": parameter_set_id,
        "parameter_contract_sha256": "locked",
        "scientific_onset_ms": onset,
        "all_checks_pass": passed,
    }


def test_three_all_pass_seeds_form_eligible_family_and_lock_median_onset():
    rows = [_row(1, onset=700), _row(2, onset=500), _row(3, onset=600)]
    family = confirmation_families(rows)[0]
    assert family["eligible_multi_seed_family"]
    assert representative_from_family(family)["seed"] == 3


def test_one_failed_seed_blocks_family():
    rows = [_row(1), _row(2), _row(3, passed=False)]
    assert not confirmation_families(rows)[0]["eligible_multi_seed_family"]


def test_config_drift_blocks_family():
    rows = [_row(1), _row(2), _row(3)]
    rows[-1]["parameter_contract_sha256"] = "drift"
    assert not confirmation_families(rows)[0]["eligible_multi_seed_family"]
