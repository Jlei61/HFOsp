import pytest

from scripts.paper_figures.plot_fig5a_spatial_zm_qigk_dynamics import (
    select_confirmed_representative,
)


def _aggregate(*, eligible=True, n_seeds=3):
    family = {
        "eligible_multi_seed_family": eligible,
        "parameter_set_id": "locked_v1",
        "n_unique_seeds": n_seeds,
        "minimum_confirmation_seeds": 3,
        "single_frozen_config": True,
        "parameter_contract_sha256": "config-a",
    }
    record = {
        "all_checks_pass": True,
        "full_edge": True,
        "mode": "hybrid",
        "run_role": "confirmation",
        "parameter_set_id": "locked_v1",
        "parameter_contract_sha256": "config-a",
    }
    return {"primary_hybrid_family": family,
            "primary_hybrid_candidate": record}


def test_selection_accepts_eligible_confirmation_family():
    family, record = select_confirmed_representative(_aggregate())
    assert family["parameter_set_id"] == record["parameter_set_id"]


@pytest.mark.parametrize("eligible,n_seeds", [(False, 3), (True, 2)])
def test_selection_fails_closed_without_multi_seed_family(eligible, n_seeds):
    with pytest.raises(RuntimeError):
        select_confirmed_representative(
            _aggregate(eligible=eligible, n_seeds=n_seeds))
