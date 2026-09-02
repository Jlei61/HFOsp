import pytest

from scripts.paper_figures.plot_fig5a_spatial_zm_ou_tonic import (
    select_tonic_representative,
)


def _record(role):
    return {
        "all_checks_pass": True,
        "run_role": role,
        "mode": "hybrid",
        "parameter_set_id": "tonic_v1" if role == "confirmation" else None,
        "parameter_contract_sha256": "locked",
    }


def test_formal_selection_requires_eligible_confirmation_family():
    family = {
        "eligible_multi_seed_family": True,
        "n_unique_seeds": 3,
        "minimum_confirmation_seeds": 3,
        "single_frozen_config": True,
        "parameter_set_id": "tonic_v1",
        "parameter_contract_sha256": "locked",
    }
    selected = select_tonic_representative({
        "primary_confirmation_family": family,
        "primary_confirmation_candidate": _record("confirmation"),
    })
    assert selected[2] is False


def test_formal_selection_fails_closed_on_discovery_only_aggregate():
    with pytest.raises(RuntimeError, match="formal Fig5A is blocked"):
        select_tonic_representative({
            "primary_discovery_candidate": _record("discovery"),
        })


def test_preview_must_be_explicit_and_is_labelled():
    _, record, preview = select_tonic_representative({
        "primary_discovery_candidate": _record("discovery"),
    }, allow_discovery_preview=True)
    assert record["run_role"] == "discovery"
    assert preview is True
