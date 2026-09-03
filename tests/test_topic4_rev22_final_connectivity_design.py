import pytest

from scripts.audit_topic4_rev22_final_connectivity_design import (
    _fit_seeds,
    summarize,
    validate_inputs,
)


def _candidate(identifier="p0"):
    return {
        "candidate_id": identifier,
        "block": "reference",
        "mechanisms": {"Z_M": "off"},
        "node_field": {"field_sha256": "field"},
    }


def _seeds():
    return {
        "response_design_manifest_sha256": "design",
        "fit": {"units": [{"topology_seed": seed, "dynamics_seed": seed}
                           for seed in (2511, 2512, 2513, 2514)]},
    }


def test_contract_validation_rejects_validation_visibility():
    response = {"schema_id": "topic4_rev22_dci_response_design_manifest_v1",
                "candidate_count": 1, "candidates": [_candidate()],
                "node_field_sha256": "field"}
    with pytest.raises(ValueError, match="validation-only"):
        validate_inputs(
            response, _seeds(),
            {"schema_id": "topic4_rev22_dci_transition_execution_v1",
             "inputs": {"patient_heldout_npz": {}}},
            response_sha256="design",
        )


def test_fit_seed_contract_is_exact():
    assert _fit_seeds(_seeds()) == [2511, 2512, 2513, 2514]
    broken = _seeds()
    broken["fit"]["units"][0]["dynamics_seed"] = 99
    with pytest.raises(ValueError, match="topology=dynamics"):
        _fit_seeds(broken)


def test_summary_requires_complete_candidate_by_topology_grid():
    candidates = [_candidate("p0"), _candidate("p1")]
    complete = {
        1: {"rows": [{"candidate_id": "p0", "topology_seed": 1, "passes": True},
                     {"candidate_id": "p1", "topology_seed": 1, "passes": True}]},
        2: {"rows": [{"candidate_id": "p0", "topology_seed": 2, "passes": True},
                     {"candidate_id": "p1", "topology_seed": 2, "passes": False}]},
    }
    rows, passed = summarize(complete, candidates)
    assert len(rows) == 4
    assert passed is False
    complete[2]["rows"].pop()
    with pytest.raises(RuntimeError, match="incomplete"):
        summarize(complete, candidates)
