import copy

import pytest

from scripts.freeze_topic4_rev22_transition_contract import (
    FORBIDDEN_INPUT_TOKENS,
    REQUIRED_INPUTS,
    sanitized_transition,
)


def _source():
    inputs = {key: {"path": f"x/{key}", "sha256": key} for key in REQUIRED_INPUTS}
    inputs.update({
        "patient_heldout_npz": {"path": "secret", "sha256": "secret"},
        "direction_classifier_manifest": {"path": "labels", "sha256": "labels"},
        "shaft_aware_target_npz": {"path": "training", "sha256": "training"},
    })
    return {"schema_id": "old", "inputs": inputs, "zm": {"mode": "z_plus_m"},
            "simulation": {"duration_ms": 20000.0},
            "local_connectivity_basis": {}, "engine_detector": {}, "spatial_ou": {},
            "validation": {"maximum_ood_fraction": 0.5}}


def test_sanitized_transition_removes_validation_and_freezes_slow_state_off():
    result = sanitized_transition(_source(), source_sha256="abc")
    assert tuple(result["inputs"]) == REQUIRED_INPUTS
    assert result["zm"]["mode"] == "off"
    assert result["zm"]["use_z"] is False
    assert result["source_transition_config_sha256"] == "abc"
    assert "validation" not in result
    declared = " ".join(result["inputs"]).lower()
    assert all(token not in declared for token in FORBIDDEN_INPUT_TOKENS)


def test_sanitized_transition_requires_every_substrate_input():
    source = copy.deepcopy(_source())
    del source["inputs"][REQUIRED_INPUTS[0]]
    with pytest.raises(ValueError, match="misses required"):
        sanitized_transition(source, source_sha256="abc")
