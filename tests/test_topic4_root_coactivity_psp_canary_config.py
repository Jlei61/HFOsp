import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev12_root_coactivity_canary import (
    validate_event_canary_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return json.loads(
        (ROOT / "config/topic4_rev12_nd_root_coactivity_psp_canary.json").read_text()
    )


def _causal_config():
    return json.loads(
        (ROOT / "config/topic4_rev12_nd_causal_root_canary.json").read_text()
    )


def test_psp_canary_is_engine_derived_and_nonselective():
    config = _config()
    validate_event_canary_contract(config)
    event_unit = config["event_unit"]
    assert event_unit["causal_memory_method"] == "local_ee_psp_tail"
    assert event_unit["psp_tail_fraction"] == 0.1
    assert event_unit["sensitivity_psp_tail_fractions"] == [0.2, 0.1, 0.05]
    assert config["field_search"]["new_field_parameters_released"] is False
    assert config["search"]["fit_network_seeds"] == []
    assert config["search"]["selection_network_seeds"] == []
    assert config["search"]["confirmation_network_seeds"] == []


def test_causal_root_canary_keeps_compounds_outside_kmeans():
    config = _causal_config()
    validate_event_canary_contract(config)
    event_unit = config["event_unit"]
    assert event_unit["name"] == "causal_root_observation"
    assert event_unit["psp_tail_fraction"] == 0.5
    assert event_unit["sensitivity_psp_tail_fractions"] == [0.8, 0.5, 0.2]
    assert event_unit["sensitivity_minimum_dominances"] == [0.6, 0.7, 0.8]
    assert "never enter A or B" in event_unit["compound_rule"]


@pytest.mark.parametrize(("path", "value"), [
    (("event_unit", "causal_memory_method"), "global_fast_state_decay"),
    (("event_unit", "local_ee_delay_quantile"), 0.99),
    (("event_unit", "contact_geometry_used_for_boundary"), True),
    (("field_search", "new_field_parameters_released"), True),
])
def test_psp_canary_rejects_scientific_contract_drift(path, value):
    config = copy.deepcopy(_config())
    config[path[0]][path[1]] = value
    with pytest.raises(RuntimeError):
        validate_event_canary_contract(config)


def test_psp_canary_rejects_duplicate_or_invalid_tail_fractions():
    for fractions in ([0.2, 0.1, 0.1], [0.2, 1.0, 0.05], [0.2, 0.05]):
        config = copy.deepcopy(_config())
        config["event_unit"]["sensitivity_psp_tail_fractions"] = fractions
        with pytest.raises(RuntimeError):
            validate_event_canary_contract(config)
