from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.paper_figures import plot_topic4_rev17_node_final_fig4 as figure


def _accepted_layers():
    candidate = "joint_rev17_candidate"
    return {
        "config": {
            "schema_id": "topic4_rev17_node_postselection_v1",
            "selected_candidate": {"candidate_id": candidate},
        },
        "postselection": {
            "status": "REV17_NODE_POSTSELECTION_ACCEPTED",
            "candidate_id": candidate,
            "acceptance": {"accepted": True},
            "boundaries": {"EE_EtoI_ZM": "off"},
        },
        "final_science": {
            "status": "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION",
            "candidate_id": candidate,
            "decision": {
                "accepted_for_same_checkpoint_intervention": True,
                "node_freeze_permitted": False,
            },
        },
        "intervention": {
            "schema_id": "topic4_rev17_node_crossed_intervention_aggregate_v1",
            "status": "REV17_NODE_FIELD_FROZEN",
            "candidate_id": candidate,
            "node_freeze_permitted": True,
            "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        },
        "freeze_manifest": {
            "schema_id": "topic4_rev17_frozen_node_field_v1",
            "status": "FROZEN", "candidate_id": candidate,
            "EE_EtoI_ZM": "off",
        },
    }


def test_final_fig4_gate_requires_all_scientific_layers():
    layers = _accepted_layers()
    assert figure.validate_final_figure_gate(**layers) == "joint_rev17_candidate"


@pytest.mark.parametrize(
    ("layer", "path", "value"),
    [
        ("postselection", ("acceptance", "accepted"), False),
        (
            "final_science",
            ("decision", "accepted_for_same_checkpoint_intervention"),
            False,
        ),
        ("intervention", ("node_freeze_permitted",), False),
        ("freeze_manifest", ("status",), "NOT_FROZEN"),
    ],
)
def test_final_fig4_gate_fails_closed(layer, path, value):
    layers = deepcopy(_accepted_layers())
    target = layers[layer]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError):
        figure.validate_final_figure_gate(**layers)


def test_final_fig4_gate_rejects_candidate_identity_drift():
    layers = _accepted_layers()
    layers["intervention"]["candidate_id"] = "different_candidate"
    with pytest.raises(RuntimeError, match="candidate identity"):
        figure.validate_final_figure_gate(**layers)
