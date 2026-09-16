"""Exact condition identity for Topic 4 v2.1 nomination.

Execution-unit cache identity is intentionally separate: equality of arrays on
one training topology does not prove equality of parameterized conditions on
new topologies.
"""
from __future__ import annotations

import json


def condition_key(candidate):
    mechanisms = candidate["mechanisms"]
    mapping = candidate["node_mapping"]
    field = candidate["node_field"]
    payload = {
        "domain": candidate.get("domain"),
        "node_field": {
            "field_type": field.get("field_type"),
            "centers_mm": field["centers_mm"],
            "target_count": field["target_count"],
        },
        "node_mapping": {
            "signed_depth_shrinkage": mapping.get("signed_depth_shrinkage", 1.0),
            "node_gain": mapping.get("node_gain", 1.0),
        },
        "node_dispersion_field": candidate.get("node_dispersion_field"),
        "dynamic_parameters": candidate["dynamic_parameters"],
        "mechanisms": {
            key: mechanisms.get(key) for key in (
                "g_EE", "g_EtoI", "Z_M", "ellipse_angle_deg",
                "ellipse_aspect_ratio", "ellipse_reference_angle_deg",
                "ellipse_reference_aspect_ratio",
            )
        },
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
