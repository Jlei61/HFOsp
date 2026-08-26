import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_topic4_rev12_node_axis_inventory.py"
SPEC = importlib.util.spec_from_file_location("node_axis_inventory", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _manifest(tmp_path, candidates, event="edge_supported_causal_family_observation"):
    stage = tmp_path / "node_stage_test"
    stage.mkdir()
    path = stage / "candidate_manifest.json"
    path.write_text(json.dumps({
        "status": "DONE", "event_unit": {"name": event},
        "candidates": candidates,
    }))
    return path


def test_geometry_amplitude_is_not_misread_as_node_gain(tmp_path):
    path = _manifest(tmp_path, [{
        "node_field": {
            "field_sha256": "field",
            "residual_coordinates": {"observed_surface_rms": 2.0},
        }
    }])
    row = MODULE.inspect_manifest(path, "edge_supported_causal_family_observation", {
        "node_gain", "threshold_gain",
    })
    assert row["surface_geometry_amplitudes"] == [2.0]
    assert not row["has_scalar_node_gain"]


def test_signed_depth_and_global_gain_are_separate_axes(tmp_path):
    path = _manifest(tmp_path, [{
        "node_field": {"field_sha256": "field"},
        "node_mapping": {"signed_depth_shrinkage": 0.5, "node_gain": 1.25},
    }])
    row = MODULE.inspect_manifest(path, "edge_supported_causal_family_observation", {
        "node_gain", "threshold_gain",
    })
    assert row["signed_depth_shrinkage_values"] == [0.5]
    assert row["has_scalar_node_gain"]
    assert row["scalar_node_gain_records"] == [{"key": "node_gain", "value": 1.25}]


def test_event_unit_compatibility_is_explicit(tmp_path):
    path = _manifest(tmp_path, [], event="directed_spatiotemporal_lineage")
    row = MODULE.inspect_manifest(path, "edge_supported_causal_family_observation", set())
    assert not row["current_causal_family_compatible"]
