import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "finalize_topic4_fcxr_lc3.py"
    spec = importlib.util.spec_from_file_location("finalize_topic4_fcxr_lc3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_noncanonical_axial_fallback_cannot_promote_spatiotemporal_candidate():
    module = _module()
    life = {"status": "TEMPORAL_LIFECYCLE_CANDIDATE"}
    spatial = {
        "canonical_spatial_interpretation_authorized": False,
        "state_labels": {"fallback": "AXIAL_LOCAL_DIRECT_RESPONSE"},
    }
    assert module._overall_verdict(life, spatial) == (
        "TEMPORAL_LIFECYCLE_CANDIDATE_SPATIAL_SUBSTITUTION_NONCANONICAL")


def test_canonical_axial_response_can_promote_spatiotemporal_candidate():
    module = _module()
    life = {"status": "TEMPORAL_LIFECYCLE_CANDIDATE"}
    spatial = {
        "canonical_spatial_interpretation_authorized": True,
        "state_labels": {"onset": "AXIAL_LOCAL_DIRECT_RESPONSE"},
    }
    assert module._overall_verdict(life, spatial) == "SPATIOTEMPORAL_LIFECYCLE_CANDIDATE"
