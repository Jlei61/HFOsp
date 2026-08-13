import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_map", ROOT / "scripts/run_topic4_fcxr_lc5v2p1_phase_map.py"
)
MAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MAP)


def test_locked_manifest_is_exact_3x3_and_has_one_control():
    _, manifest, cells = MAP.load_manifest()
    assert len(cells) == 9
    assert set(manifest["matrix"]["tau_ms"]) == {3000.0, 8000.0, 15000.0}
    assert set(manifest["matrix"]["gamma"]) == {0.005, 0.01, 0.02}
    assert manifest["reuse"]["control"].endswith("gamma_milli000")


def test_manifest_has_only_three_hard_stop_families():
    _, manifest, _ = MAP.load_manifest()
    assert manifest["hard_stops"] == [
        "CONTROL_OR_INPUT_PREFIX_MISMATCH",
        "CALIBRATION_MANIFEST_OR_MECHANISM_HASH_MISMATCH",
        "NUMERICAL_OR_RESOURCE_FAILURE",
    ]


def test_unknown_cell_is_rejected_before_any_simulation():
    with pytest.raises(ValueError, match="not in the locked manifest"):
        MAP.run_cell("tau6000_gamma0010")


def test_boundary_manifest_is_the_locked_irregular_11_cell_patch():
    path = ROOT / "config/topic4_fcxr_lc5v2p1_boundary_patch.json"
    _, manifest, cells = MAP.load_manifest(path)
    assert manifest["experiment_id"] == MAP.BOUNDARY_EXPERIMENT
    assert len(cells) == 11
    assert set(cells.values()) == MAP.BOUNDARY_CELLS
    assert all(source is None for source in manifest["reuse"]["eligible_cells"].values())


def test_boundary_manifest_rejects_outcome_adaptive_cell_edit(tmp_path):
    source = ROOT / "config/topic4_fcxr_lc5v2p1_boundary_patch.json"
    payload = json.loads(source.read_text())
    payload["matrix"]["cells"].pop()
    path = tmp_path / "drifted.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="boundary manifest drifted"):
        MAP.load_manifest(path)


def test_reuse_requires_onset_and_full_post_onset_window(tmp_path):
    arm = tmp_path / "arm"
    arm.mkdir()
    summary = {
        "tau_ms": 3000.0, "gamma_nominal_dose": 0.01, "p0_policy": "q099",
        "onset_ms": None, "T_ms": 18000.0, "external_input_sha256": "same",
        "outcome": "NO_NATURAL_ONSET", "spike_sha256": "spikes",
    }
    (arm / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="no-onset"):
        MAP.validate_reuse(
            tmp_path, Path("arm"), tau_ms=3000.0, gamma=0.01,
            observation={"min_end_ms": 18000.0, "post_onset_ms": 7000.0, "max_end_ms": 25000.0},
            expected_input="same",
        )
    summary["onset_ms"] = 12000.0
    (arm / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="post-onset"):
        MAP.validate_reuse(
            tmp_path, Path("arm"), tau_ms=3000.0, gamma=0.01,
            observation={"min_end_ms": 18000.0, "post_onset_ms": 7000.0, "max_end_ms": 25000.0},
            expected_input="same",
        )


def test_control_requires_exact_spike_and_rate_parity(tmp_path):
    arm = tmp_path / "control"
    arm.mkdir()
    summary = {
        "T_ms": 18000.0, "onset_ms": 11000.0, "outcome": "ESCALATING_SATURATION",
        "spike_sha256": "spikes",
        "control_parity": {"spike_exact": True, "rate_exact": True, "rate_max_abs_diff_hz": 0.0},
    }
    (arm / "summary.json").write_text(json.dumps(summary))
    got = MAP.validate_control(
        tmp_path, Path("control"),
        {"min_end_ms": 18000.0, "post_onset_ms": 7000.0, "max_end_ms": 25000.0},
    )
    assert got["status"] == "REUSED_CONTROL_EXACT"
    summary["control_parity"]["spike_exact"] = False
    (arm / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="exact spike/rate"):
        MAP.validate_control(
            tmp_path, Path("control"),
            {"min_end_ms": 18000.0, "post_onset_ms": 7000.0, "max_end_ms": 25000.0},
        )
def test_active_prefix_tag_is_separate_from_historical_artifacts():
    tag = MAP.PREFIX._tag(0.005, "q099", 3000.0, "lc5v2p1_timescale_dose_map")
    assert tag.startswith("lc5v2p1_timescale_dose_map_q099_")
    assert tag != MAP.PREFIX._tag(0.005, "q099", 3000.0)
