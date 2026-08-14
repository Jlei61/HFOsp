import importlib.util
import json
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc6_functional import COMPONENTS, array_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_topic4_fcxr_lc6a_functional_probes.py"
SPEC = importlib.util.spec_from_file_location("lc6a_functional_finalize", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _write_condition(root, condition, *, input_hash="same-input"):
    arm = root / "functional_probes" / condition
    arm.mkdir(parents=True)
    edges = np.linspace(-6.0, 6.0, 49)
    components = np.zeros((3, len(COMPONENTS), 48), dtype=float)
    components[:, COMPONENTS.index("F_E")] = 0.2
    components[:, COMPONENTS.index("F_I")] = 0.1
    components[:, COMPONENTS.index("I_syn_signed")] = 0.1
    arrays = {
        "neutral_axis__delta_components": np.zeros((3, len(COMPONENTS), 4)),
        "neutral_axis__delta_axis_components": components,
        "neutral_axis__delta_axis_rate_hz": np.zeros((3, 48)),
        "neutral_axis__delta_map_components": np.zeros((3, len(COMPONENTS), 2, 2)),
        "neutral_axis__delta_map_rate_hz": np.zeros((3, 2, 2)),
        "neutral_axis__delta_axis_signed_1ms": np.zeros((300, 48)),
        "neutral_axis__axis_edges_mm": edges,
        "neutral_axis__window_edges_ms": np.array([0.0, 50.0, 150.0, 300.0]),
    }
    np.savez_compressed(arm / "responses.npz", **arrays)
    location = {
        "external_input_exact": True,
        "external_input_sha256": input_hash,
        "pulse_accounting": {"duration_ms": 50.0},
        "max_active_fraction_1ms_sham": 0.0,
        "max_active_fraction_1ms_probe": 0.0,
        "excess_spikes": 0,
        "window_zero_crossings": [
            {"forward_mm": None, "backward_mm": None},
            {"forward_mm": None, "backward_mm": None},
            {"forward_mm": 0.5, "backward_mm": -0.5},
        ],
        "latency_ms": {"center": 0.0, "forward": 10.0, "backward": 10.0},
    }
    summary = {
        "status": "COMPLETE",
        "condition": condition,
        "scientific_role": "descriptive_functional_geometry_not_trajectory_gate",
        "graph_sha256": condition * 8,
        "graph_construction_q": 1.0,
        "manifest_sha256": "manifest",
        "prelock_sha256": "prelock",
        "amplitude_lock_sha256": "amplitude",
        "start_ms": 2_100.05,
        "locations": {"neutral_axis": location},
        "arrays_sha256": {key: array_sha256(value) for key, value in arrays.items()},
    }
    (arm / "summary.json").write_text(json.dumps(summary))
    (root / "functional_probes" / f"DONE_LC6A_FUNCTIONAL_{condition}.json").write_text("{}")


def test_finalize_validates_and_writes_required_artifacts(tmp_path):
    for condition in MOD.CONDITIONS:
        _write_condition(tmp_path, condition)
    (tmp_path / "functional_probe_lock.json").write_text(json.dumps({"event_bar": 0.04}))
    payload = MOD.finalize(tmp_path)
    assert payload["status"] == "COMPLETE"
    assert payload["any_registered_zero_crossing"] is True
    assert payload["zero_crossing_is_a_gate"] is False
    assert payload["n_background_event_confounded_locations"] == 0
    assert (tmp_path / "impulse_response_audit.json").is_file()
    assert (tmp_path / "figures/lc6a_functional_response.png").is_file()
    assert (tmp_path / "figures/lc6a_functional_response.pdf").is_file()
    assert "### lc6a_functional_response.png" in (tmp_path / "figures/README.md").read_text()


def test_finalize_marks_matched_background_event_as_confounded(tmp_path):
    for condition in MOD.CONDITIONS:
        _write_condition(tmp_path, condition)
    path = tmp_path / "functional_probes/Q3/summary.json"
    summary = json.loads(path.read_text())
    summary["locations"]["neutral_axis"]["max_active_fraction_1ms_sham"] = 0.05
    summary["locations"]["neutral_axis"]["max_active_fraction_1ms_probe"] = 0.05
    path.write_text(json.dumps(summary))
    (tmp_path / "functional_probe_lock.json").write_text(json.dumps({"event_bar": 0.04}))
    payload = MOD.finalize(tmp_path)
    assert payload["background_event_confounded_locations"] == ["Q3:neutral_axis"]
    assert payload["conditions"]["Q3"]["locations"]["neutral_axis"][
        "background_population_event_present"
    ] is True


def test_finalize_rejects_cross_condition_input_mismatch(tmp_path):
    for condition in MOD.CONDITIONS:
        _write_condition(tmp_path, condition, input_hash=("different" if condition == "Q3" else "same"))
    try:
        MOD.load_and_validate(tmp_path)
    except RuntimeError as exc:
        assert "one external-input stream" in str(exc)
    else:
        raise AssertionError("input mismatch was accepted")


def test_finalize_rejects_array_hash_mismatch(tmp_path):
    for condition in MOD.CONDITIONS:
        _write_condition(tmp_path, condition)
    path = tmp_path / "functional_probes/Q2/responses.npz"
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: np.asarray(handle[key]) for key in handle.files}
    arrays["neutral_axis__delta_axis_rate_hz"][0, 0] = 1.0
    np.savez_compressed(path, **arrays)
    try:
        MOD.load_and_validate(tmp_path)
    except RuntimeError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("array hash mismatch was accepted")
