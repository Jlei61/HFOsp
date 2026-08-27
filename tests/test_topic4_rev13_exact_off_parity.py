import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_topic4_rev13_exact_off_parity import audit_exact_off_parity


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _event_rows(arrays):
    rows = []
    for index in range(len(arrays["event_t_on_ms"])):
        rows.append({
            "event_index": index,
            "t_on_ms": float(arrays["event_t_on_ms"][index]),
            "t_off_ms": float(arrays["event_t_off_ms"][index]),
            "returned": bool(arrays["event_returned"][index]),
            "n_detector_fragments": int(arrays["event_fragment_count"][index]),
            "root_count": int(arrays["event_root_count"][index]),
        })
    return rows


def _base_arrays(duration_ms, events):
    n_active = int(duration_ms)
    n_envelope = int(duration_ms // 2)
    n_sheet = int(duration_ms // 2)
    n_events = len(events)
    t_on = np.asarray([row[0] for row in events], np.float32)
    t_off = np.asarray([row[1] for row in events], np.float32)
    returned = np.asarray([row[2] for row in events], bool)
    onsets = np.arange(n_events * 3, dtype=np.float32).reshape(n_events, 3)
    ranks = onsets[:, ::-1].copy()
    source = np.arange(n_events * 8, dtype=np.float32).reshape(n_events, 2, 2, 2)
    return {
        "contact_names": np.asarray(["ICL1", "ICL2", "SCL1"], dtype="U16"),
        "shaft_ids": np.asarray(["ICL", "ICL", "SCL"], dtype="U8"),
        "contact_xy_mm": np.asarray([[0, 0], [1, 0], [0, 1]], np.float64),
        "positions_E": np.asarray([[0, 0], [1, 1]], np.float32),
        "h": np.asarray([0.25, 0.75], np.float32),
        "delta_vtheta": np.asarray([-0.1, 0.1], np.float32),
        "edge_coefficients": np.zeros((2, 6), np.float64),
        "active_fraction": np.linspace(0, 1, n_active, dtype=np.float32),
        "active_fraction_bin_ms": np.asarray(1.0, np.float64),
        "contact_envelope": np.arange(3 * n_envelope, dtype=np.float32).reshape(
            3, n_envelope
        ),
        "contact_envelope_dt_ms": np.asarray(2.0, np.float64),
        "sheet_activity_counts": np.arange(
            n_sheet * 4, dtype=np.uint16
        ).reshape(n_sheet, 2, 2),
        "sheet_activity_frame_ms": np.asarray(2.0, np.float64),
        "event_t_on_ms": t_on,
        "event_t_off_ms": t_off,
        "event_returned": returned,
        "onsets": onsets,
        "ranks": ranks,
        "source_onset_maps_ms": source,
        "source_onset_evaluable": np.ones(n_events, bool),
        "event_fragment_count": np.ones(n_events, np.int16),
        "event_root_count": np.ones(n_events, np.int32),
        "event_directed_root_id": np.arange(n_events, dtype=np.int32),
    }


def _payload(seed, duration_ms, arrays, *, rev13, mapping="mapping-hash"):
    payload = {
        "status": (
            "REV13_NODE_ZERO_SUM_WORKER_COMPLETE"
            if rev13 else "REV12ND_NODE_WORKER_COMPLETE"
        ),
        "candidate_id": "exact_off" if rev13 else "stage-ak-primary",
        "field_sha256": "field-hash",
        "seed": seed,
        "simulation": {"duration_ms": float(duration_ms)},
        "events": _event_rows(arrays),
        "event_unit": {"causal_memory_ms": 100.0},
        "contact_readout": {
            "source": "lineage_restricted_neuron_activity",
            "spatial_sampler": "exact_normalized_per_neuron_gaussian",
        },
        "mechanism_freeze": {
            "EE": "off", "E_to_I": "off", "Z_M": "off",
            "edge_coefficients_all_zero": True,
        },
        "node_mapping": {"mapping_sha256": mapping},
        "arrays": {"path": "filled-after-write", "sha256": "filled-after-write"},
    }
    if rev13:
        payload["mechanism_freeze"].update({
            "node_accessibility": "exact_off",
            "node_accessibility_active": False,
        })
        payload["node_accessibility"] = {
            "enabled": False,
            "mode": "exact_off",
            "manifest": {"mode": "exact_off", "enabled": False},
            "support": None,
            "diagnostics": {},
        }
    return payload


def _write_pair(tmp_path, *, mutate=None, mapping="mapping-hash"):
    common_events = [(100.0, 150.0, True)]
    # This Stage-AK-only event crosses common_prefix - causal_memory and must
    # not be treated as a parity failure.
    boundary_event = (1850.0, 1950.0, True)
    rev13_arrays = _base_arrays(2000.0, common_events)
    stage_arrays = _base_arrays(4000.0, common_events + [boundary_event])
    for field in ("active_fraction", "contact_envelope", "sheet_activity_counts"):
        rev = rev13_arrays[field]
        stage = stage_arrays[field]
        time_axis = 1 if field == "contact_envelope" else 0
        prefix = [slice(None)] * stage.ndim
        prefix[time_axis] = slice(0, rev.shape[time_axis])
        stage[tuple(prefix)] = rev
    # Event 0 is identical even though Stage-AK has a future boundary event.
    for field in (
        "onsets", "ranks", "source_onset_maps_ms", "source_onset_evaluable",
        "event_fragment_count", "event_root_count", "event_directed_root_id",
    ):
        stage_arrays[field][0] = rev13_arrays[field][0]
    if mutate is not None:
        field, index = mutate
        rev13_arrays[field][index] += 1

    rev13_arrays.update({
        "node_accessibility_enabled": np.asarray(False),
        "node_accessibility_mode": np.asarray("exact_off"),
        "node_accessibility_state_final_mV": np.asarray([], np.float64),
        "node_accessibility_trace_time_ms": np.asarray([], np.float64),
        "node_accessibility_threshold_time_ms": np.asarray([], np.float64),
    })
    paths = {}
    for label, duration, arrays, is_rev13 in (
        ("rev13", 2000.0, rev13_arrays, True),
        ("stage", 4000.0, stage_arrays, False),
    ):
        npz = tmp_path / f"{label}.npz"
        json_path = tmp_path / f"{label}.json"
        np.savez(npz, **arrays)
        payload = _payload(
            2311, duration, arrays, rev13=is_rev13,
            mapping=(mapping if is_rev13 else "mapping-hash"),
        )
        payload["arrays"] = {"path": str(npz), "sha256": _sha256(npz)}
        json_path.write_text(json.dumps(payload))
        paths[f"{label}_json"] = json_path
        paths[f"{label}_npz"] = npz
    return paths


def _audit(paths):
    return audit_exact_off_parity(
        paths["rev13_json"], paths["rev13_npz"],
        paths["stage_json"], paths["stage_npz"],
    )


def test_boundary_truncated_stage_event_is_not_a_false_parity_failure(tmp_path):
    result = _audit(_write_pair(tmp_path))
    assert result["status"] == "PASS"
    assert result["common_prefix_ms"] == 2000.0
    assert result["event_right_boundary_ms"] == 1900.0
    assert result["n_comparable_events"] == {"rev13": 1, "stage_ak": 1}


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("active_fraction", (123,)),
        ("contact_envelope", (1, 123)),
        ("sheet_activity_counts", (123, 1, 1)),
    ],
)
def test_any_single_signal_sample_change_fails(tmp_path, field, index):
    result = _audit(_write_pair(tmp_path, mutate=(field, index)))
    assert result["status"] == "FAIL"
    assert f"prefix_signal_{field}" in result["failed_checks"]


def test_mapping_hash_mismatch_fails(tmp_path):
    result = _audit(_write_pair(tmp_path, mapping="changed-mapping"))
    assert result["status"] == "FAIL"
    assert "identity_mapping_sha256" in result["failed_checks"]
