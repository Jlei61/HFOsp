import importlib.util
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc5 import SparseSpikeStream


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_extension", ROOT / "scripts/run_topic4_fcxr_lc5v2p1_candidate_extension.py"
)
EXT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXT)


def test_full_stream_append_preserves_cell_and_step_order(monkeypatch):
    monkeypatch.setattr(EXT.U2, "CHUNK_MS", 1.0)
    monkeypatch.setattr(EXT.U2, "DT_MS", 0.5)
    original = SparseSpikeStream([0, 3], [1, 2], 4, 3)
    chunks = [
        SparseSpikeStream([0, 1], [0, 1], 2, 3),
        SparseSpikeStream([1], [2], 2, 3),
    ]
    full = EXT._combine_full(original, chunks)
    assert full.n_steps == 8
    assert full.steps.tolist() == [0, 3, 4, 5, 7]
    assert full.cells.tolist() == [1, 2, 0, 1, 2]


def test_rate_reconstruction_uses_per_cell_hz(monkeypatch):
    monkeypatch.setattr(EXT.U2, "DT_MS", 0.5)
    stream = SparseSpikeStream([0, 0, 1], [0, 1, 1], 2, 4)
    assert np.allclose(EXT._rate_from_stream(stream), [1000.0, 500.0])


def test_chunk_mean_rate_matches_population_spike_rate(monkeypatch):
    monkeypatch.setattr(EXT.U2, "DT_MS", 0.5)
    stream = SparseSpikeStream([0, 0, 1], [0, 1, 1], 2, 4)
    assert EXT.chunk_mean_rate_hz(stream) == 750.0


def test_continuation_schedule_accepts_event_aligned_18s_source(monkeypatch):
    monkeypatch.setattr(EXT.U2, "CHUNK_MS", 1000.0)
    target, continuation = EXT.continuation_schedule(18000.0, 11000.0)
    assert target == 31000.0
    assert continuation == 13000.0


def test_continuation_schedule_preserves_legacy_25s_case(monkeypatch):
    monkeypatch.setattr(EXT.U2, "CHUNK_MS", 1000.0)
    target, continuation = EXT.continuation_schedule(25000.0, 23000.0)
    assert target == 43000.0
    assert continuation == 18000.0


def test_continuation_schedule_accepts_locked_total_horizon(monkeypatch):
    monkeypatch.setattr(EXT.U2, "CHUNK_MS", 1000.0)
    target, continuation = EXT.continuation_schedule(
        25000.0, 23000.0, target_total_ms=40000.0,
    )
    assert target == 40000.0
    assert continuation == 15000.0


def test_lc6a_manifest_locks_source_and_horizon():
    manifest, payload, contract, source = EXT._manifest_contract(
        ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json"
    )
    assert manifest.name == "topic4_fcxr_lc6a_patient_axis_surround.json"
    assert payload["graph_family"][2]["id"] == "Q1"
    assert contract["target_total_ms"] == 40000.0
    assert source.name == "summary.json"


def test_locked_config_uses_summary_not_historical_sensor_path():
    summary = {
        "Imax": 3.0,
        "a_load": 0.01,
        "tau_ms": 15000.0,
        "config_scalar": {
            "pump_Imax": 3.0,
            "pump_a_load": 0.01,
            "pump_tau_ms": 15000.0,
            "y_gate": 76.5,
        },
    }
    cfg = EXT._locked_config_from_summary(summary, np.asarray([0.1, 0.2]), 2)
    assert cfg["y_gate"] == 76.5
    assert cfg["pump_p0_E"].tolist() == [0.1, 0.2]
    assert cfg["pump_u_init_E"].tolist() == [0.0, 0.0]
    assert cfg["x_relay_frozen_E"].tolist() == [1.0, 1.0]
