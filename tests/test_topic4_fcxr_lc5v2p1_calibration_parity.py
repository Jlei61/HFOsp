import importlib.util
from pathlib import Path

import numpy as np

from src.topic4_fcxr_lc5 import SparseSpikeStream


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "lc5v2p1_parity", ROOT / "scripts/audit_topic4_fcxr_lc5v2p1_calibration_parity.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_subset_stream_remaps_selected_cells_and_stops_exactly():
    stream = SparseSpikeStream(
        np.asarray([0, 1, 2, 3]), np.asarray([4, 2, 4, 1]), 10, 5
    )
    got = AUDIT._subset_stream(stream, [2, 4], stop_ms=0.15, source_dt_ms=0.05)
    assert got.n_steps == 3
    assert got.n_cells == 2
    assert got.steps.tolist() == [0, 1, 2]
    assert got.cells.tolist() == [1, 0, 1]


def test_replay_is_exact_when_resolution_is_the_same(monkeypatch):
    monkeypatch.setattr(AUDIT, "BASELINE_MS", (0.0, 5.0))
    monkeypatch.setattr(AUDIT, "EARLY_MS", (5.0, 10.0))
    monkeypatch.setattr(AUDIT, "SAMPLE_MS", 1.0)
    stream = SparseSpikeStream(
        np.asarray([1, 4, 7]), np.asarray([0, 1, 0]), 10, 2
    )
    a = AUDIT._sampled_replay(stream, dt_ms=1.0, tau_ms=3000.0, a_load=0.1)
    b = AUDIT._sampled_replay(stream, dt_ms=1.0, tau_ms=3000.0, a_load=0.1)
    assert np.array_equal(a["p0"], b["p0"])
    assert np.array_equal(a["excess_integral_ms"], b["excess_integral_ms"])
