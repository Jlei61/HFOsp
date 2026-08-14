import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.topic4_fcxr_lc6_surround import EToIGraph, graph_sha256


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/finalize_topic4_fcxr_lc6a_graph_family.py"
SPEC = importlib.util.spec_from_file_location("lc6a_graph_finalizer", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def _artifact(path, *, embedded_hash=None):
    graph = EToIGraph(
        np.array([[0, 1]], np.int32),
        np.ones((1, 2)),
        np.ones((1, 2), np.int32),
    )
    digest = graph_sha256(graph)
    metadata = {"graph_sha256": embedded_hash or digest, "graph_legality": "PASS"}
    np.savez_compressed(
        path,
        sources=graph.sources,
        weights=graph.weights,
        delay_steps=graph.delay_steps,
        graph_sha256=np.asarray([digest]),
        metadata_json=np.asarray([json.dumps(metadata)]),
    )
    return graph


def test_verified_loader_rejects_embedded_hash_drift(tmp_path):
    path = tmp_path / "graph.npz"
    _artifact(path, embedded_hash="wrong")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MOD._load_verified_graph(path)


def test_verified_loader_accepts_atomic_graph_artifact(tmp_path):
    path = tmp_path / "graph.npz"
    expected = _artifact(path)
    actual, metadata, digest = MOD._load_verified_graph(path)
    assert graph_sha256(actual) == graph_sha256(expected) == digest
    assert metadata["graph_legality"] == "PASS"


def test_finalizer_is_graph_only_and_requires_all_five_conditions():
    source = SCRIPT.read_text()
    assert MOD.GRAPH_IDS == ("C0", "C1", "Q1", "Q2", "Q3")
    assert "trajectory_outcome_read" in source
    assert "run_fcxr_loop" not in source
    assert "DONE_LC6A_GRAPH_Q1" not in source
    assert 'f"DONE_LC6A_GRAPH_{condition}.json"' in source
